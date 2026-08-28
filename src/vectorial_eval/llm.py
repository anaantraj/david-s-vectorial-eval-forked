"""Thin, cached LLM client shared by transfer functions and the judge.

Design notes
------------
* **One client for every model.** OpenRouter exposes Claude and GPT behind the
  same OpenAI-compatible endpoint, so the "Claude rewrite vs GPT rewrite"
  comparison the meeting asked for differs only in a model string. Two vendor
  SDKs would introduce differences in retry, sampling defaults, and message
  formatting that would confound that comparison.
* **Disk cache keyed on the full request.** Generation is the expensive step and
  metrics are the step you iterate on. Caching lets you fix a metric bug and
  re-score without re-paying for the transfer outputs.
* **Failures are values, not exceptions.** A refusal or a persistent API error
  returns a result with `ok=False` and is recorded in the run, because a
  transfer function that silently drops 5% of a cell would bias every
  distributional metric toward whatever it happened to keep.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .config import LLMConfig

log = logging.getLogger(__name__)

PROVIDER_ENDPOINTS = {
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "openai": (None, "OPENAI_API_KEY"),
    "anthropic": ("https://api.anthropic.com/v1", "ANTHROPIC_API_KEY"),
}


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader so the harness works without extra dependencies.

    Existing environment variables win, which keeps CI and shell overrides
    authoritative over a checked-out file.
    """
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class LLMResult:
    text: str
    ok: bool = True
    error: str | None = None
    cached: bool = False
    usage: dict | None = None
    model: str | None = None


class LLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._cache_dir = Path(cfg.cache_dir) if cfg.cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = None

    # -- client ---------------------------------------------------------
    def _ensure_client(self):
        if self._client is not None:
            return self._client
        load_dotenv()
        if self.cfg.provider not in PROVIDER_ENDPOINTS:
            raise ValueError(f"Unknown provider {self.cfg.provider!r}")
        base_url, env_var = PROVIDER_ENDPOINTS[self.cfg.provider]
        api_key = os.environ.get(env_var)
        if not api_key:
            raise RuntimeError(
                f"{env_var} is not set. Add it to .env or export it before "
                f"running any LLM-backed transfer function or metric."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pip install openai") from exc
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        return self._client

    # -- cache ----------------------------------------------------------
    def _cache_key(self, system: str, user: str, variant: int = 0) -> str:
        payload = json.dumps(
            {
                "provider": self.cfg.provider,
                "model": self.cfg.model,
                "temperature": self.cfg.temperature,
                "max_tokens": self.cfg.max_tokens,
                "system": system,
                "user": user,
                # Multi-candidate sampling draws several completions from the
                # same prompt. Without the variant index every draw after the
                # first would be served from cache and the candidates would be
                # identical, which would defeat the purpose.
                "variant": variant,
            },
            sort_keys=True,
        )
        return hashlib.blake2b(payload.encode(), digest_size=16).hexdigest()

    def _cache_get(self, key: str) -> LLMResult | None:
        if not self._cache_dir:
            return None
        path = self._cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return LLMResult(**{**data, "cached": True})

    def _cache_put(self, key: str, result: LLMResult) -> None:
        if not self._cache_dir or not result.ok:
            return
        path = self._cache_dir / f"{key}.json"
        with self._lock:
            path.write_text(
                json.dumps(
                    {
                        "text": result.text,
                        "ok": result.ok,
                        "usage": result.usage,
                        "model": result.model,
                    }
                ),
                encoding="utf-8",
            )

    # -- generation -----------------------------------------------------
    def complete(self, system: str, user: str, variant: int = 0) -> LLMResult:
        key = self._cache_key(system, user, variant)
        hit = self._cache_get(key)
        if hit is not None:
            return hit

        client = self._ensure_client()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        last_error = "unknown"
        for attempt in range(self.cfg.max_retries):
            try:
                resp = client.chat.completions.create(
                    model=self.cfg.model,
                    messages=messages,
                    max_tokens=self.cfg.max_tokens,
                    temperature=self.cfg.temperature,
                )
                choice = resp.choices[0]
                text = (choice.message.content or "").strip()
                if not text:
                    # Empty content is usually a refusal or a truncated
                    # reasoning-only turn; both are failures for our purposes.
                    last_error = f"empty completion (finish_reason={choice.finish_reason})"
                    continue
                result = LLMResult(
                    text=text,
                    usage=resp.usage.model_dump() if getattr(resp, "usage", None) else None,
                    model=getattr(resp, "model", self.cfg.model),
                )
                self._cache_put(key, result)
                return result
            except Exception as exc:  # noqa: BLE001 - surfaced as a value below
                last_error = f"{type(exc).__name__}: {exc}"
                sleep = min(30.0, 2**attempt) * (0.5 + random.random())
                log.warning("LLM call failed (attempt %d): %s", attempt + 1, last_error)
                time.sleep(sleep)
        return LLMResult(text="", ok=False, error=last_error)

    def map(
        self,
        items: Sequence,
        build_prompt: Callable[[object], tuple[str, str]],
        progress: bool = True,
        variants: Sequence[int] | None = None,
    ) -> list[LLMResult]:
        """Run `complete` over items concurrently, preserving input order.

        `variants` gives a per-item draw index, used when several candidates are
        sampled from the same prompt.
        """
        results: list[LLMResult | None] = [None] * len(items)

        def run(i: int) -> None:
            system, user = build_prompt(items[i])
            results[i] = self.complete(system, user, variants[i] if variants else 0)

        with ThreadPoolExecutor(max_workers=self.cfg.max_concurrency) as pool:
            futures = [pool.submit(run, i) for i in range(len(items))]
            for n, fut in enumerate(futures, 1):
                fut.result()
                if progress and n % 25 == 0:
                    log.info("  %d/%d", n, len(items))
        return [r if r is not None else LLMResult("", ok=False, error="not run") for r in results]
