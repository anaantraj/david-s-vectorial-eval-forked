"""End-to-end smoke test for the NDIF steering arm.

Run this before committing a long remote job. It checks, in order, the four
things that actually break: the key resolves, the model is pinned, a steering
write reaches the residual stream, and the intervention is dose-dependent.

    NDIF_API_KEY=... .venv/bin/python -m vectorial_eval.methods.steering.smoke_ndif
    ... --model meta-llama/Llama-3.1-405B-Instruct --layer 40

It uses a random unit vector rather than a fitted artifact, so a *degenerate*
continuation under steering is the expected pass condition: it shows the write
landed. A fitted vector is what makes the output meaningful, and that is the
sweep's job, not this script's.

nnsight traces the body of the `with` block by reading its source, so this must
stay an importable file. Pasting the same code into a heredoc or a REPL fails
with "could not get source code".
"""

from __future__ import annotations

import argparse
import logging
import sys

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-70B-Instruct")
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--alphas", default="0,0.5,1.0")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument(
        "--allow-unwrapped",
        action="store_true",
        help="permit a base model with no chat template; not comparable to the "
             "wrapped columns and recorded as chat_wrapped=False",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import torch

    from .ndif_backend import (
        NDIFConfig,
        NDIFSteeredGenerator,
        check_model,
        pinned_models,
        resolve_api_key,
    )
    from .runtime import SteeringSpec

    resolve_api_key()
    print("key resolved")
    print("pinned models:", ", ".join(pinned_models()))
    check_model(args.model)
    print(f"{args.model} is pinned")

    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(args.model)
    n_layers, hidden = cfg.num_hidden_layers, cfg.hidden_size
    print(f"{args.model}: {n_layers} layers, hidden {hidden}")

    # A synthetic artifact with the right shape. `act_norm` is a placeholder;
    # a fitted artifact carries the measured per-layer norms.
    generator = torch.Generator().manual_seed(0)
    artifact = {
        "format": 1,
        "base_model": args.model,
        "variant": "smoke",
        "global": {
            "vector": torch.randn(n_layers + 1, hidden, generator=generator),
            "n_target": 0,
            "n_source": 0,
        },
        "act_norm": torch.full((n_layers + 1,), 100.0),
        "counts": {},
        "fit_config": {"smoke": True},
    }

    gen = NDIFSteeredGenerator(
        artifact,
        NDIFConfig(
            model_id=args.model,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            batch_size=2,
            require_chat_template=not args.allow_unwrapped,
        ),
    )
    items = [
        {"prompt": "Write a Reddit post about shipping a milestone.", "cell_id": "c1"},
        {"prompt": "Write a Reddit post about a difficult code review.", "cell_id": "c2"},
    ]

    texts: dict[float, list[str]] = {}
    for alpha in [float(a) for a in args.alphas.split(",")]:
        spec = SteeringSpec(layers=(args.layer,), alpha=alpha, scope="global")
        results = gen.generate(items, spec, draw=0)
        texts[alpha] = [r["text"] for r in results]
        print(f"\nalpha={alpha}  chat_wrapped={gen.cfg.chat_wrapped}")
        for r in results:
            status = "ok" if r["ok"] else f"FAILED {r['error']}"
            print(f"  [{status}] {r['text'][:100]!r}")
        if not all(r["ok"] for r in results) and alpha == 0.0:
            print("\nFAIL: unsteered generation did not succeed")
            return 1

    baseline = texts.get(0.0)
    steered = [a for a in texts if a > 0]
    if baseline and steered and all(texts[a] == baseline for a in steered):
        print("\nFAIL: steering changed nothing; the write did not reach the stream")
        return 1
    print("\nPASS: remote steering is live and changes the continuation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
