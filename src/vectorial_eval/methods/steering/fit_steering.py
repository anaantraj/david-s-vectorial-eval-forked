"""Fit activation-steering vectors from train-split posts. Runs on the cluster.

Formulation
-----------
For a post `x`, let `h_i(x)` be the residual-stream state at hidden-state index
`i`, mean-pooled over the tokens of the post body only. The prompt tokens are
excluded from the pool, so what is measured is how the model represents the post
text itself rather than how it represents the instruction.

The steering vector at layer `i` is the difference of means

    v_i = mean_{x in Reddit_train} h_i(x) - mean_{x in LinkedIn_train} h_i(x)

computed over authentic posts. There is no gradient step anywhere in this file;
the whole fit is one forward pass per train post, which is why this method
produces numbers first.

Two scopes are fitted in the same pass. The global vector uses every train post
of each platform. A per-cell vector uses only the posts of one bilateral cell,
and is written only for cells that have at least `--min-cell-posts` posts on both
sides, because a difference of means over two posts is noise. Coverage is
recorded in the artifact rather than hidden.

The context each post is read under
-----------------------------------
The post is embedded in `TARGET_LM_PROMPT`, rendered for the post's own platform,
exactly as the soft-prompt and LoRA runs render their training examples. Three
variants exist and are selected with `--variant`:

    target_lm         room, domain, topic, platform
    target_lm_paired  the above plus the most similar same-cell post from the
                      other platform, retrieved by TF-IDF within the train split
    target_lm_aspect  the above plus the aspects the post foregrounds

The prompt is imported from `build_training`, never retyped, because a method
that formats its context differently from the others is no longer comparable
with them.

Leakage
-------
Only `data/posts.train.jsonl` is read, every record's `split` field is asserted
to be `train`, and the retrieval used by the paired variant is restricted to
train-split posts on both sides. The dev and test files are never opened here.

Resuming
--------
The state is a set of running sums over posts, so a kill costs at most the posts
since the last checkpoint. `resume_or_start` restores the sums and the index of
the next post, and the post order is a deterministic sort by `post_id`, so a
resumed run consumes exactly the posts a fresh run would have.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts" / "cluster"))

from vectorial_eval.config import DEFAULT_DATA_DIR  # noqa: E402
from vectorial_eval.data.build_training import (  # noqa: E402
    read_post_aspects,
    read_posts,
    render_aspect_prompt,
    render_prompt,
)
from vectorial_eval.methods.lora.prompting import wrap_prompt  # noqa: E402
from vectorial_eval.methods.steering.runtime import ARTIFACT_FORMAT  # noqa: E402

log = logging.getLogger("fit_steering")

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
VARIANTS = ("target_lm", "target_lm_paired", "target_lm_aspect")
TARGET_PLATFORM = "reddit"
SOURCE_PLATFORM = "linkedin"


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #


def build_examples(data_dir: Path, variant: str) -> tuple[list[dict], dict]:
    """Return the fitting examples, sorted by post_id, plus a provenance dict.

    Every example is `{post_id, cell_id, room, platform, prompt, completion}`. The
    completion is the authentic post text; the prompt is the shared rendering.
    """
    posts = read_posts(data_dir, splits=("train",))
    off_split = [p.post_id for p in posts if p.split != "train"]
    if off_split:
        raise RuntimeError(
            f"posts.train.jsonl contains {len(off_split)} non-train records; refusing to fit"
        )

    source_map: dict[str, str | None] = {}
    if variant == "target_lm_paired":
        from vectorial_eval.data.build_training import pair_by_retrieval

        for tgt_pf, src_pf in ((TARGET_PLATFORM, SOURCE_PLATFORM), (SOURCE_PLATFORM, TARGET_PLATFORM)):
            targets = [p for p in posts if p.platform == tgt_pf]
            sources = [p for p in posts if p.platform == src_pf]
            source_map.update(pair_by_retrieval(targets, sources))

    aspects: dict[str, list[tuple[str, float]]] = {}
    if variant == "target_lm_aspect":
        path = data_dir / "aspects" / "aspect_embeddings.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; the aspect variant needs it")
        aspects = read_post_aspects(path)

    by_id = {p.post_id: p for p in posts}
    examples: list[dict] = []
    dropped = {"no_aspects": 0, "no_pair": 0}
    for post in sorted(posts, key=lambda p: p.post_id):
        if variant == "target_lm_aspect":
            got = aspects.get(post.post_id) or []
            if not got:
                dropped["no_aspects"] += 1
                continue
            prompt = render_aspect_prompt(
                room=post.room,
                topic=post.topic,
                domain=post.domain,
                target_platform=post.platform,
                aspects=got,
            )
        elif variant == "target_lm_paired":
            src_id = source_map.get(post.post_id)
            src = by_id.get(src_id) if src_id else None
            if src is None:
                dropped["no_pair"] += 1
            prompt = render_prompt(
                room=post.room,
                topic=post.topic,
                domain=post.domain,
                target_platform=post.platform,
                source_text=src.text if src else None,
                source_platform=src.platform if src else None,
            )
        else:
            prompt = render_prompt(
                room=post.room,
                topic=post.topic,
                domain=post.domain,
                target_platform=post.platform,
            )
        examples.append(
            {
                "post_id": post.post_id,
                "cell_id": post.cell_id,
                "room": post.room,
                "platform": post.platform,
                "prompt": prompt,
                "completion": post.text,
            }
        )
    provenance = {
        "n_examples": len(examples),
        "n_target": sum(1 for e in examples if e["platform"] == TARGET_PLATFORM),
        "n_source": sum(1 for e in examples if e["platform"] == SOURCE_PLATFORM),
        "dropped": dropped,
        "splits_read": ["train"],
    }
    return examples, provenance


# --------------------------------------------------------------------------- #
# accumulator state
# --------------------------------------------------------------------------- #


class State:
    """Running sums of pooled activations, checkpointed as a whole."""

    def __init__(self, n_layers: int, hidden: int):
        import torch

        self.n_layers = n_layers
        self.hidden = hidden
        self.index = 0
        self.sums: dict[str, torch.Tensor] = {
            TARGET_PLATFORM: torch.zeros(n_layers, hidden, dtype=torch.float64),
            SOURCE_PLATFORM: torch.zeros(n_layers, hidden, dtype=torch.float64),
        }
        self.counts: dict[str, int] = {TARGET_PLATFORM: 0, SOURCE_PLATFORM: 0}
        self.cell_sums: dict[str, torch.Tensor] = {}
        self.cell_counts: dict[str, int] = defaultdict(int)
        self.audience_sums: dict[str, torch.Tensor] = {}
        self.audience_counts: dict[str, int] = defaultdict(int)
        self.norm_sum = torch.zeros(n_layers, dtype=torch.float64)
        self.norm_count = 0
        self.skipped: dict[str, int] = defaultdict(int)

    def add(self, platform: str, cell_id: str, room: str, pooled, norms) -> None:
        import torch

        self.sums[platform] += pooled
        self.counts[platform] += 1
        key = f"{cell_id}\t{platform}"
        if key not in self.cell_sums:
            self.cell_sums[key] = torch.zeros(self.n_layers, self.hidden, dtype=torch.float64)
        self.cell_sums[key] += pooled
        self.cell_counts[key] += 1
        audience_key = f"{room}\t{platform}"
        if audience_key not in self.audience_sums:
            self.audience_sums[audience_key] = torch.zeros(
                self.n_layers, self.hidden, dtype=torch.float64
            )
        self.audience_sums[audience_key] += pooled
        self.audience_counts[audience_key] += 1
        self.norm_sum += norms
        self.norm_count += 1

    def payload(self) -> dict:
        return {
            "index": self.index,
            "n_layers": self.n_layers,
            "hidden": self.hidden,
            "sums": self.sums,
            "counts": self.counts,
            "cell_sums": self.cell_sums,
            "cell_counts": dict(self.cell_counts),
            "audience_sums": self.audience_sums,
            "audience_counts": dict(self.audience_counts),
            "norm_sum": self.norm_sum,
            "norm_count": self.norm_count,
            "skipped": dict(self.skipped),
        }

    def restore(self, payload: dict) -> None:
        self.index = int(payload["index"])
        self.n_layers = int(payload["n_layers"])
        self.hidden = int(payload["hidden"])
        self.sums = payload["sums"]
        self.counts = dict(payload["counts"])
        self.cell_sums = payload["cell_sums"]
        self.cell_counts = defaultdict(int, payload["cell_counts"])
        self.audience_sums = payload.get("audience_sums", {})
        self.audience_counts = defaultdict(int, payload.get("audience_counts", {}))
        self.norm_sum = payload["norm_sum"]
        self.norm_count = int(payload["norm_count"])
        self.skipped = defaultdict(int, payload.get("skipped", {}))


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #


def load_model(
    model_id: str, load_in_4bit: bool, initial_adapter: str | None = None
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs: dict = {"dtype": torch.bfloat16, "device_map": {"": 0}}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if initial_adapter:
        from peft import PeftModel

        adapter = Path(initial_adapter)
        if not (adapter / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"initial adapter {adapter} has no adapter_config.json"
            )
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.eval()
    return model, tok


def pooled_activations(model, tok, prompt: str, completion: str, max_length: int):
    """Mean-pool the residual stream over the completion tokens only.

    Returns `(pooled[L+1, H] float64 cpu, norms[L+1] float64 cpu)` or None when
    truncation leaves no completion token to pool over.
    """
    import torch

    ids_p = tok(wrap_prompt(tok, prompt), add_special_tokens=True).input_ids
    ids_c = tok(completion, add_special_tokens=False).input_ids
    ids = (ids_p + ids_c)[:max_length]
    start = len(ids_p)
    if start >= len(ids):
        return None
    input_ids = torch.tensor([ids], device=model.device)
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    stacked = torch.stack(out.hidden_states, dim=0)[:, 0]  # [L+1, T, H]
    body = stacked[:, start:, :].to(torch.float32)
    pooled = body.mean(dim=1)
    norms = body.norm(dim=-1).mean(dim=1)
    return pooled.double().cpu(), norms.double().cpu()


# --------------------------------------------------------------------------- #
# artifact
# --------------------------------------------------------------------------- #


def finalise(state: State, args, provenance: dict) -> dict:
    import torch

    counts = state.counts
    if counts[TARGET_PLATFORM] == 0 or counts[SOURCE_PLATFORM] == 0:
        raise RuntimeError(
            "one side of the contrast is empty "
            f"({counts}); a difference of means is not defined"
        )
    tgt_mean = state.sums[TARGET_PLATFORM] / counts[TARGET_PLATFORM]
    src_mean = state.sums[SOURCE_PLATFORM] / counts[SOURCE_PLATFORM]
    global_vec = (tgt_mean - src_mean).float()

    # Decompose platform movement from target-platform audience movement.  The
    # audience residual is Gram-Schmidt orthogonal to the platform direction at
    # every layer, which is the two-vector form of the SVD normalization called
    # for in the Study 4 protocol.  Singular values and the original overlap are
    # retained so correlation is inspectable rather than hidden.
    per_audience: dict[str, dict] = {}
    audience_ids = sorted(
        {key.rsplit("\t", 1)[0] for key in state.audience_sums}
    )
    thin_audiences: list[str] = []
    for audience in audience_ids:
        key = f"{audience}\t{TARGET_PLATFORM}"
        n_target = state.audience_counts.get(key, 0)
        if n_target < args.min_audience_posts:
            thin_audiences.append(audience)
            continue
        raw = (state.audience_sums[key] / n_target).float() - tgt_mean.float()
        p_unit = global_vec / global_vec.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        overlap = (raw * p_unit).sum(dim=-1, keepdim=True)
        residual = raw - overlap * p_unit
        raw_unit = raw / raw.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        cosine = (raw_unit * p_unit).sum(dim=-1)
        singular_values = []
        for layer in range(state.n_layers):
            pair = torch.stack((p_unit[layer], raw_unit[layer]), dim=0)
            singular_values.append(torch.linalg.svdvals(pair).tolist())
        per_audience[audience] = {
            "vector": residual,
            "raw_vector": raw,
            "n_target": n_target,
            "cosine_to_platform": cosine.tolist(),
            "projection_on_platform": overlap.squeeze(-1).tolist(),
            "singular_values": singular_values,
            "basis": "platform unit vector followed by Gram-Schmidt audience residual",
        }

    per_cell: dict[str, dict] = {}
    cell_ids = sorted({k.split("\t")[0] for k in state.cell_sums})
    thin: list[str] = []
    for cell_id in cell_ids:
        tk, sk = f"{cell_id}\t{TARGET_PLATFORM}", f"{cell_id}\t{SOURCE_PLATFORM}"
        n_t, n_s = state.cell_counts.get(tk, 0), state.cell_counts.get(sk, 0)
        if n_t < args.min_cell_posts or n_s < args.min_cell_posts:
            thin.append(cell_id)
            continue
        vec = (state.cell_sums[tk] / n_t) - (state.cell_sums[sk] / n_s)
        per_cell[cell_id] = {
            "vector": vec.float(),
            "n_target": n_t,
            "n_source": n_s,
        }

    act_norm = (state.norm_sum / max(1, state.norm_count)).float()

    # A per-cell vector is only worth using if it points somewhere the global
    # vector does not; recording the agreement makes that checkable rather than
    # assumed.
    def cos(a, b):
        na, nb = a.norm(dim=-1), b.norm(dim=-1)
        return ((a * b).sum(-1) / (na * nb).clamp_min(1e-9)).tolist()

    artifact = {
        "format": ARTIFACT_FORMAT,
        "base_model": args.model,
        "initial_adapter": args.initial_adapter or None,
        "variant": args.variant,
        "layers": list(range(state.n_layers)),
        "hidden_size": state.hidden,
        "global": {
            "vector": global_vec,
            "n_target": counts[TARGET_PLATFORM],
            "n_source": counts[SOURCE_PLATFORM],
        },
        "per_cell": per_cell,
        "per_audience": per_audience,
        "act_norm": act_norm,
        "counts": {
            **provenance,
            "n_read": state.index,
            "n_pooled": counts[TARGET_PLATFORM] + counts[SOURCE_PLATFORM],
            "n_target_pooled": counts[TARGET_PLATFORM],
            "n_source_pooled": counts[SOURCE_PLATFORM],
            "skipped": dict(state.skipped),
            "n_cells_seen": len(cell_ids),
            "n_cells_with_vector": len(per_cell),
            "n_audiences_seen": len(audience_ids),
            "n_audiences_with_vector": len(per_audience),
            "audiences_too_thin": thin_audiences,
            "min_audience_posts": args.min_audience_posts,
            "per_audience_posts": {
                audience: per_audience[audience]["n_target"]
                for audience in sorted(per_audience)
            },
            "cells_too_thin": thin,
            "min_cell_posts": args.min_cell_posts,
            "per_cell_posts": {
                cid: {"target": per_cell[cid]["n_target"], "source": per_cell[cid]["n_source"]}
                for cid in sorted(per_cell)
            },
        },
        "diagnostics": {
            "global_vector_norm_per_layer": global_vec.norm(dim=-1).tolist(),
            "act_norm_per_layer": act_norm.tolist(),
            "relative_shift_per_layer": (
                global_vec.norm(dim=-1) / act_norm.clamp_min(1e-9)
            ).tolist(),
            "per_cell_cosine_to_global": {
                cid: cos(per_cell[cid]["vector"], global_vec) for cid in sorted(per_cell)
            },
            "audience_cosine_to_platform": {
                audience: per_audience[audience]["cosine_to_platform"]
                for audience in sorted(per_audience)
            },
        },
        "fit_config": {
            "model": args.model,
            "initial_adapter": args.initial_adapter or None,
            "load_in_4bit": bool(args.load_in_4bit),
            "max_length": args.max_length,
            "variant": args.variant,
            "min_cell_posts": args.min_cell_posts,
            "min_audience_posts": args.min_audience_posts,
            "orthogonalisation": "per-layer Gram-Schmidt; pair singular values recorded",
            "pooling": "mean over completion tokens, prompt tokens excluded",
            "prompt": "render_prompt followed by lora.prompting.wrap_prompt",
            "data_dir": str(args.data_dir),
            "torch_version": torch.__version__,
        },
    }
    return artifact


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    global SOURCE_PLATFORM, TARGET_PLATFORM
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", default="target_lm", choices=VARIANTS)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument(
        "--initial-adapter",
        default="",
        help="frozen Stage A LoRA whose activation space defines the vectors",
    )
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out", type=Path, default=None,
                   help="artifact path; default runs/steering/steering.<variant>.pt")
    p.add_argument("--max-length", type=int, default=768,
                   help="matches docs/09; the 99th percentile of post length fits")
    p.add_argument("--min-cell-posts", type=int, default=3,
                   help="a cell needs this many posts on BOTH platforms for its own vector")
    p.add_argument("--min-audience-posts", type=int, default=10,
                   help="target-platform posts required for an audience residual")
    p.add_argument("--save-every", type=int, default=25, help="checkpoint every N posts")
    p.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true", default=True)
    p.add_argument("--bf16", dest="load_in_4bit", action="store_false",
                   help="ablation: read activations in bf16 instead of 4-bit NF4")
    p.add_argument("--limit", type=int, default=0, help="smoke run over the first N posts")
    p.add_argument("--ckpt-dir", default=os.environ.get("VECTORIAL_CKPT_DIR", ""))
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    from checkpointing import resume_or_start, save_checkpoint

    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    direction = manifest.get("direction", {})
    SOURCE_PLATFORM = direction.get("source", SOURCE_PLATFORM)
    TARGET_PLATFORM = direction.get("target", TARGET_PLATFORM)
    if {SOURCE_PLATFORM, TARGET_PLATFORM} != {"linkedin", "reddit"}:
        raise ValueError(f"unsupported direction: {direction}")

    root = Path(__file__).resolve().parents[4]
    tag = args.variant + ("" if args.load_in_4bit else ".bf16")
    out = args.out or (root / "runs" / "steering" / f"steering.{tag}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.ckpt_dir or str(root / "runs" / "checkpoints" / f"steering_{tag}")

    examples, provenance = build_examples(args.data_dir, args.variant)
    if args.limit:
        # A smoke run must still see both platforms, since a one-sided contrast
        # is not defined. Take the first half of the budget from each side.
        half = max(1, args.limit // 2)
        picked = [e for e in examples if e["platform"] == TARGET_PLATFORM][:half]
        picked += [e for e in examples if e["platform"] == SOURCE_PLATFORM][:half]
        examples = sorted(picked, key=lambda e: e["post_id"])
        provenance["limit"] = args.limit
        provenance["n_examples"] = len(examples)
        provenance["n_target"] = sum(1 for e in examples if e["platform"] == TARGET_PLATFORM)
        provenance["n_source"] = sum(1 for e in examples if e["platform"] == SOURCE_PLATFORM)
    log.info(
        "variant=%s examples=%d (target=%d source=%d) out=%s",
        args.variant, len(examples), provenance["n_target"], provenance["n_source"], out,
    )

    model, tok = load_model(
        args.model, args.load_in_4bit, args.initial_adapter or None
    )
    n_layers = model.config.num_hidden_layers + 1
    hidden = model.config.hidden_size
    state = State(n_layers, hidden)

    def restore(ckpt):
        state.restore(ckpt.payload)

    start_step, _ = resume_or_start(ckpt_dir, restore=restore)
    state.index = start_step
    log.info("start_step=%d of %d posts", start_step, len(examples))

    t0 = time.time()
    for i in range(start_step, len(examples)):
        ex = examples[i]
        try:
            got = pooled_activations(model, tok, ex["prompt"], ex["completion"], args.max_length)
            reason = None if got is not None else "no_completion_tokens"
        except Exception as exc:  # noqa: BLE001 - one bad post must not end the fit
            log.warning("post %s failed: %s", ex["post_id"], exc)
            got, reason = None, "forward_error"
        if got is None:
            state.skipped[reason] += 1
        else:
            pooled, norms = got
            state.add(ex["platform"], ex["cell_id"], ex["room"], pooled, norms)
        state.index = i + 1
        if state.index % args.save_every == 0 or state.index == len(examples):
            save_checkpoint(
                ckpt_dir,
                state.index,
                payload=state.payload(),
                meta={
                    "n_target": state.counts[TARGET_PLATFORM],
                    "n_source": state.counts[SOURCE_PLATFORM],
                    "elapsed_s": round(time.time() - t0, 1),
                },
                keep_last=2,
            )
            log.info(
                "step %d/%d target=%d source=%d %.1fs",
                state.index, len(examples), state.counts[TARGET_PLATFORM],
                state.counts[SOURCE_PLATFORM], time.time() - t0,
            )

    import torch

    artifact = finalise(state, args, provenance)
    # The fit runs for minutes before reaching this point, so a missing output
    # directory here discards the whole run. Create it before writing.
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".pt.tmp")
    torch.save(artifact, tmp)
    os.replace(tmp, out)
    summary = {
        k: artifact[k] for k in ("format", "base_model", "variant", "hidden_size", "counts")
    }
    summary["diagnostics"] = {
        "relative_shift_per_layer": artifact["diagnostics"]["relative_shift_per_layer"],
    }
    (out.with_suffix(".json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("wrote %s (%.1f MB)", out, out.stat().st_size / 1e6)
    log.info(
        "cells with own vector: %d of %d seen",
        artifact["counts"]["n_cells_with_vector"], artifact["counts"]["n_cells_seen"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
