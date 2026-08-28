"""Tests for the activation-steering method.

These cover the parts that can be checked without a GPU: the artifact contract,
vector selection and scaling, the forward hook, and the transfer function's
obligation to emit one output per task per draw even when nothing can be loaded.
"""

from __future__ import annotations

import json

import pytest

from vectorial_eval.data.schema import TransferTask
from vectorial_eval.methods.steering.generate import clean_continuation
from vectorial_eval.methods.steering.runtime import (
    ARTIFACT_FORMAT,
    SteeringSpec,
    load_artifact,
    parse_layers,
    resolve_vector,
    steering_deltas,
)
from vectorial_eval.methods.steering.sweep_steering import select_pareto_point
from vectorial_eval.transfer import steering as _steering  # noqa: F401  (registers "steering")
from vectorial_eval.transfer.base import Corpus, build_transfer_fn

torch = pytest.importorskip("torch")

HIDDEN = 8
LAYERS = 4


def _artifact(tmp_path, with_cell=True):
    vec = torch.zeros(LAYERS, HIDDEN)
    vec[:, 0] = 2.0
    cell_vec = torch.zeros(LAYERS, HIDDEN)
    cell_vec[:, 1] = 3.0
    audience_vec = torch.zeros(LAYERS, HIDDEN)
    audience_vec[:, 1] = 4.0
    art = {
        "format": ARTIFACT_FORMAT,
        "base_model": "test/model",
        "initial_adapter": "/frozen/stage-a",
        "variant": "target_lm",
        "layers": list(range(LAYERS)),
        "hidden_size": HIDDEN,
        "global": {"vector": vec, "n_target": 337, "n_source": 293},
        "per_cell": ({"c1": {"vector": cell_vec, "n_target": 5, "n_source": 4}} if with_cell else {}),
        "per_audience": {
            "cto": {"vector": audience_vec, "n_target": 17}
        },
        "act_norm": torch.full((LAYERS,), 10.0),
        "counts": {"n_cells_with_vector": 1, "n_cells_seen": 2},
        "fit_config": {},
    }
    path = tmp_path / "steering.target_lm.pt"
    torch.save(art, path)
    return path


def _task(task_id="t1", cell_id="c1"):
    return TransferTask(
        task_id=task_id,
        cell_id=cell_id,
        room="cto",
        topic="ai",
        domain="Software Engineering",
        split="val",
        source_platform="linkedin",
        target_platform="reddit",
        source_post_id="p1",
        source_text="a source post",
        target_reference_ids=[],
        exemplar_ids=[],
        heldout_cell=False,
    )


def test_parse_layers():
    assert parse_layers(16) == (16,)
    assert parse_layers("16") == (16,)
    assert parse_layers("12,16,20") == (12, 16, 20)
    assert parse_layers("12-20:4") == (12, 16, 20)


def test_clean_continuation_cuts_at_a_new_example():
    text = "The real post.\nAudience: cto\nDomain: x"
    assert clean_continuation(text) == "The real post."


def test_load_artifact_rejects_a_foreign_format(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"format": 99, "global": {}, "act_norm": [], "base_model": "", "variant": ""}, path)
    with pytest.raises(ValueError, match="artifact format"):
        load_artifact(path)


def test_resolve_vector_prefers_the_cell_and_reports_its_post_count(tmp_path):
    art = load_artifact(_artifact(tmp_path))
    vec, prov = resolve_vector(art, "c1", SteeringSpec(layers=(1,), alpha=1.0, scope="cell"))
    assert prov["vector_scope"] == "cell"
    assert (prov["n_target_posts"], prov["n_source_posts"]) == (5, 4)
    assert float(vec[1, 1]) == 3.0


def test_resolve_vector_falls_back_to_global_and_says_so(tmp_path):
    art = load_artifact(_artifact(tmp_path))
    _, prov = resolve_vector(art, "missing", SteeringSpec(layers=(1,), alpha=1.0, scope="cell"))
    assert prov["vector_scope"] == "global"
    assert prov["fell_back"] is True
    assert prov["n_target_posts"] == 337


def test_decomposed_vector_combines_orthogonal_platform_and_audience(tmp_path):
    art = load_artifact(_artifact(tmp_path))
    spec = SteeringSpec(
        layers=(1,), alpha=1.0, scope="decomposed", audience_alpha=2.0
    )
    vec, prov = resolve_vector(art, None, spec, audience_id="cto")
    assert float(vec[1, 0]) == pytest.approx(1.0)
    assert float(vec[1, 1]) == pytest.approx(2.0)
    assert prov["orthogonalised"] is True
    assert prov["n_audience_posts"] == 17


def test_delta_is_alpha_times_the_activation_norm(tmp_path):
    """Alpha is a fraction of the residual norm, not of the raw difference."""
    art = load_artifact(_artifact(tmp_path))
    deltas = steering_deltas(art, art["global"]["vector"], SteeringSpec(layers=(2,), alpha=0.5))
    assert set(deltas) == {2}
    assert float(deltas[2].norm()) == pytest.approx(0.5 * 10.0)


def test_sweep_rejects_style_gain_that_loses_source_content():
    baseline = {
        "alpha": 0.0,
        "centroid_cos": 0.6,
        "source_cos": 0.8,
        "degeneracy": 0.0,
    }
    loses_content = {
        "alpha": 0.5,
        "centroid_cos": 0.7,
        "source_cos": 0.7,
        "degeneracy": 0.0,
    }
    improves_both = {
        "alpha": 1.0,
        "centroid_cos": 0.65,
        "source_cos": 0.81,
        "degeneracy": 0.0,
    }
    chosen, got_baseline, eligible = select_pareto_point(
        [baseline, loses_content, improves_both], 0.15
    )
    assert got_baseline is baseline
    assert chosen is improves_both
    assert eligible == [improves_both]


def test_steering_layer_must_be_in_range(tmp_path):
    art = load_artifact(_artifact(tmp_path))
    for bad in (0, LAYERS):
        with pytest.raises(ValueError, match="out of range"):
            steering_deltas(art, art["global"]["vector"], SteeringSpec(layers=(bad,), alpha=1.0))


def test_hook_adds_the_delta_and_removes_itself(tmp_path):
    """The residual stream is shifted only for the duration of the context."""
    from vectorial_eval.methods.steering.runtime import apply_steering

    class Block(torch.nn.Module):
        def forward(self, x):
            return x

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Block() for _ in range(LAYERS)])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Decoder()

    model = Model()
    art = load_artifact(_artifact(tmp_path))
    deltas = steering_deltas(art, art["global"]["vector"], SteeringSpec(layers=(1,), alpha=1.0))
    x = torch.zeros(1, 3, HIDDEN)

    with apply_steering(model, deltas):
        steered = model.model.layers[0](x)
    after = model.model.layers[0](x)

    assert float(steered[0, 0, 0]) == pytest.approx(10.0)
    assert float(after.abs().sum()) == 0.0


def test_transfer_fn_records_a_failure_per_task_per_draw(tmp_path):
    """A missing artifact must not reduce the number of outputs."""
    fn = build_transfer_fn(
        "steering", artifact=str(tmp_path / "absent.pt"), n_samples=3
    )
    tasks = [_task("t1"), _task("t2", "c2")]
    outs = fn.run(tasks, Corpus([]))
    assert len(outs) == len(tasks) * 3
    assert [o.task_id for o in outs] == ["t1", "t1", "t1", "t2", "t2", "t2"]
    assert [o.meta["sample_index"] for o in outs] == [0, 1, 2, 0, 1, 2]
    assert all(o.output_text == "" and o.meta["ok"] is False for o in outs)
    assert all("absent.pt" in o.meta["error"] for o in outs)


def test_describe_records_layer_alpha_and_the_posts_behind_each_vector(tmp_path):
    path = _artifact(tmp_path)
    fn = build_transfer_fn("steering", artifact=str(path), layer="12,16", alpha=1.5, scope="cell")
    fn._ensure_artifact()  # noqa: SLF001 - loads the vectors only, no base model
    desc = fn.describe()
    assert desc["layers"] == [12, 16]
    assert desc["alpha"] == 1.5
    assert desc["scope"] == "cell"
    assert desc["initial_adapter"] == "/frozen/stage-a"
    assert desc["vector_posts"]["global"] == {"n_target": 337, "n_source": 293}
    assert desc["vector_posts"]["n_cells_with_vector"] == 1
    assert desc["artifact_sha256"]


def test_a_raising_generator_still_yields_one_output_per_task_per_draw(tmp_path):
    """Invariant 6: nothing inside generation may propagate out of `run`."""
    fn = build_transfer_fn("steering", artifact=str(_artifact(tmp_path)), n_samples=2)
    assert fn._ensure_artifact()  # noqa: SLF001

    class Exploding:
        def generate(self, items, spec, draw=0):
            raise RuntimeError("gpu fell over")

    fn._gen = Exploding()  # noqa: SLF001
    tasks = [_task("t1"), _task("t2", "c2")]
    outs = fn.run(tasks, Corpus([]))
    assert len(outs) == 4
    assert all(o.meta["ok"] is False and o.output_text == "" for o in outs)
    assert all("gpu fell over" in o.meta["error"] for o in outs)


def test_an_unusable_layer_is_a_recorded_failure_not_an_exception(tmp_path):
    """A layer the artifact does not hold must not cost the batch its outputs."""
    from vectorial_eval.methods.steering.generate import GenConfig, SteeredGenerator

    art = load_artifact(_artifact(tmp_path))
    gen = SteeredGenerator(art, GenConfig(model_id=art["base_model"]))
    gen.model = object()  # never reached: the failure happens before generation
    res = gen.generate(
        [{"prompt": "p", "cell_id": "c1"}, {"prompt": "q", "cell_id": "c1"}],
        SteeringSpec(layers=(LAYERS + 5,), alpha=0.5, scope="global"),
    )
    assert len(res) == 2
    assert all(r["ok"] is False and "out of range" in r["error"] for r in res)


def test_the_artifact_variant_overrides_the_label(tmp_path):
    """A mislabelled run must not be reported under the wrong conditioning."""
    fn = build_transfer_fn(
        "steering", artifact=str(_artifact(tmp_path)), variant="target_lm_aspect"
    )
    fn._ensure_artifact()  # noqa: SLF001
    assert fn.variant == "target_lm"
    assert fn.describe()["variant"] == "target_lm"


def test_the_environment_selects_the_artifact(monkeypatch, tmp_path):
    """The CLI passes no artifact, so the variant has to arrive some other way."""
    from vectorial_eval.transfer.steering import ENV_ARTIFACT, ENV_VARIANT

    path = _artifact(tmp_path)
    monkeypatch.setenv(ENV_ARTIFACT, str(path))
    monkeypatch.setenv(ENV_VARIANT, "target_lm_paired")
    fn = build_transfer_fn("steering")
    assert fn.artifact_path == path
    assert fn.variant == "target_lm_paired"


def test_chosen_json_supplies_the_defaults(tmp_path):
    path = _artifact(tmp_path)
    (tmp_path / "chosen.steering.target_lm.json").write_text(
        json.dumps({"layer": 20, "alpha": 0.5, "scope": "global"}), encoding="utf-8"
    )
    fn = build_transfer_fn("steering", artifact=str(path))
    assert fn.layers == (20,)
    assert fn.alpha == 0.5
    assert fn.selected_on_dev["layer"] == 20
