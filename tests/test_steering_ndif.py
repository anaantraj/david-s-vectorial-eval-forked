"""Tests for the NDIF steering arm.

These run offline. Nothing here contacts NDIF: the status payload is a fixture
and the envoy layout is exercised with plain tensors. The live path is covered by
`methods.steering.smoke_ndif`, which needs a key and is not part of the suite.

The test that matters most is `test_bare_tensor_write_steers_every_batch_row`.
A decoder block on these deployments returns a bare `[B, T, H]` tensor, and
indexing it as if it were a `(hidden, ...)` tuple silently yields row 0 of the
batch. Everything still runs; only the first prompt of each batch is steered, and
the rest are recorded as steered outputs that were never touched. That is a
wrong-numbers bug of exactly the kind the harness's invariants exist to prevent,
so the unwrapping rule is pinned here.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vectorial_eval.methods.steering.ndif_backend import (  # noqa: E402
    NDIFConfig,
    _hidden,
    check_model,
    ndif_status,
    pinned_models,
)

# A trimmed copy of the shape api.ndif.us/status actually returns: the model id
# is embedded in the deployment key, and one model may have several deployments.
STATUS_PAYLOAD = {
    "deployments": {
        'a:ModelActor:LanguageModel:{"repo_id": "meta-llama/Llama-3.1-405B-Instruct"}': {
            "application_state": "RUNNING", "deployment_level": "HOT", "pinned": True,
        },
        'b:ModelActor:LanguageModel:{"repo_id": "meta-llama/Llama-3.1-8B-Instruct"}': {
            "application_state": "RUNNING", "deployment_level": "HOT", "pinned": False,
        },
        'c:ModelActor:LanguageModel:{"repo_id": "meta-llama/Llama-3.1-70B"}': {
            "application_state": None, "deployment_level": "WARM", "pinned": None,
        },
        'd:ModelActor:LanguageModel:{"repo_id": "meta-llama/Llama-3.1-70B"}': {
            "application_state": "RUNNING", "deployment_level": "HOT", "pinned": True,
        },
    }
}


@pytest.fixture
def status(monkeypatch):
    import io
    import json
    import urllib.request

    def fake_urlopen(*_args, **_kwargs):
        return io.BytesIO(json.dumps(STATUS_PAYLOAD).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return ndif_status()


def test_status_collapses_deployments_per_model(status):
    assert set(status) == {
        "meta-llama/Llama-3.1-405B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-70B",
    }
    # 70B has one unpinned and one pinned deployment; pinned wins, because that
    # is the condition a standard key is actually checked against.
    assert status["meta-llama/Llama-3.1-70B"]["pinned"] is True


def test_pinned_models(status):
    assert pinned_models(status) == [
        "meta-llama/Llama-3.1-405B-Instruct",
        "meta-llama/Llama-3.1-70B",
    ]


def test_check_model_accepts_pinned(status):
    assert check_model("meta-llama/Llama-3.1-405B-Instruct", status)["pinned"] is True


def test_check_model_rejects_hot_but_unpinned(status):
    # The failure this guards is specific: 8B-Instruct is RUNNING and HOT, so the
    # status page looks like it should work, and a standard key still cannot run
    # it. The error has to name the models that would have worked.
    with pytest.raises(ValueError, match="not pinned"):
        check_model("meta-llama/Llama-3.1-8B-Instruct", status)
    try:
        check_model("meta-llama/Llama-3.1-8B-Instruct", status)
    except ValueError as exc:
        assert "meta-llama/Llama-3.1-405B-Instruct" in str(exc)


def test_check_model_rejects_undeployed(status):
    with pytest.raises(ValueError, match="not deployed"):
        check_model("mistralai/Mixtral-8x22B-v0.1", status)


def test_check_model_survives_a_status_outage(monkeypatch):
    import urllib.request

    def boom(*_args, **_kwargs):
        raise OSError("unreachable")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    # A status outage must not be fatal: the run should proceed and let the trace
    # itself fail if the model really is unavailable.
    assert check_model("meta-llama/Llama-3.1-70B")["checked"] is False


# --------------------------------------------------------------------------- #
# envoy output unwrapping
# --------------------------------------------------------------------------- #


def test_hidden_unwraps_a_tuple():
    hidden = torch.zeros(2, 3, 4)
    assert _hidden((hidden, "presents"), True) is hidden


def test_hidden_passes_a_bare_tensor_through():
    hidden = torch.zeros(2, 3, 4)
    assert _hidden(hidden, False) is hidden


def test_bare_tensor_write_steers_every_batch_row():
    """The regression guard for the silent batch-row bug.

    With `tuple_output=False` the write must broadcast over the whole
    `[B, T, H]` tensor. Treating the same tensor as a tuple selects row 0, so
    prompt 1 of every batch is steered and the rest are not.
    """
    delta = torch.ones(4)

    correct = torch.zeros(2, 3, 4)
    view = _hidden(correct, False)
    view[:] += delta
    assert correct.sum() == pytest.approx(2 * 3 * 4)
    assert correct[1].sum() > 0, "second batch row was not steered"

    wrong = torch.zeros(2, 3, 4)
    view = _hidden(wrong, True)  # what the bug did
    view[:] += delta
    assert wrong[1].sum() == 0, "this is the bug being guarded against"


# --------------------------------------------------------------------------- #
# config provenance
# --------------------------------------------------------------------------- #


def test_config_does_not_claim_the_studys_quantisation_constant():
    # Every local method runs under 4-bit NF4 and reports load_in_4bit=True.
    # NDIF serves its own precision, so recording False here would file this arm
    # under a constant it does not share; it must be None.
    out = NDIFConfig(model_id="meta-llama/Llama-3.1-70B-Instruct").to_dict()
    assert out["load_in_4bit"] is None
    assert out["backend"] == "ndif"


def test_config_records_what_a_rerun_would_need():
    cfg = NDIFConfig(model_id="m", deployment={"pinned": True})
    out = cfg.to_dict()
    for key in ("deployment", "nnsight_version", "remote_seed", "chat_wrapped"):
        assert key in out, f"{key} missing; a remote result is not reproducible without it"


def test_config_inherits_the_generation_settings():
    # The remote arm must not quietly diverge on decoding parameters.
    cfg = NDIFConfig(model_id="m", max_new_tokens=320, temperature=1.0, top_p=0.95)
    out = cfg.to_dict()
    assert (out["max_new_tokens"], out["temperature"], out["top_p"]) == (320, 1.0, 0.95)


# --------------------------------------------------------------------------- #
# registration
# --------------------------------------------------------------------------- #


def test_registered_and_distinct_from_the_local_arm():
    from vectorial_eval.transfer import available

    assert "steering_ndif" in available()
    assert "steering" in available()


def test_inherits_the_local_run_loop():
    # The two arms must differ in the generator and in nothing else; if `run` or
    # the prompt construction is ever overridden, the columns stop being
    # comparable and this should fail loudly.
    from vectorial_eval.transfer.steering import SteeringTransfer
    from vectorial_eval.transfer.steering_ndif import NDIFSteeringTransfer

    assert issubclass(NDIFSteeringTransfer, SteeringTransfer)
    for method in ("run", "_items", "_failed", "_spec"):
        assert getattr(NDIFSteeringTransfer, method) is getattr(SteeringTransfer, method)


def test_api_key_error_names_the_variable(monkeypatch, tmp_path):
    from vectorial_eval.methods.steering.ndif_backend import ENV_API_KEY, resolve_api_key

    monkeypatch.delenv(ENV_API_KEY, raising=False)
    monkeypatch.chdir(tmp_path)  # so no .env is found
    with pytest.raises(RuntimeError, match=ENV_API_KEY):
        resolve_api_key()


# --------------------------------------------------------------------------- #
# artifact provenance
# --------------------------------------------------------------------------- #


def test_remote_fit_does_not_inherit_the_bf16_ablation_label():
    """`finalise` coerces the remote arm's None to False; this undoes it.

    False is not a neutral value in the local vocabulary — it is the name of the
    bf16 ablation of the study's 4-bit constant. A remote fit that never shared
    that constant must not be filed as an ablation of it.
    """
    from vectorial_eval.methods.steering.fit_steering_ndif import (
        stamp_remote_provenance,
    )

    artifact = {"fit_config": {"load_in_4bit": False, "max_length": 768}}
    stamp_remote_provenance(
        artifact, deployment={"pinned": True}, concurrency=8, chat_wrapped=True
    )
    assert artifact["fit_config"]["load_in_4bit"] is None
    assert artifact["fit_config"]["max_length"] == 768, "unrelated keys preserved"


def test_remote_fit_records_what_a_rerun_needs():
    from vectorial_eval.methods.steering.fit_steering_ndif import (
        stamp_remote_provenance,
    )

    artifact = stamp_remote_provenance(
        {}, deployment={"pinned": True, "states": ["RUNNING"]},
        concurrency=8, chat_wrapped=True,
    )
    ndif = artifact["fit_config"]["ndif"]
    for key in ("status_url", "deployment", "nnsight_version", "chat_wrapped"):
        assert key in ndif, f"{key} missing; the fit is not attributable without it"
    assert artifact["fit_config"]["backend"] == "ndif"
