"""The one place where the soft-prompt method turns a rendered context into model input.

Training and inference must agree exactly. The context itself is fixed by
`vectorial_eval.data.build_training.TARGET_LM_PROMPT`, which every method in the
study shares, but an instruction-tuned base model also needs that context wrapped
in its own turn markers, and a disagreement between the wrapping used during
training and the wrapping used at inference would appear as a quality loss that
looks like a property of soft prompting.

The wrapping is deliberately identical to `vectorial_eval.methods.lora.prompting`:
the shared context is a user turn and the authentic target post is the assistant
reply. It is duplicated here rather than imported so that neither method can break
the other at import time, and `tests/test_soft_prompt.py` asserts the two
implementations agree byte for byte whenever both are present. If that test fails,
the two methods are no longer comparable and one of them must be brought back into
line before either number is reported.

No torch import: `vectorial_eval.transfer.soft_prompt` imports this module at
harness import time and the local environment has no torch.
"""

from __future__ import annotations

from ...data.build_training import TARGET_LM_PROMPT, render_prompt

__all__ = ["TARGET_LM_PROMPT", "prompt_tail", "render_prompt", "wrap_prompt"]


def wrap_prompt(tokenizer, prompt: str) -> str:
    """Render `prompt` as a user turn, ending at the assistant header.

    Returns text rather than token ids, so the caller tokenises it with
    `add_special_tokens=False` and the template's own beginning-of-text marker is
    the only one present.
    """
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )


def prompt_tail(target_platform: str = "reddit") -> str:
    """The final line of the shared template, used to verify a data file.

    A training file whose prompts do not end with this was not produced by
    `build_training`, which would mean this method is being trained on a
    different context from the others and is no longer comparable with them.
    """
    return render_prompt(
        room="_", topic="_", domain="_", target_platform=target_platform
    ).splitlines()[-1]
