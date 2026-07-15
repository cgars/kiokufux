from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .vlm import IMAGE_ANALYSIS_PROMPT


ROTATION_VLM_PROMPT = """Determine the correct physical display orientation of this image.

Choose exactly one clockwise rotation to apply to the supplied pixels:

0
90
180
270
UNCERTAIN

Judge which rotation makes people, faces, text, buildings, furniture, the horizon, gravity, and other scene elements naturally upright.

Do not describe the image.
Do not mentally compensate for the rotation.
Return JSON only:

{"rotation": 0}"""

ROTATION_VLM_COMPARE_PROMPT = """You are shown one contact sheet with four labeled versions of the same image. Candidate A applies 0 degrees clockwise to the supplied pixels, B applies 90 degrees clockwise, C applies 180 degrees clockwise, and D applies 270 degrees clockwise.

Choose exactly one clockwise rotation to apply to the original supplied pixels:

0
90
180
270
UNCERTAIN

Judge which candidate makes people, faces, text, buildings, furniture, the horizon, gravity, and other scene elements naturally upright.

Do not describe the image.
Do not mentally compensate for the rotation.
Return JSON only:

{"rotation": 0}"""


@dataclass(frozen=True)
class PromptDefinition:
    name: str
    topic: str
    text: str


PROMPTS: tuple[PromptDefinition, ...] = (
    PromptDefinition("rotation.direct_action", "rotation", ROTATION_VLM_PROMPT),
    PromptDefinition("rotation.candidate_comparison", "rotation", ROTATION_VLM_COMPARE_PROMPT),
    PromptDefinition("vlm-analysis.default", "vlm-analysis", IMAGE_ANALYSIS_PROMPT),
)


def iter_prompts(topic: str = "all") -> Iterable[PromptDefinition]:
    for prompt in PROMPTS:
        if topic == "all" or prompt.topic == topic:
            yield prompt
