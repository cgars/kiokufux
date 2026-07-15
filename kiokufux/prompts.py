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

ROTATION_VLM_COMPARE_PROMPT = """You are shown one contact sheet containing four labeled candidate displays of the same original image.

Candidate A = original pixels, 0 degrees clockwise applied.
Candidate B = original pixels rotated 90 degrees clockwise.
Candidate C = original pixels rotated 180 degrees clockwise.
Candidate D = original pixels rotated 270 degrees clockwise.

Choose the single candidate that already looks physically upright.
Judge people, faces, text, buildings, furniture, the horizon, gravity, and other scene elements.
Do not judge whether the contact sheet as a whole is upright.
Do not describe the image.
Do not mentally compensate for rotation.
If no candidate is clearly upright, choose UNCERTAIN.

Return JSON only, using the candidate label and its clockwise rotation from the list above:

{"selected_candidate": "A", "rotation": 0}"""


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
