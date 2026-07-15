from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .vlm import IMAGE_ANALYSIS_PROMPT


ROTATION_VLM_PROMPT = """Determine only the corrective action needed to make this image upright. Do not describe how the image currently looks except as a reason. Return only valid JSON with keys: needs_rotation (boolean), action_clockwise_degrees (the clockwise rotation to perform now; one of 0, 90, 180, 270), confidence (0.0 to 1.0), and reason (short string). If the image is already upright, return needs_rotation=false and action_clockwise_degrees=0; saying no corrective action is needed is a valid answer. If the image appears rotated 90 degrees right/clockwise, the corrective action is 270 clockwise; if it appears rotated 90 degrees left/counterclockwise, the corrective action is 90 clockwise. Do not identify people or add tags."""

ROTATION_VLM_COMPARE_PROMPT = """You are shown one contact sheet with four labeled versions of the same image. Candidate A applies 0 degrees, B applies 90 degrees clockwise, C applies 180 degrees, and D applies 270 degrees clockwise to the original. Select the candidate that looks upright/correct. Return only valid JSON with keys: selected_candidate (A, B, C, or D), action_clockwise_degrees (0, 90, 180, or 270), needs_rotation (boolean), confidence (0.0 to 1.0), and reason (short string). It is valid to choose A and needs_rotation=false when the original is already upright. Do not identify people or add tags."""


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
