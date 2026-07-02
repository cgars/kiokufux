from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .catalog import Catalog
from .models import Photo

STOPWORDS = {"img", "image", "photo", "pic", "dsc", "scan", "copy", "edited", "final"}
KEYWORD_TAGS = {
    "dog", "cat", "cow", "horse", "bird", "bike", "bicycle", "car", "truck", "boat",
    "church", "house", "garden", "party", "lake", "beach", "snow", "mountain", "forest",
    "baby", "wedding", "birthday", "holiday", "family", "school", "train", "flower",
}


@dataclass(slots=True)
class ProposedTag:
    tag: str
    confidence: float
    reason: str


class LocalAutoTagger:
    """Small local-only tag proposer for MVP review workflows.

    It intentionally creates proposals, not accepted tags. Users can review and
    accept or reject them before they become exported auto tags.
    """

    source = "local-ai"

    def propose(self, photo: Photo) -> list[ProposedTag]:
        proposals: dict[str, ProposedTag] = {}
        for tag in self._filename_tags(photo.source_path):
            proposals[tag] = ProposedTag(tag, 0.72, "filename keyword")
        color = self._dominant_color_tag(photo.source_path)
        if color and color not in proposals:
            proposals[color] = ProposedTag(color, 0.55, "dominant color")
        return sorted(proposals.values(), key=lambda p: (-p.confidence, p.tag))

    def _filename_tags(self, path: Path) -> list[str]:
        cleaned = path.stem.lower().replace("_", " ").replace("-", " ")
        tags = []
        for token in cleaned.split():
            token = "".join(ch for ch in token if ch.isalpha())
            if len(token) >= 3 and token not in STOPWORDS and token in KEYWORD_TAGS:
                tags.append(token)
        return tags

    def _dominant_color_tag(self, path: Path) -> str | None:
        try:
            from PIL import Image, ImageOps
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB").resize((1, 1))
                red, green, blue = image.getpixel((0, 0))
        except Exception:
            return None
        if max(red, green, blue) < 40:
            return "dark"
        if red > green * 1.25 and red > blue * 1.25:
            return "red"
        if green > red * 1.20 and green > blue * 1.20:
            return "green"
        if blue > red * 1.20 and blue > green * 1.20:
            return "blue"
        return None


def propose_tags(catalog: Catalog, tagger: LocalAutoTagger | None = None) -> int:
    tagger = tagger or LocalAutoTagger()
    count = 0
    for photo in catalog.list_photos():
        for proposal in tagger.propose(photo):
            catalog.propose_tag(photo.photo_id, proposal.tag, proposal.confidence, source=tagger.source)
            count += 1
    return count
