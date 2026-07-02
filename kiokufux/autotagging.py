from __future__ import annotations

from dataclasses import dataclass

from .catalog import Catalog
from .embeddings import EmbeddingBackend, default_backend
from .models import Photo
from .search import cosine

DEFAULT_AUTO_TAG_TOP_K = 5
DEFAULT_AUTO_TAG_MIN_SCORE = 0.20


@dataclass(slots=True)
class ProposedTag:
    tag: str
    confidence: float
    reason: str


class EmbeddingAutoTagger:
    """Local zero-shot tag proposer using image/text embeddings.

    The tagger embeds each image and candidate tag locally, compares them with
    cosine similarity, and creates reviewable proposals. It does not accept tags
    automatically; users review proposals before they become exported auto tags.
    """

    source = "ai-zero-shot"

    def __init__(
        self,
        backend: EmbeddingBackend | None = None,
        candidate_tags: list[str] | None = None,
        top_k: int = DEFAULT_AUTO_TAG_TOP_K,
        min_score: float = DEFAULT_AUTO_TAG_MIN_SCORE,
    ) -> None:
        self.backend = backend or default_backend()
        self.candidate_tags = normalize_candidate_tags(candidate_tags or [])
        self.top_k = top_k
        self.min_score = min_score

    def propose(self, photo: Photo) -> list[ProposedTag]:
        image_vector = self.backend.embed_image(photo.source_path)
        proposals: list[ProposedTag] = []
        for tag in self.candidate_tags:
            text_vector = self.backend.embed_text(tag)
            score = cosine(image_vector, text_vector)
            if score >= self.min_score:
                proposals.append(ProposedTag(tag, score, "zero-shot image/text similarity"))
        return sorted(proposals, key=lambda p: (-p.confidence, p.tag))[: self.top_k]


def normalize_candidate_tags(tags: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        cleaned = " ".join(tag.strip().lower().split())
        if cleaned and cleaned not in seen:
            normalized.append(cleaned)
            seen.add(cleaned)
    return normalized


def propose_tags(catalog: Catalog, tagger: EmbeddingAutoTagger | None = None) -> int:
    tagger = tagger or EmbeddingAutoTagger()
    count = 0
    for photo in catalog.list_photos():
        for proposal in tagger.propose(photo):
            catalog.propose_tag(photo.photo_id, proposal.tag, proposal.confidence, source=tagger.source)
            count += 1
    return count
