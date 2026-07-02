from __future__ import annotations

DEFAULT_AUTOTAG_CANDIDATE_TAGS = [
    "person", "man", "woman", "child", "baby", "toddler", "teenager", "elderly_person",
    "couple", "family", "friends", "group", "portrait", "group_photo", "face", "smile",
    "living_room", "kitchen", "dining_room", "bedroom", "office", "restaurant", "cafe",
    "church_interior", "museum", "garden", "park", "forest", "meadow", "field", "street",
    "square", "courtyard", "playground", "cemetery", "beach", "lake", "river", "sea",
    "mountains", "city", "village", "old_town", "house", "residential_house", "barn",
    "church", "chapel", "castle", "palace", "ruin", "train_station", "car", "bicycle",
    "motorcycle", "bus", "train", "tram", "airplane", "boat", "ship", "suitcase",
    "birthday", "wedding", "baptism", "first_communion", "confirmation", "christmas",
    "easter", "new_years_eve", "carnival", "party", "family_gathering", "vacation",
    "excursion", "hike", "food", "drinks", "coffee", "cake", "tart", "candles",
    "set_table", "buffet", "barbecue", "dog", "cat", "horse", "cow", "sheep", "bird",
    "duck", "black_and_white", "color_photo", "scan", "slide", "blurry", "scan_border",
    "text_in_image", "handwriting",
]


def default_candidate_tags_text() -> str:
    return ", ".join(DEFAULT_AUTOTAG_CANDIDATE_TAGS)
