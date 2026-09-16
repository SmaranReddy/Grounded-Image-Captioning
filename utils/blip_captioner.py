"""
BLIP-based caption generator conditioned on a structured scene graph.

Drop-in replacement for generate_causal_caption() — same call signature,
richer output via a vision-language model.

Evidence gating runs after generation:
  1. Low-confidence entity mentions are hedged with "possibly".
  2. Captions with too many unsupported entities fall back to a safe template.
"""

from __future__ import annotations

import re
import time
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from utils.relation_corrector import correct_caption_relations

Detection    = Dict   # {"label": str, "box": List[float], "score": float}
Relationship = Tuple[str, str, str]   # (subject, relation, object)

# ---------------------------------------------------------------------------
# Model singleton — loaded once on first call
# ---------------------------------------------------------------------------

_processor: BlipProcessor | None = None
_model: BlipForConditionalGeneration | None = None
_device: torch.device | None = None
_load_time: float = 0.0

_MODEL_ID = "Salesforce/blip-image-captioning-base"


def _load_model(dtype: Optional[torch.dtype] = None) -> None:
    global _processor, _model, _device, _load_time

    t0 = time.time()

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.float16 if device_str == "cuda" else torch.float32

    print(f"[BLIP] Loading {_MODEL_ID} on {device_str} …")

    try:
        _processor = BlipProcessor.from_pretrained(_MODEL_ID)
        _model     = BlipForConditionalGeneration.from_pretrained(
            _MODEL_ID, torch_dtype=dtype
        )
        _model.to(device_str)
    except Exception as e:
        if device_str == "cuda":
            print(f"[BLIP] CUDA load failed: {e}")
            print(f"[BLIP] Falling back to CPU …")
            device_str = "cpu"
            dtype = torch.float32
            torch.cuda.empty_cache()
            _processor = BlipProcessor.from_pretrained(_MODEL_ID)
            _model     = BlipForConditionalGeneration.from_pretrained(
                _MODEL_ID, torch_dtype=dtype
            )
            _model.to(device_str)
        else:
            raise e

    _device = torch.device(device_str)
    _model.eval()
    _load_time = time.time() - t0
    print(f"[BLIP] Model ready in {_load_time:.1f}s on {_device}.")


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(
    detections: List[Detection],
    relationships: List[Relationship],
) -> str:
    """Serialize scene graph into a strict, grounded prompt."""
    scores = {d["label"]: d.get("score", 0.5) for d in detections}

    obj_lines = [
        f"- {d['label']} ({d.get('score', 0.5):.2f})"
        for d in detections
    ]

    if relationships:
        rel_lines = [
            f"- {subj} {rel.replace('_', ' ')} {obj} "
            f"({min(scores.get(subj, 0.5), scores.get(obj, 0.5)):.2f})"
            for subj, rel, obj in relationships
        ]
    else:
        rel_lines = ["- None"]

    return (
        "You are describing an image based ONLY on verified detections.\n"
        "\n"
        "Detected objects (with confidence):\n"
        + "\n".join(obj_lines)
        + "\n"
        "\nDetected relationships:\n"
        + "\n".join(rel_lines)
        + "\n"
        "\nSTRICT RULES:\n"
        "- Do NOT mention time of day (e.g., night, day, morning, evening, afternoon).\n"
        "- Do NOT mention lighting or weather (e.g., bright, dark, sunny, shadow, rain, fog).\n"
        "- Do NOT introduce objects that are not listed above.\n"
        "- Do NOT infer context beyond the listed objects and relationships.\n"
        "- If uncertain, omit the detail rather than guessing.\n"
        "\n"
        "Write 2 concise sentences describing what is happening using only the listed objects and relationships."
    )


def build_grounded_prompt(
    detections: List[Detection],
    relations: List[Dict],
    min_confidence: float = 0.1,
) -> str:
    """
    Build a grounded prompt from structured relation dicts with confidence.

    Args:
        detections:   [{"label": str, "box": [...], "score": float}, ...]
        relations:    [{"subject": str, "predicate": str, "object": str,
                       "confidence": float}, ...]
        min_confidence: Minimum confidence to include a relation.

    Returns:
        Grounded prompt string biasing BLIP toward evidence-supported semantics.
    """
    SEMANTIC_PRIORITY = frozenset({
        "riding", "holding", "wearing", "carrying", "looking at",
    })

    obj_lines = [
        f"- {d['label']}"
        for d in detections
    ]

    # Filter and sort relations
    valid_rels = [r for r in relations if r.get("confidence", 0) >= min_confidence]

    def _rel_sort_key(r: Dict) -> tuple:
        is_semantic = r["predicate"] in SEMANTIC_PRIORITY
        return (is_semantic, r["confidence"])

    valid_rels.sort(key=_rel_sort_key, reverse=True)

    if valid_rels:
        rel_lines = [
            f"- {r['subject']} {r['predicate']} {r['object']}"
            for r in valid_rels
        ]
    else:
        rel_lines = ["- None detected"]

    prompt = (
        "Describe this image using ONLY the verified objects and relationships below.\n"
        "\n"
        "Detected objects:\n"
        + "\n".join(obj_lines)
        + "\n"
        "\nVerified visual relationships:\n"
        + "\n".join(rel_lines)
        + "\n"
        "\nCONSTRAINTS:\n"
        "- Only mention objects listed above.\n"
        "- Use the listed relationships to describe interactions.\n"
        "- Do NOT add objects, actions, or context not listed.\n"
        "- Do NOT describe time, weather, or lighting.\n"
        "- Write 1-2 concise factual sentences."
    )

    return prompt


# ---------------------------------------------------------------------------
# Image normalisation helper
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Semantic verbalization (Steps 2, 3)
# ---------------------------------------------------------------------------

_SEMANTIC_PREDS: frozenset = frozenset({
    "riding", "holding", "carrying", "wearing", "looking at",
    "sitting on", "standing on",
})

_PRED_VERB_BASE: Dict[str, str] = {
    "riding":       "riding",
    "holding":      "holding",
    "carrying":     "carrying",
    "wearing":      "wearing",
    "looking at":   "looking at",
    "sitting on":   "sitting on",
    "standing on":  "standing on",
}


def _indefinite_article(word: str) -> str:
    w = word.lower().strip()
    return "an" if w and w[0] in "aeiou" else "a"


def verbalize_relation(
    subject: str,
    predicate: str,
    object: str,
    confidence: float,
) -> str:
    """
    Convert a (subject, predicate, object, confidence) tuple into a
    natural-language interaction description.

    Confidence tiers:
        >= 0.7:  definite  ("the person is riding a bicycle")
        0.5-0.7: appears  ("the person appears to be riding a bicycle")
        0.15-0.5: may be  ("the person may be riding a bicycle")
        < 0.15:  generic  ("the person may be interacting with a bicycle")
    """
    obj_article = _indefinite_article(object)
    subj_ref = f"the {subject}"
    verb_base = _PRED_VERB_BASE.get(predicate, "interacting with")

    if predicate == "looking at":
        if confidence >= 0.7:
            return f"{subj_ref} is looking at {obj_article} {object}"
        elif confidence >= 0.5:
            return f"{subj_ref} appears to be looking at {obj_article} {object}"
        elif confidence >= 0.15:
            return f"{subj_ref} may be looking at {obj_article} {object}"
        else:
            return f"{subj_ref} may be interacting with {obj_article} {object}"

    if confidence >= 0.7:
        return f"{subj_ref} is {verb_base} {obj_article} {object}"
    elif confidence >= 0.5:
        return f"{subj_ref} appears to be {verb_base} {obj_article} {object}"
    elif confidence >= 0.15:
        return f"{subj_ref} may be {verb_base} {obj_article} {object}"
    else:
        return f"{subj_ref} may be interacting with {obj_article} {object}"


def build_semantic_prompt(
    detections: List[Detection],
    verbalized_relations: List[str],
) -> str:
    """
    Build a grounded prompt using natural-language interaction descriptions.

    New template (Step 6):
        Detected scene elements:
        - a person
        - a bicycle

        Likely interactions:
        - the person appears to be riding the bicycle

        Generate a concise factual caption grounded in these interactions.
        Do not mention objects or actions not supported by the scene.

    Args:
        detections:           [{"label": str, ...}, ...]
        verbalized_relations: ["the person is riding a bicycle", ...]

    Returns:
        Prompt string for BLIP.
    """
    obj_lines = [
        f"- {_indefinite_article(d['label'])} {d['label']}"
        for d in detections
    ]

    if verbalized_relations:
        rel_lines = [f"- {vr}" for vr in verbalized_relations]
    else:
        rel_lines = ["- no clear interactions detected"]

    return (
        "Detected scene elements:\n"
        + "\n".join(obj_lines)
        + "\n\n"
        "Grounded interactions:\n"
        + "\n".join(rel_lines)
        + "\n\n"
        "Generate a concise factual caption grounded ONLY in these interactions.\n"
        "Describe the scene based solely on the grounded interactions above.\n"
        "Do not mention objects, actions, or spatial relationships not listed."
    )


# ---------------------------------------------------------------------------
# Image normalisation helper
# ---------------------------------------------------------------------------

BASELINE_PREFIX = "a photo of"

# Present-participle surface form used to splice a predicate into the prefix.
_PRED_PREFIX_FORM: Dict[str, str] = {
    "holding":     "holding",
    "riding":      "riding",
    "wearing":     "wearing",
    "carrying":    "carrying",
    "sitting on":  "sitting on",
    "standing on": "standing on",
    "looking at":  "looking at",
    "eating":      "eating",
    "on":          "on",
    "in":          "in",
    "inside":      "inside",
    "under":       "under",
    "above":       "above",
    "over":        "over",
    "near":        "near",
    "next to":     "next to",
    "behind":      "behind",
    "in front of": "in front of",
    "attached to": "attached to",
    "covering":    "covering",
}

GROUNDING_MIN_CONFIDENCE = 0.35


def build_blip_prefix(
    detections: List[Detection],
    relations: List[Dict],
    min_confidence: float = GROUNDING_MIN_CONFIDENCE,
    max_relations: int = 1,
) -> Tuple[str, List[Dict]]:
    """Build the decoder prefix BLIP actually generates from.

    Why this exists
    ---------------
    ``build_semantic_prompt`` produces a multi-line INSTRUCTION block ("Detected
    scene elements: ... Do not mention objects not listed"). BLIP-base
    (``Salesforce/blip-image-captioning-base``) is not instruction-tuned: its
    text decoder continues a caption prefix, it does not follow directions. The
    previous code acknowledged this by building the semantic prompt and then
    silently generating from the constant ``"a photo of"`` instead - so the
    detections and the predicted relations had NO effect whatsoever on the
    generated text, and the "grounded" system differed from the baseline only
    in the post-hoc gating stage.

    The fix keeps the honest part of that reasoning (BLIP cannot follow the
    instruction block) and supplies grounding through the one channel BLIP does
    respect: the decoder prefix. We splice the single highest-confidence
    verified relation into a natural caption opening -

        "a photo of a person riding a bicycle"

    - and let BLIP complete the surrounding scene. Every object named in the
    prefix comes from a YOLO detection that survived verification, so the
    injected span is grounded by construction; the completion is still free
    text and is still checked by ``gate_caption``.

    Relations below ``min_confidence`` are not injected: a wrong relation in the
    prefix is a hallucination we would be creating ourselves. With no confident
    relation the prefix degrades to the plain baseline opening, which makes the
    grounded system a strict superset of the baseline rather than a different
    generator.

    Returns:
        (prefix, used_relations)
    """
    usable = [
        r for r in relations
        if r.get("adjusted_confidence", r.get("confidence", 0.0)) >= min_confidence
        and _PRED_PREFIX_FORM.get(r.get("predicate", ""))
    ]
    usable.sort(
        key=lambda r: -r.get("adjusted_confidence", r.get("confidence", 0.0)),
    )

    used: List[Dict] = []
    parts: List[str] = []
    for r in usable[:max_relations]:
        subject = str(r["subject"]).replace("_", " ")
        obj = str(r["object"]).replace("_", " ")
        verb = _PRED_PREFIX_FORM[r["predicate"]]
        parts.append(
            f"{_indefinite_article(subject)} {subject} {verb} "
            f"{_indefinite_article(obj)} {obj}"
        )
        used.append(r)

    if not parts:
        return BASELINE_PREFIX, []

    return f"{BASELINE_PREFIX} {' and '.join(parts)}", used


def build_objects_only_prefix(relation: Dict) -> str:
    """Control prefix: the SAME two objects as the relation prefix, no predicate.

    "a photo of a person riding a bicycle" -> "a photo of a person and a bicycle".
    Comparing the grounded arm against this separates the effect of the
    predicted relation from the effect of merely naming two detected objects.
    """
    subject = str(relation["subject"]).replace("_", " ")
    obj = str(relation["object"]).replace("_", " ")
    return (f"{BASELINE_PREFIX} {_indefinite_article(subject)} {subject} and "
            f"{_indefinite_article(obj)} {obj}")


# Decoding parameters shared by EVERY arm of the caption experiment. These are
# the values generate_blip_baseline / generate_blip_semantic_caption already
# used; only the text prefix differs between arms.
EXPERIMENT_GENERATION_CONFIG: Dict = {
    "max_new_tokens": 128,
    "num_beams": 4,
    "early_stopping": True,
    "do_sample": False,
}


def _normalise_space(text: str) -> str:
    return " ".join(text.lower().split())


def blip_model_info() -> Dict:
    if _model is None:
        return {"loaded": False}
    return {"loaded": True, "model_id": _MODEL_ID, "device": str(_device),
            "dtype": str(next(_model.parameters()).dtype).replace("torch.", "")}


def generate_blip_from_prefix(
    image,
    prefix: str,
    dtype: Optional[torch.dtype] = None,
    generation_config: Optional[Dict] = None,
) -> Dict:
    """Caption `image` starting from `prefix`, with no post-processing at all.

    The experiment's only generation entry point: the baseline, grounded and
    control arms all call this with the same image preprocessing and the same
    decoding parameters, one image at a time (batched beam search can differ
    numerically from unbatched, which would be a confound). No relation
    correction and no evidence gating run here - both depend on YOLO and would
    change captions for reasons other than the relation.

    Returns the decoded caption, the exact prefix, the token ids BLIP was
    conditioned on, and whether the caption actually begins with the prefix.
    """
    global _processor, _model, _device

    if _model is None:
        _load_model(dtype=dtype)
    elif dtype is not None and next(_model.parameters()).dtype != dtype:
        raise RuntimeError(f"BLIP already loaded as {next(_model.parameters()).dtype}, "
                           f"requested {dtype}")

    cfg = dict(EXPERIMENT_GENERATION_CONFIG if generation_config is None else generation_config)
    pil_image = _to_pil(image)
    inputs = _processor(images=pil_image, text=prefix, return_tensors="pt").to(_device)
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=next(_model.parameters()).dtype)

    with torch.inference_mode():
        output_ids = _model.generate(**inputs, **cfg)

    caption = " ".join(_processor.decode(output_ids[0], skip_special_tokens=True).split())
    return {
        "caption": caption,
        "prefix": prefix,
        "input_ids": [int(t) for t in inputs["input_ids"][0].tolist()],
        "prefix_echoed": _normalise_space(caption).startswith(_normalise_space(prefix)),
        "generation_config": cfg,
    }


def _strip_prefix_artifacts(caption: str, prefix: str) -> str:
    """Normalise a decoded caption that starts with the conditioning prefix.

    BLIP echoes the prefix back in the decoded string, which is what we want -
    the grounded span is part of the caption. This only tidies the join
    (duplicated prefix, stray whitespace, missing capital).
    """
    text = " ".join(caption.split())
    lowered = text.lower()
    plain = prefix.lower().strip()
    while lowered.startswith(plain + " " + plain):
        text = text[len(plain) + 1:]
        lowered = text.lower()
    return text.strip()


def _to_pil(image) -> Image.Image:
    """Accept a PIL Image or a CHW/HWC torch.Tensor and return a PIL Image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    t = image.float()
    if t.ndim == 3 and t.shape[0] in (1, 3, 4):   # CHW → HWC
        t = t.permute(1, 2, 0)
    t = t.cpu().numpy()

    if t.max() <= 1.0:
        t = (t * 255).clip(0, 255).astype("uint8")
    else:
        t = t.clip(0, 255).astype("uint8")

    if t.shape[2] == 1:
        t = t[:, :, 0]

    return Image.fromarray(t).convert("RGB")


# ---------------------------------------------------------------------------
# Evidence-gating helpers
# ---------------------------------------------------------------------------

# Words that are always legitimate in a scene caption regardless of detections.
# Covers verbs, prepositions, and common scene-description adjectives.
_STOP_WORDS: frozenset = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "to", "of", "in",
    "on", "at", "by", "for", "with", "from", "into", "and", "or", "but",
    "not", "no", "it", "its", "this", "that", "there", "they", "them",
    "he", "she", "him", "her", "his", "we", "us", "our",
    "what", "which", "who", "when", "where", "why", "how", "all", "both",
    "each", "some", "such", "than", "too", "very", "just", "also",
    "while", "during", "before", "after", "until",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "other", "another", "else", "every", "many", "several", "few", "more", "most",
    "along", "away", "back", "out", "up", "down", "off", "around",
    "over", "under", "through", "across", "onto", "into", "past",
})

# Action / scene / relationship words — never penalised as unsupported.
_SCENE_WORDS: frozenset = frozenset({
    "riding", "walking", "running", "standing", "sitting", "holding",
    "carrying", "chasing", "playing", "looking", "parked", "near",
    "next", "possibly", "likely", "appears", "seem", "seems", "suggests",
    "suggesting", "indicating", "showing", "observed", "visible",
    "scene", "outdoor", "indoor", "background", "interaction", "unclear",
    "stationary", "moving", "using", "beside", "behind", "front",
    "left", "right",
    # Additional interaction verb forms
    "interacting", "interact", "sits", "stands", "hold", "holds",
    "ride", "rides", "wear", "wears", "carry", "carries", "look", "looks",
    "touch", "touches", "reach", "reaches", "grab", "grabs",
    "talk", "talking", "speak", "speaking", "eat", "eating", "drink",
    "drinking", "read", "reading", "write", "writing", "call", "calling",
    "text", "texting", "watch", "watching", "push", "pushing", "pull",
    "pulling", "open", "opening", "close", "closing",
    "fly", "flies", "flying", "driving", "drive", "swim", "swimming",
    "throw", "throwing", "catch", "catching", "kick", "kicking",
    "climb", "climbing", "jump", "jumping", "dance", "dancing",
    "sing", "singing", "paint", "painting", "draw", "drawing",
    "cook", "cooking", "pour", "pouring", "cut", "cutting",
    "ski", "skis", "skiing", "snowboard", "surf", "surfing",
    "skate", "skating", "skateboard",
    # Common action verbs in BLIP captions
    "smile", "smiling", "wave", "waving", "taking", "giving",
    "point", "pointing", "raise", "raising", "wait", "waiting",
    # Meta verbs common in BLIP captions
    "depicts", "depicted", "shows", "showing", "features", "made",
    # Material / attribute descriptors
    "wooden", "metal", "plastic", "glass", "fabric", "leather",
    "wood", "stone", "concrete", "brick", "tile",

    # Person reference synonyms
    "person", "people", "woman", "women", "man", "men", "child",
    "children", "girl", "boy", "adult", "kid", "individual",
    "someone", "somebody",
})

# Ambient / environmental / descriptive words that add colour but are not
# object-level claims. A caption saying "on a sunny afternoon" is not
# hallucinating even if "sunny" is absent from detections.
_SAFE_CONTEXT_WORDS: frozenset = frozenset({
    # environment & weather
    "street", "road", "sidewalk", "pavement", "path", "trail", "lane",
    "park", "field", "yard", "garden", "plaza", "square", "area",
    "sky", "ground", "grass", "tree", "trees", "bush", "bushes",
    "sunny", "cloudy", "rainy", "snowy", "foggy", "bright", "dark",
    "sunny", "overcast", "clear", "windy",
    # time / light
    "day", "daytime", "morning", "afternoon", "evening", "night",
    "sunlight", "shadow", "light", "shade",
    # spatial / directional scene setters
    "outside", "inside", "nearby", "away", "forward", "toward",
    "distance", "foreground", "horizon",
    # general descriptors that never name a new object
    "busy", "quiet", "crowded", "empty", "urban", "rural", "public",
    "open", "narrow", "wide", "flat", "steep", "paved", "dirt",
    "warm", "cool", "wet", "dry",
    # generic abstract nouns never worth penalising
    "care", "attention", "focus", "view", "way", "side", "top",
    "bottom", "middle", "center", "edge", "end", "part", "piece",
    # architectural / indoor scene elements
    "room", "floor", "wall", "ceiling", "corner", "door", "window",
    "hallway", "entrance", "exit", "stair", "stairs", "staircase",
    "roof", "fence", "gate", "curb", "gutter",
    # pet / animal accessories commonly mentioned in captions
    "leash", "collar", "harness", "chain",
    # generic countable nouns that never name a specific object
    "objects", "items", "things", "stuff", "people", "animals",
    "surfaces", "surface", "area", "spot", "section", "portion",
    # image meta references
    "image", "photo", "photograph", "picture",
})

_HEDGE_MARKERS: Tuple[str, ...] = ("possibly", "appears to be", "might be", "may be")

# ---------------------------------------------------------------------------
# Synonym mapping — relaxes strict word matching so that alternative
# object names (e.g. "woman" → "person") are accepted as grounded.
# ---------------------------------------------------------------------------

_SYNONYMS: Dict[str, str] = {
    # Person references
    "woman": "person", "man": "person", "people": "person",
    "child": "person", "children": "person", "girl": "person",
    "boy": "person", "adult": "person", "kid": "person",
    "lady": "person", "guy": "person", "individual": "person",
    "someone": "person", "somebody": "person",
    # Vehicles
    "bike": "bicycle",
    "motorcycle": "motorbike",
    "truck": "car", "automobile": "car", "vehicle": "car",
    "lorry": "truck",
    # Electronics
    "cell": "cell phone", "mobile": "cell phone",
    "telephone": "cell phone", "phone": "cell phone",
    "smartphone": "cell phone", "iphone": "cell phone",
    "cellphone": "cell phone",
    # Furniture
    "table": "dining table", "desk": "dining table",
    "sofa": "couch", "settee": "couch", "loveseat": "couch",
    "chair": "chair",
    # Objects
    "drink": "bottle", "soda": "bottle", "pop": "bottle",
    "luggage": "suitcase", "bag": "handbag",
    "purse": "handbag", "backpack": "backpack",
    "shoe": "shoes", "trainer": "shoes", "sneaker": "shoes",
    "footwear": "shoes",
    "television": "tv", "telly": "tv", "monitor": "tv",
    "screen": "tv",
    "sunglasses": "glasses", "eyeglasses": "glasses",
    # Animals
    "puppy": "dog", "canine": "dog", "pooch": "dog",
    "kitten": "cat", "feline": "cat", "kitty": "cat",
    # Sports
    "football": "sports ball", "soccer": "sports ball",
    "baseball": "sports ball", "basketball": "sports ball",
    "tennis": "sports ball", "volleyball": "sports ball",
    "ball": "sports ball",
    # scene
    "sidewalk": "pavement", "path": "trail",
    # Food/Drink containers
    "cup": "cup", "mug": "cup",
    "wineglass": "wine glass",
    # Furniture
    "bed": "bed",
    "toilet": "toilet",
    # Kitchen
    "fridge": "refrigerator",
    "stove": "oven",
    # Clothing/accesories
    "suitcase": "suitcase",
    "umbrella": "umbrella",
    "tie": "tie",
    "helmet": "helmet", "hardhat": "helmet",
}

_INVERTED_SYNONYMS: Dict[str, set] = {}
for syn, canon in _SYNONYMS.items():
    if canon not in _INVERTED_SYNONYMS:
        _INVERTED_SYNONYMS[canon] = set()
    _INVERTED_SYNONYMS[canon].add(syn)
    for part in syn.split():
        _INVERTED_SYNONYMS[canon].add(part)

# ---------------------------------------------------------------------------
# Concrete object nouns — words that refer to physical, detectable objects.
# If any of these appear in a caption they MUST correspond to a detection
# (or a synonym of a detection), otherwise they are unsupported hallucinations.
# ---------------------------------------------------------------------------

_CONCRETE_OBJECT_NOUNS: frozenset = frozenset({
    # COCO class names (the full canonical list)
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
    # Multi-word COCO variants
    "trafficlight", "firehydrant", "stop_sign", "parkingmeter",
    "pottedplant", "diningtable", "cellphone", "wineglass",
    "baseballbat", "baseballglove", "tennisracket", "snowboard",
    "sportsball", "teddybear", "hairdrier",
    # Additional common physical objects that BLIP hallucinates
    "ball", "toy", "hat", "cap", "helmet", "glove", "mitt",
    "shoe", "shoes", "boot", "boots", "sandal", "sandals",
    "jacket", "coat", "shirt", "pants", "shorts", "skirt",
    "dress", "uniform", "sock", "socks", "scarf", "belt",
    "glasses", "sunglasses", "goggles",
    "watch", "wristwatch",
    "ring", "necklace", "earring", "bracelet",
    "key", "keys",
    "tablet", "ipad", "camera", "headphone", "headphones",
    "earphone", "earphones", "speaker", "microphone",
    "plate", "mug", "pitcher", "jug", "tray",
    "pan", "pot", "skillet",
    "newspaper", "magazine", "paper", "document", "letter",
    "envelope", "package", "box",
    "stick", "paddle", "club",
    "net", "goal", "hoop",
    "wheel", "tire", "seat", "handlebar", "pedal",
    "chain", "lock",
    "flag", "sign", "poster", "billboard",
    "coin", "money", "cash", "card", "ticket",
    "flower", "flowers",
    "lamp", "lantern", "candle", "flashlight",
    "pipe", "pole", "wire", "cable", "cord",
    "button", "switch", "handle", "knob", "lever",
    "basket", "bin",
    "pillow", "cushion", "blanket", "towel",
    "mat", "rug", "carpet", "curtain",
    "shelf", "shelves", "cabinet", "drawer", "wardrobe",
    "mirror", "painting",
    "stove", "cooktop", "dishwasher", "washer", "dryer",
    "vacuum", "broom", "mop", "bucket",
    "fan", "heater",
    "soap", "tissue", "napkin",
    "pen", "pencil", "marker", "crayon", "brush",
    "back pack", "hand bag",
    "mobile phone", "smart phone",
    "suit case",
    "snow board",
    "base ball", "base ball bat", "base ball glove",
    "tennis racket",
    "wine glass",
    "dinner table",
    "cell phone",
    "stop sign",
    "parking meter",
    "fire hydrant",
    "traffic light",
    "potted plant",
    "teddy bear",
    "hair drier",
    "hot dog",
    # YOLO detection output names with underscore
    "cell_phone", "dining_table", "potted_plant",
    "traffic_light", "fire_hydrant", "stop_sign",
    "parking_meter", "sports_ball", "baseball_bat",
    "baseball_glove", "tennis_racket", "wine_glass",
    "hot_dog", "teddy_bear", "hair_drier",
    "cellphone", "backpack", "handbag", "suitcase",
    "skateboard", "surfboard", "snowboard",
    "toothbrush",
    # Furniture / household
    "desk", "table", "stool", "ottoman", "armchair",
    "bookshelf", "bookcase", "nightstand", "dresser",
    "coffee table", "end table",
    "lamp", "floor lamp", "table lamp",
    # Common small objects
    "water bottle", "coffee cup", "tea cup",
    "paper cup", "plastic cup",
    "wine bottle", "beer bottle", "beer can",
    "soda can", "can",
    "paper bag", "plastic bag", "shopping bag",
    "grocery bag", "trash bag", "garbage bag",
    "trash can", "garbage can", "waste bin",
    # Animal-related
    "leash", "collar", "harness",
    "bone", "treat",
    # Baby/kid items
    "stroller", "pram", "car seat", "high chair",
    "baby bottle", "pacifier", "diaper",
    # Sports equipment
    "racket", "racquet", "bat", "glove", "mitt",
    "helmet", "pads", "uniform", "jersey",
    "cleats", "skates", "skate",
    # Outdoor objects
    "fountain", "statue", "sculpture",
    "bench", "picnic table", "grill", "barbecue",
    "mailbox", "fire hydrant", "lamppost", "streetlight",
    "crosswalk", "bridge", "fence", "railing",
    # Office objects
    "stapler", "tape", "ruler", "calculator",
    "folder", "binder", "notebook", "notepad",
    "clipboard", "printer", "scanner",
})

# Build expanded object set that includes all synonym canon values
_OBJECT_NOUNS: frozenset = _CONCRETE_OBJECT_NOUNS | frozenset(
    canon for canon in _SYNONYMS.values()
)
for canon in _SYNONYMS.values():
    parts = canon.replace("_", " ").split()
    _OBJECT_NOUNS = _OBJECT_NOUNS | frozenset(parts) | frozenset({canon.replace(" ", "_")})


_CANON_KEY_RE = re.compile(r"[^a-z]+")


def _canonical_object_key(phrase: str) -> str:
    """Collapse an object name to a spacing/punctuation-insensitive key.

    "surfboard", "surf board", "surf-board" and "Surf Board" all map to
    "surfboard". Without this, a detected `surfboard` and a caption that says
    "surf board" were treated as two unrelated objects: the caption phrase was
    scored as an unsupported hallucination and pushed the caption towards the
    safe-fallback template, while the real detection went unmatched.
    """
    return _CANON_KEY_RE.sub("", phrase.lower())


def _expand_object_aliases(name: str) -> set:
    """Every surface form that should be treated as the same object as `name`.

    Closes the synonym table in BOTH directions - a caption saying "motorbike"
    about a detected "motorcycle" was previously flagged as a hallucination
    because only synonym -> canonical was resolved, never canonical -> synonym.
    """
    name = name.lower().replace("_", " ").strip()
    forms = {name}
    canonical = _SYNONYMS.get(name)
    if canonical:
        forms.add(canonical)
    for syn, canon_form in _SYNONYMS.items():
        if canon_form == name or canon_form == canonical or syn == name:
            forms.add(syn)
            forms.add(canon_form)
    for f in list(forms):
        forms.update(f.split())
    return {f for f in forms if f}


def _build_label_word_set(detections: List[Detection]) -> frozenset:
    """All individual words across every detection label, lower-cased,
    plus known synonyms so that e.g. 'woman' matches a 'person' detection."""
    words: set = set()
    for d in detections:
        label = d["label"].lower().replace("_", " ")
        words.update(_expand_object_aliases(label))
    for w in list(words):
        if w in _INVERTED_SYNONYMS:
            words.update(_INVERTED_SYNONYMS[w])
    # Spacing-insensitive keys so "surf board" matches a "surfboard" detection.
    words.update(_canonical_object_key(w) for w in list(words))
    return frozenset(w for w in words if w)


def _extract_content_words(text: str) -> List[str]:
    """
    Lower-case tokens that could represent hallucinated objects.

    Excluded from penalisation:
      - stop words (function words / grammar)
      - scene words (relationship verbs, stance words)
      - safe context words (ambient descriptors: weather, time, terrain)
    Only substantive tokens that name concrete entities remain.
    """
    tokens = re.findall(r"[a-z]+", text.lower())
    return [
        t for t in tokens
        if t not in _STOP_WORDS
        and t not in _SCENE_WORDS
        and t not in _SAFE_CONTEXT_WORDS
        and len(t) > 2
    ]


def _unsupported_fraction(caption: str, label_words: frozenset) -> float:
    """
    Fraction of content words in the caption not covered by any detection label.

    Returns 0.0 when there are no content words (avoids divide-by-zero).
    """
    content = _extract_content_words(caption)
    if not content:
        return 0.0

    # A detection label written as two words in the caption ("surf board" for a
    # `surfboard` detection) is invisible to single-token matching, so both
    # halves used to count as unsupported. Mark the members of any adjacent
    # token pair whose concatenation matches a detection as supported.
    raw_tokens = re.findall(r"[a-z]+", caption.lower())
    split_label_parts: set = set()
    for first, second in zip(raw_tokens, raw_tokens[1:]):
        if _canonical_object_key(first + second) in label_words:
            split_label_parts.add(first)
            split_label_parts.add(second)

    unsupported = [
        w for w in content
        if w not in label_words
        and _canonical_object_key(w) not in label_words
        and w not in split_label_parts
    ]
    return len(unsupported) / len(content)


def _hedge_low_confidence(
    caption: str,
    detections: List[Detection],
    threshold: float = 0.5,
) -> str:
    """
    Prefix 'possibly' before article+label phrases for low-confidence detections.

    Processes longest labels first so "sports ball" is matched before "ball".
    Skips spans already preceded by a hedge marker.
    """
    low_conf = [
        d for d in sorted(detections, key=lambda x: -len(x["label"]))
        if d.get("score", 1.0) < threshold
    ]
    if not low_conf:
        return caption

    result = caption
    for d in low_conf:
        label = d["label"].replace("_", " ")
        pat   = re.compile(
            r'\b(a|an|the)\s+(' + re.escape(label) + r')\b',
            re.IGNORECASE,
        )
        matches = list(pat.finditer(result))
        # Reverse order so earlier offsets stay valid after insertions.
        for m in reversed(matches):
            prefix_chunk = result[max(0, m.start() - 30) : m.start()].lower()
            if not any(hw in prefix_chunk for hw in _HEDGE_MARKERS):
                result = result[: m.start()] + "possibly " + result[m.start():]
    return result


def _safe_fallback(detections: List[Detection]) -> str:
    """Conservative caption listing only confirmed detections."""
    labels = list(dict.fromkeys(
        d["label"].replace("_", " ") for d in detections
    ))
    if not labels:
        return "No objects detected in the scene."
    if len(labels) == 1:
        art = "an" if labels[0][0].lower() in "aeiou" else "a"
        return f"The scene contains {art} {labels[0]}."

    parts = []
    for lbl in labels:
        art = "an" if lbl[0].lower() in "aeiou" else "a"
        parts.append(f"{art} {lbl}")

    if len(parts) == 2:
        enumerated = f"{parts[0]} and {parts[1]}"
    else:
        enumerated = ", ".join(parts[:-1]) + f", and {parts[-1]}"
    return f"The scene contains {enumerated}, but the interaction is unclear."


def _classify_content_words(
    content_words: List[str],
    label_words: frozenset,
) -> Tuple[List[str], List[str]]:
    """Split content words into grounded and unsupported.

    Returns:
        (grounded, unsupported)
    """
    grounded: List[str] = []
    unsupported: List[str] = []
    for w in content_words:
        if w in label_words:
            grounded.append(w)
        else:
            unsupported.append(w)
    return grounded, unsupported


# ---------------------------------------------------------------------------
# Object noun extraction and caption repair
# ---------------------------------------------------------------------------

_GLOBAL_KNOWN_OBJECTS: frozenset | None = None


def _get_global_known_objects() -> frozenset:
    """Build and cache the set of all known object names + synonyms + parts."""
    global _GLOBAL_KNOWN_OBJECTS
    if _GLOBAL_KNOWN_OBJECTS is not None:
        return _GLOBAL_KNOWN_OBJECTS

    objs = set(_OBJECT_NOUNS)
    for syn, canon in _SYNONYMS.items():
        objs.add(syn)
        objs.add(canon)
        objs.update(canon.split())
        objs.update(canon.replace("_", " ").split())
        objs.update(syn.split())
    _GLOBAL_KNOWN_OBJECTS = frozenset(objs)
    return _GLOBAL_KNOWN_OBJECTS


def _extract_object_phrases(caption: str) -> List[str]:
    """Extract concrete object noun phrases from a caption.

    Matches multi-word phrases first (longest match), then single-word
    object nouns. Only returns phrases that correspond to known
    concrete, detectable objects.

    Returns:
        List of object phrase strings (lowercase, deduplicated, in order of appearance).
    """
    known = _get_global_known_objects()
    caption_lower = caption.lower()
    tokens = re.findall(r"[a-z]+", caption_lower)

    # Sort multi-word known objects by length descending for greedy matching
    multi_word = sorted([o for o in known if len(o.split()) > 1], key=lambda x: -len(x))

    found_phrases: List[str] = []
    remaining = caption_lower

    # Greedy multi-word match + replace matched spans so sub-words
    # (e.g. "phone" in "cell phone") are not extracted again.
    for phrase in multi_word:
        while phrase in remaining:
            found_phrases.append(phrase)
            remaining = remaining.replace(phrase, " " * len(phrase), 1)

    # Single-word match against REMAINING (non-replaced) text only
    remaining_tokens = set(re.findall(r"[a-z]+", remaining))
    for token in sorted(set(tokens)):
        if token in known and token in remaining_tokens and token not in found_phrases:
            found_phrases.append(token)

    return found_phrases


def _repair_caption(caption: str, hallucinated_phrase: str) -> str:
    """Remove a hallucinated object phrase from a caption, preserving grammar.

    Attempts to remove the object mention along with its surrounding
    grammatical context (preposition, article, etc.). Falls back to
    simply removing the bare phrase if no grammatical context matches.

    Returns:
        Repaired caption string, or original if repair not possible.
    """
    escaped = re.escape(hallucinated_phrase)

    # Patterns ordered from most to least aggressive removal
    patterns = [
        # "interacting with a cell phone" / "with a cell phone"
        rf'\b(interacting|playing|holding|using|carrying|wearing|'
        rf'watching|looking at|talking on|texting on|calling on|'
        rf'reaching for|grabbing|touching|holding onto|'
        rf'with|and)\s+(?:a|an|the|his|her|their)\s+{escaped}\b',
        # "a wooden table" / "the red chair" (article + adjective + noun)
        rf'\b(?:a|an|the|his|her|their|my|your|our)\s+\w+\s+{escaped}\b',
        # "a cell phone" / "the cell phone" / "his cell phone"
        rf'\b(?:a|an|the|his|her|their|my|your|our)\s+{escaped}\b',
        # "and cell phone" (without article)
        rf'\band\s+{escaped}\b',
        # bare phrase
        rf'\b{escaped}\b',
    ]

    result = caption
    for pat in patterns:
        match = re.search(pat, result, re.IGNORECASE)
        if match:
            before = result[:match.start()].rstrip()
            after = result[match.end():].lstrip()

            # Clean trailing "and", comma
            before = re.sub(r',?\s*and\s*$', '', before).rstrip()
            before = re.sub(r',\s*$', '', before).rstrip()
            before = re.sub(r'\s+and\s*$', '', before).rstrip()

            result = (before + " " + after).strip()
            break

    # Final cleanup
    result = re.sub(r'\s+', ' ', result).strip()
    result = re.sub(r',\s*,', ',', result)
    result = re.sub(r'\s+\.', '.', result)
    result = re.sub(r'^\s*,', '', result)
    result = re.sub(r',\s*$', '', result)
    result = re.sub(r'\.\s*\.', '.', result)
    # Remove dangling auxiliaries ("woman is" → "woman")
    result = re.sub(r'\s+(is|are|was|were|been|being)\s*$', '', result)
    result = re.sub(r'\s+(is|are|was|were|been|being)\s+', ' ', result)
    # Remove dangling prepositions at end (including "on." / "on,")
    result = re.sub(r'\s+(a|an|the|to|for|with|at|on|in|by)\s*[.,;!?]*\s*$', '', result)
    # Remove trailing dangling adjectives ("wooden", "old", etc.)
    result = re.sub(r'\s+\w+[enly]{2}\s*$', '', result)
    result = result.strip().strip(",").strip()
    # Remove leading/trailing "and", commas after removal
    result = re.sub(r'^\s*and\s+', '', result)
    result = re.sub(r'\s+and\s*$', '', result)
    result = re.sub(r',\s*,', ',', result)
    result = re.sub(r'^\s*,', '', result)
    result = re.sub(r',\s*$', '', result)

    return result if result else caption


def _check_unsupported_objects(
    caption: str,
    label_words: frozenset,
) -> Tuple[List[str], List[str]]:
    """Strictly check which object phrases in the caption are unsupported.

    For each extracted object phrase, checks if it (or its canonical
    synonym form) exists in the detected label set.

    Returns:
        (unsupported_phrases, supported_phrases)
    """
    object_phrases = _extract_object_phrases(caption)

    # Build reverse mapping: synonym_part → canonical_form
    synonym_to_canon: Dict[str, str] = {}
    for syn, canon in _SYNONYMS.items():
        synonym_to_canon[syn] = canon
        for part in syn.split():
            synonym_to_canon[part] = canon
        for part in canon.split():
            synonym_to_canon[part] = canon

    unsupported: List[str] = []
    supported: List[str] = []

    for phrase in object_phrases:
        # Check label_words directly (includes full phrases + individual words + synonyms)
        if phrase in label_words:
            supported.append(phrase)
            continue

        # Spacing/punctuation-insensitive match ("surf board" vs "surfboard").
        if _canonical_object_key(phrase) in label_words:
            supported.append(phrase)
            continue

        # Check canonical form, in both synonym directions.
        canon = synonym_to_canon.get(phrase)
        if canon and (canon in label_words
                      or _canonical_object_key(canon) in label_words):
            supported.append(phrase)
            continue
        if _expand_object_aliases(phrase) & set(label_words):
            supported.append(phrase)
            continue

        # Check if any part of a multi-word phrase is in label_words
        if len(phrase.split()) > 1:
            parts_covered = any(part in label_words for part in phrase.split())
            if parts_covered:
                supported.append(phrase)
                continue

        unsupported.append(phrase)

    return unsupported, supported


def _print_gate_decision(
    decision: str,
    reasons_accept: List[str],
    reasons_reject: List[str],
    caption: str,
    detections: List[Detection],
) -> None:
    """Print structured debug output for the gating decision."""
    det_labels = [d["label"] for d in detections]
    print(f"\n[gate] {'=' * 50}")
    print(f"[gate] DECISION: {decision}")
    print(f"[gate] {'=' * 50}")
    print(f"[gate] Raw caption:  {caption[:120]}")
    print(f"[gate] Detections:   {det_labels}")
    if reasons_accept:
        for r in reasons_accept:
            print(f"[gate]   + {r}")
    if reasons_reject:
        for r in reasons_reject:
            print(f"[gate]   - {r}")
    print(f"[gate] {'=' * 50}")


def gate_caption(
    caption: str,
    detections: List[Detection],
    unsupported_threshold: float = 0.4,
    confidence_threshold: float = 0.5,
    _depth: int = 0,
) -> str:
    """
    Post-process a raw BLIP caption using balanced grounded validation.

    STRICT on object nouns: any concrete object mentioned must be detected.
    SOFT on verbs/scene: interactions are never penalised.
    PERMISSIVE on context: ambient/scene words pass through.
    Partial repair: single hallucinated object → remove that phrase, keep rest.

    Args:
        caption:               Raw decoded model output.
        detections:            Detection list used for grounding.
        unsupported_threshold: Maximum tolerated fraction of non-object
                               unsupported content words (default 0.40).
        confidence_threshold:  Detections below this score get hedged (default 0.50).
        _depth:                Internal recursion guard (default 0).

    Returns:
        Grounded, clean caption string.
    """
    if not detections or _depth > 3:
        return caption if _depth <= 3 else _safe_fallback(detections)

    reasons_accept: List[str] = []
    reasons_reject: List[str] = []

    label_words = _build_label_word_set(detections)

    # Step 1 — hedge low-confidence entities.
    caption = _hedge_low_confidence(caption, detections, threshold=confidence_threshold)

    # Step 2 — STRICT object noun checking.
    # If a caption mentions a concrete object that is NOT detected, it is an
    # unsupported hallucination — regardless of the overall content ratio.
    unsupported_objects, supported_objects = _check_unsupported_objects(caption, label_words)

    if unsupported_objects:
        for obj in unsupported_objects:
            reasons_reject.append(f"unsupported object: '{obj}'")

        if len(unsupported_objects) == 1:
            # ── Single unsupported object → attempt lightweight repair ──
            repaired = _repair_caption(caption, unsupported_objects[0])
            obj_still_present = unsupported_objects[0] in repaired.lower()

            if repaired != caption and not obj_still_present:
                reasons_accept.append(
                    f"repair applied: removed unsupported object phrase"
                )
                _print_gate_decision(
                    "REPAIRED", reasons_accept, reasons_reject,
                    caption, detections,
                )
                # Re-check repaired caption (recursion depth guarded)
                return gate_caption(
                    repaired, detections,
                    unsupported_threshold=unsupported_threshold,
                    confidence_threshold=confidence_threshold,
                    _depth=_depth + 1,
                )
            else:
                reasons_reject.append("repair failed to remove object")
                _print_gate_decision(
                    "REJECTED", reasons_accept, reasons_reject,
                    caption, detections,
                )
                return _safe_fallback(detections)
        else:
            # ── Multiple unsupported objects → full fallback ──
            reasons_reject.append(
                f"caption rejected: multiple unsupported objects"
            )
            _print_gate_decision(
                "REJECTED", reasons_accept, reasons_reject,
                caption, detections,
            )
            return _safe_fallback(detections)

    # Step 3 — SOFT verb / general content check.
    # Only applies to non-object content words (actions, descriptors, etc.)
    # using the configured unsupported_threshold.
    content_words = _extract_content_words(caption)
    if not content_words:
        if supported_objects:
            reasons_accept.append(f"objects grounded ({', '.join(supported_objects)})")
        _print_gate_decision(
            "ACCEPTED", reasons_accept, reasons_reject,
            caption, detections,
        )
        return caption

    grounded, unsupported = _classify_content_words(content_words, label_words)
    unsup_frac = len(unsupported) / len(content_words)

    if supported_objects:
        reasons_accept.append(f"objects grounded ({', '.join(supported_objects)})")
    if grounded:
        reasons_accept.append(f"content grounded ({', '.join(grounded[:5])})")
    if unsupported:
        reasons_accept.append(f"minor ungrounded words ({', '.join(unsupported[:3])})")

    if unsup_frac > unsupported_threshold:
        reasons_reject.append(
            f"excessive unsupported content ({unsup_frac:.2f})"
        )
        _print_gate_decision(
            "REJECTED", reasons_accept, reasons_reject,
            caption, detections,
        )
        return _safe_fallback(detections)

    _print_gate_decision(
        "ACCEPTED", reasons_accept, reasons_reject,
        caption, detections,
    )
    return caption


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Candidate generation for CLIP reranking (BLIP-2 + CLIP baseline)
# ---------------------------------------------------------------------------

def generate_blip_candidates(
    image,
    max_new_tokens: int = 128,
    num_beams: int = 5,
    num_candidates: int = 5,
    use_sampling: bool = False,
    temperature: float = 0.7,
) -> List[str]:
    """
    Generate multiple caption candidates from BLIP-2 for CLIP reranking.

    IMPORTANT: This is the UNGROUNDED BLIP-2 baseline - no detections,
    no prompt constraints. Generates free-form captions like the pure
    BLIP-2 baseline, but produces multiple candidates for reranking.

    This creates a proper semantic enhancement baseline:
    - Start with implicit captioning (BLIP-2)
    - Use CLIP for semantic visual alignment (reranking)
    - No explicit grounding, no scene graphs

    Args:
        image: PIL Image or CHW torch.Tensor
        max_new_tokens: Token budget for generation
        num_beams: Beam search width (when using beam search)
        num_candidates: Number of candidates to generate
        use_sampling: If True, use multinomial sampling instead of beam search
        temperature: Sampling temperature (only used if use_sampling=True)

    Returns:
        List of candidate caption strings (deduplicated)
    """
    global _processor, _model, _device

    if _model is None:
        _load_model()
    else:
        print("[BLIP] Reusing cached caption model")

    t0 = time.time()
    pil_image = _to_pil(image)

    # Ungrounded baseline: the prefix is a constant, independent of the image
    # content, detections and relations.
    inputs = _processor(
        images=pil_image,
        text=BASELINE_PREFIX,
        return_tensors="pt",
    ).to(_device)

    with torch.inference_mode():
        if use_sampling:
            output_ids = _model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=num_candidates,
            )
        else:
            output_ids = _model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=max(num_beams, num_candidates),
                num_return_sequences=num_candidates,
                early_stopping=True,
            )

    candidates = [
        _processor.decode(ids, skip_special_tokens=True).strip()
        for ids in output_ids
    ]

    seen = set()
    unique_candidates = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            unique_candidates.append(c)

    gen_time = time.time() - t0
    print(f"[BLIP] Generated {len(unique_candidates)} candidates in {gen_time:.2f}s on {_device}.")

    if _device.type == "cuda":
        torch.cuda.empty_cache()

    return unique_candidates


def generate_blip_baseline(
    image,
    max_new_tokens: int = 128,
    num_beams: int = 4,
) -> str:
    """
    Generate a single ungrounded caption from BLIP-2 — PURE BASELINE.

    This is the clean implicit captioning baseline:
    - No detections
    - No relations
    - No prompt constraints
    - No evidence gating

    Uses the SAME ungrounded prompt as generate_blip_candidates()
    for consistency in the BLIP-2 + CLIP reranking comparison.

    Pipeline: Image → BLIP-2 → Caption

    Args:
        image: PIL Image or CHW torch.Tensor
        max_new_tokens: Token budget for generation
        num_beams: Beam search width

    Returns:
        Single caption string (deterministic beam search output)
    """
    global _processor, _model, _device

    if _model is None:
        _load_model()
    else:
        print("[BLIP] Reusing cached caption model")

    t0 = time.time()
    pil_image = _to_pil(image)

    # Ungrounded baseline: the prefix is a constant, independent of the image
    # content, detections and relations.
    inputs = _processor(
        images=pil_image,
        text=BASELINE_PREFIX,
        return_tensors="pt",
    ).to(_device)

    with torch.inference_mode():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
        )

    caption = _processor.decode(output_ids[0], skip_special_tokens=True).strip()
    gen_time = time.time() - t0
    print(f"[BLIP] Baseline caption generated in {gen_time:.2f}s on {_device}.")

    if _device.type == "cuda":
        torch.cuda.empty_cache()

    return caption


def generate_blip_caption(
    image,
    detections: List[Detection],
    relationships: List[Relationship],
    max_new_tokens: int = 128,
    num_beams: int = 4,
    unsupported_threshold: float = 0.4,
    confidence_threshold: float = 0.5,
) -> str:
    """
    Generate a grounded scene caption using BLIP.

    Runs evidence gating after decoding so the returned string only contains
    claims supported by the provided detections.

    Args:
        image:                 PIL Image or CHW torch.Tensor.
        detections:            [{"label": str, "box": [...], "score": float}, ...]
        relationships:         [(subject, relation, object), ...]
        max_new_tokens:        Token budget for the generated caption.
        num_beams:             Beam-search width; 1 = greedy (fastest).
        unsupported_threshold: Passed to gate_caption (default 0.40).
        confidence_threshold:  Passed to gate_caption (default 0.50).

    Returns:
        A clean, grounded caption string.
    """
    global _processor, _model, _device

    if _model is None:
        _load_model()
    else:
        print("[BLIP] Reusing cached caption model")

    t0 = time.time()
    pil_image = _to_pil(image)

    # Condition on the verified scene graph via the decoder prefix. The
    # (subject, relation, object) triples reaching this entry point are plain
    # tuples with no confidence attached, so they are treated as verified and
    # injected directly. Passing the constant "a photo of" here - as this
    # function used to - made the "grounded" caption byte-identical to the
    # ungrounded baseline for every image.
    rel_dicts = [
        {"subject": subj, "predicate": rel.replace("_", " "), "object": obj,
         "confidence": 1.0}
        for subj, rel, obj in (relationships or [])
    ]
    prefix, _used = build_blip_prefix(detections, rel_dicts)
    inputs = _processor(
        images=pil_image,
        text=prefix,
        return_tensors="pt",
    ).to(_device)

    with torch.inference_mode():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
        )

    raw_caption = _strip_prefix_artifacts(
        _processor.decode(output_ids[0], skip_special_tokens=True), prefix,
    )
    gen_time = time.time() - t0
    print(f"[BLIP] Grounded caption generated in {gen_time:.2f}s on {_device}.")
    print(f"[BLIP]   conditioning prefix: {prefix!r}")

    if _device.type == "cuda":
        torch.cuda.empty_cache()

    return gate_caption(
        raw_caption,
        detections,
        unsupported_threshold=unsupported_threshold,
        confidence_threshold=confidence_threshold,
    )


def generate_blip_semantic_caption(
    image,
    detections: List[Detection],
    relations: List[Dict],
    max_new_tokens: int = 128,
    num_beams: int = 4,
    unsupported_threshold: float = 0.4,
    confidence_threshold: float = 0.5,
) -> Tuple[str, str, str, List[str]]:
    """
    Generate a grounded caption using the redesigned semantic prompt.

    Pipeline:
        1. Verbalize structured relations into natural interaction descriptions
        2. Build semantic prompt with scene elements + likely interactions
        3. Generate with BLIP
        4. Evidence gating (same as existing pipeline)

    Args:
        image:                 PIL Image or torch.Tensor.
        detections:            [{"label": str, "box": [...], "score": float}, ...]
        relations:             [{"subject": str, "predicate": str, "object": str,
                                "confidence": float}, ...]
        max_new_tokens:        Token budget.
        num_beams:             Beam search width.
        unsupported_threshold: Gating threshold.
        confidence_threshold:  Hedging threshold.

    Returns:
        (raw_caption, gated_caption, prompt, verbalized_relations)
    """
    global _processor, _model, _device

    if _model is None:
        _load_model()
    else:
        print("[BLIP] Reusing cached caption model")

    t0 = time.time()
    pil_image = _to_pil(image)

    # Step 2+3 - Verbalize relations with confidence-aware language.
    # Use adjusted_confidence (which includes semantic prior) when available,
    # falling back to raw model confidence.
    verbalized = [
        verbalize_relation(
            r["subject"], r["predicate"], r["object"],
            r.get("adjusted_confidence", r.get("confidence", 0.5)),
        )
        for r in relations
    ]

    # Step 6 - Build the human-readable evidence block. This is reported and
    # logged, but it is NOT what BLIP generates from: BLIP-base is not
    # instruction-tuned and cannot follow it (see build_blip_prefix).
    prompt = build_semantic_prompt(detections, verbalized)

    # Step 7 - Condition generation on the evidence through the decoder prefix,
    # which is the channel BLIP-base actually responds to. Falls back to the
    # plain baseline opening when no relation clears the confidence bar.
    prefix, grounding_relations = build_blip_prefix(detections, relations)
    inputs = _processor(
        images=pil_image,
        text=prefix,
        return_tensors="pt",
    ).to(_device)

    with torch.inference_mode():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
        )

    raw_caption = _strip_prefix_artifacts(
        _processor.decode(output_ids[0], skip_special_tokens=True), prefix,
    )
    gen_time = time.time() - t0
    print(f"[BLIP] Semantic caption generated in {gen_time:.2f}s on {_device}.")
    print(f"[BLIP]   conditioning prefix: {prefix!r} "
          f"({len(grounding_relations)} relation(s) injected)")

    if _device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Relation-grounded caption correction (Steps 1-7) ──────────────
    # Correct unsupported actions using grounded relations before gating.
    corrected_raw, correction_log = correct_caption_relations(
        raw_caption, detections, relations, debug=True,
    )

    # Step 9 - Evidence gating (unchanged)
    gated = gate_caption(
        corrected_raw,
        detections,
        unsupported_threshold=unsupported_threshold,
        confidence_threshold=confidence_threshold,
    )

    return raw_caption, gated, prompt, verbalized
