"""Deterministic COCO-80 object-mention extraction for the caption experiment.

One mapper, used on BOTH sides of the hallucination comparison
--------------------------------------------------------------
CHAIR and POPE compare "objects a caption mentions" against "objects a human
annotated". If those two sets are produced by different vocabularies, the
metric measures the vocabulary gap rather than hallucination: e.g. a caption
saying "a man at a table" is scored as hallucinating `dining table` whenever
the ground-truth mapper does not also turn the VG name "table" into
`dining table`. So `mentions()` is applied to captions AND to every Visual
Genome object name in build_caption_eval_set.py.

Why not utils.metrics.detect_coco_objects
-----------------------------------------
That function (used by the legacy hallucination_eval.py) has defects that bias
CHAIR in ways that differ between captions of different content:

* no plural handling: "two dogs" and "cars" are not detected at all, so a
  hallucinated plural object is invisible to CHAIR;
* multi-word labels are matched as substrings and their words are then matched
  AGAIN as single tokens, so "a hot dog" is scored as `hot dog` + `dog` and
  "a teddy bear" as `teddy bear` + `bear` - a phantom hallucination;
* open-ended WordNet hypernym walks decide what a word means, which is neither
  inspectable nor stable across NLTK/WordNet versions, and it needs NLTK.

This module is a closed, explicit table: every surface form that maps to a
COCO class is listed below, noun-noun compounds whose first word is a COCO
name but which do not assert that object ("bus stop", "train tracks",
"baseball game") are listed as exclusions, and matching is greedy
longest-phrase-first over tokens, consuming the tokens it matches. The synonym
choices follow the spirit of the CHAIR synonym list (Rohrbach et al., 2018)
but are deliberately more conservative for ambiguous words (e.g. "computer",
"plant", "seat", "baseball" alone are NOT mapped). Known remaining ambiguity:
"orange" is always the fruit, "mouse" always the device.
"""

from __future__ import annotations

import re
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

MENTION_MAPPER_VERSION = "coco_mentions/1"

COCO_80: FrozenSet[str] = frozenset({
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
})

# canonical COCO class -> singular surface forms (lower-case, may be phrases)
_SURFACE_FORMS: Dict[str, Tuple[str, ...]] = {
    "person": (
        "person", "man", "woman", "boy", "girl", "child", "kid", "lady", "guy",
        "gentleman", "adult", "baby", "toddler", "teenager", "teen", "people",
        "player", "tennis player", "baseball player", "soccer player",
        "skier", "snowboarder", "surfer", "skateboarder", "skater", "rider",
        "biker", "cyclist", "bicyclist", "motorcyclist", "pedestrian",
        "worker", "chef", "police officer", "policeman", "officer", "soldier",
        "catcher", "batter", "umpire", "passenger", "driver", "bride",
        "groom", "student", "doctor", "nurse", "farmer", "cowboy", "athlete",
        "jockey", "fireman", "firefighter", "businessman", "customer",
        "spectator",
    ),
    "bicycle": ("bicycle", "bike"),
    "car": (
        "car", "automobile", "sedan", "suv", "taxi", "cab", "van", "minivan",
        "jeep", "police car", "race car", "sports car", "limousine", "limo",
        "hatchback",
    ),
    "motorcycle": (
        "motorcycle", "motorbike", "motor bike", "motor cycle", "scooter",
        "moped", "dirt bike",
    ),
    "airplane": (
        "airplane", "plane", "aeroplane", "air plane", "jet", "airliner",
        "jetliner", "aircraft", "fighter jet",
    ),
    "bus": ("bus", "minibus", "school bus", "double decker"),
    "train": ("train", "locomotive", "tram"),
    "truck": (
        "truck", "pickup", "pickup truck", "lorry", "fire truck", "firetruck",
        "semi truck",
    ),
    "boat": (
        "boat", "ship", "sailboat", "yacht", "canoe", "kayak", "ferry",
        "rowboat", "row boat", "speedboat", "lifeboat", "life boat",
    ),
    "traffic light": ("traffic light", "traffic signal", "stop light", "stoplight"),
    "fire hydrant": ("fire hydrant", "hydrant"),
    "stop sign": ("stop sign",),
    "parking meter": ("parking meter",),
    "bench": ("bench", "park bench", "pew"),
    "bird": (
        "bird", "pigeon", "seagull", "gull", "duck", "duckling", "goose",
        "swan", "parrot", "sparrow", "crow", "raven", "eagle", "hawk", "owl",
        "pelican", "flamingo", "rooster", "hen", "heron", "penguin", "ostrich",
        "peacock", "robin",
    ),
    "cat": ("cat", "kitten", "kitty", "feline"),
    "dog": (
        "dog", "puppy", "pup", "doggy", "doggie", "canine", "labrador",
        "retriever", "poodle", "terrier", "bulldog", "beagle", "chihuahua",
        "husky", "sheepdog", "dalmatian", "pug", "corgi",
    ),
    "horse": ("horse", "pony", "foal", "stallion", "mare", "colt"),
    "sheep": ("sheep", "lamb", "ewe"),
    "cow": ("cow", "cattle", "calf", "bull", "ox", "heifer"),
    "elephant": ("elephant",),
    "bear": ("bear", "grizzly", "polar bear"),
    "zebra": ("zebra",),
    "giraffe": ("giraffe",),
    "backpack": ("backpack", "back pack", "knapsack", "rucksack"),
    "umbrella": ("umbrella", "parasol"),
    "handbag": ("handbag", "hand bag", "purse"),
    "tie": ("tie", "necktie", "neck tie", "bow tie", "bowtie"),
    "suitcase": ("suitcase", "suit case", "luggage"),
    "frisbee": ("frisbee", "flying disc"),
    "skis": ("skis", "ski"),
    "snowboard": ("snowboard", "snow board"),
    "sports ball": ("ball", "soccer ball", "tennis ball", "beach ball"),
    "kite": ("kite",),
    "baseball bat": ("baseball bat", "bat"),
    "baseball glove": ("baseball glove", "baseball mitt", "mitt"),
    "skateboard": ("skateboard", "skate board"),
    "surfboard": ("surfboard", "surf board"),
    "tennis racket": ("tennis racket", "tennis racquet", "racket", "racquet"),
    "bottle": ("bottle",),
    "wine glass": ("wine glass", "wineglass"),
    "cup": ("cup", "mug", "teacup", "coffee cup"),
    "fork": ("fork",),
    "knife": ("knife",),
    "spoon": ("spoon",),
    "bowl": ("bowl",),
    "banana": ("banana",),
    "apple": ("apple",),
    "sandwich": ("sandwich", "burger", "hamburger", "cheeseburger"),
    "orange": ("orange",),
    "broccoli": ("broccoli",),
    "carrot": ("carrot",),
    "hot dog": ("hot dog", "hotdog"),
    "pizza": ("pizza",),
    "donut": ("donut", "doughnut"),
    "cake": ("cake", "cupcake", "cheesecake"),
    "chair": ("chair", "stool", "armchair", "high chair", "highchair", "bar stool"),
    "couch": ("couch", "sofa", "loveseat", "love seat", "settee", "futon"),
    "potted plant": ("potted plant", "houseplant", "house plant"),
    "bed": ("bed",),
    "dining table": (
        "dining table", "table", "desk", "coffee table", "kitchen table",
        "picnic table",
    ),
    "toilet": ("toilet", "urinal", "commode", "toilet seat"),
    "tv": ("tv", "television", "televison", "monitor", "computer monitor"),
    "laptop": ("laptop", "laptop computer", "notebook computer", "netbook", "macbook"),
    "mouse": ("mouse", "computer mouse"),
    "remote": (
        "remote", "remote control", "remote controller", "wii remote",
        "wii controller", "game controller", "controller",
    ),
    "keyboard": ("keyboard", "computer keyboard"),
    "cell phone": (
        "cell phone", "cellphone", "phone", "mobile phone", "smartphone",
        "smart phone", "iphone",
    ),
    "microwave": ("microwave", "microwave oven"),
    "oven": ("oven", "stove", "stovetop", "stove top"),
    "toaster": ("toaster", "toaster oven"),
    "sink": ("sink",),
    "refrigerator": ("refrigerator", "fridge", "freezer"),
    "book": ("book", "novel", "textbook"),
    "clock": ("clock",),
    "vase": ("vase",),
    "scissors": ("scissors", "scissor"),
    "teddy bear": ("teddy bear", "teddy", "teddybear"),
    "hair drier": ("hair drier", "hair dryer", "hairdryer", "blow dryer", "blow drier"),
    "toothbrush": ("toothbrush", "tooth brush"),
}

# Compounds containing a COCO surface form that do NOT assert that object, or
# that assert a different one. Value None = no object; a string = that object.
_COMPOUNDS: Dict[str, Optional[str]] = {
    "bus stop": None, "bus station": None, "bus lane": None,
    "train station": None, "train track": None, "train platform": None,
    "railroad track": None,
    "car seat": None, "cable car": None, "train car": "train",
    "truck bed": "truck",
    "street light": None, "streetlight": None,
    "toilet paper": None,
    "flower bed": None, "bed sheet": None,
    "book shelf": None, "book store": None,
    "ski slope": None, "ski lift": None, "ski resort": None, "ski pole": None,
    "ski jacket": None, "ski boot": None, "ski suit": None, "ski goggle": None,
    "ski mask": None,
    "tennis court": None, "baseball game": None, "baseball field": None,
    "baseball cap": None, "baseball hat": None, "baseball uniform": None,
    "baseball team": None, "baseball diamond": None, "baseball stadium": None,
    "soccer field": None, "soccer game": None,
    "jet ski": None, "jet stream": None,
    "horse drawn carriage": "horse", "horse race": "horse",
    "dog bed": "dog", "dog leash": "dog", "cat food": None, "dog food": None,
    "pizza box": None, "pizza cutter": None, "cake stand": None,
    "clock tower": "clock",
    "phone booth": None, "phone case": None,
    "tie dye": None,
    "table cloth": None,
    "bull dog": "dog", "pit bull": "dog",
    "baby carriage": None, "baby stroller": None,
    "chair lift": None, "wheel chair": None, "table tennis": None,
    "orange juice": None, "remote area": None,
    "wine bottle": "bottle",
    "hot dog bun": "hot dog",
}

_ANIMALS = ("bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
            "zebra", "giraffe")

# "baby elephant" is an elephant, not a person plus an elephant.
for _animal in _ANIMALS:
    for _form in _SURFACE_FORMS[_animal]:
        if " " not in _form:
            _COMPOUNDS.setdefault(f"baby {_form}", _animal)

_IRREGULAR_PLURALS: Dict[str, str] = {
    "men": "man", "women": "woman", "children": "child", "people": "people",
    "persons": "person", "mice": "mouse", "knives": "knife", "geese": "goose",
    "oxen": "ox", "teeth": "tooth", "feet": "foot", "calves": "calf",
    "wolves": "wolf", "shelves": "shelf", "leaves": "leaf", "buses": "bus",
    "busses": "bus", "policemen": "policeman", "firemen": "fireman",
    "businessmen": "businessman", "gentlemen": "gentleman", "ladies": "lady",
}

_TOKEN_RE = re.compile(r"[a-z]+")
_MAX_PHRASE = 3


def _build_phrase_table() -> Dict[Tuple[str, ...], Optional[str]]:
    table: Dict[Tuple[str, ...], Optional[str]] = {}
    for canonical, forms in _SURFACE_FORMS.items():
        if canonical not in COCO_80:
            raise ValueError(f"surface-form table names non-COCO class {canonical!r}")
        for form in forms:
            key = tuple(form.split())
            if key in table and table[key] != canonical:
                raise ValueError(f"surface form {form!r} maps to two classes")
            table[key] = canonical
    for phrase, canonical in _COMPOUNDS.items():
        if canonical is not None and canonical not in COCO_80:
            raise ValueError(f"compound {phrase!r} maps to non-COCO {canonical!r}")
        table[tuple(phrase.split())] = canonical
    if max(len(k) for k in table) > _MAX_PHRASE:
        raise ValueError("phrase longer than _MAX_PHRASE")
    return table


_PHRASES: Dict[Tuple[str, ...], Optional[str]] = _build_phrase_table()
_VOCAB_WORDS: FrozenSet[str] = frozenset(w for key in _PHRASES for w in key)


def _singular(token: str) -> str:
    """Map a token onto the vocabulary's singular form when one exists.

    Only returns a changed token if the change lands on a word the table knows,
    so ordinary words ("was", "glass", "this") are never mangled into something
    that matches.
    """
    if token in _VOCAB_WORDS:
        return token
    irregular = _IRREGULAR_PLURALS.get(token)
    if irregular is not None and irregular in _VOCAB_WORDS:
        return irregular
    candidates: List[str] = []
    if token.endswith("ies") and len(token) > 4:
        candidates.append(token[:-3] + "y")
    if token.endswith("ves") and len(token) > 4:
        candidates += [token[:-3] + "f", token[:-3] + "fe"]
    if token.endswith("es") and len(token) > 3:
        candidates.append(token[:-2])
    if token.endswith("s") and not token.endswith("ss") and len(token) > 2:
        candidates.append(token[:-1])
    for cand in candidates:
        if cand in _VOCAB_WORDS:
            return cand
    return token


def tokenize(text: str) -> List[str]:
    return [_singular(t) for t in _TOKEN_RE.findall(text.lower())]


def mention_spans(text: str) -> List[Tuple[str, Optional[str]]]:
    """Every matched phrase in order: (matched singular phrase, COCO class or None)."""
    tokens = tokenize(text)
    spans: List[Tuple[str, Optional[str]]] = []
    i = 0
    while i < len(tokens):
        for n in range(min(_MAX_PHRASE, len(tokens) - i), 0, -1):
            key = tuple(tokens[i:i + n])
            if key in _PHRASES:
                spans.append((" ".join(key), _PHRASES[key]))
                i += n
                break
        else:
            i += 1
    return spans


def mentions(text: str) -> Set[str]:
    """The set of COCO-80 classes a piece of text asserts."""
    return {cls for _, cls in mention_spans(text) if cls is not None}


def objects_from_names(names: Iterable[str]) -> Set[str]:
    """COCO-80 classes named by a collection of annotation names (GT side)."""
    found: Set[str] = set()
    for name in names:
        if name:
            found |= mentions(str(name))
    return found
