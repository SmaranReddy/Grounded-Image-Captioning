"""
Inference for the learned relation predictor.

Two public entry points:

1.  predict_relation(subject, obj_label, box1, box2, ...)
        Returns the most likely predicate string for a single pair,
        or None when the model is not confident enough.

2.  infer_relationships_learned(detections, ...)
        Drop-in replacement for utils.causal_caption.infer_relationships().
        Accepts the same detection-dict list and returns the same
        List[Tuple[str, str, str]] format.

Loading
-------
The module loads the trained checkpoint lazily on first call.
The model can be trained in two modes:
    - Geometry-only (clip_dim=0): uses only label embeddings + geometry
    - Visual-semantic (clip_dim>0): adds CLIP embeddings from image crops

When using a visual-semantic model, you MUST pass the image to inference.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from .model import RelationMLP
from .relation_transformer import RelationTransformer
from .vg_dataset import (
    ALLOWED_PREDICATES,
    Vocab,
    extract_geo_features,
    normalize_label,
    GEO_DIM,
    GEO_DIM_EXT,
    geo_extractor,
    POSE_FEATURE_DIM,
    UNION_FEATURE_DIM,
)
from .clip_extractor import CLIPExtractor, CLIP_DIM
from .pose_extractor import PoseExtractor, POSE_FEATURE_DIM


# ---------------------------------------------------------------------------
# Step 1 — Object type categories (lightweight semantic type groups)
# ---------------------------------------------------------------------------

ANIMATE: frozenset = frozenset({
    "person", "dog", "horse", "cat", "bird",
    "cow", "sheep", "elephant", "bear", "zebra", "giraffe",
})

WEARABLE: frozenset = frozenset({
    "backpack", "handbag", "tie", "suitcase",
})

RIDEABLE: frozenset = frozenset({
    "bicycle", "horse", "motorcycle", "skateboard", "surfboard",
    "skis", "snowboard",
})

HANDHELD: frozenset = frozenset({
    "cell phone", "umbrella", "bottle", "cup", "book",
    "fork", "knife", "spoon", "bowl", "frisbee",
    "kite", "baseball bat", "baseball glove", "tennis racket",
    "remote", "keyboard", "mouse", "scissors", "toothbrush",
    "wine glass", "hot dog", "apple", "banana", "orange",
    "sandwich", "donut", "cake", "pizza", "carrot", "broccoli",
})

FURNITURE: frozenset = frozenset({
    "chair", "couch", "bench", "bed", "dining table", "toilet",
})

_OBJECT_CATEGORIES: Dict[str, frozenset] = {
    "animate": ANIMATE,
    "wearable": WEARABLE,
    "rideable": RIDEABLE,
    "handheld": HANDHELD,
    "furniture": FURNITURE,
}


def _get_categories(label: str) -> List[str]:
    cats: List[str] = []
    for cat_name, cat_set in _OBJECT_CATEGORIES.items():
        if label in cat_set:
            cats.append(cat_name)
    return cats


# ---------------------------------------------------------------------------
# Step 2 — Predicate validity priors (soft compatibility rules)
#
# These are SOFT scoring adjustments, NOT hard rules.
# The learned MLP remains the primary predictor.
# ---------------------------------------------------------------------------

_PREDICATE_PRIORS: Dict[str, Dict] = {
    "riding": {
        "requires_animate_subject": True,
        "preferred_object_cats": {"rideable"},
        "unsuitable_object_cats": {"furniture", "wearable", "handheld"},
        "animate_subject_bonus": 0.12,
        "inanimate_subject_penalty": -0.35,
        "preferred_object_bonus": 0.10,
        "unsuitable_object_penalty": -0.20,
    },
    "wearing": {
        "requires_animate_subject": True,
        "preferred_object_cats": {"wearable"},
        "unsuitable_object_cats": {"furniture", "rideable", "animate"},
        "animate_subject_bonus": 0.12,
        "inanimate_subject_penalty": -0.35,
        "preferred_object_bonus": 0.10,
        "unsuitable_object_penalty": -0.25,
    },
    "holding": {
        "requires_animate_subject": True,
        "preferred_object_cats": {"handheld"},
        "unsuitable_object_cats": {"furniture", "animate"},
        "animate_subject_bonus": 0.12,
        "inanimate_subject_penalty": -0.35,
        "preferred_object_bonus": 0.08,
        "unsuitable_object_penalty": -0.20,
    },
    "carrying": {
        "requires_animate_subject": True,
        "preferred_object_cats": {"handheld", "wearable"},
        "unsuitable_object_cats": {"furniture"},
        "animate_subject_bonus": 0.12,
        "inanimate_subject_penalty": -0.35,
        "preferred_object_bonus": 0.08,
        "unsuitable_object_penalty": -0.15,
    },
    "looking at": {
        "requires_animate_subject": True,
        "preferred_object_cats": set(),
        "unsuitable_object_cats": set(),
        "animate_subject_bonus": 0.10,
        "inanimate_subject_penalty": -0.30,
        "preferred_object_bonus": 0.0,
        "unsuitable_object_penalty": 0.0,
    },
    "sitting on": {
        "requires_animate_subject": True,
        "preferred_object_cats": {"furniture", "rideable"},
        "unsuitable_object_cats": {"handheld", "wearable"},
        "animate_subject_bonus": 0.12,
        "inanimate_subject_penalty": -0.35,
        "preferred_object_bonus": 0.08,
        "unsuitable_object_penalty": -0.15,
    },
    "standing on": {
        "requires_animate_subject": True,
        "preferred_object_cats": {"furniture"},
        "unsuitable_object_cats": {"handheld", "wearable"},
        "animate_subject_bonus": 0.10,
        "inanimate_subject_penalty": -0.30,
        "preferred_object_bonus": 0.06,
        "unsuitable_object_penalty": -0.12,
    },
}

# ── Directionality penalties (Step 3) ────────────────────────────────
# These penalize unnatural subject-object assignments.
# Example: "backpack on person" → backpack should not be the subject.

_INANIMATE_SUBJECT_PENALTY: float = -0.50
_WEARABLE_SUBJECT_PENALTY: float = -0.60
_FURNITURE_SUBJECT_PENALTY: float = -0.50
_HANDHELD_SUBJECT_PENALTY: float = -0.40
_RIDEABLE_SUBJECT_PENALTY: float = -0.40

# Step 7 — Consistency margin for semantic vs spatial predicate selection
# When the best semantic score is within this margin of the best spatial score,
# the semantic predicate is preferred.
_SEMANTIC_CONSISTENCY_MARGIN: float = 0.10
_SEMANTIC_CANDIDATE_THRESHOLD: float = 0.12

# Hard negative predicate rules (Step 5)
# Format: (subject_category, predicate, object_category)
# "any" matches any category. These are semantically absurd triples.
_HARD_NEGATIVE_RULES: List[Tuple[str, str, str]] = [
    ("furniture", "holding", "any"),
    ("any", "holding", "furniture"),
    ("furniture", "wearing", "any"),
    ("any", "wearing", "furniture"),
    ("any", "wearing", "rideable"),
    ("rideable", "holding", "any"),
    ("rideable", "wearing", "any"),
    ("handheld", "wearing", "any"),
    ("handheld", "riding", "any"),
    ("handheld", "sitting on", "any"),
    ("handheld", "standing on", "any"),
    ("wearable", "riding", "any"),
    ("wearable", "sitting on", "any"),
    ("wearable", "standing on", "any"),
    ("furniture", "riding", "any"),
    ("furniture", "sitting on", "any"),
    ("furniture", "standing on", "any"),
    ("rideable", "sitting on", "any"),
    ("rideable", "standing on", "any"),
    ("any", "sitting on", "handheld"),
    ("any", "standing on", "handheld"),
    ("any", "wearing", "handheld"),
    # Animate subject riding another animate (non-rideable) → absurd
    ("animate", "riding", "animate"),
    # Holding another animate → unusual, penalize
    ("animate", "holding", "animate"),
]

# Default temperature for calibrated inference (Step 3)
_DEFAULT_TEMPERATURE: float = 2.0


def _calibrate_scores(logits: torch.Tensor, temperature: float = _DEFAULT_TEMPERATURE) -> torch.Tensor:
    """Temperature-scaled softmax for calibrated confidence.
    
    Higher temperature spreads probability mass, preventing the
    near-0/near-1 collapse that makes semantic priors ineffective.
    """
    if temperature <= 0:
        return F.softmax(logits, dim=-1)
    return F.softmax(logits / temperature, dim=-1)


def _compute_prior_adjustment(subject: str, predicate: str, object: str) -> Tuple[float, float, float]:
    """Compute semantic prior bonus/penalty for a (subject, predicate, object) triple.
    
    Returns:
        (bonus, penalty, total_adjustment)
    """
    prior = _PREDICATE_PRIORS.get(predicate)
    is_semantic = predicate in SEMANTIC_PREDS

    subj_cats = _get_categories(subject)
    obj_cats = _get_categories(object)

    bonus = 0.0
    penalty = 0.0

    if prior is not None:
        if prior["requires_animate_subject"]:
            if "animate" in subj_cats:
                bonus += prior["animate_subject_bonus"]
            else:
                penalty += prior["inanimate_subject_penalty"]

        if prior["preferred_object_cats"]:
            if any(cat in obj_cats for cat in prior["preferred_object_cats"]):
                bonus += prior["preferred_object_bonus"]

        if prior["unsuitable_object_cats"]:
            if any(cat in obj_cats for cat in prior["unsuitable_object_cats"]):
                penalty += prior["unsuitable_object_penalty"]

    # ── Step 3 — Directionality penalties ──────────────────────────────
    # Penalize unnatural subject-object assignments.
    # Inanimate subject with animate object → likely reversed direction.
    if "animate" not in subj_cats and "animate" in obj_cats:
        if is_semantic:
            # Semantic preds with reversed direction are absurd
            # e.g. "tie wearing person"
            penalty += _INANIMATE_SUBJECT_PENALTY
        elif predicate in NEUTRAL_SPATIAL:
            # "backpack on person" → bad direction for spatial pred
            penalty += _INANIMATE_SUBJECT_PENALTY * 0.6
        elif predicate in WEAK_SPATIAL:
            # "backpack near person" → still bad direction
            penalty += _INANIMATE_SUBJECT_PENALTY * 0.4

    # Wearable as subject is always wrong direction for non-spatial preds
    if "wearable" in subj_cats:
        if is_semantic or predicate in NEUTRAL_SPATIAL:
            penalty += _WEARABLE_SUBJECT_PENALTY
        elif predicate in WEAK_SPATIAL:
            penalty += _WEARABLE_SUBJECT_PENALTY * 0.5

    # Furniture as subject for semantic predicates
    if "furniture" in subj_cats and is_semantic:
        penalty += _FURNITURE_SUBJECT_PENALTY

    # Handheld as subject for semantic predicates
    if "handheld" in subj_cats and is_semantic:
        penalty += _HANDHELD_SUBJECT_PENALTY

    # Rideable as subject for semantic predicates
    if "rideable" in subj_cats and is_semantic:
        penalty += _RIDEABLE_SUBJECT_PENALTY

    return bonus, penalty, bonus + penalty


def _is_extreme_nonsense(subject: str, predicate: str, object: str) -> bool:
    """Hard-filter semantically impossible triples (Step 5).
    
    Rejects:
    1. Any predicate requiring animate subject when subject is inanimate.
    2. Hard negative rule violations.
    3. Same-object semantic absurdities.
    """
    # Check 1: Predicate requires animate subject but subject isn't
    prior = _PREDICATE_PRIORS.get(predicate)
    if prior is not None and prior.get("requires_animate_subject", False):
        subj_cats = _get_categories(subject)
        if "animate" not in subj_cats:
            return True

    # Check 2: Hard negative rules
    if _check_hard_negative(subject, predicate, object):
        return True

    # Check 3: Same-object nonsense
    if subject == object:
        # Animate self-relations: "person holding person" → nonsense
        if "animate" in _get_categories(subject):
            if predicate in {"holding", "wearing", "carrying",
                             "sitting on", "standing on", "riding"}:
                return True
        # Inanimate self-relations with semantic preds → nonsense
        elif predicate in SEMANTIC_PREDS:
            return True

    return False


def _check_hard_negative(subject: str, predicate: str, object: str) -> bool:
    """Check if a triple matches a hard negative rule (Step 5).
    
    These are semantically absurd interactions that should never
    appear in grounded captions, e.g. "chair holding phone",
    "backpack riding bicycle", "tie wearing person".
    
    Exceptions:
    - animate+riding+animate is allowed when object is rideable
      (e.g. "person riding horse" — horse is both animate and rideable)
    
    Returns:
        True if the triple should be rejected.
    """
    subj_cats = _get_categories(subject)
    obj_cats = _get_categories(object)

    for subj_cat, pred, obj_cat in _HARD_NEGATIVE_RULES:
        if pred != predicate:
            continue
        subj_match = subj_cat == "any" or subj_cat in subj_cats
        obj_match = obj_cat == "any" or obj_cat in obj_cats
        if not (subj_match and obj_match):
            continue

        # Exception: animate+riding+animate allowed when object is rideable
        # e.g. "person riding horse" — horse is both animate and rideable
        if (predicate == "riding" and subj_cat == "animate" and obj_cat == "animate"
                and "rideable" in obj_cats):
            continue

        return True

    # Additional semantic absurdities:
    # Cell phone cannot be "under" a person (tiny object over big = nonsense)
    if "handheld" in subj_cats and "animate" in obj_cats:
        if predicate in {"under", "above", "covering"}:
            return True

    # Wearable as subject with spatial predicate → reversed direction
    if "wearable" in subj_cats and "animate" in obj_cats:
        if predicate in NEUTRAL_SPATIAL:
            return True

    return False


# ---------------------------------------------------------------------------
# Step 4 — Feature contribution analysis utilities
# ---------------------------------------------------------------------------

def _get_feature_group_norms(model: nn.Module) -> Dict[str, float]:
    """Analyze feature group contributions.

    For MLP: L2 norm of first-layer weight columns per feature group.
    For Transformer: cross-attention weight aggregation per modality.

    Returns:
        Dict mapping group name to contribution score.
    """
    if isinstance(model, RelationTransformer):
        return _get_transformer_feature_norms(model)

    # MLP path
    first_weight = model.mlp[0].weight  # (H, in_dim)
    embed_dim = model.label_emb.weight.shape[1]
    clip_dim = model.clip_dim
    union_dim = model.union_dim
    pose_dim = model.pose_dim

    # Use the model's OWN geometry width, not the legacy 5. With a 19-dim
    # checkpoint the constant sliced the wrong 5 columns as "geo" and shifted
    # every later group by 14, so the reported CLIP / union / pose modality
    # shares were read off the wrong weight columns entirely.
    geo_dim = getattr(model, "geo_dim", GEO_DIM)

    groups = {
        "subj_label": (0, embed_dim),
        "obj_label":  (embed_dim, 2 * embed_dim),
        "geo":        (2 * embed_dim, 2 * embed_dim + geo_dim),
    }
    offset = 2 * embed_dim + geo_dim
    if clip_dim > 0:
        groups["subj_clip"] = (offset, offset + clip_dim)
        offset += clip_dim
        groups["obj_clip"]  = (offset, offset + clip_dim)
        offset += clip_dim
    if union_dim > 0:
        groups["union_clip"] = (offset, offset + union_dim)
        offset += union_dim
    if pose_dim > 0:
        groups["pose"] = (offset, offset + pose_dim)

    norms = {}
    for name, (start, end) in groups.items():
        group_weight = first_weight[:, start:end]
        norms[name] = group_weight.norm().item()

    return norms


def _get_transformer_feature_norms(model: RelationTransformer) -> Dict[str, float]:
    """Estimate feature importance via projection weight norms.

    For the transformer, each modality has a linear projection layer.
    We use the L2 norm of each projection's weight as a proxy for
    how much information flows through that modality.

    This mirrors the MLP approach structurally.
    """
    norms = {}
    norms["subj_label"] = model.subj_label_proj.weight.norm().item()
    norms["obj_label"] = model.obj_label_proj.weight.norm().item()
    norms["geo"] = model.geo_proj.weight.norm().item()
    if hasattr(model, 'subj_clip_proj'):
        norms["subj_clip"] = model.subj_clip_proj.weight.norm().item()
        norms["obj_clip"] = model.obj_clip_proj.weight.norm().item()
    if hasattr(model, 'union_proj'):
        norms["union_clip"] = model.union_proj.weight.norm().item()
    if hasattr(model, 'pose_proj'):
        norms["pose"] = model.pose_proj.weight.norm().item()
    return norms


def _measure_feature_ablation(
    model: nn.Module,
    subj_idx: torch.Tensor,
    obj_idx: torch.Tensor,
    geo: torch.Tensor,
    subj_feat: Optional[torch.Tensor],
    obj_feat: Optional[torch.Tensor],
    pred_vocab: Vocab,
    union_feat: Optional[torch.Tensor] = None,
    pose_feat: Optional[torch.Tensor] = None,
) -> Dict:
    """Measure prediction changes when feature groups are ablated.

    Compares full model prediction to predictions with:
    - CLIP features zeroed
    - Union features zeroed
    - Pose features zeroed
    - Geometry features zeroed

    Returns:
        Dict with prediction changes and logit differences.
    """
    with torch.no_grad():
        def _forward(sf, of, uf, pf, geo_t):
            return model(subj_idx, obj_idx, geo_t,
                         subj_feat=sf, obj_feat=of,
                         union_feat=uf, pose_feat=pf)

        full_logits = _forward(subj_feat, obj_feat, union_feat, pose_feat, geo)

        full_pred_idx = full_logits[0].argmax(dim=-1).item()
        full_pred = pred_vocab.token(full_pred_idx)
        full_probs = F.softmax(full_logits[0], dim=-1)
        full_max_prob = full_probs.max().item()

        result = {
            "full_prediction": full_pred,
            "full_confidence": round(full_max_prob, 4),
        }

        # Ablate CLIP
        if subj_feat is not None and obj_feat is not None:
            zeros_c = torch.zeros_like(subj_feat)
            no_clip_logits = _forward(zeros_c, zeros_c, union_feat, pose_feat, geo)
            no_clip_pred_idx = no_clip_logits[0].argmax(dim=-1).item()
            no_clip_pred = pred_vocab.token(no_clip_pred_idx)
            no_clip_probs = F.softmax(no_clip_logits[0], dim=-1)
            no_clip_max_prob = no_clip_probs.max().item()
            logit_diff = (full_logits[0] - no_clip_logits[0]).norm().item()

            result["ablate_clip_prediction"] = no_clip_pred
            result["ablate_clip_confidence"] = round(no_clip_max_prob, 4)
            result["clip_logit_norm_diff"] = round(logit_diff, 4)
            result["pred_changed_without_clip"] = full_pred != no_clip_pred

        # Ablate union
        if union_feat is not None and model.union_dim > 0:
            zeros_u = torch.zeros_like(union_feat)
            no_union_logits = _forward(subj_feat, obj_feat, zeros_u, pose_feat, geo)
            no_union_pred_idx = no_union_logits[0].argmax(dim=-1).item()
            no_union_pred = pred_vocab.token(no_union_pred_idx)
            no_union_probs = F.softmax(no_union_logits[0], dim=-1)
            no_union_max_prob = no_union_probs.max().item()
            union_logit_diff = (full_logits[0] - no_union_logits[0]).norm().item()

            result["ablate_union_prediction"] = no_union_pred
            result["ablate_union_confidence"] = round(no_union_max_prob, 4)
            result["union_logit_norm_diff"] = round(union_logit_diff, 4)
            result["pred_changed_without_union"] = full_pred != no_union_pred

        # Ablate pose
        if pose_feat is not None and model.pose_dim > 0:
            zeros_p = torch.zeros_like(pose_feat)
            no_pose_logits = _forward(subj_feat, obj_feat, union_feat, zeros_p, geo)
            no_pose_pred_idx = no_pose_logits[0].argmax(dim=-1).item()
            no_pose_pred = pred_vocab.token(no_pose_pred_idx)
            no_pose_probs = F.softmax(no_pose_logits[0], dim=-1)
            no_pose_max_prob = no_pose_probs.max().item()
            pose_logit_diff = (full_logits[0] - no_pose_logits[0]).norm().item()

            result["ablate_pose_prediction"] = no_pose_pred
            result["ablate_pose_confidence"] = round(no_pose_max_prob, 4)
            result["pose_logit_norm_diff"] = round(pose_logit_diff, 4)
            result["pred_changed_without_pose"] = full_pred != no_pose_pred

        # Ablate geometry
        zero_geo = torch.zeros_like(geo)
        no_geo_logits = _forward(subj_feat, obj_feat, union_feat, pose_feat, zero_geo)
        no_geo_pred_idx = no_geo_logits[0].argmax(dim=-1).item()
        no_geo_pred = pred_vocab.token(no_geo_pred_idx)
        no_geo_probs = F.softmax(no_geo_logits[0], dim=-1)
        no_geo_max_prob = no_geo_probs.max().item()
        geo_logit_diff = (full_logits[0] - no_geo_logits[0]).norm().item()

        result["ablate_geo_prediction"] = no_geo_pred
        result["ablate_geo_confidence"] = round(no_geo_max_prob, 4)
        result["geo_logit_norm_diff"] = round(geo_logit_diff, 4)
        result["pred_changed_without_geo"] = full_pred != no_geo_pred

    return result


def _analyze_feature_utilization(
    model: nn.Module,
    subj_idx: torch.Tensor,
    obj_idx: torch.Tensor,
    geo: torch.Tensor,
    subj_feat: Optional[torch.Tensor],
    obj_feat: Optional[torch.Tensor],
    pred_vocab: Vocab,
    union_feat: Optional[torch.Tensor] = None,
    pose_feat: Optional[torch.Tensor] = None,
) -> Dict:
    """Full feature utilization analysis for a single pair.

    Combines:
    1. Feature group weight norms
    2. Feature ablation study
    3. Contribution breakdown across ALL feature groups

    Returns:
        Dict with all analysis results.
    """
    norms = _get_feature_group_norms(model)
    ablation = _measure_feature_ablation(
        model, subj_idx, obj_idx, geo, subj_feat, obj_feat,
        pred_vocab, union_feat=union_feat, pose_feat=pose_feat,
    )

    total_norm = sum(norms.values()) or 1.0
    contribution = {k: round(v / total_norm, 4) for k, v in norms.items()}

    clip_pct = round(
        (norms.get("subj_clip", 0) + norms.get("obj_clip", 0)) / total_norm * 100, 1
    )
    union_pct = round(norms.get("union_clip", 0) / total_norm * 100, 1)
    pose_pct = round(norms.get("pose", 0) / total_norm * 100, 1)
    geo_pct = round(norms.get("geo", 0) / total_norm * 100, 1)
    label_pct = round(
        (norms.get("subj_label", 0) + norms.get("obj_label", 0)) / total_norm * 100, 1
    )

    return {
        "feature_norms": norms,
        "feature_contribution": contribution,
        "ablation": ablation,
        "clip_contribution_pct": clip_pct,
        "union_contribution_pct": union_pct,
        "pose_contribution_pct": pose_pct,
        "geo_contribution_pct": geo_pct,
        "label_contribution_pct": label_pct,
    }


# ---------------------------------------------------------------------------
# Step 7 — Semantic consistency checking
# ---------------------------------------------------------------------------

def _apply_semantic_consistency(
    candidates: List[Dict],
) -> List[Dict]:
    """Apply consistency checking to ensure coherent semantic interpretations.
    
    For each (subject, object) pair, if a semantic predicate has strong
    enough support, suppress weaker conflicting spatial alternatives.
    
    The goal is: only one coherent semantic interpretation per pair.
    If the model thinks "person riding bicycle" is plausible, we should
    not also produce "person on bicycle" or "bicycle under person".
    
    Args:
        candidates: List of candidate relation dicts (multiple per pair possible).
        
    Returns:
        Filtered list with at most one predicate per pair (the best one).
    """
    # Group by (subject, object) pair (ordered)
    pair_groups: Dict[Tuple[str, str], List[Dict]] = {}
    for c in candidates:
        # Use the subject/object pair as-is (not sorted)
        # so direction is preserved
        key = (c["subject"], c["object"])
        if key not in pair_groups:
            pair_groups[key] = []
        pair_groups[key].append(c)

    result = []
    for key, group in pair_groups.items():
        if len(group) == 1:
            result.append(group[0])
            continue

        # Find best semantic and spatial scores
        best_semantic = None
        best_semantic_score = -float('inf')
        best_spatial = None
        best_spatial_score = -float('inf')
        best_any = None
        best_any_score = -float('inf')

        for c in group:
            score = c.get("adjusted_confidence", c["confidence"])
            if score > best_any_score:
                best_any_score = score
                best_any = c
            if c["predicate"] in SEMANTIC_PREDS:
                if score > best_semantic_score:
                    best_semantic_score = score
                    best_semantic = c
            else:
                if score > best_spatial_score:
                    best_spatial_score = score
                    best_spatial = c

        # Decision logic:
        # 1. If semantic has good support and is close to spatial best → prefer semantic
        # 2. If only spatial exists → keep best spatial
        # 3. If only semantic exists → keep best semantic
        if (best_semantic is not None and
            best_semantic_score >= _SEMANTIC_CANDIDATE_THRESHOLD and
            (best_spatial is None or
             best_semantic_score >= best_spatial_score - _SEMANTIC_CONSISTENCY_MARGIN)):
            result.append(best_semantic)
            if best_spatial is not None:
                print(f"  [consistency] PREFERRED semantic '{best_semantic['predicate']}' "
                      f"({best_semantic_score:.3f}) over spatial "
                      f"'{best_spatial['predicate']}' ({best_spatial_score:.3f})")
        else:
            result.append(best_any)

    return result


# ---------------------------------------------------------------------------
# Step 8 — Relation quality evaluation
# ---------------------------------------------------------------------------

def evaluate_relation_quality(
    relations: List[Dict],
    raw_debug: List[Dict],
) -> Dict:
    """Evaluate quality of extracted relations.
    
    Computes metrics for:
    - Semantic relation precision (higher is better)
    - Inanimate subject rate (lower is better)
    - Reversed direction rate (lower is better)
    - Weak spatial clutter rate (lower is better)
    
    Tracks separately:
    - Semantic predicates (riding, holding, wearing, carrying, looking at,
      sitting on, standing on)
    - Spatial predicates (on, in, under, above, near, next to, etc.)
    
    Args:
        relations: Final selected relations.
        raw_debug: All evaluated pairs from infer_relationships_semantic.
        
    Returns:
        Dict with all computed metrics.
    """
    metrics: Dict = {
        "total_relations": len(relations),
        "semantic_relations": 0,
        "spatial_relations": 0,
        "animate_subject": 0,
        "inanimate_subject": 0,
        "reversed_direction_suspected": 0,
        "weak_spatial": 0,
        "neutral_spatial": 0,
    }

    predicate_breakdown: Dict[str, int] = {}

    for r in relations:
        pred = r["predicate"]
        predicate_breakdown[pred] = predicate_breakdown.get(pred, 0) + 1

        subj_cats = _get_categories(r["subject"])
        obj_cats = _get_categories(r["object"])

        if pred in SEMANTIC_PREDS:
            metrics["semantic_relations"] += 1
        elif pred in WEAK_SPATIAL:
            metrics["weak_spatial"] += 1
            metrics["spatial_relations"] += 1
        elif pred in NEUTRAL_SPATIAL:
            metrics["neutral_spatial"] += 1
            metrics["spatial_relations"] += 1
        else:
            metrics["spatial_relations"] += 1

        if "animate" in subj_cats:
            metrics["animate_subject"] += 1
        else:
            metrics["inanimate_subject"] += 1
            if "animate" in obj_cats:
                metrics["reversed_direction_suspected"] += 1

    total = max(len(relations), 1)
    metrics["semantic_precision"] = round(metrics["semantic_relations"] / total, 4)
    metrics["animate_subject_rate"] = round(metrics["animate_subject"] / total, 4)
    metrics["reversed_direction_rate"] = round(
        metrics["reversed_direction_suspected"] / total, 4
    )
    metrics["weak_spatial_rate"] = round(metrics["weak_spatial"] / total, 4)
    metrics["predicate_breakdown"] = predicate_breakdown

    # Pair-level stats from raw_debug
    total_evaluated = len([d for d in raw_debug if d.get("status") == "candidate"])
    metrics["total_pairs_evaluated"] = total_evaluated
    metrics["selection_ratio"] = round(
        len(relations) / max(total_evaluated, 1), 4
    )

    return metrics


# ---------------------------------------------------------------------------
# Singleton state
# ---------------------------------------------------------------------------

_model:         Optional[nn.Module] = None
_label_vocab:   Optional[Vocab]      = None
_pred_vocab:    Optional[Vocab]      = None
_device:        Optional[torch.device] = None
_clip_model:    Optional[CLIPExtractor] = None
_pose_model:    Optional[PoseExtractor] = None
_model_clip_dim: int = 0
_model_pose_dim: int = 0
_model_union_dim: int = 0
_model_type:     str = "mlp"
_model_geo_dim:  int = GEO_DIM
_model_geo_mode: str = "basic"
_model_geo_fn = extract_geo_features

_DEFAULT_CKPT_DIR = os.environ.get("REL_CKPT_DIR", "./checkpoints")


def load_relation_model(checkpoint_dir: str = _DEFAULT_CKPT_DIR) -> None:
    global _model, _label_vocab, _pred_vocab, _device, _model_clip_dim
    global _model_pose_dim, _model_union_dim, _model_type
    global _model_geo_dim, _model_geo_mode, _model_geo_fn

    model_path = os.path.join(checkpoint_dir, "relation_mlp.pt")
    lv_path    = os.path.join(checkpoint_dir, "label_vocab.json")
    pv_path    = os.path.join(checkpoint_dir, "pred_vocab.json")

    for p, name in [(model_path, "relation_mlp.pt"),
                    (lv_path,    "label_vocab.json"),
                    (pv_path,    "pred_vocab.json")]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"[relation_prediction] Checkpoint file not found: {p}\n"
                f"  Run training first:\n"
                f"    python -m relation_prediction.train --vg-root ./data/visual_genome"
            )

    _label_vocab = Vocab.load(lv_path)
    _pred_vocab  = Vocab.load(pv_path)
    _device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_data = torch.load(model_path, map_location=_device, weights_only=True)

    # Determine model type from config or infer from state_dict keys.
    if isinstance(raw_data, dict) and "model_config" in raw_data:
        config = raw_data.get("model_config", {})
        model_type = config.get("model_type", "mlp")
    else:
        model_type = "mlp"

    if model_type == "transformer":
        state = raw_data["model_state_dict"]
        clip_dim = config.get("clip_dim", 0)
        pose_dim = config.get("pose_dim", 0)
        union_dim = config.get("union_dim", 0)
        embed_dim = config.get("embed_dim", 64)
        d_model = config.get("d_model", 256)
        geo_dim = config.get("geo_dim", GEO_DIM)
        geo_norm = bool(config.get("geo_norm", False))

        _model_clip_dim = clip_dim
        _model_pose_dim = pose_dim
        _model_union_dim = union_dim
        _model_type = "transformer"
        _set_geo_mode(config, geo_dim)

        _model = RelationTransformer(
            num_labels=len(_label_vocab),
            num_predicates=len(_pred_vocab),
            d_model=d_model,
            embed_dim=embed_dim,
            clip_dim=clip_dim,
            pose_dim=pose_dim,
            union_dim=union_dim,
            geo_dim=geo_dim,
            geo_norm=geo_norm,
        )
        _model.load_state_dict(state)
        _model.to(_device)
        _model.eval()

        mode_parts = []
        if clip_dim > 0:
            mode_parts.append("visual-semantic")
        if union_dim > 0:
            mode_parts.append("union")
        if pose_dim > 0:
            mode_parts.append("pose")
        mode_str = "+".join(mode_parts) if mode_parts else "geometry-only"
        print(
            f"[relation_prediction] Loaded TRANSFORMER model from {checkpoint_dir} "
            f"({len(_label_vocab):,} labels, {len(_pred_vocab):,} predicates, "
            f"{mode_str}, d_model={d_model})"
        )

    else:
        # Support both new format (with config) and old format (bare state_dict).
        if isinstance(raw_data, dict) and "model_state_dict" in raw_data:
            state = raw_data["model_state_dict"]
            config = raw_data.get("model_config", {})
            pose_dim = config.get("pose_dim", 0)
            union_dim = config.get("union_dim", 0)
            clip_dim = config.get("clip_dim", 0)
            embed_dim = config.get("embed_dim", state["label_emb.weight"].shape[1])
            hidden_dims = _infer_hidden_dims(state)
            geo_dim = config.get("geo_dim", GEO_DIM)
            geo_norm = bool(config.get("geo_norm", "geo_norm.weight" in state))
        else:
            state = raw_data
            config = {}
            embed_dim = state["label_emb.weight"].shape[1]
            hidden_dims = _infer_hidden_dims(state)
            geo_norm = "geo_norm.weight" in state
            geo_dim = (int(state["geo_norm.weight"].shape[0]) if geo_norm else GEO_DIM)
            clip_dim = _infer_clip_dim(state, embed_dim, geo_dim)
            pose_dim = 0
            union_dim = 0

        _model_clip_dim = clip_dim
        _model_pose_dim = pose_dim
        _model_union_dim = union_dim
        _model_type = "mlp"
        _set_geo_mode(config, geo_dim)

        _model = RelationMLP(
            num_labels=len(_label_vocab),
            num_predicates=len(_pred_vocab),
            embed_dim=embed_dim,
            hidden_dims=hidden_dims,
            clip_dim=clip_dim,
            pose_dim=pose_dim,
            union_dim=union_dim,
            geo_dim=geo_dim,
            geo_norm=geo_norm,
        )
        _model.load_state_dict(state)
        _model.to(_device)
        _model.eval()

        input_dim = 2 * embed_dim + geo_dim + 2 * clip_dim + union_dim + pose_dim
        mode_parts = []
        if clip_dim > 0:
            mode_parts.append("visual-semantic")
        if union_dim > 0:
            mode_parts.append("union")
        if pose_dim > 0:
            mode_parts.append("pose")
        mode_str = "+".join(mode_parts) if mode_parts else "geometry-only"
        print(
            f"[relation_prediction] Loaded model from {checkpoint_dir} "
            f"({len(_label_vocab):,} labels, {len(_pred_vocab):,} predicates, "
            f"{mode_str}, input_dim={input_dim})"
        )
        print(f"[RelationMLP]")
        print(f"  Loaded config:")
        print(f"    input_dim={input_dim}")
        print(f"    geo_mode={_model_geo_mode} (geo_dim={geo_dim}, batchnorm={geo_norm})")
        print(f"    hidden_dims={hidden_dims}")
        print(f"    pose_dim={pose_dim}")
        print(f"    union_dim={union_dim}")


def _set_geo_mode(config: dict, geo_dim: int) -> None:
    """Select the geometry extractor that matches the loaded checkpoint.

    A checkpoint trained on the 19-dim descriptor must be fed the 19-dim
    descriptor at inference; feeding it the 5-dim one is a silent shape error
    at best and a silent accuracy collapse at worst.
    """
    global _model_geo_dim, _model_geo_mode, _model_geo_fn
    mode = config.get("geo_mode")
    if mode not in ("none", "basic", "ext"):
        # geo_dim=0 is the clip_only ablation arm; without the "none" case its
        # checkpoint was read back as "basic" and refused to load.
        mode = "ext" if geo_dim == GEO_DIM_EXT else ("none" if geo_dim == 0 else "basic")
    _model_geo_fn, _model_geo_dim = geo_extractor(mode)
    _model_geo_mode = mode
    if _model_geo_dim != geo_dim:
        raise ValueError(
            f"checkpoint geo_dim={geo_dim} does not match geo_mode={mode!r} "
            f"({_model_geo_dim}-dim). Refusing to build mismatched features."
        )


def _infer_hidden_dims(state: dict) -> Tuple[int, ...]:
    hidden: List[int] = []
    idx = 0
    while True:
        key = f"mlp.{idx}.weight"
        if key not in state:
            break
        hidden.append(state[key].shape[0])
        idx += 3
    return tuple(hidden[:-1])


def _infer_clip_dim(state: dict, embed_dim: int, geo_dim: int = GEO_DIM) -> int:
    """
    Infer clip_dim from the first MLP layer's input weight shape.
    clip_dim = (in_features - 2*embed_dim - geo_dim) // 2
    """
    first_weight = state["mlp.0.weight"]  # (out_features, in_features)
    in_features = first_weight.shape[1]
    clip_portion = in_features - 2 * embed_dim - geo_dim
    if clip_portion <= 0:
        return 0
    # clip_portion should be 2 * clip_dim
    if clip_portion % 2 != 0:
        print(f"[WARNING] Unexpected clip_portion={clip_portion}, treating as 0")
        return 0
    return clip_portion // 2


def _ensure_loaded() -> None:
    if _model is None:
        load_relation_model()


def _ensure_clip_model() -> None:
    global _clip_model
    if _clip_model is None and _model_clip_dim > 0:
        _clip_model = CLIPExtractor(device=_device)


def _ensure_pose_model() -> None:
    global _pose_model
    if _pose_model is None and _model_pose_dim > 0:
        _pose_model = PoseExtractor()


# ---------------------------------------------------------------------------
# Step 6 — Directionality preference helper
# ---------------------------------------------------------------------------

def _directionality_preference(candidate: Dict) -> Tuple:
    """Score tuple for direction naturalness.
    
    Higher tuple = more natural subject→object direction.
    Used during symmetric dedup to prefer animate subjects
    and semantic predicates.
    """
    subj_cats = _get_categories(candidate["subject"])
    is_animate = "animate" in subj_cats
    is_semantic = candidate["predicate"] in SEMANTIC_PREDS
    adj_conf = candidate.get("adjusted_confidence", candidate.get("confidence", 0.0))
    return (is_animate, is_semantic, adj_conf)


# ---------------------------------------------------------------------------
# Raw logits extraction (for Step 3 calibration + Step 8 debug)
# ---------------------------------------------------------------------------

def _resolve_image_size(
    img_w: float,
    img_h: float,
    image: Optional[Image.Image] = None,
) -> Tuple[float, float]:
    """Return the image size used to normalise the geometry features.

    ``extract_geo_features`` divides the subject/object centre offsets by the
    image width/height. Training always passes the real VG image size, so dx/dy
    live in roughly [-1, 1]. Every caller in this repository used to fall back
    on the ``img_w=img_h=1.0`` defaults, which turned dx/dy into RAW PIXEL
    offsets (order 1e2-1e3) at inference time — a ~1000x train/inference
    distribution shift on two of the five geometry inputs.

    When the caller did not supply a real size we recover it from the PIL image
    it already passes in. An explicit size > 1 is always respected.
    """
    if img_w > 1.0 and img_h > 1.0:
        return float(img_w), float(img_h)
    if image is not None:
        return float(image.width), float(image.height)
    return float(max(img_w, 1.0)), float(max(img_h, 1.0))


def _get_raw_logits(
    subject: str,
    obj_label: str,
    box1: List[float],
    box2: List[float],
    img_w: float = 1.0,
    img_h: float = 1.0,
    image: Optional[Image.Image] = None,
) -> Tuple[Optional[torch.Tensor], List[str], int, int]:
    """Get raw logits from the MLP for a single pair, before any softmax.

    Automatically extracts union-region CLIP and pose features
    when the loaded model supports them.

    Returns:
        (logits_tensor, pred_tokens_list, subj_idx, obj_idx)
        or (None, [], -1, -1) on lookup failure.
    """
    _ensure_loaded()

    subj_norm = normalize_label(subject)
    obj_norm  = normalize_label(obj_label)
    if subj_norm == "UNK" or obj_norm == "UNK":
        return None, [], -1, -1

    subj_idx = _label_vocab[subj_norm]
    obj_idx  = _label_vocab[obj_norm]

    subj_box = (float(box1[0]), float(box1[1]), float(box1[2]), float(box1[3]))
    obj_box  = (float(box2[0]), float(box2[1]), float(box2[2]), float(box2[3]))
    img_w, img_h = _resolve_image_size(img_w, img_h, image)
    geo      = _model_geo_fn(subj_box, obj_box, img_w, img_h)

    with torch.no_grad():
        s = torch.tensor([subj_idx], dtype=torch.long,    device=_device)
        o = torch.tensor([obj_idx],  dtype=torch.long,    device=_device)
        g = torch.tensor([geo],      dtype=torch.float32, device=_device)

        subj_emb = None
        obj_emb  = None
        union_emb = None
        pose_feat = None

        if _model_clip_dim > 0 or _model_union_dim > 0:
            if image is not None:
                _ensure_clip_model()
                if _model_clip_dim > 0:
                    subj_emb = _clip_model.extract_crop(image, subj_box).to(_device).unsqueeze(0)
                    obj_emb  = _clip_model.extract_crop(image, obj_box).to(_device).unsqueeze(0)

                if _model_union_dim > 0:
                    uemb = _clip_model.extract_union_embedding(image, subj_box, obj_box).to(_device)
                    union_emb = uemb.unsqueeze(0)
            else:
                if _model_clip_dim > 0:
                    subj_emb = torch.zeros((1, _model_clip_dim), device=_device)
                    obj_emb  = torch.zeros((1, _model_clip_dim), device=_device)
                if _model_union_dim > 0:
                    union_emb = torch.zeros((1, _model_union_dim), device=_device)

        if _model_pose_dim > 0 and image is not None and subj_norm == "person":
            _ensure_pose_model()
            if PoseExtractor.is_available():
                pf = _pose_model.extract_pose_features(image, subj_box)
                if pf is not None:
                    pose_feat = pf.to(_device).unsqueeze(0)

        if pose_feat is None and _model_pose_dim > 0:
            pose_feat = torch.zeros((1, _model_pose_dim), device=_device)

        logits = _model(s, o, g,
                        subj_feat=subj_emb, obj_feat=obj_emb,
                        union_feat=union_emb, pose_feat=pose_feat)

    pred_tokens = [_pred_vocab.token(i) for i in range(len(_pred_vocab))]
    return logits[0], pred_tokens, subj_idx, obj_idx


# ---------------------------------------------------------------------------
# Single-pair prediction
# ---------------------------------------------------------------------------

def predict_relation(
    subject: str,
    obj_label: str,
    box1: List[float],
    box2: List[float],
    img_w: float = 1.0,
    img_h: float = 1.0,
    threshold: float = 0.15,
    image: Optional[Image.Image] = None,
    temperature: float = _DEFAULT_TEMPERATURE,
) -> Optional[Tuple[str, float]]:
    _ensure_loaded()

    subj_norm = normalize_label(subject)
    obj_norm  = normalize_label(obj_label)
    if subj_norm == "UNK" or obj_norm == "UNK":
        return None

    subj_idx = _label_vocab[subj_norm]
    obj_idx  = _label_vocab[obj_norm]

    subj_box = (float(box1[0]), float(box1[1]), float(box1[2]), float(box1[3]))
    obj_box  = (float(box2[0]), float(box2[1]), float(box2[2]), float(box2[3]))
    img_w, img_h = _resolve_image_size(img_w, img_h, image)
    geo      = _model_geo_fn(subj_box, obj_box, img_w, img_h)

    with torch.no_grad():
        s = torch.tensor([subj_idx], dtype=torch.long,    device=_device)
        o = torch.tensor([obj_idx],  dtype=torch.long,    device=_device)
        g = torch.tensor([geo],      dtype=torch.float32, device=_device)

        subj_emb = None; obj_emb = None; union_emb = None; pose_feat = None

        if _model_clip_dim > 0 or _model_union_dim > 0:
            if image is not None:
                _ensure_clip_model()
                if _model_clip_dim > 0:
                    subj_emb = _clip_model.extract_crop(image, subj_box).to(_device).unsqueeze(0)
                    obj_emb  = _clip_model.extract_crop(image, obj_box).to(_device).unsqueeze(0)
                if _model_union_dim > 0:
                    uemb = _clip_model.extract_union_embedding(image, subj_box, obj_box).to(_device)
                    union_emb = uemb.unsqueeze(0)
            else:
                if _model_clip_dim > 0:
                    subj_emb = torch.zeros((1, _model_clip_dim), device=_device)
                    obj_emb  = torch.zeros((1, _model_clip_dim), device=_device)
                if _model_union_dim > 0:
                    union_emb = torch.zeros((1, _model_union_dim), device=_device)

        if _model_pose_dim > 0 and image is not None and subj_norm == "person":
            _ensure_pose_model()
            if PoseExtractor.is_available():
                pf = _pose_model.extract_pose_features(image, subj_box)
                if pf is not None:
                    pose_feat = pf.to(_device).unsqueeze(0)

        if pose_feat is None and _model_pose_dim > 0:
            pose_feat = torch.zeros((1, _model_pose_dim), device=_device)

        logits = _model(s, o, g,
                        subj_feat=subj_emb, obj_feat=obj_emb,
                        union_feat=union_emb, pose_feat=pose_feat)

        # Step 3 — Temperature-calibrated softmax
        probs  = _calibrate_scores(logits, temperature=temperature)
        top_prob, top_idx = probs[0].max(dim=-1)

    confidence = top_prob.item()
    if confidence < threshold:
        return None

    pred_token = _pred_vocab.token(top_idx.item())
    if pred_token in (Vocab.PAD, Vocab.UNK):
        return None
    if pred_token not in ALLOWED_PREDICATES:
        return None
    return (pred_token, confidence)


def predict_relation_topk(
    subject: str,
    obj_label: str,
    box1: List[float],
    box2: List[float],
    img_w: float = 1.0,
    img_h: float = 1.0,
    k: int = 3,
    image: Optional[Image.Image] = None,
    temperature: float = _DEFAULT_TEMPERATURE,
) -> List[Tuple[str, float]]:
    _ensure_loaded()

    subj_norm = normalize_label(subject)
    obj_norm  = normalize_label(obj_label)
    if subj_norm == "UNK" or obj_norm == "UNK":
        return []

    subj_idx = _label_vocab[subj_norm]
    obj_idx  = _label_vocab[obj_norm]
    img_w, img_h = _resolve_image_size(img_w, img_h, image)
    geo = _model_geo_fn(
        (float(box1[0]), float(box1[1]), float(box1[2]), float(box1[3])),
        (float(box2[0]), float(box2[1]), float(box2[2]), float(box2[3])),
        img_w, img_h,
    )

    with torch.no_grad():
        s = torch.tensor([subj_idx], dtype=torch.long,    device=_device)
        o = torch.tensor([obj_idx],  dtype=torch.long,    device=_device)
        g = torch.tensor([geo],      dtype=torch.float32, device=_device)

        subj_emb = None; obj_emb = None; union_emb = None; pose_feat = None

        if _model_clip_dim > 0 or _model_union_dim > 0:
            if image is not None:
                subj_box = (float(box1[0]), float(box1[1]), float(box1[2]), float(box1[3]))
                obj_box  = (float(box2[0]), float(box2[1]), float(box2[2]), float(box2[3]))
                _ensure_clip_model()
                if _model_clip_dim > 0:
                    subj_emb = _clip_model.extract_crop(image, subj_box).to(_device).unsqueeze(0)
                    obj_emb  = _clip_model.extract_crop(image, obj_box).to(_device).unsqueeze(0)
                if _model_union_dim > 0:
                    uemb = _clip_model.extract_union_embedding(image, subj_box, obj_box).to(_device)
                    union_emb = uemb.unsqueeze(0)
            else:
                if _model_clip_dim > 0:
                    subj_emb = torch.zeros((1, _model_clip_dim), device=_device)
                    obj_emb  = torch.zeros((1, _model_clip_dim), device=_device)
                if _model_union_dim > 0:
                    union_emb = torch.zeros((1, _model_union_dim), device=_device)

        if _model_pose_dim > 0 and image is not None and normalize_label(subject) == "person":
            _ensure_pose_model()
            if PoseExtractor.is_available():
                pf = _pose_model.extract_pose_features(image, subj_box)
                if pf is not None:
                    pose_feat = pf.to(_device).unsqueeze(0)

        if pose_feat is None and _model_pose_dim > 0:
            pose_feat = torch.zeros((1, _model_pose_dim), device=_device)

        logits = _model(s, o, g,
                        subj_feat=subj_emb, obj_feat=obj_emb,
                        union_feat=union_emb, pose_feat=pose_feat)

        # Step 3 — Temperature-calibrated softmax
        probs = _calibrate_scores(logits, temperature=temperature)[0]

    topk_probs, topk_idxs = probs.topk(min(k, len(probs)))
    results = []
    for prob, idx in zip(topk_probs.tolist(), topk_idxs.tolist()):
        token = _pred_vocab.token(idx)
        if token not in (Vocab.PAD, Vocab.UNK):
            results.append((token, prob))
    return results


# ---------------------------------------------------------------------------
# Drop-in replacement for utils.causal_caption.infer_relationships
# ---------------------------------------------------------------------------

Detection    = Dict
Relationship = Tuple[str, str, str]


def infer_relationships_learned(
    detections: List[Detection],
    threshold: float = 0.15,
    img_w: float = 1.0,
    img_h: float = 1.0,
    top_k: int = 2,
    image: Optional[Image.Image] = None,
) -> List[Relationship]:
    """
    Drop-in replacement for utils.causal_caption.infer_relationships().

    When using a visual-semantic model (trained with clip_dim > 0),
    pass the image to enable CLIP feature extraction from crops.

    Args:
        detections: [{"label": str, "box": [x1,y1,x2,y2], "score": float}, ...]
        threshold:  Minimum model confidence to include a relationship.
        img_w:      Image width for normalisation.
        img_h:      Image height for normalisation.
        top_k:      Maximum number of relations to return (default 2).
        image:      PIL Image (required for visual-semantic model).

    Returns:
        [(subject_label, predicate, object_label), ...]
    """
    if len(detections) < 2:
        return []

    candidates: List[Tuple[Relationship, float]] = []
    seen: set = set()

    n = len(detections)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            a = detections[i]
            b = detections[j]

            subj_norm = normalize_label(a["label"])
            obj_norm  = normalize_label(b["label"])
            if subj_norm == "UNK" or obj_norm == "UNK":
                continue

            result = predict_relation(
                subj_norm, obj_norm,
                a["box"], b["box"],
                img_w=img_w, img_h=img_h,
                threshold=threshold,
                image=image,
            )
            if result is None:
                continue

            pred, confidence = result
            triple = (subj_norm, pred, obj_norm)
            if triple not in seen:
                seen.add(triple)
                candidates.append((triple, confidence))

    if not candidates:
        return []

    candidates.sort(key=lambda x: -x[1])

    seen_pair: set = set()
    deduped: List[Relationship] = []
    for triple, _ in candidates:
        s, p, o = triple
        pair_key = (s, o)
        if pair_key not in seen_pair:
            seen_pair.add(pair_key)
            deduped.append(triple)

    final = deduped[:top_k]

    total_candidates = len(candidates)
    total_deduped    = len(deduped)
    discarded = [t for t in deduped if t not in final]
    print(f"[relation_prediction] infer_relationships_learned: "
          f"{total_candidates} candidates -> {total_deduped} after dedup "
          f"-> {len(final)} selected (top_k={top_k})")
    if discarded:
        print(f"[relation_prediction]   discarded: "
              f"{[f'{s} {p} {o}' for s, p, o in discarded]}")
    print(f"[relation_prediction]   final relations: "
          f"{[f'{s} {p} {o}' for s, p, o in final]}")

    return final


def infer_relationships_structured(
    detections: List[Detection],
    threshold: float = 0.1,
    img_w: float = 1.0,
    img_h: float = 1.0,
    top_k: int = 3,
    image: Optional[Image.Image] = None,
    semantic_boost: bool = True,
    temperature: float = _DEFAULT_TEMPERATURE,
) -> List[Dict]:
    """
    Structured relation inference with confidence scores.

    Returns structured dicts with subject, predicate, object, confidence.
    Prioritises semantic predicates (riding, holding, wearing, carrying, looking at)
    over spatial relations.

    Args:
        detections: [{"label": str, "box": [x1,y1,x2,y2], "score": float}, ...]
        threshold:  Minimum model confidence.
        img_w:      Image width.
        img_h:      Image height.
        top_k:      Max relations to return.
        image:      PIL Image (required for visual-semantic model).
        semantic_boost: When True, applies lower threshold for semantic predicates.

    Returns:
        [{"subject": str, "predicate": str, "object": str, "confidence": float,
          "subject_box": [x1,y1,x2,y2], "object_box": [x1,y1,x2,y2]}, ...]
    """
    SEMANTIC_PREDS = frozenset({
        "riding", "holding", "wearing", "carrying", "looking at",
    })

    if len(detections) < 2:
        return []

    candidates: List[Dict] = []
    seen: set = set()

    n = len(detections)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            a = detections[i]
            b = detections[j]

            subj_norm = normalize_label(a["label"])
            obj_norm  = normalize_label(b["label"])
            if subj_norm == "UNK" or obj_norm == "UNK":
                continue

            result = predict_relation(
                subj_norm, obj_norm,
                a["box"], b["box"],
                img_w=img_w, img_h=img_h,
                threshold=threshold,
                image=image,
                temperature=temperature,
            )
            if result is None:
                continue

            pred, confidence = result
            triple = (subj_norm, pred, obj_norm)
            if triple not in seen:
                seen.add(triple)
                candidates.append({
                    "subject": subj_norm,
                    "predicate": pred,
                    "object": obj_norm,
                    "confidence": round(confidence, 4),
                    "subject_box": a["box"],
                    "object_box": b["box"],
                    "subject_score": a.get("score", 1.0),
                    "object_score": b.get("score", 1.0),
                })

    if not candidates:
        return []

    # Sort by confidence, but boost semantic predicates
    if semantic_boost:
        def _sort_key(x: Dict) -> float:
            boost = 0.2 if x["predicate"] in SEMANTIC_PREDS else 0.0
            return x["confidence"] + boost
        candidates.sort(key=_sort_key, reverse=True)
    else:
        candidates.sort(key=lambda x: x["confidence"], reverse=True)

    # Deduplicate by (subject, object) pair, keep highest confidence
    seen_pair: set = set()
    deduped: List[Dict] = []
    for c in candidates:
        pair_key = (c["subject"], c["object"])
        if pair_key not in seen_pair:
            seen_pair.add(pair_key)
            deduped.append(c)

    final = deduped[:top_k]

    total_candidates = len(candidates)
    total_deduped    = len(deduped)
    print(f"[relation_prediction] infer_relationships_structured: "
          f"{total_candidates} candidates -> {total_deduped} after dedup "
          f"-> {len(final)} selected (top_k={top_k})")
    for r in final:
        print(f"[relation_prediction]   {r['subject']} {r['predicate']} {r['object']} "
              f"(conf={r['confidence']:.3f})")

    return final


# ---------------------------------------------------------------------------
# Semantic relation inference (Steps 1, 4, 5)
# ---------------------------------------------------------------------------

SEMANTIC_PREDS: frozenset = frozenset({
    "riding", "holding", "carrying", "wearing", "looking at",
    "sitting on", "standing on",
})

WEAK_SPATIAL: frozenset = frozenset({
    "under", "above", "over", "inside", "next to", "near",
    "attached to", "behind", "in front of", "covering",
})

NEUTRAL_SPATIAL: frozenset = frozenset({
    "on", "in",
})

_RERANK_GAP: float = 0.15
_TOP_K_CANDIDATES: int = 6

# ── Precision filtering thresholds (Steps 1, 5) ────────────────────────
# These transform the pipeline from "many noisy relations" to
# "small set of high-confidence grounded semantic constraints".
#
# MIN_SEMANTIC_SCORE: minimum adjusted_confidence for non-semantic predicates.
#                     Semantic predicates (riding, holding, etc.) bypass this
#                     because they have their own prior-based scoring.
MIN_SEMANTIC_SCORE: float = 0.30

# Weak spatial predicates need extremely high confidence to survive.
# This suppresses relations like "near", "next to", "behind", etc.
# unless the model is very certain.
WEAK_SPATIAL_THRESHOLD: float = 0.85

# Spatial relations between two inanimate objects are always rejected.
# This eliminates pure geometry clutter like "chair near chair"
# or "book under table" that have no semantic meaning.
REJECT_INANIMATE_SPATIAL: bool = True


# ── Improved semantic priors (Phase 7) ──────────────────────────────
_IMPROVED_PRIORS = {
    "sitting on": {
        "reject_objects": {"dog", "cat", "bird", "cell phone", "bottle", "cup",
                           "book", "frisbee", "kite", "baseball bat",
                           "tennis racket", "sports ball"},
        "soft_reject_penalty": -0.35,
    },
    "riding": {
        "reject_objects": {"chair", "couch", "bench", "bed", "dining table",
                           "dog", "cat", "cell phone", "bottle", "cup", "book",
                           "frisbee", "kite", "backpack", "handbag", "suitcase",
                           "tie", "umbrella", "sports ball"},
        "soft_reject_penalty": -0.40,
    },
    "holding": {
        "reject_objects": {"chair", "couch", "bench", "bed", "dining table",
                           "car", "bicycle", "truck", "bus", "train",
                           "horse", "elephant", "cow", "toilet", "sink",
                           "refrigerator"},
        "soft_reject_penalty": -0.30,
    },
    "wearing": {
        "reject_objects": {"chair", "couch", "bench", "bed", "dining table",
                           "car", "bicycle", "bus", "truck", "train",
                           "cell phone", "bottle", "cup", "book",
                           "dog", "cat", "horse", "cow", "elephant",
                           "frisbee", "kite", "sports ball", "skateboard",
                           "surfboard", "toilet", "sink", "refrigerator"},
        "soft_reject_penalty": -0.35,
    },
    "carrying": {
        "reject_objects": {"chair", "couch", "bench", "bed", "dining table",
                           "car", "bicycle", "bus", "truck", "train",
                           "toilet", "sink", "refrigerator",
                           "horse", "cow", "elephant"},
        "soft_reject_penalty": -0.25,
    },
    "standing on": {
        "reject_objects": {"cell phone", "bottle", "cup", "book",
                           "dog", "cat", "bird", "frisbee", "kite",
                           "sports ball"},
        "soft_reject_penalty": -0.30,
    },
}


def _apply_refined_priors(subject: str, predicate: str, object: str) -> float:
    """Apply object-level soft semantic penalties (Phase 7)."""
    prior = _IMPROVED_PRIORS.get(predicate)
    if not prior:
        return 0.0
    obj_lower = object.lower().replace("_", " ")
    reject_set = prior.get("reject_objects", set())
    if obj_lower in reject_set:
        return prior.get("soft_reject_penalty", -0.25)
    return 0.0


def infer_relationships_semantic(
    detections: List[Detection],
    threshold: float = 0.05,
    img_w: float = 1.0,
    img_h: float = 1.0,
    top_k: int = 3,
    image: Optional[Image.Image] = None,
    temperature: float = _DEFAULT_TEMPERATURE,
    debug: bool = True,
    improved_priors: bool = False,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Calibrated semantic relation inference with lightweight priors (Steps 3-6).

    Full pipeline per pair:
      1. Extract raw logits from trained MLP (no softmax)
      2. Apply temperature-scaled softmax → calibrated scores
      3. Compute semantic prior adjustment per predicate
      4. Combine: final = calibrated_score + prior_adjustment
      5. Filter extreme nonsense (inanimate subjects with animate-requiring preds)
      6. Directionality-aware symmetric dedup

    Args:
        detections: [{"label", "box", "score"}, ...]
        threshold:  Minimum final score to include a relation.
        img_w:      Image width.
        img_h:      Image height.
        top_k:      Maximum relations to return.
        image:      PIL Image for CLIP crop extraction.
        temperature: Softmax temperature for calibration (default 2.0).
        debug:      Print detailed per-pair debug info.

    Returns:
        (filtered_relations, debug_info)
    """
    if len(detections) < 2:
        return [], []

    all_pred_tokens: List[str] = []
    raw_debug: List[Dict] = []
    candidates: List[Dict] = []
    n = len(detections)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            a = detections[i]
            b = detections[j]

            subj_norm = normalize_label(a["label"])
            obj_norm  = normalize_label(b["label"])
            if subj_norm == "UNK" or obj_norm == "UNK":
                continue

            # Step 1 — Get raw logits
            logits, pred_tokens, _, _ = _get_raw_logits(
                subj_norm, obj_norm,
                a["box"], b["box"],
                img_w=img_w, img_h=img_h,
                image=image,
            )
            if logits is None:
                continue

            all_pred_tokens = pred_tokens

            # Step 1b — DEBUG: Print raw logits BEFORE softmax/temperature/priors
            if debug:
                print(f"\n  ═══ RAW LOGITS: {subj_norm}+{obj_norm} ═══")
                sorted_raw = sorted(
                    [(pred_tokens[i], logits[i].item()) for i in range(len(pred_tokens))
                     if pred_tokens[i] not in (Vocab.PAD, Vocab.UNK)
                     and pred_tokens[i] in ALLOWED_PREDICATES],
                    key=lambda x: -x[1],
                )
                for pname, rval in sorted_raw:
                    marker = ""
                    if pname in SEMANTIC_PREDS:
                        marker = " [SEM]"
                    elif pname in WEAK_SPATIAL:
                        marker = " [WEAK_SP]"
                    elif pname in NEUTRAL_SPATIAL:
                        marker = " [NEUTRAL_SP]"
                    print(f"    {pname:20s}: {rval:8.4f}{marker}")

                # Logit statistics
                raw_vals = torch.tensor([v for _, v in sorted_raw])
                print(f"  ── RAW LOGIT STATS ──")
                print(f"    max={raw_vals.max():.4f}  min={raw_vals.min():.4f}  "
                      f"mean={raw_vals.mean():.4f}  std={raw_vals.std():.4f}")
                if len(raw_vals) >= 2:
                    top2_values, top2_indices = raw_vals.topk(2)
                    gap = (top2_values[0] - top2_values[1]).item()
                    print(f"    top1-top2 gap={gap:.4f}")
                    print(f"    top1={sorted_raw[0][0]} ({sorted_raw[0][1]:.4f})  "
                          f"top2={sorted_raw[1][0]} ({sorted_raw[1][1]:.4f})")

            # Step 2 — Temperature-calibrated softmax
            calibrated = _calibrate_scores(logits.unsqueeze(0), temperature=temperature)[0]

            # Step 4 — DEBUG: Temperature scaling comparison
            if debug and temperature != 1.0:
                valid_mask = torch.tensor([
                    0 if pred_tokens[i] in (Vocab.PAD, Vocab.UNK) or pred_tokens[i] not in ALLOWED_PREDICATES
                    else 1 for i in range(len(pred_tokens))
                ], device=logits.device)
                valid_logits = logits.clone()
                valid_logits[valid_mask == 0] = -float('inf')
                top3_raw = valid_logits.argsort(descending=True)[:3]
                print(f"  ── TEMPERATURE SCALING ({subj_norm}+{obj_norm}) ──")
                for T in [1.0, 2.0, 5.0, 10.0]:
                    scaled = _calibrate_scores(logits.unsqueeze(0), temperature=T)[0]
                    probs_str = "  ".join([f"{pred_tokens[i.item()]}: {scaled[i].item():.6f}" for i in top3_raw])
                    print(f"    T={T:.1f}:  {probs_str}")

            # Step 3+4 — Compute prior adjustments and final scores
            per_pred_debug: List[Dict] = []
            best_pred = None
            best_final = -float("inf")
            best_calib = 0.0
            best_prior_total = 0.0
            best_prior_idx = -1

            for pidx, pred_name in enumerate(pred_tokens):
                if pred_name in (Vocab.PAD, Vocab.UNK):
                    continue
                if pred_name not in ALLOWED_PREDICATES:
                    continue

                calib_score = calibrated[pidx].item()
                bonus, penalty, prior_total = _compute_prior_adjustment(subj_norm, pred_name, obj_norm)
                if improved_priors:
                    refined_penalty = _apply_refined_priors(subj_norm, pred_name, obj_norm)
                    penalty += refined_penalty
                    prior_total = bonus + penalty
                final_score = calib_score + prior_total

                per_pred_debug.append({
                    "predicate": pred_name,
                    "calibrated": round(calib_score, 4),
                    "prior_bonus": round(bonus, 4),
                    "prior_penalty": round(penalty, 4),
                    "prior_total": round(prior_total, 4),
                    "final": round(final_score, 4),
                })

                if final_score > best_final:
                    best_final = final_score
                    best_pred = pred_name
                    best_calib = calib_score
                    best_prior_total = prior_total
                    best_prior_idx = pidx

            if best_pred is None:
                continue

            # ═══════════════════════════════════════════════════════════
            # STEP 2+7 — Semantic consistency override
            # ═══════════════════════════════════════════════════════════
            # If a semantic predicate has reasonable support and is close
            # to the best spatial predicate, prefer the semantic one.
            # This strengthens "riding" over "on" for person+bicycle,
            # "sitting on" over "on" for person+chair, etc.
            #
            # Uses TWO complementary checks:
            # 1. Calibrated score margin (works when T spreads probability)
            # 2. Raw logit relative support (works when T can't spread)
            _best_sem_name = None
            _best_sem_final = -float('inf')
            _best_spatial_final = -float('inf')
            for pp in per_pred_debug:
                pf = pp["final"]
                if pp["predicate"] in SEMANTIC_PREDS:
                    if pf > _best_sem_final:
                        _best_sem_final = pf
                        _best_sem_name = pp["predicate"]
                else:
                    if pf > _best_spatial_final:
                        _best_spatial_final = pf

            # Check 1: Calibrated score margin
            consistency_triggered = False
            if (best_pred is not None and
                best_pred not in SEMANTIC_PREDS and
                _best_sem_name is not None and
                _best_sem_final >= _SEMANTIC_CANDIDATE_THRESHOLD and
                _best_sem_final >= _best_spatial_final - _SEMANTIC_CONSISTENCY_MARGIN):
                consistency_triggered = True

            # Check 2: Raw logit relative support (handles extreme saturation)
            if not consistency_triggered and best_pred is not None and _best_sem_name is not None:
                # ONLY apply when subject is animate — inanimate subjects
                # should not get semantic predicates (caught by extreme nonsense)
                if "animate" in _get_categories(subj_norm):
                    raw_logit_values = []
                    for pidx, pname in enumerate(pred_tokens):
                        if pname in ALLOWED_PREDICATES and pname not in (Vocab.PAD, Vocab.UNK):
                            raw_logit_values.append((pname, logits[pidx].item()))
                    if raw_logit_values:
                        max_raw_logit = max(v for _, v in raw_logit_values)
                        best_raw_spatial = 0.0
                        best_raw_semantic = 0.0
                        for pname, rval in raw_logit_values:
                            if max_raw_logit > 0:
                                relative = rval / max_raw_logit
                            else:
                                relative = 0.0
                            if pname in SEMANTIC_PREDS and relative > best_raw_semantic:
                                best_raw_semantic = relative
                            elif pname not in SEMANTIC_PREDS and relative > best_raw_spatial:
                                best_raw_spatial = relative

                        # Trigger if semantic has strong relative support vs spatial
                        if (best_raw_semantic > 0.3 and
                            best_raw_semantic >= best_raw_spatial * 0.5):
                            consistency_triggered = True

            if consistency_triggered:
                # Override: use semantic predicate instead of spatial
                if debug:
                    print(f"  [consistency] OVERRIDE: {subj_norm} {best_pred} {obj_norm} → "
                          f"{_best_sem_name} "
                          f"(sem final={_best_sem_final:.3f} vs spatial final={best_final:.3f})")
                best_pred = _best_sem_name
                for pp in per_pred_debug:
                    if pp["predicate"] == best_pred:
                        best_final = pp["final"]
                        best_calib = pp["calibrated"]
                        best_prior_total = pp["prior_total"]
                        break

            # Step 5 — Filter extreme nonsense
            if _is_extreme_nonsense(subj_norm, best_pred, obj_norm):
                if debug:
                    print(f"[semantic]  ✗ EXTREME NONSENSE: {subj_norm} {best_pred} {obj_norm}")
                raw_debug.append({
                    "subject": subj_norm,
                    "object": obj_norm,
                    "status": "rejected_extreme_nonsense",
                    "best_predicate": best_pred,
                    "per_predicate": per_pred_debug,
                })
                continue

            # Threshold check
            if best_final < threshold:
                continue

            candidates.append({
                "subject": subj_norm,
                "predicate": best_pred,
                "object": obj_norm,
                "confidence": round(best_calib, 4),
                "adjusted_confidence": round(best_final, 4),
                "prior_adjustment": round(best_prior_total, 4),
                "subject_box": a["box"],
                "object_box": b["box"],
                "subject_score": a.get("score", 1.0),
                "object_score": b.get("score", 1.0),
                "best_predicate_idx": best_prior_idx,
            })

            raw_debug.append({
                "subject": subj_norm,
                "object": obj_norm,
                "status": "candidate",
                "best_predicate": best_pred,
                "best_calibrated": round(best_calib, 4),
                "best_prior_adjustment": round(best_prior_total, 4),
                "best_final_score": round(best_final, 4),
                "per_predicate": per_pred_debug,
            })

    # ═══════════════════════════════════════════════════════════════════
    # STEP 1+3+5 — Precision filtering
    # ═══════════════════════════════════════════════════════════════════
    filtered_candidates: List[Dict] = []
    precision_log: List[Dict] = []

    for c in candidates:
        pred = c["predicate"]
        subj = c["subject"]
        obj = c["object"]
        adj_conf = c.get("adjusted_confidence", c["confidence"])
        subj_cats = _get_categories(subj)
        is_animate_subj = "animate" in subj_cats

        # Step 3 — Animate subject enforcement for semantic predicates
        if pred in SEMANTIC_PREDS and not is_animate_subj:
            precision_log.append({
                "subject": subj, "predicate": pred, "object": obj,
                "adj_conf": adj_conf, "status": "rejected",
                "reason": "semantic predicate requires animate subject",
            })
            continue

        # Step 1+5 — Suppress geometry-only relations
        is_inanimate_pair = (not is_animate_subj and "animate" not in _get_categories(obj))

        if pred in WEAK_SPATIAL:
            # Reject spatial relations between two inanimates (pure clutter)
            if REJECT_INANIMATE_SPATIAL and is_inanimate_pair:
                precision_log.append({
                    "subject": subj, "predicate": pred, "object": obj,
                    "adj_conf": adj_conf, "status": "rejected",
                    "reason": "spatial relation between inanimate objects (no semantic value)",
                })
                continue
            if adj_conf < WEAK_SPATIAL_THRESHOLD:
                precision_log.append({
                    "subject": subj, "predicate": pred, "object": obj,
                    "adj_conf": adj_conf, "status": "rejected",
                    "reason": f"weak spatial relation (adj_conf={adj_conf:.3f} < {WEAK_SPATIAL_THRESHOLD})",
                })
                continue

        # Reject ALL neutral spatial relations between inanimates
        # "on"/"in" between two inanimate objects is pure geometry clutter
        if pred in NEUTRAL_SPATIAL and is_inanimate_pair:
            precision_log.append({
                "subject": subj, "predicate": pred, "object": obj,
                "adj_conf": adj_conf, "status": "rejected",
                "reason": f"neutral spatial between inanimates (no semantic value, adj_conf={adj_conf:.3f})",
            })
            continue

        # Directionality: reject spatial predicates with inanimate subject + animate object
        # This catches "bicycle on person", "book above person", etc.
        if (pred in WEAK_SPATIAL or pred in NEUTRAL_SPATIAL
                and not is_animate_subj and "animate" in _get_categories(obj)):
            precision_log.append({
                "subject": subj, "predicate": pred, "object": obj,
                "adj_conf": adj_conf, "status": "rejected",
                "reason": f"spatial predicate with inanimate subject + animate object "
                          f"(reversed direction, adj_conf={adj_conf:.3f})",
            })
            continue

        # Step 5 — MIN_SEMANTIC_SCORE for non-semantic predicates
        if pred not in SEMANTIC_PREDS:
            threshold_for_this = MIN_SEMANTIC_SCORE
            if adj_conf < threshold_for_this:
                precision_log.append({
                    "subject": subj, "predicate": pred, "object": obj,
                    "adj_conf": adj_conf, "status": "rejected",
                    "reason": f"below min semantic score (adj_conf={adj_conf:.3f} < {threshold_for_this})",
                })
                continue

        precision_log.append({
            "subject": subj, "predicate": pred, "object": obj,
            "adj_conf": adj_conf, "status": "kept",
            "reason": "passed precision filters",
        })
        filtered_candidates.append(c)

    candidates = filtered_candidates

    # ═══════════════════════════════════════════════════════════════════
    # STEP 7 — Semantic consistency: prefer semantic over spatial per pair
    # ═══════════════════════════════════════════════════════════════════
    # For each (subject, object) pair, if there are multiple candidates
    # (shouldn't happen normally, but just in case), keep only the best
    # semantic one if scores are close.
    if len(candidates) > 1:
        pre_consistency = len(candidates)
        candidates = _apply_semantic_consistency(candidates)
        if debug and len(candidates) < pre_consistency:
            print(f"\n[consistency] Semantic consistency removed "
                  f"{pre_consistency - len(candidates)} candidates")

    if not candidates:
        if debug:
            print(f"\n[precision filter] ALL RELATIONS REJECTED")
            for pl in precision_log:
                if pl["status"] == "kept":
                    continue
                print(f"  REJECTED: {pl['subject']} {pl['predicate']} {pl['object']}")
                print(f"    reason: {pl['reason']}")
            print(f"\n[semantic] 0 relations after precision filtering")
        return [], raw_debug

    # ═══════════════════════════════════════════════════════════════════
    # STEP 2 — Strong semantic priority sort
    # ═══════════════════════════════════════════════════════════════════
    def _sort_key(x: Dict) -> float:
        is_animate = "animate" in _get_categories(x["subject"])
        is_sem = x["predicate"] in SEMANTIC_PREDS
        adj_conf = x.get("adjusted_confidence", x["confidence"])
        # Large gap so semantic predicates always rank above spatial
        sem_boost = 10.0 if is_sem else 0.0
        anim_boost = 5.0 if is_animate else 0.0
        return anim_boost + sem_boost + adj_conf

    candidates.sort(key=_sort_key, reverse=True)

    # Step 6 — Directionality-aware symmetric dedup
    seen_pairs: Dict[tuple, Dict] = {}
    for c in candidates:
        pair = tuple(sorted([c["subject"], c["object"]]))
        if pair not in seen_pairs:
            seen_pairs[pair] = c
        else:
            existing = seen_pairs[pair]
            if _directionality_preference(c) > _directionality_preference(existing):
                seen_pairs[pair] = c

    deduped = list(seen_pairs.values())

    # ═══════════════════════════════════════════════════════════════════
    # STEP 4 — One primary relation per animate subject
    # ═══════════════════════════════════════════════════════════════════
    subject_best: Dict[str, Dict] = {}
    for c in deduped:
        subj = c["subject"]
        subj_cats = _get_categories(subj)
        if "animate" not in subj_cats:
            key = f"__{subj}__{c['object']}"
            subject_best[key] = c
        else:
            if subj not in subject_best:
                subject_best[subj] = c
            else:
                existing = subject_best[subj]
                existing_key = (
                    existing["predicate"] in SEMANTIC_PREDS,
                    existing.get("adjusted_confidence", existing["confidence"]),
                )
                current_key = (
                    c["predicate"] in SEMANTIC_PREDS,
                    c.get("adjusted_confidence", c["confidence"]),
                )
                if current_key > existing_key:
                    subject_best[subj] = c

    selected_relations = list(subject_best.values())[:top_k]

    # ═══════════════════════════════════════════════════════════════════
    # STEP 8 — Debug output: precision filter transparency
    # ═══════════════════════════════════════════════════════════════════
    if debug:
        rejected = [p for p in precision_log if p["status"] == "rejected"]
        kept = [p for p in precision_log if p["status"] == "kept"]
        if rejected:
            print(f"\n[precision filter] ─── PRECISION FILTER ───")
            for pl in rejected:
                print(f"  REJECTED: {pl['subject']} {pl['predicate']} {pl['object']}")
                print(f"    reason: {pl['reason']}")
        if kept:
            kept_selected = [p for p in precision_log
                             if p["status"] == "kept" and
                             f"{p['subject']} {p['predicate']} {p['object']}" in
                             {f'{r["subject"]} {r["predicate"]} {r["object"]}' for r in selected_relations}]
            if kept_selected:
                print(f"  KEPT:")
                for pl in kept_selected:
                    print(f"  KEPT: {pl['subject']} {pl['predicate']} {pl['object']}")
                    print(f"    reason: {pl['reason']}")

        # Per-pair debug
        print(f"\n[calibrated] ─── Relation Debug ───")
        for rd in raw_debug:
            if rd["status"] == "rejected_extreme_nonsense":
                continue
            s, o = rd["subject"], rd["object"]
            print(f"  {s}+{o}:")
            top_pp = sorted(rd["per_predicate"], key=lambda x: -x["final"])[:5]
            for pp in top_pp:
                p = pp["predicate"]
                cal = pp["calibrated"]
                prior = pp["prior_total"]
                final_val = pp["final"]
                marker = " ◀" if p == rd["best_predicate"] else ""
                print(f"    {p}:")
                print(f"      calibrated={cal:.4f}")
                print(f"      prior={prior:+.4f}")
                print(f"      final={final_val:.4f}{marker}")
            print(f"  SELECTED: {rd['best_predicate']} "
                  f"(calib={rd['best_calibrated']:.4f} → "
                  f"final={rd['best_final_score']:.4f})")
            print()

        print(f"[calibrated] {len(raw_debug)} pairs evaluated -> "
              f"{len(precision_log)} after precision filter -> "
              f"{len(candidates)} after calibration+prior -> "
              f"{len(deduped)} after dedup -> "
              f"{len(selected_relations)} selected (top_k={top_k})")
        for r in selected_relations:
            sem_marker = " ★" if r["predicate"] in SEMANTIC_PREDS else ""
            print(f"[calibrated]   {r['subject']} {r['predicate']} {r['object']} "
                  f"(calib={r['confidence']:.3f}, adj={r['adjusted_confidence']:.3f}, "
                  f"prior={r['prior_adjustment']:+.3f}){sem_marker}")

        # ═══════════════════════════════════════════════════════════════
        # STEP 4 — Feature utilization analysis
        # ═══════════════════════════════════════════════════════════════
        if _model is not None:
            try:
                feature_norms = _get_feature_group_norms(_model)
                total_fn = sum(feature_norms.values()) or 1.0
                header = "FEATURE GROUP NORMS (projection weights)" if _model_type == "transformer" else "FEATURE GROUP NORMS (first layer)"
                print(f"\n  ─── {header} ───")
                for name, norm in sorted(feature_norms.items(), key=lambda x: -x[1]):
                    pct = norm / total_fn * 100
                    print(f"    {name:15s}: {norm:8.4f}  ({pct:5.1f}%)")
                print(f"    {'─' * 35}")
                clip_pct = (feature_norms.get("subj_clip", 0) + feature_norms.get("obj_clip", 0)) / total_fn * 100
                union_pct = feature_norms.get("union_clip", 0) / total_fn * 100
                pose_pct = feature_norms.get("pose", 0) / total_fn * 100
                geo_pct = feature_norms.get("geo", 0) / total_fn * 100
                label_pct = (feature_norms.get("subj_label", 0) + feature_norms.get("obj_label", 0)) / total_fn * 100
                print(f"    Subject CLIP:      {clip_pct:5.1f}%  "
                      f"{'(underused)' if clip_pct < 30 else '(active)'}")
                if _model_union_dim > 0:
                    print(f"    Union-region CLIP: {union_pct:5.1f}%  "
                          f"{'(underused)' if union_pct < 10 else '(active)'}")
                if _model_pose_dim > 0:
                    print(f"    Pose features:     {pose_pct:5.1f}%  "
                          f"{'(underused)' if pose_pct < 5 else '(active)'}")
                print(f"    Geometry:          {geo_pct:5.1f}%  "
                      f"{'(dominant)' if geo_pct > 15 else '(controlled)'}")
                print(f"    Label embeddings:  {label_pct:5.1f}%")

                # For transformer, also show attention-based modality usage
                if _model_type == "transformer":
                    try:
                        attn_contribs = _model.get_feature_contributions(
                            torch.tensor([0], device=_device),
                            torch.tensor([0], device=_device),
                            torch.zeros((1, _model_geo_dim), device=_device),
                        )
                        if attn_contribs:
                            total_attn = sum(attn_contribs.values()) or 1.0
                            print(f"\n  ─── ATTENTION-BASED MODALITY USAGE ───")
                            for name, val in sorted(attn_contribs.items(), key=lambda x: -x[1]):
                                pct = val / total_attn * 100
                                print(f"    {name:15s}: {pct:5.1f}%")
                    except Exception as e2:
                        print(f"  [attention analysis] ({e2})")
            except Exception as e:
                print(f"  [feature analysis] Skipped ({e})")

        # ═══════════════════════════════════════════════════════════════
        # STEP 8 — Quality evaluation
        # ═══════════════════════════════════════════════════════════════
        quality = evaluate_relation_quality(selected_relations, raw_debug)
        print(f"\n  ─── RELATION QUALITY METRICS ───")
        print(f"    Total relations:           {quality['total_relations']}")
        print(f"    Semantic precision:        {quality['semantic_precision']:.2%} "
              f"({quality['semantic_relations']}/{quality['total_relations']})")
        print(f"    Animate subject rate:      {quality['animate_subject_rate']:.2%}")
        print(f"    Reversed direction rate:   {quality['reversed_direction_rate']:.2%}")
        print(f"    Weak spatial rate:         {quality['weak_spatial_rate']:.2%}")
        if quality.get("predicate_breakdown"):
            print(f"    Predicate breakdown:       {quality['predicate_breakdown']}")

    return selected_relations, raw_debug
