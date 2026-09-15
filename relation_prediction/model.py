"""
MLP relation classifier with visual-semantic support.

Architecture
------------
    [subj_label_emb | obj_label_emb | geo | subj_visual | obj_visual |
     union_visual* | pose_features*]
    → Linear → ReLU → Dropout
    → Linear → ReLU → Dropout
    → Linear → logits (num_predicates)

Visual features (CLIP embeddings) are optional:
    - When clip_dim=0:  input = 2*embed_dim + geo_dim        (geometry-only)
    - When clip_dim>0:  input = 2*embed_dim + geo_dim + 2*clip_dim  (visual-semantic)

geo_dim is 5 for the legacy descriptor and 19 for the extended one
(see relation_prediction.vg_dataset.extract_geo_features_ext).

* union_visual and pose_features are optional interaction-aware augmentations:
    - union_dim: CLIP embedding of the union region covering both subject and object
    - pose_dim:  Compact 20-dim pose features for interaction understanding

The label embeddings provide coarse class information.
The CLIP visual embeddings provide appearance-based interaction cues.
The union-region embedding captures contact, posture, and interaction context.
The pose features provide body interaction cues (sitting, riding, holding, etc).

All new features are OPTIONAL and backward-compatible.

The exact block list is RelationMLP.feature_blocks(). The first Linear is sized
from it and every forward() checks the batch against it, so a missing or
mis-sized block raises FeatureDimensionError naming the block instead of a bare
"mat1 and mat2 shapes cannot be multiplied".
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from .vg_dataset import GEO_DIM, GEO_DIM_EXT, POSE_FEATURE_DIM, UNION_FEATURE_DIM


class FeatureDimensionError(ValueError):
    """The feature blocks handed to the model do not match the widths it was built for.

    Raised before concatenation. The union ablation arm used to reach the first
    Linear with its union block silently dropped and fail as
    "mat1 and mat2 shapes cannot be multiplied (512x1683 and 2451x256)", which
    names neither the block nor the cause.
    """


def format_feature_blocks(blocks: List[Tuple[str, int]]) -> str:
    """Render [(name, width), ...] as "name:width + name:width"."""
    return " + ".join(f"{name}:{width}" for name, width in blocks)


class RelationMLP(nn.Module):
    """
    Args:
        num_labels:     Vocabulary size for subject / object labels.
        num_predicates: Number of predicate classes to predict.
        embed_dim:      Embedding dimension for label tokens.
        hidden_dims:    Sequence of hidden layer widths.
        dropout:        Dropout probability applied after each hidden ReLU.
        clip_dim:       Dimension of CLIP visual embeddings (0 = no visual features).
        pose_dim:       Dimension of pose features (0 = no pose features).
        union_dim:      Dimension of union-region CLIP embedding (0 = no union).
        geo_dim:        Geometry descriptor width (GEO_DIM=5 legacy,
                        GEO_DIM_EXT=19 extended).
        geo_norm:       Standardise the geometry block with a BatchNorm1d.
                        The running statistics live in the state_dict, so the
                        exact training-time normalisation is restored at
                        inference with no extra checkpoint plumbing.
    """

    def __init__(
        self,
        num_labels: int,
        num_predicates: int,
        embed_dim: int = 64,
        hidden_dims: tuple = (256, 128),
        dropout: float = 0.3,
        clip_dim: int = 0,
        pose_dim: int = 0,
        union_dim: int = 0,
        geo_dim: int = GEO_DIM,
        geo_norm: bool = False,
    ) -> None:
        super().__init__()

        self.label_emb = nn.Embedding(num_labels, embed_dim, padding_idx=0)
        self.embed_dim = embed_dim
        self.clip_dim = clip_dim
        self.pose_dim = pose_dim
        self.union_dim = union_dim
        self.geo_dim = geo_dim
        if geo_norm and geo_dim == 0:
            raise ValueError("geo_norm=True is meaningless with geo_dim=0")
        self.geo_norm = nn.BatchNorm1d(geo_dim) if geo_norm else None

        # Sized from the same block list check_inputs() and forward() walk, so
        # the width allocated here and the width a batch is held to cannot
        # drift apart. Plain int attribute: the state_dict is unchanged.
        self.input_dim = sum(width for _, width in self.feature_blocks())
        layers: list = []
        prev = self.input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, num_predicates))
        self.mlp = nn.Sequential(*layers)

    def feature_blocks(self) -> List[Tuple[str, int]]:
        """Ordered (name, width) of every block forward() concatenates."""
        blocks = [
            ("subj_label", self.embed_dim),
            ("obj_label", self.embed_dim),
            ("geometry", self.geo_dim),
        ]
        if self.clip_dim > 0:
            blocks += [("subj_clip", self.clip_dim), ("obj_clip", self.clip_dim)]
        if self.union_dim > 0:
            blocks.append(("union_clip", self.union_dim))
        if self.pose_dim > 0:
            blocks.append(("pose", self.pose_dim))
        return blocks

    def check_inputs(
        self,
        geo: torch.Tensor,
        subj_feat: torch.Tensor = None,
        obj_feat: torch.Tensor = None,
        union_feat: torch.Tensor = None,
        pose_feat: torch.Tensor = None,
    ) -> List[Tuple[str, int]]:
        """Check a batch's feature blocks against feature_blocks().

        Returns the (name, width) blocks the batch supplies, in concatenation
        order. Raises FeatureDimensionError naming every enabled block that is
        missing (None) or has the wrong width.

        A block the model was built WITHOUT is ignored even when a tensor is
        passed. That is deliberate: the geometry control of the feature
        ablation loads the CLIP cache (to fix the sample population) but is
        built with clip_dim=0, so the loader still hands it subj/obj feats.
        """
        supplied = {
            "geometry": geo,
            "subj_clip": subj_feat,
            "obj_clip": obj_feat,
            "union_clip": union_feat,
            "pose": pose_feat,
        }
        expected = self.feature_blocks()
        actual: List[Tuple[str, int]] = []
        problems: List[str] = []
        for name, width in expected:
            if name not in supplied:          # label embeddings: built internally
                actual.append((name, width))
                continue
            tensor = supplied[name]
            if tensor is None:
                problems.append(f"{name} expects {width} columns but was not passed")
                continue
            got = int(tensor.shape[-1])
            actual.append((name, got))
            if got != width:
                problems.append(f"{name} expects {width} columns, got {got}")
        if problems:
            raise FeatureDimensionError(
                f"RelationMLP expects input_dim={self.input_dim} "
                f"[{format_feature_blocks(expected)}] but the batch supplies "
                f"{sum(w for _, w in actual)} [{format_feature_blocks(actual)}]: "
                + "; ".join(problems)
            )
        return actual

    def forward(
        self,
        subj_idx: torch.Tensor,   # (B,) long
        obj_idx:  torch.Tensor,   # (B,) long
        geo:      torch.Tensor,   # (B, GEO_DIM) float
        subj_feat: torch.Tensor = None,  # (B, clip_dim) float or None
        obj_feat:  torch.Tensor = None,  # (B, clip_dim) float or None
        union_feat: torch.Tensor = None, # (B, union_dim) float or None
        pose_feat:  torch.Tensor = None, # (B, pose_dim) float or None
    ) -> torch.Tensor:            # (B, num_predicates)
        # Every enabled block must be present at its declared width. Before
        # this check an enabled block passed as None was silently skipped, so
        # the concatenation came up short and the failure surfaced one layer
        # later as an anonymous matmul shape error.
        self.check_inputs(geo, subj_feat, obj_feat, union_feat, pose_feat)

        se = self.label_emb(subj_idx)         # (B, embed_dim)
        oe = self.label_emb(obj_idx)          # (B, embed_dim)

        if self.geo_norm is not None:
            geo = self.geo_norm(geo)

        components = [se, oe, geo]

        # The clip_dim guard matters: the geometry control of the feature
        # ablation loads the CLIP cache (to fix the sample population) but is
        # built with clip_dim=0, so the loader still hands it subj/obj feats.
        # Without this check those 1536 columns would be concatenated onto a
        # model whose first Linear never allocated weights for them, and the
        # run would die in the first batch instead of training the control.
        if self.clip_dim > 0:
            components.append(subj_feat)
            components.append(obj_feat)

        if self.union_dim > 0:
            components.append(union_feat)

        if self.pose_dim > 0:
            components.append(pose_feat)

        x = torch.cat(components, dim=-1)     # (B, input_dim)
        return self.mlp(x)
