"""Reference-free CLIPScore (Hessel et al., 2021) for the caption experiment.

    CLIPScore(image, caption) = 2.5 * max(cos(E_img(image), E_txt("A photo depicts " + caption)), 0)

with CLIP ViT-B/32 - the definition, weight w=2.5 and text prompt of the paper.
It is computed with the HuggingFace CLIP port (utils.clip_scorer), whose image
preprocessing is close to but not bit-identical with the OpenAI `clip`
package, so absolute values can differ slightly from published CLIPScores;
comparisons between arms in this experiment are unaffected because every arm
goes through the same code.

It measures image-text alignment, not hallucination and not agreement with
human reference captions (none exist for these images).
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
import torch.nn.functional as F

CLIPSCORE_WEIGHT = 2.5
CLIPSCORE_PROMPT = "A photo depicts "


def _features(output) -> torch.Tensor:
    return output.pooler_output if hasattr(output, "pooler_output") else output


@torch.no_grad()
def compute_clipscores(image_files: Mapping[str, str],
                       captions_by_arm: Mapping[str, Mapping[str, str]],
                       device: Optional[str] = None,
                       batch_size: int = 32) -> Dict[str, Dict[str, float]]:
    """arm -> image_id -> CLIPScore. Each image is encoded once for all arms."""
    from PIL import Image
    from utils.clip_scorer import get_clip_scorer

    scorer = get_clip_scorer()
    model, processor = scorer._model, scorer._processor
    dev = next(model.parameters()).device if device is None else torch.device(device)

    ids = list(image_files)
    image_emb: Dict[str, torch.Tensor] = {}
    for s in range(0, len(ids), batch_size):
        chunk = ids[s:s + batch_size]
        images = [Image.open(image_files[i]).convert("RGB") for i in chunk]
        inputs = processor(images=images, return_tensors="pt").to(dev)
        emb = F.normalize(_features(model.get_image_features(**inputs)).float(), dim=-1).cpu()
        for i, e in zip(chunk, emb):
            image_emb[i] = e

    scores: Dict[str, Dict[str, float]] = {}
    for arm, caps in captions_by_arm.items():
        scores[arm] = {}
        for s in range(0, len(ids), batch_size):
            chunk = ids[s:s + batch_size]
            texts = [CLIPSCORE_PROMPT + caps[i] for i in chunk]
            inputs = processor(text=texts, return_tensors="pt", padding=True,
                               truncation=True, max_length=77).to(dev)
            emb = F.normalize(_features(model.get_text_features(**inputs)).float(), dim=-1).cpu()
            for i, t in zip(chunk, emb):
                cos = float(torch.dot(image_emb[i], t))
                scores[arm][i] = CLIPSCORE_WEIGHT * max(cos, 0.0)
    return scores
