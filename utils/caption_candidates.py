"""Multi-candidate BLIP generation and uniform candidate scoring.

Candidate pool (per image)
--------------------------
Beam search is run once per PREFIX and asked for all of its finished beams:

    baseline      "a photo of"                                    always
    objects_only  "a photo of a <subject> and a <object>"          if a relation was selected
    relation      "a photo of a <subject> <predicate> a <object>"  if a relation was selected

The relation prefix is used at ANY confidence - the confidence enters the
score, not a gate. That is the point of reranking: the system sees what the
caption would look like if it stated the relation, and may decline.

`num_candidates` beams are returned per prefix with the SAME decoding
parameters the frozen experiment used (`num_beams=4, max_new_tokens=128,
early_stopping=True, do_sample=False`, float32). With the default
`num_candidates=4 == num_beams`, beam rank 0 of each prefix is bit-identical
to what a `num_return_sequences=1` run of that prefix produces, so the pool
provably CONTAINS the baseline, objects-only and grounded arms' captions: the
reranked arms are a choice over the existing arms plus their runners-up, not a
different generator. `smoke` verifies this identity on the real model.

Uniform candidate score
-----------------------
Beam search's own `sequences_scores` cover only the tokens a prefix did not
supply, so a candidate produced from the relation prefix is scored over fewer
tokens than one produced from "a photo of" - they are not comparable, and
using them would systematically favour the longest prefix. Every candidate is
therefore re-scored identically:

    lm(c) = mean over tokens of log P(token | image, preceding tokens)

teacher-forced from the caption's first token, conditioned on the image alone.
The prefix that happened to generate a candidate leaves no trace in its score.
The image is encoded once and its embedding is shared by the whole batch, so
scoring a pool costs one vision forward and one text forward per image.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

import torch

from utils.blip_captioner import (BASELINE_PREFIX, EXPERIMENT_GENERATION_CONFIG,
                                  build_blip_prefix, build_objects_only_prefix)
from utils.caption_rerank import SOURCE_ORDER

DEFAULT_NUM_CANDIDATES = 4
DEFAULT_SCORE_BATCH_SIZE = 12


def candidate_prefixes(relation: Optional[Mapping]) -> Dict[str, str]:
    """source -> prefix. Deterministic; no thresholds are applied here."""
    prefixes = {"baseline": BASELINE_PREFIX}
    if relation:
        prefixes["objects_only"] = build_objects_only_prefix(relation)
        prefix, used = build_blip_prefix([], [dict(relation)], min_confidence=0.0)
        if not used:
            raise ValueError(f"relation {dict(relation)} has no injectable surface form")
        prefixes["relation"] = prefix
    return {s: prefixes[s] for s in SOURCE_ORDER if s in prefixes}


def _blip(dtype: Optional[torch.dtype]):
    import utils.blip_captioner as blip
    if blip._model is None:
        blip._load_model(dtype=dtype)
    elif dtype is not None and next(blip._model.parameters()).dtype != dtype:
        raise RuntimeError(f"BLIP already loaded as {next(blip._model.parameters()).dtype}, "
                           f"requested {dtype}")
    return blip._processor, blip._model, blip._device


@torch.no_grad()
def generate_candidates(image, prefixes: Mapping[str, str],
                        num_candidates: int = DEFAULT_NUM_CANDIDATES,
                        dtype: Optional[torch.dtype] = None,
                        generation_config: Optional[Mapping] = None) -> List[Dict]:
    """All finished beams of every prefix, in (source, beam_rank) order.

    `lm_score` is left unset here; `score_candidates` fills it so that every
    candidate is scored the same way regardless of which prefix produced it.
    """
    processor, model, device = _blip(dtype)
    cfg = dict(EXPERIMENT_GENERATION_CONFIG if generation_config is None else generation_config)
    if num_candidates < 1:
        raise ValueError(f"num_candidates must be >= 1, got {num_candidates}")
    if num_candidates > cfg["num_beams"]:
        raise ValueError(f"num_candidates {num_candidates} exceeds num_beams "
                         f"{cfg['num_beams']}: beam search cannot return more finished "
                         "beams than it keeps")

    out: List[Dict] = []
    for source in SOURCE_ORDER:
        if source not in prefixes:
            continue
        prefix = prefixes[source]
        inputs = processor(images=image, text=prefix, return_tensors="pt").to(device)
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=next(model.parameters()).dtype)
        generated = model.generate(**inputs, **cfg, num_return_sequences=num_candidates,
                                   output_scores=True, return_dict_in_generate=True)
        seq_scores = (generated.sequences_scores.tolist()
                      if getattr(generated, "sequences_scores", None) is not None
                      else [None] * len(generated.sequences))
        for rank, (ids, beam_score) in enumerate(zip(generated.sequences, seq_scores)):
            text = " ".join(processor.decode(ids, skip_special_tokens=True).split())
            out.append({
                "source": source,
                "prefix": prefix,
                "beam_rank": rank,
                "text": text,
                "beam_sequence_score": None if beam_score is None else float(beam_score),
            })
    return out


@torch.no_grad()
def score_candidates(image, candidates: Sequence[Mapping],
                     batch_size: int = DEFAULT_SCORE_BATCH_SIZE,
                     dtype: Optional[torch.dtype] = None) -> List[Dict]:
    """Attach `lm_score` (mean per-token log-likelihood given the image) to each candidate.

    Identical conditioning for every candidate: the image plus the caption's
    own tokens from its first token onwards. Candidates are deduplicated by
    text before the forward pass and the score is copied back, so repeated
    beams cost nothing and always receive exactly the same number.
    """
    import torch.nn.functional as F

    if not candidates:
        return []
    processor, model, device = _blip(dtype)
    model_dtype = next(model.parameters()).dtype
    bos = model.config.text_config.bos_token_id

    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(
        device=device, dtype=model_dtype)
    image_embeds = _encode_image(model, pixel_values)

    unique: List[str] = []
    seen = set()
    for c in candidates:
        text = str(c["text"])
        if text not in seen:
            seen.add(text)
            unique.append(text)

    scores: Dict[str, float] = {}
    for start in range(0, len(unique), max(1, batch_size)):
        chunk = unique[start:start + max(1, batch_size)]
        enc = processor.tokenizer(chunk, return_tensors="pt", padding=True,
                                  truncation=True, max_length=512)
        input_ids = enc["input_ids"].clone().to(device)
        attention_mask = enc["attention_mask"].to(device)
        # BLIP's own `generate` replaces the tokenizer's leading [CLS] with the
        # decoder BOS token; mirror it so scoring and generation see one prefix.
        input_ids[:, 0] = bos
        logits = _decode_logits(model, input_ids, attention_mask, image_embeds,
                                pixel_values, len(chunk))
        logprobs = F.log_softmax(logits[:, :-1].float(), dim=-1)
        targets = input_ids[:, 1:]
        gathered = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        mask = attention_mask[:, 1:].to(gathered.dtype)
        n_tokens = mask.sum(dim=1).clamp(min=1)
        means = ((gathered * mask).sum(dim=1) / n_tokens).tolist()
        for text, value in zip(chunk, means):
            scores[text] = float(value)

    return [dict(c, lm_score=scores[str(c["text"])]) for c in candidates]


def _encode_image(model, pixel_values):
    return model.vision_model(pixel_values=pixel_values)[0]


def _decode_logits(model, input_ids, attention_mask, image_embeds, pixel_values, batch: int):
    """Text-decoder logits for `batch` captions over one shared image embedding.

    The image is encoded once and broadcast, which is what makes scoring a pool
    affordable on 4 GB. If a transformers version ever changes the decoder's
    signature this falls back to the public forward with the pixel values
    repeated - slower, numerically identical (verified to 1e-5 on CPU).
    """
    encoder_attention_mask = torch.ones(image_embeds.shape[:-1], dtype=torch.long,
                                        device=image_embeds.device)
    try:
        return model.text_decoder(
            input_ids=input_ids, attention_mask=attention_mask,
            encoder_hidden_states=image_embeds.expand(batch, -1, -1),
            encoder_attention_mask=encoder_attention_mask.expand(batch, -1),
        ).logits
    except (AttributeError, TypeError):
        return model(pixel_values=pixel_values.expand(batch, -1, -1, -1),
                     input_ids=input_ids, attention_mask=attention_mask).logits


def generate_and_score(image, relation: Optional[Mapping],
                       num_candidates: int = DEFAULT_NUM_CANDIDATES,
                       batch_size: int = DEFAULT_SCORE_BATCH_SIZE,
                       dtype: Optional[torch.dtype] = None) -> Dict:
    """The whole per-image generation stage: prefixes -> beams -> uniform scores."""
    prefixes = candidate_prefixes(relation)
    candidates = generate_candidates(image, prefixes, num_candidates=num_candidates, dtype=dtype)
    candidates = score_candidates(image, candidates, batch_size=batch_size, dtype=dtype)
    return {"prefixes": prefixes, "candidates": candidates}
