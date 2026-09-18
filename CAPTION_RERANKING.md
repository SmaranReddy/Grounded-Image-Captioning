# Grounding-aware candidate reranking

This document covers the second caption experiment. The first one
(`CAPTION_EXPERIMENTS.md`) asked whether injecting a predicted relation into
BLIP's prefix reduces object hallucination. It does not. This one asks whether
using the same visual evidence to **choose between captions** — rather than to
**dictate** one — is a better form of grounding.

Read `CAPTION_EXPERIMENTS.md` first: its image set, detector, CLIP
verification, relation checkpoint, decoding parameters and evaluation
protocol are reused here unchanged.

---

## 1. Why the hard relation prefix was rejected as the final method

The frozen 250-image test run produced:

| metric | baseline | grounded (relation prefix) | objects_only (control) |
|---|---|---|---|
| CHAIR_i | 4.11% | **8.30%** | 8.35% |
| CHAIR_s | 5.20% | **15.60%** | 15.60% |
| hallucinated objects / caption | 0.056 | **0.160** | 0.160 |
| mentioned objects / caption | 1.364 | 1.928 | 1.916 |
| object recall | 40.67% | 54.98% | 54.60% |
| POPE random F1 | 60.85% | 74.04% | 73.85% |
| POPE popular F1 | 60.79% | 73.54% | 73.42% |
| POPE adversarial F1 | 60.72% | 73.33% | 73.28% |
| CLIPScore | 0.7115 | 0.6733 | 0.6776 |

Relation injected in 227/250 images (90.8%). Paired grounded − baseline, 95%
bootstrap CI over images: CHAIR_i **+4.19 pt [1.49, 6.89]**, CHAIR_s **+10.40 pt
[6.00, 14.80]**, POPE-adversarial F1 **+12.61 pt [9.49, 15.77]**, object recall
**+14.30 pt [11.46, 17.21]**, CLIPScore −0.04. Grounded − objects_only: no
meaningful difference on any metric.

Two conclusions follow, and only these two:

1. **Naming verified objects helps object-centric metrics** (recall, POPE) and
   **hurts hallucination metrics** (CHAIR). It is a "say more" control, and it
   costs image-text alignment (CLIPScore).
2. **The predicate contributed nothing measurable.** Everything the grounded
   arm achieved, the objects-only control achieved too. There is no evidence
   in that experiment that relations help captioning.

The mechanism explains both. A prefix is a **mandatory assertion**: whatever is
spliced into `"a photo of ..."` is stated by the caption whether or not it is
true, so every detector error and every relation error is converted
one-for-one into caption content. The system cannot decline. This is visible
in single images — on a picture of a cat and a laptop, the prefix
`"a photo of a person riding a bicycle"` makes BLIP produce *"a photo of a
person riding a bicycle with a laptop"*.

The relation model is not the bottleneck. At 64.24% mean top-1 over 19
predicates it is a usable component, and the earlier relation experiments
established that more capacity, more training and richer geometry do not move
it. The bottleneck is that a 64%-accurate predictor is being used as if it
were certain.

**Nothing in this document withdraws the frozen experiment.** Its numbers,
run directory and conclusions stand as recorded.

---

## 2. Why reranking

Replace the mandate with a choice. BLIP generates several candidate captions;
a deterministic scoring function picks one using the system's own visual
evidence. A wrong or weak relation then costs the relation candidate a few
points of score instead of appearing verbatim in the output.

Alternatives considered and why they lost:

| option | verdict |
|---|---|
| **A. hard relation prefix** (current) | rejected: measured above. Mandatory assertion. |
| **B. objects-only prefix** | kept as a control, not as a method: same failure mode, no relation. |
| **C. tune the confidence threshold** | the exploratory sweep in the frozen run already varies it over 0.0–0.9; it trades usage against damage but never separates `grounded` from `objects_only`, because the failure is the mandate, not the cut-off. |
| **D. post-hoc caption repair / gating** | exists in the legacy path (`utils/blip_captioner.gate_caption`, `utils/relation_corrector.py`). It rewrites BLIP's text with hand-written rules and a template fallback, which is neither reproducible nor defensible, and it is what the frozen experiment deliberately switched off. |
| **E. fine-tune BLIP on relation-conditioned captions** | no caption supervision exists for these images (Visual Genome has no reference captions here), and 4 GB of VRAM is not a training budget. Out of scope. |
| **F. multi-candidate + grounding-aware reranking** | **chosen.** Reuses every existing component, adds no model, no training and no hand-written text rules; its only free parameters are three scalars selected on a disjoint validation split. |

One property makes F cheap to defend: with the default settings the candidate
pool **provably contains the captions of all three existing arms** (§3), so
reranking is a choice *over* the frozen experiment's own outputs rather than a
different generator.

---

## 3. Candidate generation (exact procedure)

`utils/caption_candidates.py`, stage `python run_caption_experiment.py candidates`.

For each image, beam search runs once per prefix and returns all of its
finished beams:

| source | prefix | when |
|---|---|---|
| `baseline` | `a photo of` | always |
| `objects_only` | `a photo of a <subject> and a <object>` | whenever a relation was selected |
| `relation` | `a photo of a <subject> <predicate> a <object>` | whenever a relation was selected |

The relation prefix is generated from at **any** confidence. The confidence
enters the score, not a gate — that is what makes the relation evidence rather
than an instruction.

Decoding is the frozen experiment's, unchanged: `num_beams=4`,
`max_new_tokens=128`, `early_stopping=True`, `do_sample=False`, float32, one
image at a time. `--num-candidates` (default **4**) beams are returned per
prefix, giving a pool of at most 12 candidates, exactly 4 for an image with no
relation.

**Why K = 4 rather than 5.** `num_return_sequences` does not change beam
search; it only decides how many of the beams it already keeps are returned.
With `K = num_beams = 4`, beam rank 0 of each prefix is bit-identical to what
that prefix produces on its own — so the pool contains the `baseline`,
`objects_only` and `grounded` arms' captions by construction. `smoke` asserts
this identity against the real model and the real arms, and it held on every
image of the local end-to-end check. A larger K would need a larger beam
width, which would change the decoding and break the comparison with the
frozen arms.

### Uniform candidate scoring

Beam search's own `sequences_scores` cover only the tokens a prefix did *not*
supply, so a candidate from the relation prefix is scored over fewer tokens
than one from `"a photo of"`. They are not comparable, and using them would
systematically favour the longest prefix. Every candidate is therefore
re-scored identically:

    lm(c) = mean over tokens of log P(token | image, preceding tokens)

teacher-forced from the caption's own first token, conditioned on the image
alone. Which prefix produced a candidate leaves no trace in its score. The
image is encoded once and its embedding is shared across the batch
(`--batch-size`, default 12), so scoring a whole pool costs one vision forward
and one text forward per image. The shared-embedding path is checked against
the public `BlipForConditionalGeneration.forward` in `smoke`; they agreed to
7.6e-6 on CPU.

---

## 4. The scoring function (exact formula)

`utils/caption_rerank.py`. For a candidate caption `c`, verified-detection
object set `V` and selected relation `r = (subject, predicate, object, confidence)`:

```
score(c) =       lm(c)
         + w_obj  * support(c)                     support(c)     = |mentions(c) & V|
         - w_hall * unsupported(c)                 unsupported(c) = |mentions(c) - V|
         + w_rel  * confidence * rel(c)            rel(c)         = 1 if c states r else 0
```

* `lm` carries weight 1 **by definition**. Only the ratios between weights
  matter, so fixing it removes one redundant parameter.
* `mentions()` is `utils/coco_mentions.py` — the same mapper the evaluation
  applies to captions *and* to the human ground truth. "Supported" and
  "hallucinated" are therefore measured in one vocabulary, and a caption
  saying "a man" is credited for a detected `person`.
* `V` comes from the CLIP-verified YOLO detections, mapped through that same
  mapper. Detections outside COCO-80 contribute nothing.
* `rel(c)` is a **mention-level** match, not a string match: the subject class
  must be mentioned, the object class must be mentioned later, and the
  predicate's words must lie strictly between them. So "a man riding a
  motorcycle" states `(person, riding, motorcycle)` but "a person next to a
  bicycle" does not state `(person, riding, bicycle)`.
* The relation term is **soft and confidence-weighted**. A 0.05-confidence
  relation contributes 0.05·w_rel; a 0.9-confidence one contributes 0.9·w_rel.
  There are no hand-set bonuses anywhere (the legacy path's +0.22-style priors
  are not used, and neither is its predicate override).

**No CLIP image–text similarity term.** CLIPScore is one of the reported
metrics and is computed with CLIP ViT-B/32; scoring candidates by that same
similarity and then reporting CLIPScore as a result would be circular. `lm`
supplies image–text compatibility from the generator itself, at no extra model
cost, and leaves CLIPScore an independent measurement.

**No tuned length term.** `lm` is already length-normalised and the object
terms are counts that scale with content, so a fourth weight fitted on a few
hundred images would buy little. Verbosity is handled by fixed, pre-declared
hygiene bounds instead (`is_degenerate`): a candidate is dropped if it is
empty, shorter than 3 words, longer than 60 words, or repeats a 4-gram three
times. Dropped candidates are counted and reported. If every candidate is
dropped the stage raises rather than inventing text.

**Tie-breaking** is deterministic and conservative: scores equal to 9 decimal
places tie, and ties go to the `baseline` candidate first, then
`objects_only`, then `relation`; within a source, to the earlier beam; finally
by text. The reranker therefore never asserts extra content "for free" — it
departs from BLIP's own first choice only when the evidence strictly prefers
something else.

---

## 5. How the weights were selected, and on what data

Three free scalars, selected **only** on the validation split.

* **Tuning set**: `splits/caption_eval_val_200.json`, built by the same
  `build_caption_eval_set.py` with `--split val --limit 200 --seed 42`. The
  frozen E0 split is image-disjoint (`tests/test_split_integrity.py`), so no
  image of the 250-image test set can appear here;
  `tests/test_caption_eval_set.py` asserts the two manifests are disjoint
  whenever both exist.
* **Grid** (pre-declared): `w_obj ∈ {0, 0.1, 0.25, 0.5, 1.0}`,
  `w_hall ∈ {0, 0.25, 0.5, 1.0, 2.0}`, `w_rel ∈ {0, 0.25, 0.5, 1.0, 2.0}`.
* **Objective** (pre-declared): maximise **POPE-adversarial F1** on the
  validation images; ties → lower CHAIR_i; ties → smaller L1 norm of the
  weights; ties → lexicographically smallest `(w_obj, w_hall, w_rel)`.

  POPE F1 is used because it is *symmetric*: asserting an absent object costs a
  false positive and staying silent about a present one costs a false negative.
  A caption cannot win it by saying less — which CHAIR_i alone would reward,
  and which would make the empty caption optimal.
* **Procedure**: `w_rel` is held at 0 and `(w_obj, w_hall)` are chosen; then
  those two are **frozen** and only `w_rel` is chosen. The two reranking arms
  therefore differ by exactly one parameter and one scoring term.

The result is written to `results_caption/rerank_weights.json` together with
the whole grid, the objective, the validation run it came from and a sha256 of
each weight vector. `tune-rerank` refuses to overwrite a lock with different
weights, and `rerank` refuses to re-select a run directory with different
weights. The chosen weights and the file's hash are recorded in
`captions_<arm>.meta.json` inside the test run.

**Because the reranker is a pure function of the cached candidate pool, the
whole grid search runs on CPU over `candidates.jsonl` — no GPU, no
regeneration.** That is also why tuning cannot accidentally touch the test
images: the stage refuses to run on a run directory whose eval set is the test
split.

### What is allowed to see what

| | may read |
|---|---|
| candidate generation | image |
| reranking | CLIP-verified YOLO detections, predicted relation, BLIP's own likelihood |
| reranking | **never** the Visual Genome annotations |
| evaluation | human Visual Genome annotations |
| weight selection | validation images and their annotations |
| weight selection | **never** the 250 test images |

`tests/test_caption_rerank_stage.py::test_rerank_output_is_unchanged_when_the_ground_truth_is_poisoned`
replaces every human annotation in the evaluation manifest with nonsense and
asserts that the rerank stage's output is byte-identical. That makes "the
system grounds its caption in its own perception" a checked property rather
than a claim.

---

## 6. Frozen test protocol

The 250-image test set, its detections and its three prefix arms are
**unchanged**; the new stages are added to the same run directory so every arm
is scored on identical images with identical detections.

1. Build the validation set and run `detect` + `candidates` on it.
2. `tune-rerank` → `results_caption/rerank_weights.json`. **Locked.**
3. `candidates` on `results_caption/main` (the frozen test run).
4. `rerank` on `results_caption/main` with the locked file.
5. `hallucination_eval.py --run-dir results_caption/main` — **once**.

No weight, threshold, K or formula may be revisited after step 5. If the
result is negative it is reported as negative.

---

## 7. Arms

| arm | conditioning | selection |
|---|---|---|
| `baseline` | `a photo of` | — |
| `objects_only` | objects prefix when a relation passes 0.5 | — |
| `grounded` | relation prefix when a relation passes 0.5 | — |
| `object_reranked` | none (pool of all prefixes) | score with `w_rel = 0` |
| `relation_reranked` | none (pool of all prefixes) | score with the tuned `w_rel` |

The two reranking arms share **one** candidate pool, generated once. They
differ only in whether the relation term is in the score. That is deliberate:
it isolates "does relation evidence add value beyond object evidence?" from
"does having relation-conditioned candidates available help?". Note the
consequence — `object_reranked` may still select a relation-worded candidate
if the object evidence favours it; the report gives the source distribution of
the selected candidate for each arm, so this is visible rather than hidden.

---

## 8. What is reported

Everything the frozen experiment reported (CHAIR_i, CHAIR_s, hallucinated and
mentioned objects per caption, object recall, caption length, POPE
random/popular/adversarial accuracy/F1/FP-rate/yes-ratio, CLIPScore, paired
95% bootstrap CIs and exact McNemar on CHAIR_s), plus, per reranking arm:

* candidates per image (pool size) and candidate generation success rate;
* candidates dropped as degenerate, with reasons;
* selected-score distribution and the margin over the runner-up;
* which prefix the selected candidate came from;
* **% of images where the top-ranked candidate differs from BLIP's first
  candidate** (the rank-0 baseline beam);
* **% of images where the relation term changed the selection** (the same pool
  rescored with `w_rel = 0`);
* % of captions that state the predicted relation, and their mean confidence;
* the locked weights and the sha256 of the weight file.

Paired deltas are produced against `baseline`, `objects_only`, `grounded`, and
`relation_reranked − object_reranked`.

---

## 9. Commands (GTX 1650, Windows)

Run from the repository root, after `CAPTION_EXPERIMENTS.md` steps 1–9 have
produced `results_caption/main`.

```
git pull
python -m pytest tests/ -q
python run_caption_experiment.py preflight
```

Build the tuning set (needs `objects.json`, already fetched):

```
python build_caption_eval_set.py --split val --limit 200 --output splits/caption_eval_val_200.json
```

Validation run — detections and candidates:

```
python run_caption_experiment.py detect --run-dir results_caption/val_tuning --eval-set splits/caption_eval_val_200.json --tuning
```
```
python run_caption_experiment.py candidates --run-dir results_caption/val_tuning
```

Lock the weights (CPU, seconds):

```
python run_caption_experiment.py tune-rerank --run-dir results_caption/val_tuning
```

Frozen test run — candidates, then selection with the locked weights:

```
python run_caption_experiment.py candidates --run-dir results_caption/main
```
```
python run_caption_experiment.py rerank --run-dir results_caption/main
```

Score all five arms, once:

```
python hallucination_eval.py --run-dir results_caption/main
```

Optional: a 5-image end-to-end check first (it also verifies the pool/arm
identity and the scorer against the reference forward):

```
python run_caption_experiment.py smoke --force
```

Knobs that matter: `--num-candidates` (beams per prefix, ≤ `num_beams` = 4),
`--batch-size` (candidates per scoring forward; lower it if VRAM is tight),
`--seed`, `--device`, `--blip-dtype` (leave at float32 — GTX 16xx cards NaN in
fp16).

**Memory.** `candidates` holds BLIP-base only, in float32 (~1 GB of weights).
Beam search runs on one image; scoring runs on one image embedding and at most
`--batch-size` short token sequences. Peak stays far below 4 GB. `detect` is
unchanged from the frozen run.

**Time.** `candidates` costs three beam searches plus one scoring forward per
image — roughly what the three prefix arms cost together in the frozen run, so
expect the same order: minutes on the GPU for 250 images, plus the same again
for the 200 validation images. `rerank` and `tune-rerank` are CPU-only and
take seconds.

---

## 10. Limitations

* **The relation model saw the validation images during model selection.** The
  relation checkpoint's epoch and seed were chosen on validation top-1, so its
  predictions on the validation split may be slightly optimistic relative to
  test, which could bias `w_rel` upward. Tuning on the training split would be
  worse (the model was fitted there). This is the standard choice and it is
  disclosed rather than hidden.
* **Three weights fitted on a few hundred images.** The grid is coarse on
  purpose and the objective prefers smaller weights on ties, but the selected
  values should be read as "a working operating point", not as estimates.
* **`V` is only as good as YOLO + CLIP verification.** An object the detector
  misses is counted as unsupported, so `w_hall` penalises some *correct*
  mentions. That trade-off is exactly what the validation search resolves, and
  it is why `w_hall` is a free parameter rather than a fixed rule.
* **The score measures object grounding, not the plausibility of the relation
  as English.** A candidate that names two detected objects scores well even
  if the phrasing is odd. Implausible triples are removed upstream by the
  repository's semantic filter, not by the reranker.
* **The pool can only rank what beam search produced.** If none of the ≤12
  candidates is well grounded, the reranker cannot fix the image.
* **Visual Genome annotations are not exhaustive**, so absolute CHAIR and POPE
  numbers remain upper bounds for every arm, and some "absent" POPE probes may
  in fact be present. This applies identically to all five arms.
* **The relation-conditioned candidates are in both reranking arms' pools.**
  The ablation isolates the scoring term, not the pool. A third arm whose pool
  excludes the relation prefix would separate the two effects; it is a cheap
  follow-up (the pool is already cached) and is deliberately not run here to
  keep the pre-registered comparison small.

---

## 11. The claim this experiment is allowed to make

Before the frozen run, the defensible statement is:

> Visual relation prediction provides useful relational evidence, but forcing a
> predicted relation into a caption generator's input makes the generator
> assert it unconditionally and introduces hallucinations. We therefore treat
> the prediction as soft evidence for reranking candidate captions instead.

It may be strengthened to "relation-aware reranking reduces hallucination"
**only** if the frozen test run shows `relation_reranked` beating
`object_reranked` on CHAIR with a CI that excludes 0, without a significant
loss of object recall. If `relation_reranked ≈ object_reranked`, the honest
conclusion is that the relation still adds nothing at caption level and that
the contribution of this work is the reranking mechanism and the negative
result about the relation — which is a result, and is reported as one.
