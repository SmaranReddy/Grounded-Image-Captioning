# Caption experiment: does relation grounding reduce BLIP's object hallucination?

This runbook is for the GPU machine that holds the trained relation checkpoints
(`checkpoints_gpu/`). **Do not retrain anything.** Every command is one line
and works the same in PowerShell and bash; run them from the repository root.

Step 6 is a gate: **if the smoke test does not print `SMOKE TEST: PASS`, stop.**

---

## What is being compared

Same 250 images, same BLIP checkpoint (`Salesforce/blip-image-captioning-base`),
same preprocessing, same decoding (`num_beams=4, max_new_tokens=128,
early_stopping=True, do_sample=False`, float32, one image at a time). The only
thing that differs is the text BLIP continues from:

| arm | prefix given to BLIP | role |
|---|---|---|
| `baseline` | `a photo of` | reference |
| `grounded` | `a photo of a person riding a bicycle` if the selected relation's confidence ≥ 0.5, otherwise `a photo of` | **the system under test** |
| `objects_only` | `a photo of a person and a bicycle` (same two objects, same trigger) | control: separates "naming two detected objects" from "stating their relation" |

No caption gating, repair, relation correction or template fallback runs in any
arm. Images where the grounded arm does not inject a relation are scored with the
baseline caption, so the arms can only differ where a relation was injected.

### How the relation is produced (pre-declared)

1. YOLO11m detections, confidence ≥ 0.5, top 10.
2. CLIP verification (`utils/detection_verifier.py`, unchanged).
3. Detections whose label is in the relation vocabulary and whose box is ≥ 10 px
   on both sides (the training population's filter).
4. Every ordered pair goes through the selected **geometry + CLIP + union** MLP,
   with features built exactly as for training/evaluation: 19-D geometry on the
   real image size, CLIP (`CLIPExtractor.encode_crops`) on clamped crops, union =
   bounding box of the two raw boxes, clamped. Each crop is encoded once per image.
5. Prediction = argmax over the 19 real predicates (as `eval_gt_relations.py`);
   confidence = softmax probability at T=1 (uncalibrated).
6. Pairs whose top-1 triple fails the repository's plausibility rules
   (`predict._is_extreme_nonsense`) are discarded; the predicate is never swapped.
7. The single most confident surviving pair is the image's relation. It is
   injected iff confidence ≥ **0.5** (fixed before any caption was generated).

### Which checkpoint (no test-set selection)

`select-checkpoint` reads only `checkpoints_gpu/geometry_clip_union_seed{42,43,44}/training_meta.json`
and picks the highest **validation** top-1 (`val_acc`) — the same metric the
trainer used to choose the saved epoch — breaking ties by validation macro-F1,
then by lower seed. It never opens `results_gpu/`. Seed 44 is not chosen
because of its test score; whichever seed is chosen is chosen by validation.

---

## 1. Update the code

```
git pull
```

## 2. Check the environment

```
python check_environment.py
```

```
python run_caption_experiment.py preflight
```

On this first run `preflight` will report `caption eval set` and
`relation checkpoint selection` as `MISS` (steps 4–5 create them) and
`VG objects.json` as `MISS` until step 4. Everything else must be `OK`, and
`CUDA GPU` should name the GTX 1650. If `ultralytics` or `transformers` is
missing: `pip install -r requirements.txt`.

## 3. Run the test suite

```
python -m pytest tests/ -q
```

Expected: **327 passed, 1 skipped** (the skip is the legacy `test_pope.py`, which
needs `nltk`; if `nltk` is installed it runs and you get 328 passed). No test
needs a GPU, the VG images or the trained checkpoints.

## 4. Verify the trained relation checkpoint

Human object annotations (≈55 MB download, once):

```
python download_vg.py --objects
```

Select the checkpoint by validation top-1:

```
python run_caption_experiment.py select-checkpoint
```

Expected: a table of the three seeds with validation top-1 and macro-F1, one
marked `<- selected`, then `loaded checkpoints_gpu/geometry_clip_union_seedNN:
input_dim=2451 blocks=[['subj_label', 64], ['obj_label', 64], ['geometry', 19],
['subj_clip', 768], ['obj_clip', 768], ['union_clip', 768]]`. It writes
`results_caption/relation_checkpoint_selection.json`. Any other `input_dim` is a
refusal, not a warning.

## 5. Build the caption evaluation set

```
python build_caption_eval_set.py
```

Expected (the image ids are deterministic; the counts below assume all 27,145
experiment images were downloaded for the relation experiment):

```
  eligible test images : 3,939
  requested images     : 250
  usable images        : 250
  missing images       : 0
  corrupt images       : 0
  final count          : 250
STATUS: OK
```

Writes `splits/caption_eval_test_250.json`. If it prints `STOP` (fewer than 100
usable images) run `python prepare_visual_genome.py --download`, then rebuild
with `python build_caption_eval_set.py --force`. Do not continue on `STOP`.

Now `python run_caption_experiment.py preflight` must print `PREFLIGHT: READY`.

## 6. Smoke test on 5 images — GATE

```
python run_caption_experiment.py smoke
```

It runs YOLO, CLIP verification, the relation model, a synthetic relation through
the real BLIP, all three arms and the hallucination evaluator on the first 5
images, and writes `results_caption/smoke/smoke_report.json`. Expected last line:

```
SMOKE TEST: PASS  (0 failed, N warnings)
```

`WARN` on "relations above threshold" / "grounded path uses relation prefix" only
means none of these 5 images crossed 0.5; the synthetic check still proves
injection. If you want to see a real injection before the full run:
`python run_caption_experiment.py smoke --n 20 --force`.

**Any `FAIL`: stop and send `results_caption/smoke/smoke_report.json`.**

## 7. Detect relations and generate baseline captions

```
python run_caption_experiment.py detect
```

```
python run_caption_experiment.py generate --arm baseline
```

## 8. Generate grounded and control captions

```
python run_caption_experiment.py generate --arm grounded
```

```
python run_caption_experiment.py generate --arm objects_only
```

Every stage is resumable: if one is interrupted, run the same command again and
it continues. Stages refuse to mix settings (different threshold, eval set,
checkpoint or dtype) inside one run directory.

## 9. Run the hallucination evaluation

```
python hallucination_eval.py --run-dir results_caption/main
```

It refuses to score if any arm's image set differs, if a prefix does not match
its rule, if a relation that should have been injected did not reach BLIP, or if
a placeholder/template caption appears. It prints and writes
`results_caption/main/caption_results.md`.

## 10. Generate qualitative examples

```
python run_caption_experiment.py examples --n 16
```

Writes `results_caption/main/examples/examples.md` plus one annotated image per
example (boxes, relation, all three captions, hallucinated objects). Examples
are drawn with a fixed seed from five strata — relation removed a hallucination,
relation added one, relation changed the caption without changing the count,
relation below threshold, no relation — so failures appear with successes.
**They illustrate; they are not evidence.**

## 11. Collect the results

```
python run_caption_experiment.py report
```

```
python -c "import shutil; shutil.make_archive('caption_results_bundle', 'zip', '.', 'results_caption')"
```

Send back `caption_results_bundle.zip` and `splits/caption_eval_test_250.json`.

---

## Rough time on the GTX 1650

Measured on a CPU laptop (no GPU): detect ≈1.2–3.2 s/image, BLIP ≈1.5–1.9 s per
caption. The GPU should be faster; not measured. Worst case per stage for 250
images at CPU speed: detect ≈5–13 min; baseline ≈8 min; grounded and objects_only
up to ≈16 min each (a second caption is generated for images whose relation is
below threshold, for the exploratory sweep). Peak GPU memory stays well under
4 GB: `detect` holds YOLO11m + two CLIP ViT-B/32 copies + the MLP; `generate`
holds BLIP-base only. BLIP runs in float32 because GTX 16xx cards are known to
produce NaNs in float16.

## Files produced

```
results_caption/
  relation_checkpoint_selection.json    seed candidates (validation only), chosen seed,
                                        weights sha256, verified feature blocks
  smoke/smoke_report.json               PASS/FAIL per check
  main/
    run_config.json                     git commit, versions, eval-set hash, image ids,
                                        threshold, checkpoint hash, relation policy
    relations.jsonl                     per image: raw + verified detections, every pair's
                                        top-3 predicates, selected relation, decision
    captions_baseline.jsonl             per image: prefix, BLIP input ids, caption
    captions_grounded.jsonl             ... + relation, relation_used, fallback_reason,
                                        relation_caption (any confidence)
    captions_objects_only.jsonl
    captions_<arm>.meta.json            decoding parameters, dtype, BLIP device
    per_image.jsonl                     per image, per arm: caption, mentioned and
                                        hallucinated objects
    caption_results.json / .md          all metrics below
    examples/examples.md + *.jpg
splits/caption_eval_test_250.json       the 250 image ids + human GT objects
```

Only the small JSON/MD summaries are git-trackable; captions, relations and
example images are ignored by `.gitignore`.

---

## Metrics

### Hallucination (the question being asked)

Ground truth: Visual Genome **human** object annotations (`objects.json` names,
plus relation entities), mapped to COCO-80. YOLO is never used as ground truth.
Captions and ground-truth names go through the same mapper
(`utils/coco_mentions.py`: explicit synonym table, plurals, "hot dog" is not a
dog, "bus stop" is not a bus).

| metric | definition | direction |
|---|---|---|
| **CHAIR_i** (primary) | hallucinated / mentioned COCO objects, summed over all captions | lower better |
| CHAIR_s | share of captions with ≥ 1 hallucinated object | lower better |
| hallucinated objects / caption | mean count | lower better |
| hallucinations introduced by injected prefix vs in BLIP continuation | where the hallucinated objects came from | diagnostic |
| POPE (random / popular / adversarial) | balanced yes/no probes built from the ground truth only — k ≤ 3 present objects and exactly as many absent ones per image, identical for all arms; "yes" = caption mentions the object. Accuracy, precision, recall, F1, yes-ratio, and false-positive rate on absent objects | FP rate lower better; F1 higher better |

Always read CHAIR next to "mentioned objects / caption", "object recall" and
"caption length": a caption that says less hallucinates less. POPE here uses a
probe set that does not depend on the caption (the legacy `utils/pope.py` built
its probes from the caption and made every negative a true negative; it is not
used).

### Caption quality (reported separately)

| metric | definition |
|---|---|
| object recall | correctly mentioned GT objects / GT objects |
| CLIPScore | 2.5 · max(cos(CLIP-image, CLIP-text("A photo depicts " + caption)), 0), ViT-B/32 (Hessel et al., 2021) |
| caption length | words |

These images have no human reference captions, so BLEU / METEOR / CIDEr / SPICE
are **not** reported. CLIPScore is reference-free image–text alignment.

### Relation usage

Share of images with a verified detection, with ≥ 2 eligible detections, with a
selected relation, and with the relation **injected**; fallback share and reasons
(`fewer_than_2_eligible_detections`, `all_pairs_rejected_by_semantic_filter`,
`below_confidence_threshold`); mean confidence of injected relations; predicate
counts.

---

## Reading the result (fixed before the run)

Primary comparison: **grounded − baseline, CHAIR_i, all 250 images**, 95% paired
bootstrap CI over images (10,000 resamples). Everything else is secondary and
descriptive. The same differences are also reported on the images where a
relation was injected (where all of the effect must be, since the others share
the baseline caption).

| outcome | what the numbers must show |
|---|---|
| **A. Grounding reduces hallucination** | CHAIR_i CI entirely below 0, **and** grounded − objects_only CHAIR_i not above 0 (otherwise the gain is from naming detected objects, not from the relation), **and** object recall not significantly lower (otherwise it is saying less, not hallucinating less) |
| **B. No reduction** | CHAIR_i CI includes 0 |
| **B'. Grounding increases hallucination** | CHAIR_i CI entirely above 0 — check "introduced by injected prefix" to see whether YOLO/relation errors cause it |
| **C. Mixed** | hallucination and a quality metric (object recall, CLIPScore, POPE F1) move in opposite, significant directions |
| **D. Rarely triggered** | relation injected in < 20% of images: the all-images CI will be dominated by identical captions; judge from the injected subset and state that the pipeline rarely changes captions |
| **E. Useful relations, no caption effect** | relation usage is substantial but no caption metric moves |

The threshold sweep in the report re-uses the same generated captions at
thresholds 0.0–0.9 and is **exploratory**: it may suggest a follow-up, it does
not replace the pre-declared 0.5.

Known limitations that apply to all arms: Visual Genome annotations are not
exhaustive, so absolute hallucination rates are upper bounds and some "absent"
POPE probes may be present; YOLO may name a real object with a COCO label the
annotators did not use (e.g. a laptop bag as `handbag`), which is counted
against the arm that says it — this is why the prefix-vs-continuation split is
reported; relation confidences are uncalibrated.

No caption-level conclusion exists until this has been run.

---

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `select-checkpoint`: missing training_meta.json / relation_mlp.pt | `checkpoints_gpu/` incomplete or elsewhere | pass `--checkpoint-root <dir>` |
| `checkpoint does not have the geometry+CLIP+union feature signature` | wrong variant directory | point at `geometry_clip_union_seedNN` |
| `build_caption_eval_set.py`: `objects.json not found` | step 4 skipped | `python download_vg.py --objects` |
| `STOP: only N usable images` | VG test images missing | `python prepare_visual_genome.py --download`, then rebuild with `--force` |
| `was not built from the frozen E0 split` | split manifest modified | restore `splits/e0_image_split.json` from git |
| smoke `FAIL: synthetic relation reaches BLIP` | BLIP not conditioning on the prefix (transformers version) | send `smoke_report.json` and `pip show transformers` |
| `refusing to mix them in one arm` / `records a different ...` | re-running a stage with other settings | use a new `--run-dir` |
| `run is not scoreable` | a prefix/relation/caption inconsistency | send the printed problem list; do not report metrics |
| NaN / empty captions with `--blip-dtype float16` | fp16 on GTX 16xx | use the default float32 |
| CUDA out of memory in `detect` | other processes on the GPU | close them; or `--device cpu` (slow but identical) |

## Legacy scripts (not part of this experiment)

`grounded_caption_pipeline.py`, `evaluate.py` and the default mode of
`hallucination_eval.py` are the earlier demo path: they use temperature-2
softmax plus hand-set priors and a predicate override, YOLO-based caption gating
and templates, YOLO detections as default ground truth, and the caption-dependent
POPE. Do not report numbers from them. `grounded_caption_pipeline.py` now loads
visual checkpoints correctly (`REL_CKPT_DIR=checkpoints_gpu/geometry_clip_union_seedNN`)
and no longer emits placeholder text as captions.
