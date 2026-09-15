# Running the visual relation experiment on a GPU machine

Follow these steps in order. **Step 5 is a gate: if the baseline does not
reproduce, stop and report it before training anything.**

Everything below has been verified to run on CPU except the steps marked
**needs GPU** (which run on CPU too, just slowly) and **needs VG assets**.

---

## What this experiment is for

An audit of this repo established:

* The reported result — Top-1 **63.89%** on a frozen image-disjoint test split —
  is real and reproduces exactly.
* But a lookup table with **no geometry at all**, `argmax P(predicate | subject
  class, object class)`, scores **57.4%** on the same test set. The 19 geometry
  features are worth ~6.5 points over a Python dict.
* Training accuracy plateaus at 65.5% against 62.4% validation. Widening the
  model 4x or training 60 epochs changes nothing beyond seed noise. The model
  is not capacity-limited; its inputs do not determine the label.
* Seed variance is ±0.0033, so **any difference below ~0.7 points is noise**.
  This is why every configuration is run with three seeds.
* The reported run had `use_visual: false`. **Actual pixels have never been
  measured on this split.** That is the one untested lever, and it is what
  these four variants measure:

| variant | features |
|---|---|
| `geometry` | label embeddings + 19-D geometry (control) |
| `geometry_clip` | + subject and object CLIP crops |
| `geometry_clip_union` | + CLIP of the union region spanning the pair |
| `clip_only` | CLIP crops, **no** geometry (control) |

All four run on the **same** frozen split, the same predicate scheme, the same
loss, and — importantly — the **same sample population**. Turning visual
features on drops pairs whose crops are missing, so the geometry control runs
with `--visual-filter-only`: it loads the same CLIP cache and keeps the same
samples but is built with `clip_dim=0`. Without that the comparison would vary
the features *and* the test set at once and neither number would mean anything.

---

## 1. Clone

```bash
git clone https://github.com/SmaranReddy/Image-Detection-using-GNNs.git
cd Image-Detection-using-GNNs
```

## 2. Create an environment

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1
```

## 3. Install dependencies

Install a **CUDA** build of PyTorch first — the default PyPI wheel is CPU-only:

```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Then confirm the machine is usable:

```bash
python check_environment.py
```

It prints `READY` / `MISSING` / `OPTIONAL` per item. At this point `VG images`
and `CLIP cache` will read `MISSING` — that is expected; steps 6 and 7 fix
them. Everything else should read `READY`, and the `Accelerator` section should
name your GPU. If it says `no CUDA device`, the CUDA torch install did not take.

`nltk`, `mediapipe`, `ultralytics` and `matplotlib` are `OPTIONAL`: none of the
four variants use them.

## 4. Run the test suite

```bash
python -m pytest tests/ -q
```

Expected: **240 passed, 1 skipped** (the skip is `test_pope.py`, which needs
`nltk` — not used by this experiment).

## 5. Verify the baseline — THIS IS A GATE

You need `data/visual_genome/relationships.json` and `image_data.json` for this
(~730 MB, no images yet):

```bash
python download_vg.py
```

Then reproduce both frozen results:

```bash
python eval_gt_relations.py \
  --checkpoint-dir checkpoints_e2_geo \
  --split-manifest splits/e0_image_split.json \
  --results-dir results_check --output-name e2_repro --force

python eval_gt_relations.py \
  --checkpoint-dir checkpoints \
  --split-manifest splits/e0_image_split.json \
  --results-dir results_check --output-name e0_repro --force
```

**Expected, exactly:**

| | E2 (`checkpoints_e2_geo`) | E0 (`checkpoints`) |
|---|---|---|
| Top-1 | `0.638897` | `0.553437` |
| Recall@3 | `0.923047` | `0.848049` |
| Recall@5 | `0.965386` | `0.889606` |
| Macro-F1 | `0.381597` | `0.301131` |
| Weighted-F1 | `0.614064` | `0.511180` |

and in both runs:

```
kept samples      : 68,900
qualifying images : 27,145
n_samples_train   : 48,191
n_samples_val     : 10,482
n_samples_test    : 10,227
overlap check     : PASS
```

> **If any of these numbers differ, STOP. Do not train.** Report the mismatch
> — a changed baseline means the data pipeline or the label space moved, and
> every number produced afterwards would be measuring something else.

## 6. Prepare the Visual Genome images — **needs VG assets**

The experiment needs **27,145 images**, not all 108,077. After the repo's
filters (COCO-80 labels, 19-predicate allowlist, 10px minimum box) only those
carry a usable relation, and the frozen split names exactly them. That is
roughly **3 GB** rather than ~25 GB.

```bash
python prepare_visual_genome.py --report                    # what is needed
python prepare_visual_genome.py --download --workers 16     # fetch (resumable)
python prepare_visual_genome.py --validate --delete-corrupt # decode every file
python prepare_visual_genome.py --download                  # refetch any deleted
```

Safe to interrupt and re-run: existing files are never refetched, and partial
downloads land in `.part` files that are discarded rather than mistaken for
finished images. Rough time: 20–40 minutes on a decent connection.

Stop when it prints `STATUS: READY`.

## 7. Build the CLIP cache — **needs GPU**

```bash
python build_clip_cache.py --include-union --batch-size 128
```

This encodes ~78,000 object crops and ~48,000 union regions (about 126,000
CLIP forward passes) into
`data/visual_genome/clip_cache_proper.pt`. Resumable: progress is checkpointed
to shards, and re-running after an interrupt continues where it stopped.
Rough time: 10–30 minutes on a laptop GPU, several hours on CPU.

`--include-union` is required — without it the `geometry_clip_union` variant
has nothing to read.

## 8. Verify cache coverage

```bash
python build_clip_cache.py --include-union --report
python check_environment.py
```

Both object and union coverage must read **100.00%** and the status must be
`READY`. The experiment runner refuses to start below 100%, because a run that
quietly trains on a fraction of the intended samples produces a number that
looks fine and means nothing.

## 9–12. Run the experiment — **needs GPU**

Everything at once (4 variants x 3 seeds = 12 runs):

```bash
python run_visual_experiment.py --all
```

Or one at a time, if you want to watch them:

```bash
python run_visual_experiment.py --variant geometry            --seed 42
python run_visual_experiment.py --variant geometry_clip       --seed 42
python run_visual_experiment.py --variant geometry_clip_union --seed 42
python run_visual_experiment.py --variant clip_only           --seed 42
# then repeat each with --seed 43 and --seed 44
```

Add `--dry-run` first if you want to see the exact commands without running
them. Each run performs a preflight (split integrity, image coverage, cache
coverage, NaN/inf/zero-row checks on the cache, device) and **aborts before
training** if anything is short.

Each run writes:

```
results_gpu/<variant>_seed<seed>/
    run.json     provenance: git commit, split sha256, seed, feature config,
                 model config, dataset coverage, training config, metrics
    <name>.json  the evaluation result
    train.log
    eval.log
checkpoints_gpu/<variant>_seed<seed>/    weights + vocabularies
```

Rough time: a few minutes per run on GPU once the cache exists.

## 13. Collect the results

```bash
python run_visual_experiment.py --collect
```

Prints a per-run table and the mean ± std Top-1 per variant, writes
`results_gpu/summary.json`, and **warns loudly if the arms were scored on
different test-set sizes** — which would mean the comparison is confounded and
the numbers should not be compared.

---

## How to read the outcome

Compare `geometry` against `geometry_clip`. Given the ±0.0033 seed noise
measured in the audit:

* **> ~1 point** — visual features carry real signal this pipeline was missing.
  That is the interesting result and worth pursuing.
* **within ~0.7 points** — a null result, and an informative one: it would mean
  object-pair priors plus geometry are close to everything this formulation can
  extract, and the remaining error is in the label space rather than the
  features. (The audit already found 49.4% of errors are confusions *inside* a
  semantic cluster, and that crediting any human-annotated predicate for the
  pair lifts the same checkpoint from 63.9% to 70.4%.)

Report the number either way. A negative result here is a finding, not a
failure — and it is the difference between "we did not try" and "we measured
it".

`clip_only` says how much of any gain is CLIP rather than geometry;
`geometry_clip_union` says whether the pair's shared region adds anything the
two isolated crops do not.

---

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `check_environment.py` says `no CUDA device` | CPU-only torch wheel | reinstall torch from the cu124 index (step 3) |
| baseline numbers differ in step 5 | pipeline or label space changed | **stop and report** — do not train |
| preflight: `N required images missing` | step 6 incomplete | `python prepare_visual_genome.py --download` |
| preflight: `union CLIP coverage < 100%` | cache built without unions | `python build_clip_cache.py --include-union` |
| preflight: `cache contains N all-zero rows` | degenerate crops got stored | delete the cache and its `.shards/` dir, rebuild |
| `--collect` warns about different test-set sizes | an arm ran without `--require-visual` | rerun that arm through `run_visual_experiment.py`, not by hand |
| cache build is very slow | running on CPU | check step 3; `--device cuda` |

## What is NOT part of this experiment

* **Caption / hallucination evaluation** (CHAIR, POPE) needs `nltk` plus its
  WordNet corpus (`python -c "import nltk; nltk.download('wordnet');
  nltk.download('omw-1.4')"`) and a separate caption ground-truth image set
  that is not currently available. `splits/e0_caption_gt_test.json` names 250
  images; none are on disk.
* **Pose features** (`--use-pose`) need `mediapipe`, which has no wheel for
  several recent Python versions. No variant here uses them.
* **The transformer model** (`--model transformer`) is not part of the
  ablation; all four variants use the MLP so the comparison is about features,
  not architecture.
