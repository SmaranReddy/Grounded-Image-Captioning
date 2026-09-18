# Grounded Image Captioning

Object detection -> CLIP verification -> visual relation prediction ->
caption grounding, with an evaluation that measures object hallucination
against human Visual Genome annotations.

## Pipeline

```
image -> YOLO11m -> CLIP verification -> relation prediction (geometry + CLIP + union)
      -> semantic plausibility filter -> caption conditioning -> BLIP-base
      -> CHAIR / POPE / CLIPScore evaluation
```

## Documents

| file | what it covers |
|---|---|
| `GPU_EXPERIMENTS.md` | the relation experiment: four feature variants, three seeds, frozen image-disjoint split |
| `CAPTION_EXPERIMENTS.md` | caption experiment 1 - does injecting a predicted relation into BLIP's prefix reduce hallucination? **Closed; the answer is no.** |
| `CAPTION_RERANKING.md` | caption experiment 2 - grounding-aware reranking of multiple BLIP candidates |

## Results so far

**Relation prediction** (frozen image-disjoint test split, 3 seeds, mean top-1):

| features | mean top-1 |
|---|---|
| CLIP only | 62.17% |
| geometry | 63.07% |
| geometry + CLIP | 63.96% |
| **geometry + CLIP + union** | **64.24%** |

Seed variance is +-0.33 pt, so differences below ~0.7 pt are noise.

**Captioning** (250 frozen test images, human VG object annotations):
injecting the predicted relation into BLIP's prefix raises object recall
(+14.3 pt) and POPE F1 (+12.6 pt) but also raises hallucination
(CHAIR_i +4.19 pt, CHAIR_s +10.40 pt), and an objects-only control matches it
on every metric - the predicate itself contributes nothing. Forcing a
64%-accurate prediction into the generator's input makes the generator assert
it unconditionally. `CAPTION_RERANKING.md` is the response: use the prediction
as soft evidence to choose between candidate captions instead.

## Reproducing

Everything runs on a 4 GB GPU (GTX 1650) or on CPU. Start with
`python check_environment.py`, then follow the runbook for the experiment you
want. `python -m pytest tests/ -q` needs neither a GPU nor the dataset.
