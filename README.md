# Graph-Enhanced Multimodal Sign Language Translation

Reproducible implementation of the graph-enhanced multimodal Transformer described in the accompanying IJIES manuscript.

**Important:** this repository contains **no synthetic benchmark data and no fallback/mock feature generation**. If an annotation, video, RGB feature, pose file, or face-landmark file is missing, preprocessing/training fails with an explicit error.

The repository implements:

- official train/dev/test annotation consumption for PHOENIX-2014T and CSL-Daily;
- raw-video frame sampling and RGB feature extraction with a configurable ImageNet-pretrained ResNet-50 (2048-D pooled features by default);
- OpenPose BODY_25 pose parsing and 68-point facial landmark parsing;
- shoulder-centered / shoulder-distance pose normalization;
- the three-stream RGB/pose/face architecture;
- learnable graph convolution for the 25-joint pose stream;
- confidence-aware dynamic cross-modal fusion;
- temporal saliency gating;
- 4-layer, 8-head Transformer encoder and 4-layer translation decoder;
- joint CTC recognition + label-smoothed translation training;
- 4000-step warmup followed by inverse-square-root learning-rate decay;
- AMP, gradient accumulation, gradient clipping, three-seed evaluation;
- canonical SacreBLEU BLEU-4 and `rouge-score` ROUGE-L evaluation;
- WER, ablation, robustness, attention, convergence, and throughput analysis scripts.

## Reproducibility boundary

The manuscript does not specify every low-level dependency needed to uniquely reconstruct raw-video feature extraction (for example, the exact RGB CNN checkpoint and OpenPose release). Therefore these choices are explicit configuration values rather than hidden assumptions. **The reported numerical results must only be claimed after running this repository on the official datasets with the same extractor versions/checkpoints used for the paper.** The code never silently substitutes invented data.

## Repository layout

```text
configs/
  phoenix2014t.yaml
  csldaily.yaml
scripts/
  prepare_annotations.py
  extract_features.py
  train.py
  evaluate.py
  run_ablation.py
  run_robustness.py
  benchmark.py
  export_attention.py
src/graph_multimodal_slt/
  config.py
  data/
    annotations.py
    datasets.py
    features.py
    vocab.py
  evaluation/
    metrics.py
  models/
    architecture.py
  training/
    engine.py
    scheduler.py
  utils/
    seed.py
    io.py
    logging.py
    checkpoint.py
requirements.txt
pyproject.toml
LICENSE
CITATION.cff
```

## 1. Environment

Python 3.10+ is recommended. Install PyTorch separately if your CUDA version requires a specific wheel, then:

```bash
pip install -r requirements.txt
pip install -e .
```

## 2. Dataset preparation

You must obtain the datasets from their official sources and comply with their licenses/terms. This repository does not redistribute PHOENIX-2014T or CSL-Daily.

Example directory structure:

```text
/data/phoenix2014t/
  videos/
  annotations/
    PHOENIX-2014-T.train.corpus.csv
    PHOENIX-2014-T.dev.corpus.csv
    PHOENIX-2014-T.test.corpus.csv
  features/

/data/csldaily/
  videos/
  annotations/
    csl_daily.train.json
    csl_daily.dev.json
    csl_daily.test.json
  features/
```

Do not create placeholder annotations. The preparation scripts validate sample counts and IDs.

## 3. Feature extraction

The model consumes three synchronized streams:

- RGB: `(T, 2048)` ResNet-50 global-average-pooled features;
- pose: `(T, 25, 2)` OpenPose BODY_25 coordinates;
- face: `(T, 68, 2)` facial landmark coordinates.

Run preprocessing against real videos and real OpenPose outputs. A missing file is an error.

```bash
python scripts/extract_features.py \
  --dataset phoenix2014t \
  --config configs/phoenix2014t.yaml \
  --data-root /data/phoenix2014t \
  --openpose-json-root /data/phoenix2014t/openpose_json \
  --output-root /data/phoenix2014t/features
```

The same command can be used for CSL-Daily with its config and roots.

## 4. Training

The manuscript uses 100 epochs, batch size 16, gradient accumulation 2 (effective batch 32), seeds 42/123/456, AdamW `(0.9, 0.998)`, weight decay `1e-4`, maximum learning rate `3e-4`, 4000 warmup steps, inverse-square-root decay, AMP, dropout 0.1, label smoothing 0.1, and gradient clipping 1.0.

```bash
python scripts/train.py \
  --dataset phoenix2014t \
  --config configs/phoenix2014t.yaml \
  --data-root /data/phoenix2014t \
  --output-dir runs/phoenix2014t
```

For CSL-Daily:

```bash
python scripts/train.py \
  --dataset csldaily \
  --config configs/csldaily.yaml \
  --data-root /data/csldaily \
  --output-dir runs/csldaily
```

## 5. Evaluation

```bash
python scripts/evaluate.py \
  --dataset phoenix2014t \
  --config configs/phoenix2014t.yaml \
  --data-root /data/phoenix2014t \
  --checkpoint runs/phoenix2014t/seed_42/best.pt \
  --split test
```

The evaluator reports **SacreBLEU BLEU-4, ROUGE-L, and WER**. The exact SacreBLEU signature and package versions are saved with every run.

## 6. No hidden fallback behavior

The following are deliberately prohibited:

- generating annotation rows when annotations are missing;
- generating random RGB features;
- generating random pose coordinates;
- generating random facial landmarks;
- using test data to build vocabularies;
- silently changing official splits.

Every one of these conditions raises an exception.

## 7. Important scientific note

The paper reports 25.40 BLEU-4 / 34.80 ROUGE-L on PHOENIX-2014T and 22.80 BLEU-4 / 31.20 ROUGE-L on CSL-Daily. These numbers are **reported manuscript values, not values fabricated by this repository**. This repository is an executable reproduction pipeline. The numbers must be re-established by actually training/evaluating on the official datasets. The repository cannot honestly guarantee those values without the exact raw datasets, extractor versions, and original feature checkpoints used in the study.
