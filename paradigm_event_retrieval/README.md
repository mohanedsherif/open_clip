# Paradigm Event Retrieval

An isolated, event-level text-to-image retrieval adaptation project for the checked-out OpenCLIP source. It does not include identity recognition, OCR, uploading, APIs, or external integrations.

## First-iteration plan

1. Inspect the checkout and installed runtime.
2. Validate images, report duplicate candidates, and split strictly by event.
3. Produce deterministic manifests with one sampled caption per image; the templates are editable YAML.
4. Evaluate one pretrained checkpoint with normalized image/text embeddings and frozen event-held-out queries.
5. Only then benchmark candidate checkpoints and decide whether Stage A is warranted.

## Setup

Run from the repository root. The exact install command below is aligned with this checkout's `pyproject.toml`; it could not be executed in the initial environment because PyTorch was absent.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[training]"
python -m pip install -r paradigm_event_retrieval\requirements.txt
python -m open_clip_train.main --help
cd paradigm_event_retrieval
python -m scripts.inspect_environment
```

The environment inspector records Python, Git commit, PyTorch, OpenCLIP, CUDA, and GPU memory. Pin the resulting PyTorch build in your deployment lockfile because its CUDA variant is hardware-specific.

## Dataset preparation

Put private images under `data/raw/event_id/category/image.ext` and keep them out of Git. Edit `configs/caption_templates.yaml` to change wording without Python edits.

```powershell
python -m scripts.prepare_dataset --raw-dir data/raw --manifests-dir data/manifests --templates configs/caption_templates.yaml --report data/manifests/dataset_report.json
```

This accepts jpg/jpeg/png/webp/tif/tiff, decodes every file, reports (but never deletes) exact SHA-256 and perceptual-hash near duplicates, and writes `train.csv`, `validation.csv`, and `test.csv`. An event is assigned to exactly one split. Use only unseen test events for `data/evaluation/queries.json`.

## Baseline and evaluation

First inspect the locally available checkpoint tags—no model is selected or downloaded by this command:

```powershell
python -m scripts.list_models
```

After reviewing the checkpoint source, purpose, and hardware requirement, create a baseline with a real returned `model/tag` pair. `ViT-B-32/openai` is configured only as an initial pipeline-validation baseline; it has not been downloaded here.

```powershell
python -m scripts.evaluate_checkpoint --manifest data/manifests/test.csv --queries data/evaluation/queries.json --model ViT-B-32 --pretrained openai
```

The result includes Precision@1/@5/@10/@50, Recall@1/@5/@10/@50, MRR, and nDCG@10. For final reporting use Precision@1/@5/@10 and Recall@5/@10/@50.

## Fine-tuning, inference, export

The Stage A/B YAML files encode conservative locking and BF16 defaults but deliberately do not launch training during this first iteration. Before wiring a run, inspect the locally verified `open_clip_train.main --help` and use its `--lock-image`, `--lock-image-unlocked-groups`, `--lock-text`, `--lock-text-unlocked-layers`, `--grad-checkpointing`, and `--precision amp_bf16` flags. Stage B is permitted only after an actual Stage A validation improvement. Full fine-tuning is guarded by configuration.

`ParadigmEmbeddingModel` supports batched normalized embeddings, CPU/CUDA fallback, model preprocessing, and float32 output. Export is intentionally deferred until a chosen checkpoint has frozen test metrics.

## Tests

```powershell
python -m pytest tests -q
```

Model-loading tests skip until OpenCLIP is installed; dataset and retrieval-metric tests use generated local fixtures.
