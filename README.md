Video-based Terrain and Scene Classification for Off-road Autonomous Driving

This repository contains code for multi-view spatiotemporal multi-task perception (terrain + scene). It includes training, evaluation, and experimental/demo scripts.

Quick Start

1) Install dependencies (recommended inside a virtual environment):

```bash
pip install -r requirements.txt  # or install: torch, torchvision, pyyaml, scikit-learn, pillow
```

2) Expected dataset layout (example):

```
dataset/rellis/
	images/
		00000/
			pylon_camera_node/
				*.jpg
	labels/
		00000/
			pylon_camera_node_label_id/
				*.png
```

3) Training

Run training with the default config:

```bash
python train.py --config configs/train.yaml
```

Override parameters from the command line, e.g.:

```bash
python train.py --config configs/train.yaml --epochs 10 --batch-size 8
```

4) Evaluation (generate result tables)

```bash
python eval_all_tables.py --config configs/eval_all_tables.yaml
```

5) Experimental demo

Experimental/demo script has been moved to `experiments/run_pipeline.py`:

```bash
python experiments/run_pipeline.py
```

Repository structure

```
checkpoint_offroad_net.pth
checkpoints/
configs/
	train.yaml
	eval_all_tables.yaml
data/
	rugd_lite/
		front/  left/  right/
video_stream/
	front/  left/  right/
dataset/
	rellis/
		images/
		labels/
		ontology/
datasets/
	__init__.py
	rellis_dataset.py
models/
	__init__.py
	offroad_net.py
	tsm_module.py
utils/
	__init__.py
	metrics.py
train.py
eval_all_tables.py
experiments/
	run_pipeline.py
README.md
```

Files overview

- `train.py`: main training script. Supports `--config` to load YAML configs.
- `eval_all_tables.py`: evaluation script that generates various result tables. Supports config.
- `experiments/run_pipeline.py`: experimental/demo script for quick local tests and latency measurements.
- `datasets/rellis_dataset.py`: dataset parser and label handling.
- `models/offroad_net.py`: main model `DecoupledSpatiotemporalNet` implementation.
- `configs/`: YAML configuration files for training and evaluation.

Notes

- If no GPU is available, scripts will fall back to CPU automatically (slower).
- YAML files in `configs/` are loadable via `--config`; CLI args override config defaults.

Common commands

```bash
# Train
python train.py --config configs/train.yaml

# Evaluate
python eval_all_tables.py --config configs/eval_all_tables.yaml

# Run demo experiment
python experiments/run_pipeline.py
```
