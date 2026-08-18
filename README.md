# FROM: Occluded Face Recognition by Masking Corrupted Features

PyTorch implementation of *End2End Occluded Face Recognition by Masking Corrupted
Features* (TPAMI 2021), refactored for training on a **custom dataset** with
**distributed multi-GPU training + PartialFC**.

The network learns to detect which spatial regions of a face's feature map are
corrupted by occlusion (masks, sunglasses, hands, etc.) and masks them out before
computing the 512-d embedding, so recognition relies only on the visible face.

## What's in this repo

```
FROM/
├── Dockerfile / docker-compose.yml     # reproducible CUDA 12.2 environment
├── train_distributed.py                # DDP + PartialFC training entrypoint
├── start_distributed.sh                # launches the 3 training stages
├── INTEGRATION_GUIDE.md                # detailed setup + occlusion explanation
├── experiments/
│   ├── custom-112x112-Clean.yaml       # Stage 1 config
│   ├── custom-112x112-Occ.yaml         # Stage 2 config
│   ├── custom-112x112-Mask.yaml        # Stage 3 config
│   └── smoke-test.yaml                 # fast validation config (--max_steps)
├── lib/
│   ├── models/   fpn_112.py · iresnet_fpn.py · partial_fc_v2.py · margin_losses.py · lr_scheduler.py
│   ├── datasets/ dataset_rec.py        # RecordIO + on-the-fly occlusion
│   └── core/     config.py · utils.py
└── data/datasets/occluder/             # occluder patches (download separately)
```

**Backbones** (`TRAIN.MODEL`): `iResNet50_FPN` (InsightFace iResNet50 + FROM mask
branch) or the original `LResNet50E_IR_FPN`. Both return `(fc_mask, mask, vec, fc)`
and support the full 3-stage Clean→Occ→Mask method.

> **Note:** there is **no in-training evaluation** — this build saves a checkpoint
> every epoch and you evaluate them with your own pipeline (see *Outputs* below).

## Requirements

Everything is pinned in the `Dockerfile` (PyTorch 2.3.1, mxnet 1.9.1, CUDA 12.2).
Build the image:

```bash
docker build -t from-pfc .
```

Training data must be **MXNet RecordIO** (`train.rec` + `train.idx`, InsightFace
format). Occluder patches are downloaded separately — see below.

## Data preparation

1. Prepare your identity-labeled faces as a RecordIO dataset (`train.rec`,
   `train.idx`) aligned to 112×112.
2. Download the occluder patches from the
   [Google Drive link](https://drive.google.com/drive/folders/12r0QEQFb8MOxh1ZtX679Pnx4g8hknLOg?usp=sharing)
   and place them at `data/datasets/occluder/` with the `occluder.txt` manifest.
   *(Without these, Stages 2 & 3 silently train without occlusion.)*
3. Edit the three `experiments/custom-112x112-*.yaml` configs — see *Config* below.

## Training

Occlusion is generated **on-the-fly** by the dataloader — no separate occluded
dataset is needed. Training runs in three stages, each initialized from a
checkpoint of the previous stage:

| Stage | Mode | Occlusion | Mask branch | Loads from |
|-------|------|-----------|-------------|------------|
| 1 | Clean | off | off | scratch (or a pretrained backbone) |
| 2 | Occ | on-the-fly | off | Stage 1 `backbone_epoch_*.pth.tar` |
| 3 | Mask | on-the-fly | on | Stage 2 `backbone_epoch_*.pth.tar` |

```bash
docker run --gpus all --shm-size=32g \
    -v /path/to/rec/data:/data/dataset \
    -v /path/to/occluders:/workspace/FROM/data/datasets/occluder \
    -v /home/shared_scripts:/home/shared_scripts \   # only if USE_AZURE=True
    -it from-pfc

# inside the container:
bash start_distributed.sh 1   # Stage 1: clean baseline (backbone + PartialFC)
# pick the best epoch from Stage 1 and set it as NETWORK.PRETRAINED in custom-112x112-Occ.yaml
bash start_distributed.sh 2   # Stage 2: occlusion training
# pick the best epoch from Stage 2 and set it as NETWORK.PRETRAINED in custom-112x112-Mask.yaml
bash start_distributed.sh 3   # Stage 3: full FROM mask training
```

`NETWORK.PRETRAINED` should point at a specific `backbone_epoch_NNN.pth.tar` from
the previous stage's output dir (only the backbone transfers between stages — a
fresh PartialFC head, optimizer, and LR schedule start each stage, so you may seed
from **any** epoch, not just the last). Adjust `NUM_GPUS` / `BATCH_SIZE` at the top
of `start_distributed.sh`. See **INTEGRATION_GUIDE.md** for the occlusion pipeline
and stage details.

## Smoke test

Before a multi-hour run, validate the model/data/checkpoint plumbing in seconds
with `--max_steps` (runs only N steps per epoch):

```bash
torchrun --nproc_per_node=3 train_distributed.py --cfg experiments/smoke-test.yaml \
    --pattern 5 --ratio 3 --batch_size 16 --workers 4 --max_steps 20
```
Check that: no shape errors, loss is finite and decreasing, `margin = 0.0000` at
epoch 0, LR ramps up, and `checkpoint_gpu_{0,1,2}.pt` + `training_metrics.csv`
appear in the output dir. Azure upload is off in this config.

## Config

Edit each `experiments/custom-112x112-*.yaml`:

| Field | Description |
|-------|-------------|
| `TRAIN.MODEL` | backbone: `iResNet50_FPN` or `LResNet50E_IR_FPN` |
| `DATASET.REC_PATH` | dir with `train.rec` / `train.idx` (e.g. `/data/dataset`) |
| `DATASET.NUM_CLASS` | number of identities |
| `NUM_IMAGE` | total images (drives `cosine`/`polynomial` schedule length) |
| `SAMPLE_RATE` | PartialFC negative sampling: `1.0` (<50K ids), `0.1` (100K+), `0.01` (1M+) |
| `FP16` | mixed-precision training |
| `LR_SCHEDULER` | `cosine` · `polynomial` · `multistep` |
| `WARMUP_ITERS` / `WARMUP_LR` | linear LR warmup: steps and start LR |
| `LOSS.TYPE` | `CosFace` · `ArcFace` · `Combined` |
| `LOSS.SCALE` / `LOSS.MARGIN` | margin-softmax scale `s` and target margin `m` |
| `LOSS.MARGIN_WARMUP_EPOCHS` | epochs to linearly ramp margin `0 → MARGIN` (`0` = off) |
| `LOSS.WEIGHT_PRED` | weight of the mask-prediction loss (Stage 3 only) |
| `USE_AZURE` | upload checkpoints/logs to Azure File Share each epoch |
| `AZURE_REMOTE_DIR` | remote base path on the share |

**Schedulers:** `cosine` and `polynomial` are per-step (need `NUM_IMAGE`) and warm
up over `WARMUP_ITERS`; `multistep` drops LR by `LR_FACTOR` at `LR_STEP` epochs.

**Margin warmup:** with `MARGIN_WARMUP_EPOCHS: 5`, the margin ramps linearly from
`0.0` to `LOSS.MARGIN` over the first 5 epochs, then holds — a gentler start than
full margin from epoch 0.

## Outputs

Every epoch writes to the run's output dir `output/<dataset>/<model>-<mode>/<...>/`:

```
backbone_epoch_000.pth.tar … _NNN.pth.tar   # rank 0 — backbone only, for eval / PRETRAINED
checkpoint_gpu_0.pt / _1.pt / _2.pt         # one PER RANK — for --resume
training_metrics.csv                        # per-epoch: loss, cls, pred, lr, time, img/s
<timestamp>_train.log                       # full training log
```

- **`backbone_epoch_*`** — the model for **inference / your evaluation pipeline**.
  The backbone is DDP-synced (identical on every rank), so the single rank-0 file
  is the complete model. All epochs are kept so you can evaluate each.
- **`checkpoint_gpu_<rank>.pt`** — resumable state. PartialFC shards the classifier
  across GPUs, so **each rank saves its own shard** (+ optimizer, LR scheduler, FP16
  scaler). All shards are needed to resume.
- If `USE_AZURE: True`, all of the above are uploaded (best-effort, never blocking)
  to `training-checkpoint/from/<run_name>/`; the log is uploaded at the end of the run.

## Resume

`--resume` continues an **interrupted run of the same stage** (crash, preemption,
time limit). It is *not* how you move between stages — for that, use
`NETWORK.PRETRAINED`. Each rank loads its own shard:

```bash
torchrun --nproc_per_node=3 train_distributed.py \
    --cfg experiments/custom-112x112-Occ.yaml \
    --resume output/.../   # each rank picks up its checkpoint_gpu_<rank>.pt
```

`--resume` accepts a directory (→ `checkpoint_gpu_<rank>.pt`), a `.../gpu_N.pt`
file (remapped per rank), or a plain file (loaded as-is; single-GPU). It restores
backbone + PartialFC shard + optimizer + LR-scheduler position + FP16 scaler.

## Inference

Only the backbone is needed at test time (PartialFC is training-only):

```python
from lib.models.iresnet_fpn import iresnet50_occ   # or fpn_112.LResNet50E_IR_Occ_112

model = iresnet50_occ(num_mask=226)                 # match TRAIN.MODEL used in training
ckpt = torch.load('output/.../backbone_epoch_039.pth.tar')
model.load_state_dict(ckpt['state_dict'])
model.eval()

# forward returns (fc_mask, mask, vec, fc); use fc_mask as the 512-d embedding
```

## Acknowledgement

The occlusion pipeline and occluder images are from
[PDSN](https://github.com/linserSnow/PDSN). Distributed training uses PartialFC
from [InsightFace](https://github.com/deepinsight/insightface).

## Citation

```bibtex
@article{qiu2021end2end,
  title={End2End occluded face recognition by masking corrupted features},
  author={Qiu, Haibo and Gong, Dihong and Li, Zhifeng and Liu, Wei and Tao, Dacheng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2021},
  publisher={IEEE}
}
```
