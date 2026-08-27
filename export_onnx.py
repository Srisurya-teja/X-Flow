"""
Convert FROM training checkpoints (.pth.tar / .pt) to ONNX.

Exports the face-embedding head of the model. For a checkpoint dir it converts
every matching file; for a single file it converts just that one.

    # convert a whole Stage-1 output dir
    python export_onnx.py --input output/Custom/iResNet50_FPN_CosFace-Clean/<run>/ \
        --model iResNet50_FPN --num_mask 226

    # convert one checkpoint, phase-3 (Mask) model -> masked embedding
    python export_onnx.py --input .../backbone_epoch_020.pth.tar --embedding fc_mask

Which output is the embedding depends on the training stage:
  * Clean / Occ (Stage 1 / 2): use `fc`      (default) — the mask branch is untrained
  * Mask       (Stage 3):      use `fc_mask`            — the masked embedding

Only `fc` (or `fc_mask`) is kept as the ONNX output; ONNX dead-code elimination
then prunes the branch that doesn't feed it (so a Stage-1 `fc` export becomes a
clean iResNet50 -> 512-d graph, with no mask branch).

Preprocessing the exported model expects (must match training):
  RGB, 112x112, normalized to [-1, 1]  (i.e. (pixel/255 - 0.5) / 0.5), NCHW.
"""

import os
import sys
import glob
import argparse

import torch
import torch.nn as nn

# Make `lib` importable regardless of where the script is run from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.models.iresnet_fpn import iresnet50_occ
from lib.models.fpn_112 import (
    LResNet50E_IR_Occ_112,
    LResNet50E_IR_Occ_2D_112,
    LResNet50E_IR_Occ_FC_112,
)

MODEL_FACTORY = {
    'iResNet50_FPN': iresnet50_occ,
    'LResNet50E_IR_FPN': LResNet50E_IR_Occ_112,
    'LResNet50E_IR_Occ_2D': LResNet50E_IR_Occ_2D_112,
    'LResNet50E_IR_Occ_FC': LResNet50E_IR_Occ_FC_112,
}


class EmbeddingWrapper(nn.Module):
    """Expose a single embedding output so the ONNX graph is a clean extractor.

    The underlying model returns (fc_mask, mask, vec, fc); we keep only the chosen
    embedding, and ONNX prunes the rest of the graph.
    """
    def __init__(self, model, which='fc'):
        super().__init__()
        self.model = model
        self.which = which

    def forward(self, x):
        fc_mask, mask, vec, fc = self.model(x)
        return fc_mask if self.which == 'fc_mask' else fc


def _extract_state_dict(ckpt):
    """Pull the model weights out of any of the checkpoint formats we save."""
    if isinstance(ckpt, dict):
        for key in ('state_dict', 'state_dict_backbone'):
            if key in ckpt:
                ckpt = ckpt[key]
                break
    # strip a DataParallel/DDP 'module.' prefix if present
    return {k.replace('module.', '', 1) if k.startswith('module.') else k: v
            for k, v in ckpt.items()}


def build_model(model_name, num_mask):
    if model_name not in MODEL_FACTORY:
        raise ValueError(f'Unknown model {model_name}; choose from {list(MODEL_FACTORY)}')
    return MODEL_FACTORY[model_name](num_mask=num_mask)


def convert_one(path, out_path, args):
    model = build_model(args.model, args.num_mask)
    ckpt = torch.load(path, map_location='cpu')
    state_dict = _extract_state_dict(ckpt)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f'  [warn] {len(missing)} missing keys (e.g. {missing[:3]})')
    if unexpected:
        print(f'  [warn] {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})')

    model.eval()
    wrapper = EmbeddingWrapper(model, which=args.embedding).eval()

    H = W = args.image_size
    dummy = torch.randn(1, 3, H, W)

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {'input': {0: 'batch'}, 'embedding': {0: 'batch'}}

    torch.onnx.export(
        wrapper,
        dummy,
        out_path,
        input_names=['input'],
        output_names=['embedding'],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f'  -> {out_path}')

    if not args.no_verify:
        _verify(wrapper, dummy, out_path)


def _verify(wrapper, dummy, out_path):
    """Optional: check the ONNX graph and compare outputs against PyTorch."""
    try:
        import onnx
        onnx.checker.check_model(onnx.load(out_path))
    except ImportError:
        print('  [verify] onnx not installed; skipping graph check')
    except Exception as e:
        print(f'  [verify] onnx check failed: {e}')

    try:
        import numpy as np
        import onnxruntime as ort
        with torch.no_grad():
            ref = wrapper(dummy).numpy()
        sess = ort.InferenceSession(out_path, providers=['CPUExecutionProvider'])
        got = sess.run(['embedding'], {'input': dummy.numpy()})[0]
        max_diff = float(np.abs(ref - got).max())
        status = 'OK' if max_diff < 1e-3 else 'MISMATCH'
        print(f'  [verify] max|torch-onnx| = {max_diff:.2e}  ({status})')
    except ImportError:
        print('  [verify] onnxruntime not installed; skipping numerical check')
    except Exception as e:
        print(f'  [verify] runtime check failed: {e}')


def parse_args():
    p = argparse.ArgumentParser(description='Convert FROM .pth.tar checkpoints to ONNX')
    p.add_argument('--input', required=True,
                   help='directory of checkpoints, or a single .pth.tar/.pt file')
    p.add_argument('--output_dir', default=None,
                   help='where to write .onnx files (default: alongside the inputs)')
    p.add_argument('--pattern', default='*.pth.tar',
                   help='glob for checkpoints when --input is a directory (default: *.pth.tar)')
    p.add_argument('--model', default='iResNet50_FPN', choices=list(MODEL_FACTORY),
                   help='model architecture used in training')
    p.add_argument('--num_mask', type=int, default=226,
                   help='grid classes the model was built with (PATTERN 5 -> 226)')
    p.add_argument('--image_size', type=int, default=112)
    p.add_argument('--embedding', default='fc', choices=['fc', 'fc_mask'],
                   help="'fc' for Clean/Occ (Stage 1/2), 'fc_mask' for Mask (Stage 3)")
    p.add_argument('--opset', type=int, default=13)
    p.add_argument('--dynamic_batch', action='store_true',
                   help='export a dynamic batch dimension')
    p.add_argument('--no_verify', action='store_true',
                   help='skip the onnx / onnxruntime verification step')
    return p.parse_args()


def main():
    args = parse_args()

    if os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, args.pattern)))
        if not files:
            raise FileNotFoundError(f'No files matching {args.pattern} in {args.input}')
    elif os.path.isfile(args.input):
        files = [args.input]
    else:
        raise FileNotFoundError(args.input)

    out_dir = args.output_dir or (args.input if os.path.isdir(args.input)
                                  else os.path.dirname(args.input))
    os.makedirs(out_dir, exist_ok=True)

    print(f'Converting {len(files)} checkpoint(s) | model={args.model} '
          f'num_mask={args.num_mask} embedding={args.embedding}')
    for path in files:
        stem = os.path.basename(path)
        for ext in ('.pth.tar', '.pt', '.pth'):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        out_path = os.path.join(out_dir, stem + '.onnx')
        print(f'[{os.path.basename(path)}]')
        try:
            convert_one(path, out_path, args)
        except Exception as e:
            print(f'  [error] failed: {e}')

    print('Done.')


if __name__ == '__main__':
    main()
