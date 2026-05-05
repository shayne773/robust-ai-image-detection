from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets import GeneratorFolderDataset, build_generator_records
from src.datasets.transforms import (
    build_detector_transforms,
    build_size_constrained_detector_transforms,
    resolve_compression_quality,
)
from src.models import build_resnet_classifier
from src.training.trainer import evaluate
from src.utils.io import load_yaml
from src.utils.paths import ensure_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/default.yaml')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--eval-generators', nargs='+', default=None)
    p.add_argument('--test-compression', default=None)
    p.add_argument('--preprocess-mode', choices=['standard', 'size-constrained'], default=None)
    p.add_argument('--crop-size', type=int, default=None)
    p.add_argument('--inference-resize-size', type=int, default=None)
    p.add_argument('--device', default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    dcfg, pcfg, mcfg, tcfg = cfg['dataset'], cfg['preprocessing'], cfg['model'], cfg['training']
    xcfg = cfg.get('size_constraint', {})

    eval_generators = args.eval_generators or dcfg['eval_generators']
    test_compression = args.test_compression or pcfg['test_compression']
    resolve_compression_quality(test_compression)
    preprocess_mode = args.preprocess_mode or pcfg.get('preprocess_mode', 'standard')

    root = Path(dcfg['root'])
    allowed_ext = {e.lower() for e in dcfg.get('allowed_extensions', ['.jpg', '.jpeg', '.png'])}
    records = build_generator_records(root, dcfg['generator_dirs'], eval_generators, dcfg['eval_split'], dcfg['real_class_name'], dcfg['fake_class_name'], allowed_ext)
    if preprocess_mode == 'size-constrained':
        crop_size = int(args.crop_size or xcfg.get('crop_size', 450))
        inference_resize_size = int(args.inference_resize_size or xcfg.get('inference_resize_size', 512))
        transform = build_size_constrained_detector_transforms(
            test_compression,
            int(cfg['transforms']['image_size']),
            crop_size,
            str(cfg['transforms'].get('interpolation', 'bilinear')),
            is_train=False,
            inference_resize_size=inference_resize_size,
        )
    elif preprocess_mode == 'standard':
        transform = build_detector_transforms(test_compression, int(cfg['transforms']['image_size']), str(cfg['transforms'].get('interpolation', 'bilinear')), is_train=False)
    else:
        raise ValueError(f"Unsupported preprocess mode: {preprocess_mode}")
    dataset = GeneratorFolderDataset(records, transform)
    loader = DataLoader(dataset, batch_size=int(tcfg['batch_size']), shuffle=False, num_workers=int(tcfg['num_workers']))

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = build_resnet_classifier(str(mcfg['architecture']), num_classes=2, pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    metrics = evaluate(model, loader, nn.CrossEntropyLoss(), device)
    payload = {
        'checkpoint_path': args.checkpoint,
        'eval_generators': eval_generators,
        'test_compression': test_compression,
        'loss': metrics.loss,
        'accuracy': metrics.accuracy,
        'roc_auc': metrics.roc_auc,
        'recall': metrics.recall,
        'preprocess_mode': preprocess_mode,
        'crop_size': crop_size if preprocess_mode == 'size-constrained' else None,
        'inference_resize_size': inference_resize_size if preprocess_mode == 'size-constrained' else None,
        'class_mapping': {dcfg['real_class_name']: 0, dcfg['fake_class_name']: 1},
        'num_examples': len(dataset),
    }

    # Parse training details from checkpoint filename
    ckpt_name = Path(args.checkpoint).stem  # e.g., 'best_resnet50_train-biggan_comp-raw_pct-0.1'
    train_generators = 'unknown'
    train_compression = 'unknown'
    if '_train-' in ckpt_name and '_comp-' in ckpt_name:
        parts = ckpt_name.split('_')
        for i, part in enumerate(parts):
            if part.startswith('train-'):
                train_generators = part.replace('train-', '').replace('-', ' ')
            elif part.startswith('comp-'):
                train_compression = part.replace('comp-', '')

    out_dir = ensure_dir('outputs/eval')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = out_dir / f"{mcfg['architecture']}_train-{train_generators.replace(' ', '-')}_comp-{train_compression}_eval-{'-'.join(eval_generators)}_test-{test_compression}_{timestamp}_metrics.json"
    out_file.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
