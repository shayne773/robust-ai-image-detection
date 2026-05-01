from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets import GeneratorFolderDataset, build_generator_records
from src.datasets.transforms import SUPPORTED_COMPRESSION_MODES, build_detector_transforms
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
    p.add_argument('--device', default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    dcfg, pcfg, mcfg, tcfg = cfg['dataset'], cfg['preprocessing'], cfg['model'], cfg['training']

    eval_generators = args.eval_generators or dcfg['eval_generators']
    test_compression = args.test_compression or pcfg['test_compression']
    if test_compression not in SUPPORTED_COMPRESSION_MODES:
        raise ValueError(f'Unsupported test compression: {test_compression}')

    root = Path(dcfg['root'])
    allowed_ext = {e.lower() for e in dcfg.get('allowed_extensions', ['.jpg', '.jpeg', '.png'])}
    records = build_generator_records(root, dcfg['generator_dirs'], eval_generators, dcfg['eval_split'], dcfg['real_class_name'], dcfg['fake_class_name'], allowed_ext)
    dataset = GeneratorFolderDataset(records, build_detector_transforms(test_compression, int(cfg['transforms']['image_size']), str(cfg['transforms'].get('interpolation', 'bilinear')), is_train=False))
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
        'class_mapping': {dcfg['real_class_name']: 0, dcfg['fake_class_name']: 1},
        'num_examples': len(dataset),
    }
    out_dir = ensure_dir('outputs/eval')
    out_file = out_dir / f"{mcfg['architecture']}_eval-{'-'.join(eval_generators)}_test-{test_compression}_metrics.json"
    out_file.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
