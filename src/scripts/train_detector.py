from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.datasets import GeneratorFolderDataset, build_generator_records, class_balanced_subsample
from src.datasets.transforms import SUPPORTED_COMPRESSION_MODES, build_detector_transforms
from src.models import build_resnet_classifier
from src.training.trainer import evaluate, save_checkpoint, train_one_epoch
from src.utils.io import load_yaml
from src.utils.paths import ensure_dir
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/default.yaml')
    p.add_argument('--train-generators', nargs='+', default=None)
    p.add_argument('--eval-generators', nargs='+', default=None)
    p.add_argument('--train-compression', default=None)
    p.add_argument('--train-percent', type=float, default=None)
    p.add_argument('--model', default=None)
    p.add_argument('--device', default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    dcfg, pcfg, mcfg, tcfg = cfg['dataset'], cfg['preprocessing'], cfg['model'], cfg['training']
    set_global_seed(int(tcfg.get('seed', 42)))

    train_generators = args.train_generators or dcfg['train_generators']
    eval_generators = args.eval_generators or dcfg['eval_generators']
    train_compression = args.train_compression or pcfg['train_compression']
    train_percent = float(args.train_percent if args.train_percent is not None else dcfg['train_percent'])
    model_name = (args.model or mcfg['architecture']).lower()

    if train_compression not in SUPPORTED_COMPRESSION_MODES:
        raise ValueError(f'Unsupported train compression: {train_compression}')

    root = Path(dcfg['root'])
    allowed_ext = {e.lower() for e in dcfg.get('allowed_extensions', ['.jpg', '.jpeg', '.png'])}
    train_records = build_generator_records(root, dcfg['generator_dirs'], train_generators, dcfg['train_split'], dcfg['real_class_name'], dcfg['fake_class_name'], allowed_ext)
    train_records = class_balanced_subsample(train_records, train_percent, int(tcfg.get('seed', 42)))
    val_records = build_generator_records(root, dcfg['generator_dirs'], eval_generators, dcfg['eval_split'], dcfg['real_class_name'], dcfg['fake_class_name'], allowed_ext)

    train_ds = GeneratorFolderDataset(train_records, build_detector_transforms(train_compression, int(cfg['transforms']['image_size']), str(cfg['transforms'].get('interpolation', 'bilinear')), is_train=True))
    val_ds = GeneratorFolderDataset(val_records, build_detector_transforms(pcfg['test_compression'], int(cfg['transforms']['image_size']), str(cfg['transforms'].get('interpolation', 'bilinear')), is_train=False))

    train_loader = DataLoader(train_ds, batch_size=int(tcfg['batch_size']), shuffle=True, num_workers=int(tcfg['num_workers']))
    val_loader = DataLoader(val_ds, batch_size=int(tcfg['batch_size']), shuffle=False, num_workers=int(tcfg['num_workers']))

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = build_resnet_classifier(model_name, num_classes=2, pretrained=bool(mcfg['pretrained'])).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=float(tcfg['lr']))

    ckpt_dir = ensure_dir(tcfg['checkpoint_dir'])
    gen_slug = '-'.join(train_generators)
    best_path = ckpt_dir / f'best_{model_name}_train-{gen_slug}_comp-{train_compression}_pct-{train_percent}.pt'
    best = -1.0
    for epoch in range(1, int(tcfg['epochs']) + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vm = evaluate(model, val_loader, criterion, device)
        print(f'epoch={epoch} train_loss={tr_loss:.4f} val_loss={vm.loss:.4f} val_acc={vm.accuracy:.4f} val_auc={vm.roc_auc} val_recall={vm.recall:.4f}')
        score = vm.roc_auc if vm.roc_auc is not None else vm.accuracy
        if score > best:
            best = score
            save_checkpoint(best_path, model, optimizer, epoch, best, 'roc_auc', extra={
                'architecture': model_name,
                'train_generators': train_generators,
                'train_compression': train_compression,
                'train_percent': train_percent,
                'class_mapping': {dcfg['real_class_name']: 0, dcfg['fake_class_name']: 1},
            })
    print(f'saved best checkpoint: {best_path}')


if __name__ == '__main__':
    main()
