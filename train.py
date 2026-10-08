"""python train.py --stage pretrain|adapt --config configs/la_bay.json --data-root data"""
import argparse
import json
from pathlib import Path
from gift.engine import train
from gift.models import VARIANTS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['pretrain', 'adapt'], required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--data-root', default='data')
    p.add_argument('--seed', type=int, default=0, choices=range(5))
    p.add_argument('--variant', choices=VARIANTS)
    p.add_argument('--policy', choices=['DEFAULT_CE', 'FULL', 'HEAD_ONLY', 'INTERACTION_ONLY', 'RESET_QK'])
    p.add_argument('--output')
    p.add_argument('--source-checkpoint')
    p.add_argument('--device', default='cuda')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--skip-data-hash', action='store_true')
    p.add_argument('--epochs', type=int, help='Diagnostic override; output is marked non-formal')
    p.add_argument('--max-updates', type=int, help='Diagnostic override; output is marked non-formal')
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.variant: config['variant'] = args.variant
    if args.policy: config['adaptation']['policy'] = args.policy
    output = args.output or str(Path('outputs') / config['direction'] / config['variant'] / f'seed{args.seed}')
    source = args.source_checkpoint or str(Path(output) / 'source.pt')
    result = train(config, args.seed, args.stage, args.data_root, output, args.device,
                   source_checkpoint=source if args.stage == 'adapt' else None, verify=not args.skip_data_hash,
                   epochs=args.epochs, max_updates=args.max_updates, threads=args.threads)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
