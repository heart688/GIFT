"""Print reproducibility commands; --execute runs the requested finite queue."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directions', nargs='+', default=['LA_BAY', 'BAY_LA', 'CD_SZ', 'SZ_CD'],
                        choices=['LA_BAY', 'BAY_LA', 'CD_SZ', 'SZ_CD'])
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(5)), choices=range(5))
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output-root', default='outputs')
    parser.add_argument('--execute', action='store_true', help='Actually train; without this flag print commands only')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    for direction in dict.fromkeys(args.directions):
        for seed in dict.fromkeys(args.seeds):
            output = Path(args.output_root) / direction / f'seed{seed}'
            common = ['--config', str(root / 'configs' / (direction.lower() + '.json')),
                      '--data-root', args.data_root, '--seed', str(seed),
                      '--device', args.device]
            source = output / 'source.pt'
            commands = [(source, [sys.executable, str(root / 'train.py'), '--stage', 'pretrain',
                                  *common, '--output', str(output)]),
                        (output / 'adapted.pt', [sys.executable, str(root / 'train.py'), '--stage', 'adapt',
                                  *common, '--output', str(output),
                                  '--source-checkpoint', str(source)])]
            for checkpoint, command in commands:
                print(json.dumps({'checkpoint': str(checkpoint), 'argv': command}), flush=True)
                if args.execute:
                    if checkpoint.exists():
                        print('Existing checkpoint retained: ' + str(checkpoint), flush=True)
                    else:
                        subprocess.run(command, cwd=root, check=True)


if __name__ == '__main__': main()
