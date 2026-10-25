import argparse
import json
from pathlib import Path
from gift.engine import evaluate, write_json


def main():
    p = argparse.ArgumentParser(description='Evaluate a locally trained final GIFT checkpoint')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data-root', default='data')
    p.add_argument('--device', default='cuda')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--output', default='evaluation.json')
    p.add_argument('--zero-day', action='store_true', help='Use a source checkpoint and its statistics; no target fitting')
    p.add_argument('--save-predictions', help='Optional local .npy output; large files are ignored by Git')
    args = p.parse_args()
    import torch
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = json.loads(json.dumps(state['config']))
    if args.zero_day:
        if state['stage'] != 'pretrain': raise ValueError('--zero-day requires a source checkpoint')
        config['support_days'] = [0, 0]
    result = evaluate(config, state['seed'], args.checkpoint, args.data_root, args.device,
                      threads=args.threads, save_predictions=args.save_predictions)
    write_json(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
