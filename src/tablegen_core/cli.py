"""Command line interface; explicit flags override JSON configuration fields."""
import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog='tablegen-core')
    sub = parser.add_subparsers(dest='command', required=True)
    listing = sub.add_parser('list-priors', help='List the 24 selectable priors')
    listing.add_argument('--json', action='store_true')
    gen = sub.add_parser('generate', help='Generate CSV, NPY, metadata and an offline HTML preview')
    gen.add_argument('--rows', type=int)
    gen.add_argument('--features', type=int)
    gen.add_argument('--targets', type=int)
    gen.add_argument('--seed', type=int)
    gen.add_argument('--prior', action='append', help='Name, prior-N, default8, robust16 or all24; repeat or comma-separate')
    gen.add_argument('--mode', choices=['direct', 'dag'])
    gen.add_argument('--output', required=True, help='New output directory (must not already exist)')
    gen.add_argument('--config', help='JSON object or path to a JSON file; explicit flags take precedence')
    args = parser.parse_args(argv)
    try:
        if args.command == 'list-priors':
            from .registry import list_priors
            entries = list_priors()
            if args.json:
                print(json.dumps(entries, ensure_ascii=False, indent=2))
            else:
                for entry in entries:
                    print(f"{entry['name']}\t{entry['id']}\t{entry['description']}")
            return 0
        from .api import GenerationConfig, generate
        config = {}
        if args.config:
            raw = args.config.strip()
            config = json.loads(raw if raw.startswith('{') else Path(raw).read_text(encoding='utf-8'))
            config = vars(GenerationConfig.from_dict(config))
        for name in ('rows', 'features', 'targets', 'seed', 'mode'):
            if getattr(args, name) is not None:
                config[name] = getattr(args, name)
        if args.prior is not None:
            config['priors'] = [name.strip() for group in args.prior for name in group.split(',')]
        files = generate(config).write(args.output)
        print(json.dumps(files, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
