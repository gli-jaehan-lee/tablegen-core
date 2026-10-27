"""Read-only checks for a standalone publication tree; never uploads files.

Usage: python tools/check_release.py /path/to/repository --edition core
This heuristic scan complements, rather than replaces, a manual release review.
"""
import argparse
import json
import re
from pathlib import Path

CORE_NAMES = {
    'mlp', 'tree', 'disc', 'lin', 'quad', 'gp', 'em', 'prod',
    'piecewise_route_engine', 'nori_anchor_rho_copy', 'gev_tail_warp',
    'ewens_sampling_formula', 'markov_state_error_emission',
    'informative_censoring_hazard', 'eos_excess_mixing_enthalpy',
    'modulated_satellite_structure_factor', 'paws_timestamp_drop',
    'niche_model_food_web', 'island_biogeography_equilibrium',
    'bourgouin_precipitation_type', 'temperature_dependent_instar_transition',
    'tetragonal_interplanar_spacing', 'bch_error_locator', 'moe_linear_experts',
}
PRIVATE_PATH = re.compile(r'(?:/Users/|/home/)[A-Za-z0-9_.-]+/|[A-Z]:\\Users\\[^\\]+\\')
SECRET = re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|(?:ghp_|github_pat_)[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16}')
INTERNAL = re.compile(r'(?:Brain/|_meta/|proj:tfm|proj:tablegen|d\d{6}_|\[\[)')
FORBIDDEN = {'.env', '.git-credentials', 'id_rsa', 'id_ed25519', '.DS_Store'}
TEXT_SUFFIXES = {'.py', '.md', '.txt', '.json', '.toml', '.yml', '.yaml', '.sh', '.command', '.bat', '.html', '.csv'}


def check_release(root, edition):
    root = Path(root)
    issues = []
    files = 0
    if not root.is_dir():
        return ['Repository directory does not exist'], 0
    required = ['README.md', 'pyproject.toml', '.gitignore', 'THIRD_PARTY_NOTICES.md']
    if not any((root / name).is_file() for name in ('LICENSE', 'LICENSE.md', 'LICENSE.txt')):
        issues.append('Missing top-level LICENSE')
    modules = ['tablegen_core'] if edition == 'core' else ['tablegen', 'tablegen_addons', 'tablegen_extensions']
    required += [f'src/{module}/__init__.py' for module in modules]
    for relative in required:
        if not (root / relative).is_file():
            issues.append(f'Missing required file: {relative}')
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if '.git' in relative.parts:
            continue
        if path.is_symlink():
            issues.append(f'Symlink not allowed: {relative}')
            continue
        if not path.is_file():
            continue
        files += 1
        if path.name in FORBIDDEN or path.suffix in {'.pem', '.key', '.pyc'} or '__pycache__' in relative.parts:
            issues.append(f'Private/generated file: {relative}')
        if path.suffix not in TEXT_SUFFIXES and path.name not in {'LICENSE', '.gitignore'}:
            continue
        try:
            content = path.read_text(encoding='utf-8')
        except UnicodeError:
            issues.append(f'Non-UTF-8 text: {relative}')
            continue
        # Report locations only: never echo a potential secret value.
        for label, pattern in [('personal absolute path', PRIVATE_PATH), ('secret-like value', SECRET)]:
            if pattern.search(content):
                issues.append(f'{label}: {relative}')
        # Third-party source/history is preserved. Public guides must be portable.
        if path.suffix == '.md' and 'licenses' not in relative.parts and path.name != 'THIRD_PARTY_NOTICES.md':
            if INTERNAL.search(content):
                issues.append(f'Internal documentation reference: {relative}')
    if edition == 'core':
        index = root / 'src/tablegen_core/prior_index.json'
        try:
            data = json.loads(index.read_text())
            names = list(data['legacy']) + list(data['by_name'])
            if len(names) != 24 or set(names) != CORE_NAMES:
                issues.append('Core roster is not the approved unique 8 + 16 names')
        except (OSError, ValueError, KeyError, TypeError):
            issues.append('Core prior_index.json missing or malformed')
    return issues, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repository', type=Path)
    parser.add_argument('--edition', choices=('core', 'full'), required=True)
    args = parser.parse_args()
    issues, files = check_release(args.repository, args.edition)
    print(json.dumps({'edition': args.edition, 'files_checked': files,
                      'passed': not issues, 'issues': issues}, indent=2))
    return int(bool(issues))


if __name__ == '__main__':
    raise SystemExit(main())
