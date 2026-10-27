#!/usr/bin/env python3
"""Prepare an isolated runnable copy using local, unpublished experimental assets.

No training is launched and the original experiment is never modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / 'experiments/batch_nll_20261027'
ORIGINAL_ROOT = '/ai/data/moe-lora'
ORIGINAL_RUN = 'Jev/runs/A_OPENJEV/batch_nll_mlp_openjev_1233_1024_20261027'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def relocate(value, assets):
    if isinstance(value, dict):
        return {key: relocate(item, assets) for key, item in value.items()}
    if isinstance(value, list):
        return [relocate(item, assets) for item in value]
    if isinstance(value, str) and value.startswith(ORIGINAL_ROOT + '/'):
        return str(assets / value[len(ORIGINAL_ROOT) + 1:])
    return value

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-root', type=Path, required=True,
                        help='Local asset tree with Jev/ and drg-llama/; defaults are never downloaded')
    parser.add_argument('--run-dir', type=Path, required=True, help='Must not already exist')
    parser.add_argument('--dry-run', action='store_true', help='Validate asset existence without writing')
    parser.add_argument('--check-weight-hashes', action='store_true', help='Also read and hash the large expert and O2 files')
    args = parser.parse_args()
    assets = args.assets_root.resolve()
    run = args.run_dir.resolve()
    c = relocate(json.loads((EXPORT / 'records/config.json').read_text()), assets)
    c['run_dir'] = str(run)
    c['method_document'] = str(ROOT / 'docs/METHOD_OPENJEV_LATEST_ZH.md')
    original = json.loads((EXPORT / 'records/experiment_contract.json').read_text())
    links = {
        'reused/expert_and_head.pt': assets / ORIGINAL_RUN / 'reused/expert_and_head.pt',
        'oracle/train.pt': assets / ORIGINAL_RUN / 'oracle/train.pt',
    }
    required = [*links.values(), Path(c['openjev_checkpoint']),
                Path(c['openjev_local_model']) / 'config.json',
                Path(c['source']['model_name_or_path']) / 'config.json',
                *[Path(c['split_manifest']['paths'][s]) for s in ('train', 'validation')]]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit('Required unpublished local assets are missing:\n' + '\n'.join(missing))
    if run.exists():
        raise SystemExit('Refusing to overwrite an existing run: ' + str(run))
    if args.check_weight_hashes:
        for path, expected in [(links['reused/expert_and_head.pt'], original['expert_sha256']),
                               (Path(c['openjev_checkpoint']), original['o2_sha256']),
                               (links['oracle/train.pt'], original['train_oracle_sha256'])]:
            if sha(path) != expected:
                raise SystemExit('Asset hash mismatch: ' + str(path))
    if args.dry_run:
        print('PASS: required local assets exist; no files written and no training launched')
        return
    run.mkdir(parents=True)
    shutil.copytree(EXPORT / 'source', run / 'source', ignore=shutil.ignore_patterns('__pycache__'))
    for relative, path in links.items():
        target = run / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(path.resolve())
    for name in ('logs', 'features', 'mlp', 'openjev'):
        (run / name).mkdir()
    shutil.copy2(EXPORT / 'records/oracle/verification.json', run / 'oracle/verification.json')
    # Rebuild the local execution contract after relocation and public score-field removal.
    configuration = {key: value for key, value in c.items() if key != 'experiment_contract'}
    contract = dict(original)
    contract['configuration'] = configuration
    contract['method_document_sha256'] = sha(Path(c['method_document']))
    c['experiment_contract'] = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    write(run / 'config.json', c)
    write(run / 'experiment_contract.json', contract)
    write(run / 'publication_origin.json', {
        'source_records': str(EXPORT / 'records'),
        'original_contract_digest': json.loads((EXPORT / 'records/config.json').read_text())['experiment_contract'],
        'local_contract_digest': c['experiment_contract'],
        'inherited_scores_omitted': True,
        'training_launched': False,
    })
    print('Prepared isolated run: ' + str(run))
    print('To launch explicitly:')
    print('python ' + str(run / 'source/launch_batch_nll_routers.py') + ' --run-dir ' + str(run) + ' --gpus 0,1,2,3,4,5,6,7 --detach')

if __name__ == '__main__':
    main()
