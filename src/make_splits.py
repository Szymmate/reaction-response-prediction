"""Create reproducible employee-disjoint partitions from the configured master seed."""
import features  # Set numerical thread limits first.
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from features import SEED, digest, load_all, person_partition, targets


def group_assignments(frame, n_splits, seed):
    """Balance outcome proportions without ever splitting an employee across folds."""
    result = np.full(len(frame), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (fit, valid) in enumerate(splitter.split(frame, targets(frame), frame.person_id)):
        assert set(frame.iloc[fit].person_id).isdisjoint(frame.iloc[valid].person_id)
        result[valid] = fold
    assert set(result) == set(range(n_splits))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, default=Path('results/splits'))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose a new empty split directory')
    args.output.mkdir(parents=True, exist_ok=True)
    people, _, _, frame = load_all(args.root / 'data')
    people[['person_id']].assign(split=people.person_id.map(person_partition)).to_csv(args.output / 'person_splits.csv', index=False)
    pool = frame[frame.person_id.map(person_partition).eq('train')].reset_index(drop=True)
    labeled = pool[pool.response_category.notna()].reset_index(drop=True)
    outer = labeled[['situation_id', 'person_id', 'response_category']].copy()
    inner = []
    for repeat in range(10):
        assignment = group_assignments(labeled, 5, SEED + repeat)
        outer[f'repeat_{repeat}_fold'] = assignment
        for fold in range(5):
            fitting = labeled[assignment != fold].reset_index(drop=True)
            table = fitting[['situation_id', 'person_id']].copy()
            table['inner_fold'] = group_assignments(fitting, 3, SEED + 1000 + 5 * repeat + fold)
            table['outer_repeat'] = repeat
            table['outer_fold'] = fold
            inner.append(table)
    outer.to_csv(args.output / 'outer_folds.csv', index=False)
    pd.concat(inner, ignore_index=True).to_csv(args.output / 'inner_folds.csv', index=False)
    final = labeled[['situation_id', 'person_id']].copy()
    final['inner_fold'] = group_assignments(labeled, 3, SEED + 2000)
    final.to_csv(args.output / 'final_inner_folds.csv', index=False)
    # Canonical relative paths let the generated directory be copied to results/splits.
    files = {str(p.relative_to(args.root)): digest(p) for p in sorted((args.root / 'data').glob('*.parquet'))}
    files.update({f'results/splits/{p.name}': digest(p) for p in sorted(args.output.glob('*.csv'))})
    manifest = dict(seed=SEED, files=files,
        source='Reproducible seed-specific employee partitions; development only',
        partition_rule='SHA256(seed:person_id): 60% train, 15% tune, 10% calibration, 15% test',
        folds='StratifiedGroupKFold, shuffled employee groups',
        seed_streams=dict(outer='seed + repeat', inner='seed + 1000 + 5*repeat + fold', final_inner='seed + 2000'),
        development_counts=dict(labeled_events=len(labeled), labeled_people=labeled.person_id.nunique(),
                                pool_events=len(pool), unlabeled_events=int(pool.response_category.isna().sum())),
        independence='Changing partitions does not remove prior development exposure.')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
