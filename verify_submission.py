"""Fail clearly when active results/samples are missing or inconsistent with the new protocol."""
from pathlib import Path
import json, numpy as np
ROOT = Path(__file__).resolve().parent
required = [ROOT / 'data.npz', ROOT / 'coast.pkl', ROOT / 'samples/kinland.npz',
            ROOT / 'results/train_kinland.json', ROOT / 'results/oppoint.json',
            ROOT / 'results/realism.json', ROOT / 'results/detect.json', ROOT / 'results/ablation.json']
missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
if missing:
    raise SystemExit('NOT FINAL: rerun experiments; missing inputs/outputs: ' + ', '.join(missing))
with open(ROOT / 'results/train_kinland.json') as fp:
    meta = json.load(fp)
if meta.get('tries') != 50 or not all(meta.get(k, {}).get('intersection_after') is not None for k in ('train','val','test')):
    raise SystemExit('NOT FINAL: incorrect or outdated kinland metadata (expected 50 and intersection_after)')
with np.load(ROOT / 'samples/kinland.npz') as s, np.load(ROOT / 'data.npz') as d:
    for split in ('train','val','test'):
        if split not in s or len(s[split]) != len(d[split]):
            raise SystemExit('NOT FINAL: coastline-aware sample split/count mismatch: '+split)
with open(ROOT / 'results/oppoint.json') as fp: op = json.load(fp)
for name in ('kinnoise','kinland'):
    if name not in op or not all(metric in op[name] for metric in ('RF','IsoForest','Rule','Land','RuleLand')):
        raise SystemExit(f'NOT FINAL: missing detector outputs for {name}')
    if op[name]['Land'].get('threshold_method') != 'fixed any-intersection rule':
        raise SystemExit('NOT FINAL: stale point-only coastline results in active oppoint.json')
print('STRUCTURAL CHECK PASSED: active outputs exist with segment-aware metadata.')
print('IMPORTANT: This check does not assert equivalence to paper values or rerun the science.')
for name in ('kinnoise','kinland'):
    print(name, {k: round(op[name][k]['tpr']*100, 3) for k in ('Rule','Land','RuleLand')})
