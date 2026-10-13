#!/usr/bin/env python3
"""Evaluate a frozen gui_demo model on four prespecified unseen synthetic trajectories."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out = ROOT/'runs/generalization_gui_demo'
    out.mkdir(exist_ok=False)
    model = ROOT/'runs/gui_demo/model'
    protected = [model/n for n in ('actuator_net.pt','actuator_net_jit.pt','config.json')]
    before = {str(p.relative_to(ROOT)):sha(p) for p in protected}
    train_log = ROOT/'runs/gui_demo/log.csv'
    meta = json.loads(train_log.with_suffix('.meta.json').read_text())
    cfg = json.loads((model/'config.json').read_text())
    opts = meta['options']
    assert sha(train_log) == meta['csv_sha256']
    assert opts['seed'] == 0 and opts['amplitude'] == .05 and opts['frequency'] == .5
    assert opts['kp'] == 10 and opts['kd'] == .5 and opts['dt'] == .001
    assert opts['torque_scale'] == .9 and opts['extra_damping'] == .1
    assert cfg['stride'] == 2 and cfg['smooth'] == 1 and cfg['steps'] == 0
    cases = [('new_phase', .05, .5), ('new_frequency', .05, .8),
             ('new_amplitude', .08, .5), ('new_both', .08, .8)]
    manifest = dict(model_sha256_before=before, training_csv_sha256=sha(train_log),
                    seed=1, cases=cases, note='No training. Same synthetic plant, new trajectories. All cases reported.')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')

    def run(case):
        name, amplitude, frequency = case
        folder = out/name
        folder.mkdir()
        collect = [sys.executable, 'scripts/collect.py', '--urdf', opts['urdf'],
                   '--out', str(folder/'log.csv'), '--seconds', '20', '--dt', '.001',
                   '--kp','10','--kd','.5','--torque-scale','.9','--extra-damping','.1',
                   '--seed','1','--amplitude',str(amplitude),'--frequency',str(frequency)]
        evaluate = [sys.executable, 'scripts/evaluate.py', str(folder/'log.csv'),
                    '--model',str(model),'--out',str(folder/'eval'), '--urdf',opts['urdf'],
                    '--kp','10','--kd','.5','--stride','2','--smooth','1','--fixed-dt','.002']
        (folder/'commands.json').write_text(json.dumps([collect,evaluate],indent=2)+'\n')
        for label,command in [('collect',collect),('evaluate',evaluate)]:
            with (folder/f'{label}.log').open('w') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert sha(folder/'log.csv') != sha(train_log)
        print(f'completed {name}: amplitude={amplitude}, frequency={frequency}',flush=True)
        return dict(name=name, amplitude=amplitude, frequency=frequency, seed=1,
                    csv_sha256=sha(folder/'log.csv'),
                    metrics=json.loads((folder/'eval/metrics.json').read_text()))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run,cases))
    after = {str(p.relative_to(ROOT)):sha(p) for p in protected}
    assert before == after, 'frozen model changed!'
    manifest.update(model_sha256_after=after, model_unchanged=True, results=results)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lines=['# Unseen synthetic trajectory evaluation', '',
           'Frozen gui_demo model; no retraining. Same 2-joint synthetic plant; no real robot data.', '',
           '| Case | amplitude (rad) | frequency (Hz) | baseline pos RMSE | augmented pos RMSE | pos reduction | velocity reduction |',
           '|---|---:|---:|---:|---:|---:|---:|']
    original=json.loads((ROOT/'runs/gui_demo/eval/metrics.json').read_text())
    for r in [dict(name='training trajectory (reference)', amplitude=.05,frequency=.5,metrics=original)]+results:
        roll=r['metrics']['rollout']; b=roll['baseline']; a=roll['augmented']
        pos=1-a['pos_rmse_rad']/b['pos_rmse_rad']
        vel=1-a['vel_rmse_rad_s']/b['vel_rmse_rad_s']
        lines.append(f"| {r['name']} | {r['amplitude']} | {r['frequency']} | {b['pos_rmse_rad']:.8g} | {a['pos_rmse_rad']:.8g} | {pos:.1%} | {vel:.1%} |")
    lines += ['', 'Every new case uses seed=1 (training seed=0), 20 s, 1 kHz physics / 500 Hz network.',
              'No sensor noise. Torque scale 0.9 and extra damping 0.1 stay unchanged.',
              'Use full-rollout RMSE above, not the inherited heldout-ticks metric to define an unseen trajectory.',
              'One new seed per condition: a limited probe, not broad generalization or real-robot validation.',
              'Training weights, TorchScript, and config SHA256 were unchanged; see manifest.json.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__ == '__main__':
    main()
