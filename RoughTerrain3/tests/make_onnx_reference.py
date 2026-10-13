"""Generate independent Python ORT outputs for C++ integration regression."""
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
root = Path(__file__).resolve().parents[1]
options = ort.SessionOptions()
options.intra_op_num_threads = 1
session = ort.InferenceSession(str(root / 'policy/model_10000.onnx'), options, providers=['CPUExecutionProvider'])
rng = np.random.default_rng(20261010)
observations = np.concatenate([np.zeros((1, 45), np.float32), rng.normal(size=(64,45)).astype(np.float32)])
cases = [{'obs': obs.tolist(), 'actions': session.run(['actions'], {'obs': obs[None]})[0][0].tolist()} for obs in observations]
(root / 'tests/onnx_reference.json').write_text(json.dumps(cases) + '\n')
