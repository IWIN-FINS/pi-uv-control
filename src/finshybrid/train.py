"""Portable G1/A training entry; never starts simulation or rewrites supplied models."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .encoder import init_encoder, encode, materialize
from .predictor import Model
from .torch_model import Predictor, G1Predictor, GraphTrainer, fixture, score
from .scales import H


def windows(data, horizon=H):
    starts = np.arange(max(0, len(data['x']) - horizon + 1))
    return starts[data['seg'][starts] == data['seg'][starts + horizon - 1]]


def load(path):
    with np.load(path, allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    for k in ('x', 'y', 'u', 'seg'):
        if k not in data:
            raise ValueError(f'Missing dataset key {k}')
    n = len(data['x'])
    if data['x'].shape != (n, 42) or data['y'].shape != (n, 42) or data['u'].shape != (n, 6):
        raise ValueError('Expected x/y=(N,42), u=(N,6), normalized legacy features')
    if len(data['seg']) != n or any(not np.isfinite(data[k]).all() for k in ('x', 'y', 'u')):
        raise ValueError('Nonfinite or inconsistent training data')
    return data


def initialize(data, seed, device):
    meta, weights = init_encoder(data, seed, 'G1')
    shell = G1Predictor(meta, weights, np.zeros((16, 16)), 'G1', device)
    with torch.no_grad():
        nv = np.concatenate([shell.fossen(shell.t(data['x'][i:i+1024, 1:13]),
                                         shell.t(data['u'][i:i+1024])).cpu().numpy()
                             for i in range(0, len(data['x']), 1024)])
    z = encode(data['x'], meta, weights, 'G1')
    target = np.c_[data['y'][:, 7:13]-nv, encode(data['y'], meta, weights, 'G1')]
    xx = np.c_[z, data['u']]
    k = np.linalg.solve(xx.T @ xx / len(xx) + .003*np.eye(16), xx.T @ target / len(xx)).T
    return meta, weights, np.r_[k, np.zeros((6, 16))], np.maximum(z.std(0), .2)


class EagerTrainer(GraphTrainer):
    """Same update/backtracking arithmetic as GraphTrainer, without CUDA capture.

    CPU/eager execution is portable; bitwise training identity across devices is
    not promised. This does not change the objective, Adam budget or acceptance.
    """
    def __init__(self, p, batch, latent_scale):
        self.p = p
        self.batch = tuple(p.t(x).clone() for x in batch)
        self.scale = p.t(latent_scale)
        self.pars = [p.k, *p.w]
        self.anchor = [v.detach().clone() for v in self.pars]
        self.first = [torch.zeros_like(v) for v in self.pars]
        self.second = [torch.zeros_like(v) for v in self.pars]

        class Replay:
            def __init__(self, callback): self.callback = callback
            def replay(self): self.callback()

        def gradient():
            self.value = self.gradient()
            self.valid = p.chart_valid

        def forward():
            self.newvalue = p.objective(*self.batch, self.scale, self.anchor)
            self.newvalid = p.chart_valid

        self.graph, self.post = Replay(gradient), Replay(forward)


def export(path, meta, weights, k):
    raw = Model().raw
    a, b = materialize(k[:16], 'G1')
    raw.update(experiment='portable_G1_A_training', a=a.ravel().tolist(), b=b.ravel().tolist(),
               **{name: value.tolist() for name, value in meta.items()})
    for i, name in enumerate(('w1','b1','w2','b2','w3','b3','layerNormGamma','layerNormBeta')):
        raw[name] = (weights[i].T.ravel() if i in (0,2,4) else weights[i]).tolist()
    path.write_text(json.dumps(raw, indent=2, allow_nan=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=281)
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    parser.add_argument('--cuda-graph', action='store_true')
    args = parser.parse_args()
    if args.steps < 1 or (args.cuda_graph and args.device != 'cuda'):
        parser.error('steps must be positive; --cuda-graph requires --device cuda')
    if args.output.exists():
        parser.error('Output must be a new directory; existing runs are never overwritten')
    if args.train.resolve() == args.validation.resolve():
        parser.error('Training and validation must be different files')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    data, valid = load(args.train), load(args.validation)
    ids, vi = windows(data), windows(valid)[::8]
    if len(ids) < 128 or not len(vi):
        parser.error('Need at least 128 train windows and one validation window')
    meta, weights, k, scale = initialize(data, args.seed, args.device)
    p = Predictor(meta, weights, k, 'A', args.device)
    initial_score = score(p, valid, vi)
    contraction = 1.
    if not np.isfinite(initial_score):
        anchor = np.zeros_like(k); anchor[6:16, :10] = np.eye(10)
        for contraction in (.5, .25, .125, .0625, 0.):
            with torch.no_grad(): p.k.copy_(p.t(anchor+contraction*(k-anchor)))
            initial_score = score(p, valid, vi)
            if np.isfinite(initial_score): break
        else: raise RuntimeError('No feasible initialization')
    args.output.mkdir(parents=True)
    rng = np.random.default_rng(args.seed+510)
    driver_type = GraphTrainer if args.cuda_graph else EagerTrainer
    driver = driver_type(p, fixture(data, rng.choice(ids, 128, False)), scale)
    best = (initial_score, p.arrays(), 0)
    trace = []; accepted = 0; start = time.monotonic()
    for tick in range(1, args.steps+1):
        loss, ok = driver.update(fixture(data, rng.choice(ids, 128, False)), tick)
        accepted += int(ok)
        # Exact historical selection checkpoints: 0 and multiples of 100.
        if tick % 100 == 0:
            value = score(p, valid, vi)
            trace.append(dict(step=tick, validation8=value if np.isfinite(value) else None,
                              loss=loss if np.isfinite(loss) else None, accepted=accepted))
            print(json.dumps(trace[-1]), flush=True)
            if value < best[0]: best = (value, p.arrays(), tick)
    export(args.output/'operator.json', meta, best[1][1:], best[1][0])
    record = dict(seed=args.seed, steps=args.steps, accepted=accepted, best_step=best[2],
                  validation8=best[0], initial_validation8=initial_score, contraction=contraction,
                  trace=trace, elapsed_s=time.monotonic()-start, device=args.device,
                  cuda_graph=args.cuda_graph,
                  note='New training run; not a reproduction claim for historical closed-loop metrics',
                  data_sha256={name:hashlib.sha256(path.read_bytes()).hexdigest()
                               for name,path in [('train',args.train),('validation',args.validation)]})
    (args.output/'training.json').write_text(json.dumps(record,indent=2,allow_nan=False))
    print(json.dumps(record,indent=2))


if __name__ == '__main__':
    main()
