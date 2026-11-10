"""Run a synthetic interface example or a small completed-interval hardware replay."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .predictor import Model
from .multistep_bias import MultiStepBiasUpdater, PERIOD_NS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--replay', type=Path)
    parser.add_argument('--backend', choices=['numpy', 'native'], default='numpy')
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.replay and args.model is None:
        parser.error('--replay requires --model with the exact frozen checkpoint')
    model = Model(args.model)
    updater = MultiStepBiasUpdater(model, depth_weight=.25,
                                  backend=args.backend, cache_dir=args.cache_dir)
    if args.replay:
        data = json.loads(args.replay.read_text())
        if data['model_sha256'] != hashlib.sha256(args.model.read_bytes()).hexdigest():
            raise ValueError('Replay requires the exact frozen hardware checkpoint')
        events = data['transitions']
        kind = 'completed_hardware_interval_replay'
    else:
        # A known synthetic velocity increment demonstrates the API only.
        state = np.r_[0., -.6, 0., 0., 2.8, 0., np.zeros(6)]
        truth_bias = np.array([.002, -.004, .001, .002, -.003, .001])
        events = []
        for k in range(30):
            wrench = np.zeros(6)
            current = model.predict(state, wrench[None], truth_bias)[0]
            events.append(dict(start_ns=k*PERIOD_NS, end_ns=(k+1)*PERIOD_NS,
                               available_ns=(k+1)*PERIOD_NS, previous=state.tolist(),
                               current=current.tolist(), mean_wrench_si=wrench.tolist()))
            state = current
        kind = 'synthetic_interface_example_not_closed_loop_evidence'
    reports = []
    for event in events:
        info = updater.update(event['start_ns'], event['end_ns'], event['previous'],
                              event['mean_wrench_si'], event['current'],
                              available_ns=event['available_ns'])
        reports.append(dict(info, bias=updater.b.tolist()))
    summary = dict(kind=kind, backend=args.backend, depth_weight=.25,
                   intervals=len(events), accepted=updater.accepted,
                   rejected=updater.rejected, final_bias=updater.b.tolist(), reports=reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='reports'}, indent=2))


if __name__ == '__main__':
    main()
