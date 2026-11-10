"""Build a live-shaped objective from a captured native configuration. No I/O to robot."""
import argparse,json
from pathlib import Path
import numpy as np
from finshybrid.deployment import ShadowCycle
from finshybrid.mpc import objective_batch,solve_reference

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solve-reference',action='store_true',help='Slow optional SciPy debug solve, not native SQP')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    source=json.loads((root/'tests/fixtures/native_objective.json').read_text())
    cycle=ShadowCycle();state=cycle.online.model.physical(np.array(source['z']))
    cycle.observe(0,state)
    # Captured nonzero bias is loaded ONLY to replay this historical target.
    # A live host obtains b from consecutive observe() calls instead.
    cycle.online.online.b=np.array(source['de83VelocityBias'])
    center=np.array(source['forceCenter']);half=np.array(source['forceHalf'])
    config=dict(force_min_n=center-half,force_max_n=center+half,
                allocation_basis=np.array(source['allocationBasis']).reshape(6,8),
                position_weight=source['positionWeight'],velocity_weight=source['velocityWeight'],effort_weight=source['effortWeight'])
    record=cycle.prepare_objective(config,np.array(source['referencePose']).reshape(8,6),
                                   np.array(source['referenceVelocity']).reshape(8,6),np.array(source['start']).reshape(8,8))
    out=dict(kind='captured objective shadow example',dispatch_permitted=False,
             objective=float(objective_batch(cycle.online.model,record,record['start'])[0]),
             captured_native_objective=source['startCost'])
    if args.solve_reference:
        def solver(model,r):
            result=solve_reference(model,r)
            return dict(success=bool(result.success),sequence=result.x.reshape(8,8))
        out['solve']=cycle.solve_shadow(record,solver)
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
