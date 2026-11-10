import json
from pathlib import Path
import numpy as np
import pytest
from finshybrid.deployment import ShadowCycle
from finshybrid.replay import replay
from finshybrid.mpc import objective_batch
from finshybrid.scales import PC

ROOT=Path(__file__).resolve().parents[1]

def test_native_completed_observation_replay():
    result=replay(ROOT/'examples/closed_loop/de92_replay/observations.jsonl')
    assert result['samples']==81
    assert max(r.get('bias_max_error',0) for r in result['records'])<2e-8
    assert max(r.get('baseline_velocity_max_error',0) for r in result['records'])<2e-8
    assert np.linalg.norm(result['records'][-1]['bias'])>0

def test_gap_duplicate_and_missing_input_do_not_advance_state():
    host=ShadowCycle();x=np.r_[PC,np.zeros(6)];host.observe(0,x)
    for t,w in [(0,np.zeros(6)),(.2,np.zeros(6)),(.1,None)]:
        with pytest.raises(ValueError):host.observe(t,x,w)
        assert host.timestamp==0 and host.online.online.tick==0
        np.testing.assert_array_equal(host.online.previous,x)

def test_live_record_matches_captured_native_objective():
    from finshybrid import Model
    fixture=json.loads((ROOT/'tests/fixtures/native_objective.json').read_text())
    host=ShadowCycle(Model());host.observe(0,host.online.model.physical(np.array(fixture['z'])))
    host.online.online.b=np.array(fixture['de83VelocityBias'])
    center=np.array(fixture['forceCenter']);half=np.array(fixture['forceHalf'])
    config=dict(force_min_n=center-half,force_max_n=center+half,
                allocation_basis=np.array(fixture['allocationBasis']).reshape(6,8),
                position_weight=fixture['positionWeight'],velocity_weight=fixture['velocityWeight'],effort_weight=fixture['effortWeight'])
    record=host.prepare_objective(config,np.array(fixture['referencePose']).reshape(8,6),
                                  np.array(fixture['referenceVelocity']).reshape(8,6))
    # Current observation must re-encode the same physical state; native z contains this encoding.
    np.testing.assert_allclose(record['z'],fixture['z'],atol=3e-7,rtol=0)
    cost=objective_batch(host.online.model,record,fixture['start'])[0]
    assert abs(cost-fixture['startCost'])/max(1,abs(fixture['startCost']))<3e-6
    original=host.online.online.b.copy()
    def solver(model,r):
        assert not r['de83VelocityBias'].flags.writeable
        return dict(success=True,sequence=np.zeros((8,8)))
    result=host.solve_shadow(record,solver,deadline_s=10)
    assert result['status']=='candidate_only' and not result['dispatch_permitted']
    np.testing.assert_array_equal(original,host.online.online.b)
    assert host.solve_shadow(record,lambda m,r:dict(success=False,sequence=np.zeros((8,8))))['candidate_force_n'] is None
    assert host.solve_shadow(record,lambda m,r:dict(success=True,sequence=np.full((8,8),np.nan)))['candidate_force_n'] is None
