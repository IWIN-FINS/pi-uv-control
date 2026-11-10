import json
from pathlib import Path
import numpy as np
import pytest
from finshybrid import Model, BiasUpdater
from finshybrid.controller import OnlinePredictor
from finshybrid.scales import PC, VS
from finshybrid.mpc import objective_batch, central_gradient

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('seed',[281,282,283])
def test_archived_encoder_and_step(seed):
    model=Model(seed=seed)
    assets=ROOT/f'src/finshybrid/assets/train{seed}'
    for case in json.loads((assets/'encoder_cases.json').read_text())['cases']:
        np.testing.assert_allclose(model.lift(np.r_[case['p'],case['v']])[0],case['z'],rtol=0,atol=3e-13)
    for case in json.loads((assets/'step_cases.json').read_text())['cases']:
        np.testing.assert_allclose(model.step(case['z'],case['u'])[0],case['next'],rtol=0,atol=3e-13)

def test_bias_independent_history():
    fixture=json.loads((ROOT/'tests/fixtures/bias_replay.json').read_text())
    online=BiasUpdater()
    for y, expected in zip(fixture['innovations'],fixture['expected']):
        previous=online.b.copy(); r=online.update(y)
        np.testing.assert_allclose(r['b'],expected['b'],rtol=0,atol=2e-14)
        assert r['accepted']==expected['accepted']
        assert r['after'] <= r['before']+1e-14
        assert np.linalg.norm(r['b'])<=.25+1e-14
        assert np.linalg.norm(r['b']-previous)<=.01+1e-14

def test_nonzero_bias_full_mpc_objective_gradient():
    record=json.loads((ROOT/'tests/fixtures/native_objective.json').read_text())
    model=Model()
    assert np.linalg.norm(record['de83VelocityBias'])>0
    value=objective_batch(model,record,record['start'])[0]
    assert abs(value-record['startCost'])/max(1,abs(value))<3e-6
    actual=np.array(record['startGradient']); calculated=central_gradient(model,record,record['start'])
    assert np.max(abs(actual-calculated)/np.maximum(1,np.maximum(abs(actual),abs(calculated))))<5e-5

def test_causal_input_and_zero_online_baseline():
    model=Model(); controller=OnlinePredictor(model)
    state=np.r_[PC,np.zeros(6)]; wrench=np.zeros(6)
    assert controller.observe(state) is None
    for tick in range(35):
        observed=model.predict(state,wrench[None])[0]; observed[6:]+=VS*.02
        report=controller.observe(observed,wrench)
        if tick<19: assert np.all(report['b']==0)
        state=observed
    b=controller.online.b.copy()
    first=controller.predict(np.zeros((8,6)))
    np.testing.assert_array_equal(controller.online.b,b)
    np.testing.assert_allclose(first,model.predict(state,np.zeros((8,6)),b),atol=0,rtol=0)
    assert np.all(b>0)
    controller.reset(); assert controller.previous is None and np.all(controller.online.b==0)

def test_invalid_observation_preserves_online_state():
    online=BiasUpdater()
    with pytest.raises(ValueError): online.update([np.nan]*6)
    assert online.tick==0

def test_torch_eight_step_and_control_gradient():
    torch=pytest.importorskip('torch')
    from finshybrid.torch_model import Predictor
    from finshybrid.encoder import unpack
    from finshybrid.train import EagerTrainer
    model=Model(); meta,w,k,_,_=unpack(model.raw)
    p=Predictor(meta,w,np.r_[k,np.zeros((6,16))],'A','cpu')
    rng=np.random.default_rng(281)
    state=np.r_[PC,np.zeros(6)];z=model.lift(state)
    x=np.c_[z[:,:36],np.zeros((1,6))];u=rng.normal(size=(8,1,6))*.01
    original_z=z.copy();values=[]
    for uk in u: z=model.step(z,uk);values.append(z.copy())
    ut=p.t(u).requires_grad_(); actual=p.rollout(p.t(x),ut)
    np.testing.assert_allclose(actual.detach().numpy(),values,atol=2e-12,rtol=0)
    actual[-1,0,1:13].square().sum().backward()
    eps=1e-6;up=u.copy();um=u.copy();up[0,0,0]+=eps;um[0,0,0]-=eps
    def cost(inputs):
        s=original_z.copy()
        for uk in inputs:s=model.step(s,uk)
        return np.sum(s[0,1:13]**2)
    np.testing.assert_allclose(ut.grad[0,0,0].item(),(cost(up)-cost(um))/(2*eps),rtol=2e-5,atol=2e-7)
    targets=np.concatenate((np.array(values)[:,:,:36],np.zeros((8,1,6))),axis=-1)
    batch=(x,u,targets)
    driver=EagerTrainer(p,batch,np.ones(10)); loss,accepted=driver.update(batch,1)
    assert np.isfinite(loss) and isinstance(accepted,bool)
