import importlib
import json
import sys
from pathlib import Path
import numpy as np
import pytest
from finshybrid.encoder import init_encoder, materialize
from finshybrid.predictor import Model
from finshybrid.multistep_bias import MultiStepBiasUpdater, PERIOD_NS

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def model(tmp_path):
    raw = json.loads((ROOT/'src/finshybrid/config/reference_metadata.json').read_text())
    x = np.random.default_rng(17).normal(size=(64,42))
    meta, weights = init_encoder({'x':x}, 281, 'G1')
    k = np.zeros((16,16)); k[6:,:10] = .5*np.eye(10)
    a,b = materialize(k,'G1')
    raw.update(a=a.ravel().tolist(),b=b.ravel().tolist(),**{n:v.tolist() for n,v in meta.items()})
    for i,n in enumerate(('w1','b1','w2','b2','w3','b3','layerNormGamma','layerNormBeta')):
        raw[n]=(weights[i].T.ravel() if i in (0,2,4) else weights[i]).tolist()
    p=tmp_path/'synthetic_operator.json';p.write_text(json.dumps(raw))
    return Model(p)

def test_explicit_model_required(monkeypatch):
    monkeypatch.delenv('FINSHYBRID_MODEL',raising=False)
    with pytest.raises(ValueError,match='Supply a model'): Model()

def test_batch_scalar_prediction_and_bias_units(model):
    from finshybrid.scales import VS
    state=np.r_[0.,-1.2,0.,0.,0.,0.,np.zeros(6)]
    z=model.lift(np.stack([state,state]))
    u=np.zeros((2,6));b=np.array([.01,-.02,.01,0.,0.,0.])
    out=model.step(z,u,b)
    assert np.allclose(out[0],model.step(z[:1],u[:1],b)[0],atol=1e-12,rtol=0)
    delta=model.physical(out)[:,6:]-model.physical(model.step(z,u))[:,6:]
    assert np.allclose(delta,b*VS,atol=1e-12,rtol=0)

def test_future_and_duplicate_intervals_rejected(model):
    updater=MultiStepBiasUpdater(model)
    state=np.r_[0.,-1.2,0.,0.,0.,0.,np.zeros(6)]
    args=(0,PERIOD_NS,state,np.zeros(6),state)
    with pytest.raises(ValueError,match='completed'): updater.update(*args,available_ns=PERIOD_NS-1)
    updater.update(*args,available_ns=PERIOD_NS)
    with pytest.raises(ValueError,match='Nonmonotonic'): updater.update(*args,available_ns=PERIOD_NS)

def test_hardware_import_closure():
    sys.path.insert(0,str(ROOT/'deployment/hardware'))
    for name in ('g1_mpc','multistep_bias','fossen_mpc','dynamics_thrust','export_dynamics_training','mpc_runtime'):
        importlib.import_module(name)
    for name in ('fossen_mpc_kernel.cpp','multistep_bias_kernel.cpp'):
        assert (ROOT/'deployment/hardware'/name).is_file()
