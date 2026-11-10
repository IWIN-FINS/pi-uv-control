import importlib.util
from pathlib import Path
import numpy as np
import pytest
from finshybrid.de101 import DE101Model
from finshybrid import Model

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'reproduction/de101'
def test_frozen_de101_models_and_full_native_objectives():
    spec=importlib.util.spec_from_file_location('published_de101_verify',BUNDLE/'verify.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.verify(full_objectives=True)
    assert result['models']==15 and len(result['full_objectives'])==18

def test_explicit_family_and_only_bias_arm_can_adapt():
    for arm in ('F','Knu','Kx','H','H_bias'):
        path=BUNDLE/f'models/train281/{arm}/operator.json'
        with pytest.raises(ValueError,match='DE101Model'):Model(path)
        model=DE101Model(path);z=model.lift([0,-1.2,0,0,0,0,0,0,0,0,0,0])
        if arm!='H_bias':
            with pytest.raises(ValueError,match='Only H_bias'):model.step(z,np.zeros((1,6)),np.ones(6)*.01)
        else:assert np.isfinite(model.step(z,np.zeros((1,6)),np.ones(6)*.01)).all()
