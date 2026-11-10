"""Portable/new updater equivalence against the hardware implementation and log."""
import ast
import json
from pathlib import Path
import shutil
from collections import deque

import numpy as np
import pytest

from finshybrid import Model, MultiStepBiasUpdater
from finshybrid.geometry import rotation
from finshybrid.scales import VS
from finshybrid.multistep_bias import WindowPredictor, NativeWindowPredictor, wrapped

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'reproduction/multistep_depth_bias'


def fixture():
    return json.loads((DATA/'hardware_intervals.json').read_text())['transitions']


def update(updater, event):
    return updater.update(event['start_ns'], event['end_ns'], event['previous'],
                          event['mean_wrench_si'], event['current'],
                          available_ns=event['available_ns'])


@pytest.fixture(scope='module')
def model():
    return Model(DATA/'hardware_operator.json')


@pytest.fixture(scope='module')
def native_cache(tmp_path_factory):
    if not shutil.which('g++'):
        pytest.skip('Optional native equivalence needs g++')
    return tmp_path_factory.mktemp('multistep_native')


def test_window_rollout_matches_public_predict_and_keeps_latent(model):
    events = fixture()
    initial = np.array([e['previous'] for e in events[:3]])
    inputs = np.array([[e['mean_wrench_si'] for e in events[i:i+8]] for i in range(3)])
    predictor = WindowPredictor(model)
    predictor.prepare(initial, inputs)
    biases = np.array([[.01, -.02, .003, .01, -.01, .003], np.zeros(6)])
    predictions = predictor.predict(biases)
    for i, bias in enumerate(biases):
        for j in range(len(initial)):
            np.testing.assert_allclose(predictions[i, j], model.predict(initial[j], inputs[j], bias),
                                       atol=2e-12, rtol=0)


def test_native_rollout_matches_numpy(model, native_cache):
    events = fixture()
    initial = np.array([events[i]['previous'] for i in range(3)])
    inputs = np.array([[e['mean_wrench_si'] for e in events[i:i+8]] for i in range(3)])
    numpy = WindowPredictor(model)
    native = NativeWindowPredictor(model, native_cache)
    for predictor in (numpy, native):
        predictor.prepare(initial, inputs)
    biases = np.array([[.01, -.02, .003, .01, -.01, .003], np.zeros(6)])
    np.testing.assert_allclose(native.predict(biases), numpy.predict(biases), atol=2e-12, rtol=0)


@pytest.mark.parametrize('backend', ['numpy', 'native'])
def test_hardware_replay_matches_recorded_bias(model, backend, native_cache):
    updater = MultiStepBiasUpdater(model, backend=backend, cache_dir=native_cache)
    count = 0
    for i, event in enumerate(fixture()):
        old = updater.b.copy()
        info = update(updater, event)
        assert info['accepted'] == event['expected']['accepted']
        np.testing.assert_allclose(updater.b, event['expected']['bias'], atol=2e-9, rtol=0)
        assert np.linalg.norm(updater.b) <= .25+1e-12
        assert np.linalg.norm(updater.b-old) <= .01+1e-12
        if i < 19:
            np.testing.assert_array_equal(updater.b, 0)
        if info['accepted']:
            count += 1
            assert info['after'] <= info['before']+1e-14
            assert info['depth_after'] <= info['depth_before']+1e-12
            assert info['velocity_after'] <= 1.05*info['velocity_before']+1e-10
            assert info['tilt_after'] <= 1.05*info['tilt_before']+1e-10
    assert count > 0


def test_original_hardware_optimizer_matches_portable(model, native_cache):
    # Extract only the original class; never import its hardware dependency tree.
    tree = ast.parse((DATA/'source/multistep_bias.py').read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == 'MultiStepBiasUpdater')
    namespace = dict(np=np, deque=deque, rotation=rotation, VS=VS, wrapped=wrapped,
                     PERIOD_NS=100_000_000,
                     WindowPredictor=lambda plant: NativeWindowPredictor(plant, native_cache))
    exec(compile(ast.Module(body=[cls], type_ignores=[]), '<hardware-original>', 'exec'), namespace)
    original = namespace['MultiStepBiasUpdater'](model, depth_weight=.25)
    portable = MultiStepBiasUpdater(model, backend='native', cache_dir=native_cache)
    for event in fixture():
        a, b = update(original, event), update(portable, event)
        assert a == b
        np.testing.assert_array_equal(original.b, portable.b)


def test_depth_increment_is_invariant_to_common_origin_shift(model):
    updater = MultiStepBiasUpdater(model)
    for event in fixture()[:20]:
        update(updater, event)
    updater._prepare()
    prediction = updater.predictor.predict(updater.b)
    residual, parts = updater.residuals(prediction)
    updater.initial[:, 1] -= .25
    updater.truth[:, :, 1] -= .25
    prediction[:, :, :, 1] -= .25
    changed, changed_parts = updater.residuals(prediction)
    np.testing.assert_allclose(residual, changed, atol=1e-12, rtol=0)
    np.testing.assert_allclose(parts['depth'], changed_parts['depth'], atol=1e-12, rtol=0)


def test_gaps_pause_reset_and_rejected_times(model):
    updater = MultiStepBiasUpdater(model)
    events = fixture()
    for event in events[:10]:
        update(updater, event)
    update(updater, events[15])
    assert len(updater.history) == 1 and updater.tick == 1
    with pytest.raises(ValueError, match='Nonmonotonic'):
        update(updater, events[15])
    event = dict(events[16], available_ns=events[16]['end_ns']-1)
    with pytest.raises(ValueError, match='completed'):
        update(updater, event)
    assert len(updater.history) == 1
    updater.b[:] = [.01, -.02, 0, 0, 0, 0]
    old = updater.b.copy()
    updater.pause()
    assert not updater.history and updater.tick == 0
    np.testing.assert_array_equal(updater.b, old)
    updater.reset()
    np.testing.assert_array_equal(updater.b, 0)


def test_models_and_operators_remain_frozen(model, native_cache):
    before = [a.copy() for a in (model.A, model.B, model.C, model.D)]
    raw = json.dumps(model.raw, sort_keys=True)
    updater = MultiStepBiasUpdater(model, backend='native', cache_dir=native_cache)
    for event in fixture():
        update(updater, event)
    for actual, expected in zip((model.A, model.B, model.C, model.D), before):
        np.testing.assert_array_equal(actual, expected)
    assert raw == json.dumps(model.raw, sort_keys=True)


@pytest.mark.parametrize('kwargs', [dict(depth_weight=-1), dict(depth_scale_m=0),
                                   dict(anchors=0), dict(guard_relative=.2), dict(backend='bad')])
def test_invalid_settings(model, kwargs):
    with pytest.raises(ValueError):
        MultiStepBiasUpdater(model, **kwargs)
