import importlib.util,json,csv
from pathlib import Path
import numpy as np
import pytest
from finshybrid.plot_trajectory import load_track,metrics,line_with_gaps,plot_tracks

ROOT=Path(__file__).resolve().parents[1]
def test_recorded_figure_eight_matches_all_ten_historical_results():
    import hashlib
    source=json.loads((ROOT/'examples/figure_eight/data/provenance.json').read_text())
    assert len(source['records'])==10
    for r in source['records']:
        path=ROOT/r['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']
        t=load_track(path);m=metrics(t);assert m['rows']==4000 and m['missing_pairs']==0
        assert abs(m['available_sample_rmse_m']-r['full_path_rmse_m'])<1e-12
        active=(t['t']>=20)&(t['t']<50)
        actual=float(np.sqrt(np.mean(np.sum((t['actual'][active]-t['reference'][active])**2,axis=1))))
        assert abs(actual-r['loaded_path_rmse_m'])<1e-12

def test_missing_data_and_clock_gaps_are_not_connected_or_filled(tmp_path):
    p=tmp_path/'real.csv';p.write_text('time_s,x_m,y_m,x_ref_m,y_ref_m\n0,0,0,0,0\n0.1,,,0,0\n0.2,1,0,0,0\n1.2,2,0,0,0\n')
    track=load_track(p);m=metrics(track);assert m['valid_pairs']==3 and m['missing_pairs']==1
    assert abs(m['available_sample_rmse_m']-np.sqrt(5/3))<1e-14
    xy,gaps,_=line_with_gaps(track['t'],track['actual']);assert gaps==1 and np.isnan(xy).all(1).sum()==2
    p.write_text('time_s,x_m,y_m,x_ref_m,y_ref_m\n0,0,0,0,0\n0,1,0,0,0\n')
    with pytest.raises(ValueError,match='strictly increasing'):load_track(p)

def test_unity_xz_adapter_and_portable_renderer(tmp_path):
    pytest.importorskip('matplotlib')
    p=tmp_path/'unity.csv';p.write_text('phase,phase_time_s,pose_0,pose_2,reference_pose_0,reference_pose_2\nBaselineMpc,0,99,99,99,99\nKoopmanMpc,0,0,0,0,0\nKoopmanMpc,0.1,1,2,1,2\n')
    t=load_track(p,'unity');assert len(t['t'])==2 and np.array_equal(t['actual'][1],[1,2])
    out=tmp_path/'plot.png';s=plot_tracks([p],['test'],out,format='unity',frame='Unity world XZ')
    assert out.exists() and out.with_suffix('.json').exists() and s['reference_curves']==1
    with pytest.raises(FileExistsError):plot_tracks([p],['test'],out,format='unity')
