import numpy as np
import pytest
import aic.spatial_pipeline as spatial

def test_ema_is_used_and_shot_reset_clears_history(monkeypatch):
    points=iter([(.2,.5,1),(.8,.5,1),(.8,.5,1)])
    monkeypatch.setattr(spatial,'gradient_saliency_center',lambda image:next(points))
    path=spatial.SpatialPath('subject_proxy_smooth',[1,3],alpha=.25)
    black=np.zeros((90,180,3),np.uint8)
    first,reset=path.step(black);assert reset
    second,reset=path.step(black);assert not reset
    assert path.center[0]==pytest.approx(.35)
    third,reset=path.step(np.full_like(black,255));assert reset
    assert path.center[0]==pytest.approx(.8)
    assert first[0]<second[0]<third[0]

def test_dense_inference_observes_all_frames_and_emits_smoothed_crop(tmp_path,monkeypatch):
    import json
    import subprocess
    from aic.inference import run_inference
    from aic.contract import load_jsonl
    video=tmp_path/'one.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i',
        'color=black:size=80x40:rate=5:duration=1','-c:v','libx264','-threads','1',str(video)],check=True)
    index=tmp_path/'index.jsonl'
    index.write_text(json.dumps({'video_id':'one','video_path':str(video),'width':80,'height':40,
                                'frame_count':5,'targetRatioWH':[1,3]})+'\n')
    def observe(image):
        observe.i+=1
        return (.2 if observe.i==1 else .8),.5,1
    observe.i=0
    monkeypatch.setattr(spatial,'gradient_saliency_center',observe)
    output=tmp_path/'pred.jsonl'
    result=run_inference(index,output,dummy=True,spatial_mode='subject_proxy_smooth')
    assert result['spatial_diagnostics'][0]['observations']==5
    assert result['validation']['valid']
    crops=[p['bboxes'] for p in load_jsonl(output)[0]['predictions']]
    assert crops[0][0]<crops[1][0]<crops[2][0]

def test_detector_missing_is_not_silent_center_fallback(tmp_path):
    from aic.inference import run_inference
    from aic.contract import ContractError
    with pytest.raises(ContractError,match='detector weight'):
        run_inference(tmp_path/'index.jsonl',tmp_path/'out',dummy=True,spatial_mode='true_face')
