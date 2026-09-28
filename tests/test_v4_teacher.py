import json
import numpy as np
from scripts.qwen_event_segments import parse_roles,event_mask,segments
from scripts.qwen_context_point import consumed,neighbours
from scripts.v4_e1_obs_eval import adapt_keys
from scripts.v4_e3_yth_eval import human_labels


def test_parse_roles_ok_and_repair():
 r,st=parse_roles('{"roles":{"3":"MAIN","4":"idle"}}',[3,4]);assert st=='ok' and r=={3:'MAIN',4:'IDLE'}
 r,st=parse_roles('{"roles":{"3":"MAIN","9":"JUNK"}}',[3,4]);assert st=='repaired' and r=={3:'MAIN',4:'UNSURE'}
 r,st=parse_roles('no json',[1]);assert st=='no_json' and r=={1:'UNSURE'}


def test_event_mask_restore_drop_guard():
 fc=np.repeat(np.arange(8),10);seg=segments(range(8))
 assert seg[0]==[0,1] and seg[3]==[6,7]
 rem,g=event_mask(fc,[7],{3:'MAIN',0:'IDLE'},seg);assert rem==[0,1] and not g
 rem,g=event_mask(fc,[7],{3:'UNSURE'},seg);assert rem==[7]
 rem,g=event_mask(fc,[7],{0:'JUNK',1:'JUNK',2:'IDLE'},seg);assert g and rem==[7]
 # TEMP guard tripped (5 of 8 cells) -> TEMP keeps all -> event default is no removal
 rem,g=event_mask(fc,[0,1,2,3,4],{3:'JUNK'},seg);assert rem==[6,7] and not g
 rem,g=event_mask(fc,[0,1,2,3,4],{0:'IDLE',1:'IDLE',2:'IDLE'},seg);assert g and rem==[]


def test_context_consumed_and_neighbours():
 keys=[0,15,30,45];face=np.array([True]*30+[False]*30);assert consumed(keys,60,face)==[False,False,True,True]
 shot=np.cumsum(np.array([True]+[False]*39+[True]+[False]*19))
 pre,post=neighbours(keys,1,shot,30.);assert pre==[0] and post==[30]
 pre,post=neighbours(keys,3,shot,30.);assert pre==[] and post==[]


def test_adapt_keys_rule():
 reset=np.zeros(40,bool);reset[0]=True
 k2=[0,10,20,30];k4=[0,5,10,15,20,25,30,35]
 p2=[[.5,.5,False],[.52,.5,False],[.9,.5,False],None]
 got=adapt_keys(k2,p2,k4,reset,0,.5)
 assert got==[15,35]  # 15: jump .38/.5 > .1; 35: held point missing; 5: .02/.5 < .1
 assert adapt_keys(k2,p2,k4,reset,0,0.)==[]


def test_qwen_centres_interp():
 from aic.max_window_path import qwen_centres,qwen_centres_interp
 reset=np.zeros(30,bool);reset[[0,20]]=True;raw=np.full(30,.9)
 keys=[0,10,20,25];pts=[[.2,.5,False],[.6,.5,False],[.1,.5,False],None]
 h,_=qwen_centres(pts,keys,reset,raw,0);c,src=qwen_centres_interp(pts,keys,reset,raw,0)
 assert np.allclose(c[:10],.2+.4*np.arange(10)/10)   # interpolated inside shot 1
 assert np.allclose(c[10:20],h[10:20])               # next keyframe after a cut: hold
 assert np.allclose(c[20:25],h[20:25])               # right end invalid: hold
 assert np.allclose(c[25:],raw[25:])                 # invalid keyframe: B0 fallback


def test_human_labels(tmp_path):
 p=tmp_path/'s.json';p.write_text(json.dumps({'segments_frames':[{'start_frame':0,'end_frame':10,'mturk_votes':0.},{'start_frame':5,'end_frame':15,'mturk_votes':1.5}]}))
 lab=human_labels(p,20);assert (lab[:5]==-1).all() and (lab[5:15]==1).all() and np.isnan(lab[15:]).all()


def test_drop_only_mask():
 from scripts.qwen_event_segments import drop_only_mask
 fc=np.repeat(np.arange(8),10);seg=segments(range(8))
 rem,g=drop_only_mask(fc,[7],{3:'MAIN',0:'IDLE',1:'SETUP'},seg);assert rem==[0,1,7] and not g


def test_crop_rerank_candidates_and_parse():
 from scripts.qwen_crop_rerank import candidates,parse_choice
 cs,c0=candidates(.5,.3);assert c0==.5 and len(cs)==5 and np.allclose(cs,[.26,.38,.5,.62,.74])
 cs,c0=candidates(.05,.3);assert c0==.15 and cs[0]==.15 and all(b-a>=.045-1e-9 for a,b in zip(cs,cs[1:]))
 cs,_=candidates(.5,1.);assert cs==[.5]
 assert parse_choice('{"choice":"B","uncertain":false}',3)==(1,'ok')
 assert parse_choice('{"choice":"D"}',3)[1]=='bad_choice' and parse_choice('{"choice":"A","uncertain":true}',3)[1]=='uncertain'
