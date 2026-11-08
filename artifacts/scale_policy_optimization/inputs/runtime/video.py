"""Bounded frame sampling with original frame IDs and presentation timestamps."""
import av,numpy as np
from pathlib import Path


COLOR_COMPAT_EVENTS=[]
def to_rgb(frame,width,height,frame_id=None):
 original_trc=int(frame.color_trc)
 try:
  return frame.reformat(width=width,height=height,format='rgb24').to_ndarray()
 except av.error.OSError as exc:
  if exc.errno!=95 or original_trc not in (9,10):raise
  # Metadata compatibility only; decoded planes, dimensions and PTS unchanged.
  source_pts=frame.pts
  try:
   frame.color_trc=2
   rgb=frame.reformat(width=width,height=height,format='rgb24',threads=2).to_ndarray()
  finally:frame.color_trc=original_trc
  assert frame.pts==source_pts
  COLOR_COMPAT_EVENTS.append(dict(frame_id=frame_id,source_pts=source_pts,source_trc=original_trc,used_trc=2,output_width=width,output_height=height,reason='errno95LOG/LOG_SQRTswscalecompatibility'))
  return rgb

def scan(path,max_duration=None,index_fps=None):
 if index_fps is not None and (not np.isfinite(index_fps) or index_fps<=0):raise ValueError('invalid explicit frame-index timebase')
 pts=[];W=H=None;origin=None;previous_source=None;source_regressions=0
 with av.open(str(path)) as container:
  stream=container.streams.video[0];stream.thread_type='SLICE';stream.codec_context.thread_count=2
  for i,frame in enumerate(container.decode(stream)):
   if frame.pts is None:raise ValueError('frame lacks PTS')
   source_t=float(frame.pts*frame.time_base)
   if previous_source is not None and source_t<previous_source:source_regressions+=1
   previous_source=source_t;t=source_t
   if origin is None:origin=t
   t=i/index_fps if index_fps is not None else t-origin
   if max_duration is not None and t>=max_duration:break
   if pts and t<pts[-1]:raise ValueError('nonmonotonic PTS')
   if W is not None and (frame.width,frame.height)!=(W,H):raise ValueError('variable frame dimensions')
   W,H=frame.width,frame.height;pts.append(t)
  fps=float(stream.average_rate) if stream.average_rate else None
  stream_end=float(stream.duration*stream.time_base) if stream.duration else None
 if len(pts)<2:raise ValueError('insufficient decoded frames')
 # Clip at 30 seconds for original DiDeMo, whose annotations only cover first 30s.
 delta=float(np.median(np.diff(pts)));duration=pts[-1]+delta
 if max_duration is not None:duration=min(duration,max_duration)
 return dict(path=str(path),pts=pts,width=W,height=H,num_frames=len(pts),duration=duration,average_fps=fps,stream_duration=stream_end,pts_origin=origin,index_fps=index_fps,source_pts_regressions=source_regressions)

def sample(path,nframes=16,max_duration=None,side=160,start=0.0,end=None,index_fps=None):
 meta=scan(path,max_duration,index_fps=index_fps);pts=np.array(meta['pts']);stop=min(end or meta['duration'],meta['duration'])
 ids=np.flatnonzero((pts>=start)&(pts<stop))
 if len(ids)==0:raise ValueError('empty window')
 targets=np.linspace(pts[ids[0]],pts[ids[-1]],nframes)
 selected=np.array([ids[np.argmin(abs(pts[ids]-t))] for t in targets]);want=set(selected.tolist());frames={}
 with av.open(str(path)) as container:
  stream=container.streams.video[0];stream.codec_context.thread_count=2
  for i,frame in enumerate(container.decode(stream)):
   if i in want:
    # Approximate aspect ratio on model's 32-pixel grid, bounded by side.
    W,H=meta['width'],meta['height'];scale=side/max(W,H)
    w=max(32,round(W*scale/32)*32);h=max(32,round(H*scale/32)*32)
    frames[i]=to_rgb(frame,w,h,frame_id=i)
   if i>=selected[-1]:break
 if len(frames)!=len(want):raise ValueError('decode failed while sampling')
 meta.update(sample_frame_ids=selected.tolist(),sample_pts=pts[selected].tolist(),window_start=start,window_end=stop)
 # Qwen processor computes timestamp = index / fps. Use a millisecond clock
 # for its metadata only; retain actual decoded IDs/PTS separately above.
 vm=dict(total_num_frames=max(1,round(meta['duration']*1000)),fps=1000.0,frames_indices=[round((x-start)*1000) for x in pts[selected]],duration=stop-start,video_backend='pyav_pts_ms')
 return np.stack([frames[i] for i in selected]),vm,meta
