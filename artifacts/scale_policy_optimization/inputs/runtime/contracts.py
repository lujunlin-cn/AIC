"""Strict competition IO. Errors are never converted into empty highlights."""
import json,math
from pathlib import Path

def finite(x): return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def load_index(path):
 text=Path(path).read_text().strip()
 rows=json.loads(text) if text.startswith('[') else [json.loads(x) for x in text.splitlines() if x.strip()]
 if not isinstance(rows,list) or not rows: raise ValueError('empty/invalid index')
 ids=set()
 for row in rows:
  vid=row.get('video_id'); ratio=row.get('targetRatioWH')
  if not isinstance(vid,(str,int)) or isinstance(vid,bool): raise ValueError('invalid video_id')
  key=str(vid)
  if key in ids: raise ValueError('duplicate video_id '+key)
  ids.add(key)
  if not isinstance(ratio,list) or len(ratio)!=2 or not all(finite(x) and x>0 for x in ratio):raise ValueError('invalid ratio')
 return rows

def parse_segments(text,duration):
 # Accept a JSON object embedded in prose, but never arbitrary number pairs.
 decoder=json.JSONDecoder();found=None
 for i,c in enumerate(text):
  if c!='{':continue
  try:o,_=decoder.raw_decode(text[i:])
  except json.JSONDecodeError:continue
  if isinstance(o,dict) and 'segments' in o:found=o['segments'];break
 if not isinstance(found,list):raise ValueError('missing segments JSON')
 out=[]
 for pair in found:
  if not isinstance(pair,list) or len(pair)!=2 or not all(finite(x) for x in pair):raise ValueError('invalid segment')
  a,b=pair
  if not 0<=a<b<=duration+0.05:raise ValueError('segment outside video or reversed')
  out.append([a,min(b,duration)])
 out.sort();merged=[]
 for a,b in out:
  if merged and a<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],b)
  else:merged.append([a,b])
 return merged

def crop_box(W,H,ratio,center=(0.5,0.5),scale=1.0):
 tw,th=ratio
 if not W>0 or not H>0 or not 0<scale<=1 or not all(finite(v) for v in (*ratio,*center,scale)):raise ValueError('invalid geometry')
 # Floor width, then derive exact evaluator height. Never round height separately.
 w=math.floor(min(W,H*tw/th)*scale)
 if w<1:raise ValueError('source too small for ratio')
 h=w*th/tw
 x=min(max(round(center[0]*W-w/2),0),math.floor(W-w))
 y=min(max(round(center[1]*H-h/2),0),math.floor(H-h))
 return [x,y,w]

def integer_predictions(predictions,W,H,ratio):
 """Export smooth internal coordinates as legal integer source-pixel boxes.

 Keep the largest feasible integer crop width and preserve its center as closely
 as rounding/bounds allow. Height remains derived from width, never serialized.
 """
 width=math.floor(min(W,H*ratio[0]/ratio[1]));height=width*ratio[1]/ratio[0]
 if width<1:raise ValueError('source too small for integer target-ratio crop')
 out=[]
 for r in predictions:
  x,y,w=r['bboxes'];h=w*ratio[1]/ratio[0]
  if not all(finite(v) for v in (x,y,w)):raise ValueError('nonfinite crop export')
  if min(x,y)<0 or w<=0 or x+w>W+1e-9 or y+h>H+1e-9:raise ValueError('invalid internal crop before export')
  px=min(max(round(x+w/2-width/2),0),W-width)
  py=min(max(round(y+h/2-height/2),0),math.floor(H-height))
  out.append(dict(frame=r['frame'],bboxes=[int(px),int(py),int(width)]))
 return out

def needs_subject_localization(W,H,ratio):
 """If the only feasible crop is the full image, subject inference cannot help."""
 width=crop_box(W,H,ratio)[2]
 return width!=W or width*ratio[1]/ratio[0]!=H

def validate_record(row,expected,n,W,H,require_integer=False):
 if str(row.get('video_id'))!=str(expected['video_id']) or row.get('targetRatioWH')!=expected['targetRatioWH']:raise ValueError('index mismatch')
 predictions=row.get('predictions');seen=set();last=-1
 if not isinstance(predictions,list):raise ValueError('predictions must be a list')
 tw,th=expected['targetRatioWH']
 for r in predictions:
  f=r.get('frame');b=r.get('bboxes')
  if not isinstance(f,int) or isinstance(f,bool) or not 0<=f<n or f in seen or f<=last:raise ValueError('bad/repeated/unsorted frame')
  seen.add(f);last=f
  if not isinstance(b,list) or len(b)!=3 or not all(finite(v) for v in b):raise ValueError('bad bbox')
  if require_integer and not all(type(v) is int for v in b):raise ValueError('submission bbox requires three integers')
  x,y,w=b;h=w*th/tw
  if min(x,y)<0 or w<=0 or x+w>W+1e-9 or y+h>H+1e-9:raise ValueError('bbox outside source')
 return len(predictions)

def frame_predictions(pts,segments,box):
 # Intervals are half-open [start,end); pts is original decoded presentation time.
 return [{'frame':i,'bboxes':list(box)} for i,t in enumerate(pts) if any(a<=t<b for a,b in segments)]

def interval_iou(pred,truth):
 def merge(a):
  out=[]
  for s,e in sorted(a):
   if out and s<=out[-1][1]:out[-1][1]=max(e,out[-1][1])
   else:out.append([s,e])
  return out
 p,g=merge(pred),merge(truth)
 inter=sum(max(0,min(b,d)-max(a,c)) for a,b in p for c,d in g)
 union=sum(b-a for a,b in p)+sum(b-a for a,b in g)-inter
 return inter/union if union else 1.0
