import copy, json, math, re
import numpy as np
from clock_rule import adjusted
from adaptive_crop import smooth_crops
from contracts import integer_predictions
from recovery_path_projection import apply_events as project

def prompt(ratio):
    return (f'Choose the composition center for a {ratio[0]}:{ratio[1]} crop of this image. '
            'Keep the important action and content visible. The crop window will be the largest '
            'rectangle of the requested aspect ratio that fits inside the original image. '
            'Choose its center, without zooming in. Return only JSON {"crop_center":[x,y]}. '
            'Both coordinates must be integers from 0 to 1000 over the complete image.')

def parse_center(raw, W, H, ratio):
    match = re.search(r'"crop_center"\s*:\s*', raw)
    if match is None or '{' not in raw[:match.start()]:
        return None
    try:
        pair, _ = json.JSONDecoder().raw_decode(raw[match.end():])
    except ValueError:
        return None
    if not isinstance(pair, list) or len(pair) != 2:
        return None
    if not all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1000 for v in pair):
        return None
    width = min(W, H * ratio[0] / ratio[1])
    height = width * ratio[1] / ratio[0]
    return [float(np.clip(pair[0]*W/1000-width/2, 0, W-width)),
            float(np.clip(pair[1]*H/1000-height/2, 0, H-height)), width]

def candidate_anchor(model, processor, rgb, W, H, original, ratio):
    import torch
    from PIL import Image
    messages = [{'role':'user','content':[{'type':'image'},
                {'type':'text','text':prompt(ratio)}]}]
    text = processor.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    batch = processor(text=[text], images=[Image.fromarray(rgb)],
        do_resize=False, return_tensors='pt').to('cuda:0')
    with torch.inference_mode():
        output = model.generate(**batch, do_sample=False, max_new_tokens=100, use_cache=True)
    raw = processor.batch_decode(output[:,batch.input_ids.shape[1]:],
                                 skip_special_tokens=True)[0]
    crop = parse_center(raw, W, H, ratio)
    valid = crop is not None
    if crop is None:
        width = min(W, H*ratio[0]/ratio[1])
        crop = [(W-width)/2, (H-width*ratio[1]/ratio[0])/2, width]
    cx = (crop[0]+crop[2]/2)/W
    cy = (crop[1]+crop[2]*ratio[1]/ratio[0]/2)/H
    return dict(original, subject_box=[cx,cy,cx,cy],
                composition_raw=raw, composition_valid=valid)

def route(W, H, ratio):
    if ratio[0]*16 == ratio[1]*9 and W > H:
        return 'a'
    if ratio[0]*9 == ratio[1]*16 and H-W*9/16 > 2:
        return 'b'
    return 'native'

def apply_vertical(meta,events,anchors,baseline,candidate,qp_floor=.15):
 assert meta['height']-meta['width']*9/16>2
 logical=dict(meta,width=meta['height'],height=meta['width'])
 nodes=[]
 for a in anchors:
  box=a.get('subject_box');nodes.append(dict(a,subject_box=None if box is None else [box[1],box[0],box[3],box[2]]))
 paths=[]
 for values in [baseline,candidate]:paths.append([dict(v,bboxes=[v['bboxes'][1],v['bboxes'][0],v['bboxes'][2]*9/16]) for v in values])
 assert [p['frame'] for p in baseline]==[p['frame'] for p in candidate]
 projected,audit=project(logical,events,nodes,*paths,qp_floor=qp_floor);out=copy.deepcopy(baseline)
 for a,b in zip(out,projected):a['bboxes'][1]=b['bboxes'][0]
 assert all(a['frame']==b['frame'] and a['bboxes'][0]==b['bboxes'][0] and a['bboxes'][2]==b['bboxes'][2] for a,b in zip(out,baseline))
 return out,audit
