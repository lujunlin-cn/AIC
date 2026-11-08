"""Execute the hash-verified P1 spatial runtime on public videos and stills.

LIVE uses original decoded PTS, the original 4 Hz FrameBank, subject anchors,
center adapter, smoothing and QP. All public clip frames are selected; temporal
inference and human crop labels play no part in Position generation.
"""
from __future__ import annotations

import ast
from contextlib import nullcontext
import hashlib
import inspect
import io
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
from PIL import Image
import torch

from aic.scale_execution import fingerprint
from aic.scale_geometry import deployed_center


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,allow_nan=False,default=str)+'\n');tmp.replace(path)


class OriginalP1Projector:
    def __init__(self,backend,config,batch_size=1):
        import adaptive_crop, generation, frame_bank
        self.backend=backend;self.config=config;self.batch_size=batch_size
        self.subject_function=adaptive_crop.subject_anchor
        self.candidate_function=generation.candidate_anchor
        self.FrameBank=frame_bank.FrameBank
        tree=ast.parse(inspect.getsource(self.subject_function))
        self.subject_prompt=next(ast.literal_eval(node.value) for node in ast.walk(tree)
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='prompt' for t in node.targets))
        self.identity=fingerprint({'parent':config['parent']['identity_sha256'],
            'runtime_sources':config['runtime']['runtime_source_hashes'],
            'video_protocol':'full_public_clip_original_4Hz_bank_original_P1_qp015',
            'still_protocol':'single_image_original_subject_center_smooth_QP_export',
            'batch_size':batch_size,'subject_prompt':self.subject_prompt,
            'projector_sha256':file_sha(__file__),
            'spatial_entrypoint_sha256':file_sha(Path(config['runtime']['runtime_dir']).parent/'infer_candidates.py')})
        source=Path(config['runtime']['runtime_dir']).parent/'infer_candidates.py'
        self.spatial_source_sha256=file_sha(source)
        if self.spatial_source_sha256!=config['runtime']['spatial_entrypoint_sha256']:
            raise ValueError('original spatial entrypoint SHA mismatch')
        self.batch_parity=[];self.archives={}
        backend.model.gradient_checkpointing_disable()

    def generate(self,images,prompt,ratio=None):
        backend=self.backend;backend.model.eval()
        backend._activate(scale=False,ratio=ratio or [9,16])
        # Preserve original singleton numerics when selected. PEFT activation is
        # reapplied before each task, but neither P1 bank is ever optimized.
        if self.batch_size == 1:
            values=[]
            with torch.inference_mode():
                for image in images:
                    if ratio is None:
                        values.append(self.subject_function(backend.model,backend.processor,image)['raw'])
                    else:
                        values.append(self.candidate_function(backend.model,backend.processor,image,
                            image.shape[1],image.shape[0],{},ratio)['composition_raw'])
            return values
        messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt}]}]
        text=backend.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        results=[]
        for offset in range(0,len(images),self.batch_size):
            part=images[offset:offset+self.batch_size]
            tokenizer=backend.processor.tokenizer;old=tokenizer.padding_side;tokenizer.padding_side='left'
            try:
                values=backend.processor(text=[text]*len(part),images=[Image.fromarray(im) for im in part],
                    do_resize=False,padding=True,return_tensors='pt').to(backend.device)
            finally:tokenizer.padding_side=old
            context=backend.model.disable_adapter() if ratio is None else nullcontext()
            with context,torch.inference_mode():
                output=backend.model.generate(**values,do_sample=False,max_new_tokens=100,use_cache=True)
            results.extend(backend.processor.batch_decode(output[:,values.input_ids.shape[1]:],skip_special_tokens=True))
        return results

    @staticmethod
    def subject(raw):
        from adaptive_crop import parse_subject
        try:return {'raw':raw,'subject_box':parse_subject(raw),'error':None}
        except ValueError as exc:return {'raw':raw,'subject_box':None,'error':str(exc)}

    @staticmethod
    def candidate(raw,W,H,original,ratio):
        from generation import parse_center
        crop=parse_center(raw,W,H,ratio);valid=crop is not None
        if crop is None:
            width=min(W,H*ratio[0]/ratio[1]);crop=[(W-width)/2,(H-width*ratio[1]/ratio[0])/2,width]
        cx=(crop[0]+crop[2]/2)/W;cy=(crop[1]+crop[2]*ratio[1]/ratio[0]/2)/H
        return dict(original,subject_box=[cx,cy,cx,cy],composition_raw=raw,composition_valid=valid)

    def public_image(self,row):
        root=Path(self.config['runtime']['image_root'])
        if row.get('input_archive'):
            path=(root/row['input_archive']).resolve()
            if not path.is_relative_to(root.resolve()):raise ValueError('public archive path escapes image root')
            if path not in self.archives:self.archives[path]=ZipFile(path)
            payload=self.archives[path].read(row['input_image'])
        else:
            path=(root/row['input_image']).resolve()
            if not path.is_relative_to(root.resolve()):raise ValueError('public image path escapes image root')
            payload=path.read_bytes()
        if hashlib.sha256(payload).hexdigest()!=row['image_sha256']:raise ValueError('public image SHA mismatch')
        with Image.open(io.BytesIO(payload)) as im:image=im.convert('RGB')
        if row.get('transposed_geometry_augmentation'):image=image.transpose(Image.Transpose.TRANSPOSE)
        if list(image.size)!=row.get('media_size',[row['image_width'],row['image_height']]):
            raise ValueError('public image/view dimensions mismatch')
        W,H=image.size;side=self.config['runtime']['max_image_side']
        return np.asarray(image.resize((max(32,round(W*side/max(W,H)/32)*32),max(32,round(H*side/max(W,H)/32)*32)),
                                       resample=Image.Resampling.BICUBIC))

    def record(self,row,crop,**evidence):
        result={key:row[key] for key in ('dataset_source','source_id','image_id','split','target_ratio',
                'image_width','image_height','image_sha256')}
        result.update(center_xy=list(deployed_center(crop,row['target_ratio'])),p1_aic_bbox_xyw=crop,
            position_source='P1_frozen',position_kind='P1_final_deployed',
            parent_identity_sha256=self.config['parent']['identity_sha256'],position_protocol_sha256=self.identity,
            **evidence)
        return result

    def video(self,path,rows,cache_path):
        import av, adaptive_crop, generation, frame_bank
        from video import to_rgb
        path=Path(path)
        original_hash=file_sha(path)
        if any(row['source_video_sha256']!=original_hash for row in rows):raise ValueError('original source SHA mismatch')
        bank=self.FrameBank(path,side=640,rate=4)
        base_rows=[r for r in rows if not r.get('transposed_geometry_augmentation')]
        wanted={r['frame_id']:r for r in base_rows};roundtrips=0
        root=Path(self.config['runtime']['image_root'])
        with av.open(str(path)) as container:
            stream=container.streams.video[0];stream.codec_context.thread_count=2
            for i,frame in enumerate(container.decode(stream)):
                if i in wanted:
                    row=wanted[i];W,H=row['media_size'];rgb=to_rgb(frame,W,H,frame_id=i)
                    payload=(root/row['input_image']).read_bytes()
                    if hashlib.sha256(payload).hexdigest()!=row['image_sha256']:raise ValueError('cached LIVE image SHA mismatch')
                    with Image.open(io.BytesIO(payload)) as im:cached=np.asarray(im.convert('RGB'))
                    if not np.array_equal(rgb,cached):raise ValueError('cached LIVE frame does not match original frame ID pixels')
                    roundtrips+=1
                if i>=max(wanted):break
        if roundtrips!=len(wanted):raise ValueError('incomplete original frame/PTS recovery')
        from importlib.util import spec_from_file_location,module_from_spec
        source=Path(self.config['runtime']['runtime_dir']).parent/'infer_candidates.py'
        spec=spec_from_file_location('aic_original_p1_spatial',source);runtime=module_from_spec(spec);spec.loader.exec_module(runtime)
        records=[];details=[]
        for ratio in ([9,16],[16,9]):
            selected=[r for r in rows if r['target_ratio']==ratio]
            if not selected:continue
            transpose=selected[0].get('transposed_geometry_augmentation',False)
            frames=bank.frames.transpose(0,2,1,3).copy() if transpose else bank.frames
            meta=dict(bank.meta)
            if transpose:meta['width'],meta['height']=meta['height'],meta['width']
            if any([r['image_width'],r['image_height']] != [meta['width'],meta['height']] for r in selected):
                raise ValueError('original video geometry differs from public annotation coordinate system')
            view=SimpleNamespace(frames=frames,ids=bank.ids,times=bank.times,meta=meta)
            count=24;targets=np.linspace(0,max(0,meta['duration']-.001),count)
            indices=sorted({int(abs(bank.times-target).argmin()) for target in targets})
            images=[frames[i] for i in indices]
            subjects=self.generate(images,self.subject_prompt)
            candidates=self.generate(images,generation.prompt(ratio),ratio)
            if not any(x['target_ratio']==ratio for x in self.batch_parity):
                self.backend._activate(scale=False,ratio=ratio)
                with torch.inference_mode():
                    one=self.subject_function(self.backend.model,self.backend.processor,images[0])
                    two=self.candidate_function(self.backend.model,self.backend.processor,images[0],meta['width'],meta['height'],{},ratio)
                passed=one['raw']==subjects[0] and two['composition_raw']==candidates[0]
                evidence={'target_ratio':ratio,'batch_size':self.batch_size,'raw_greedy_match':passed,
                    'single_subject':one,'batched_subject':self.subject(subjects[0]),
                    'single_candidate':two,'batched_candidate':self.candidate(candidates[0],meta['width'],meta['height'],{},ratio)}
                self.batch_parity.append(evidence)
                atomic_json(cache_path.parent.parent/'batch_parity.json',self.batch_parity)
                if not passed:raise ValueError('batched P1 generation differs from original single-image greedy runtime')
            lookup={hashlib.sha256(images[j].tobytes()).hexdigest():(subjects[j],candidates[j]) for j in range(len(images))}
            def sub(model,processor,rgb):return self.subject(lookup[hashlib.sha256(rgb.tobytes()).hexdigest()][0])
            def cand(model,processor,rgb,W,H,original,ratio):return self.candidate(lookup[hashlib.sha256(rgb.tobytes()).hexdigest()][1],W,H,original,ratio)
            old=(adaptive_crop.subject_anchor,generation.candidate_anchor,frame_bank.FrameBank)
            adaptive_crop.subject_anchor=sub;generation.candidate_anchor=cand;frame_bank.FrameBank=lambda *args,**kwargs:view
            try:
                with torch.inference_mode():
                    outputs,cache,audits=runtime.spatial(self.backend.model,self.backend.processor,str(path),
                        {'video_id':selected[0]['source_id'],'targetRatioWH':ratio},meta,[[0.,meta['duration']]],0)
            finally:adaptive_crop.subject_anchor,generation.candidate_anchor,frame_bank.FrameBank=old
            predictions={p['frame']:p['bboxes'] for p in outputs['P1']['predictions']}
            for row in selected:
                records.append(self.record(row,predictions[row['frame_id']],frame_id=row['frame_id'],
                    frame_pts=meta['pts'][row['frame_id']],source_video_sha256=original_hash,
                    parent_execution_context='full_public_video_original_P1_spatial',qp_floor=.15,
                    selected_frames_sha256=fingerprint(cache['target_frames']),path_cache_sha256=fingerprint(cache),
                    spatial_entrypoint_sha256=self.spatial_source_sha256))
            details.append({'target_ratio':ratio,'original_runtime_cache':cache,'original_runtime_audits':audits.get('P1',{})})
        atomic_json(cache_path,{'position_protocol_sha256':self.identity,'source_video_sha256':original_hash,
            'records':records,'details':details,'label_pixel_roundtrips':roundtrips,'batch_parity':self.batch_parity})
        return records

    def still_batch(self,rows):
        from adaptive_crop import smooth_crops
        from clock_rule import adjusted
        from contracts import integer_predictions
        from generation import route,prompt,apply_vertical
        from recovery_path_projection import apply_events
        images=[self.public_image(row) for row in rows]
        subjects=self.generate(images,self.subject_prompt)
        proposals={}
        for ratio in ([9,16],[16,9]):
            indices=[i for i,r in enumerate(rows) if r['target_ratio']==ratio]
            values=self.generate([images[i] for i in indices],prompt(ratio),ratio) if indices else []
            proposals.update(zip(indices,values))
        results=[]
        for i,row in enumerate(rows):
            W,H,ratio=row['image_width'],row['image_height'],row['target_ratio']
            meta={'width':W,'height':H,'pts':[0.]};events=[[0.,1.]]
            anchors=[dict(self.subject(subjects[i]),time=0.,sample_time=0.,sample_frame_id=0,scene_cut=False)]
            proposed=[self.candidate(proposals[i],W,H,anchors[0],ratio)]
            base,_=smooth_crops(meta['pts'],events,adjusted(anchors,events),W,H,ratio,max_speed=.15)
            cand,_=smooth_crops(meta['pts'],events,adjusted(proposed,events),W,H,ratio,max_speed=.15)
            base=integer_predictions(base,W,H,ratio);cand=integer_predictions(cand,W,H,ratio)
            arm=route(W,H,ratio)
            if arm!='native':
                final,_=(apply_events if arm=='a' else apply_vertical)(meta,events,anchors,base,cand,qp_floor=.15)
            else:final=base
            results.append(self.record(row,final[0]['bboxes'],parent_execution_context='public_still_single_frame_original_P1_projection',
                parent_subject_raw=subjects[i],parent_candidate_raw=proposals[i],parent_route=arm,qp_floor=.15,
                selected_frames_sha256=fingerprint([0]),path_cache_sha256=fingerprint({'anchors':anchors,'proposed':proposed}),
                spatial_entrypoint_sha256=self.spatial_source_sha256))
        return results
