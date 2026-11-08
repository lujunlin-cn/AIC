"""CPU-only full RetargetVid media/annotation audit, with immutable exposure.

One actual DenseCropAdapter RGB read per source also captures its complete
display-order PTS timeline. Two ratios and six raters share that single decode.
No averaged box becomes ground truth. No training split is created.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import inspect
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .common import read_jsonl, write_json, write_jsonl
from .schema import annotation_record, media_record
from .holdout import project_split
from .ids import dhf1k_id
from .media import load_timeline, save_timeline, read_frames
from .adapters.base import load_video_frames
from .adapters.spatial import DenseCropAdapter, _read_ltrb, ltrb_to_xywh
from .retargetvid_archive import RETARGET_IDS, digest, ensure_selected_media
from . import window_scorer as ws

RATIOS = ("1-3", "3-1")
RATERS = tuple(range(1, 7))
AUDIT_VERSION = "retargetvid_full_media_v1_exposed_only"


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def summary(values):
    a=np.asarray(values,float);a=a[np.isfinite(a)]
    if not len(a):return {"count":0,"mean":None,"median":None,"p5":None,"p95":None}
    return {"count":len(a),"mean":float(a.mean()),"median":float(np.median(a)),"p5":float(np.quantile(a,.05)),"p95":float(np.quantile(a,.95)),"min":float(a.min()),"max":float(a.max())}


def bootstrap_sources(values,seed=20261108,repeats=2000):
    a=np.asarray(values,float)
    if not len(a) or not np.isfinite(a).all():return {"sources":len(a),"mean":None,"bootstrap_ci95":None}
    rng=np.random.default_rng(seed)
    boot=np.array([a[rng.integers(0,len(a),len(a))].mean() for _ in range(repeats)])
    return {"sources":len(a),"mean":float(a.mean()),"median":float(np.median(a)),"p5":float(np.quantile(a,.05)),"p95":float(np.quantile(a,.95)),
            "bootstrap_ci95":np.quantile(boot,[.025,.975]).tolist() if len(a)>1 else None,"resampling_unit":"one_independent_source_video_not_frame_or_rater_pair"}


def protected_sources(evidence):
    paths={"release":Path(evidence)/"reports/releases/spatial_crop_v1.RELEASE.json",
           "quality":Path(evidence)/"reports/releases/spatial_crop_v1.quality.json",
           "ledger":Path(evidence)/"configs/holdout_exposure_v1.json"}
    values={k:json.loads(p.read_text()) for k,p in paths.items()}
    hashes={k:digest(p) for k,p in paths.items()}
    if hashes["ledger"]!=values["release"]["holdout_ledger_sha256"]:
        raise ValueError("original ledger differs from immutable spatial_crop_v1 identity")
    q=values["quality"]["counts"]["RetargetVid"]
    if q["per_split"]["exposed_eval"]["source_videos"]!=200 or any(q["per_split"][s]["source_videos"] for s in ("train","dev","confirmation")):
        raise ValueError("RetargetVid exposure proof does not cover all 200 sources")
    expected={dhf1k_id(n) for n in RETARGET_IDS}
    if not expected<=set(values["ledger"]["groups"]):raise ValueError("ledger missing RetargetVid source group")
    by_group=defaultdict(list)
    for r in values["ledger"]["exposures"]:
        if r["group_id"] in expected:by_group[r["group_id"]].append(r)
    for g in expected:
        if not {e["role"] for e in by_group[g]} & {"selection","confirmation","diagnostic","in_progress_unregistered"}:
            raise ValueError("a source has no blocking exposure evidence")
    return paths,hashes,by_group


def prepare_annotations(root):
    root=Path(root);expected={f"{n:03d}_{ratio}.txt" for n in RETARGET_IDS for ratio in RATIOS};files={};archive_hashes={}
    for ann in RATERS:
        archive=root/"annotations"/f"annotator_{ann}.zip"
        archive_hashes[archive.name]=digest(archive)
        with ZipFile(archive) as z:
            names=[n for n in z.namelist() if n.endswith(".txt")]
            if len(names)!=400 or {Path(n).name for n in names}!=expected:
                raise ValueError("incomplete/duplicate RetargetVid annotation archive")
            for name in names:
                if Path(name).is_absolute() or ".." in Path(name).parts or Path(name).parent.name!=f"annotator_{ann}":
                    raise ValueError("annotation archive path/annotator mismatch")
                payload=z.read(name);p=root/"annotations_all"/f"annotator_{ann}"/Path(name).name
                if p.exists():
                    if p.read_bytes()!=payload:raise ValueError("existing raw annotation differs; never overwrite it")
                else:
                    p.parent.mkdir(parents=True,exist_ok=True)
                    with p.open("xb") as f:f.write(payload)
                n,ratio=p.stem.split("_");files[(int(n),ratio,ann)]=p
    if len(files)!=2400:raise ValueError("expected exactly 2400 distinct annotator/ratio/source files")
    return files,archive_hashes


def read_annotations(files):
    rows={};hashes={};errors=[]
    for key,p in sorted(files.items()):
        a=_read_ltrb(str(p))
        if a.ndim!=2 or a.shape[1]!=4 or not len(a) or not np.isfinite(a).all() or not np.equal(a,np.rint(a)).all():
            errors.append({"file":str(p),"reason":"not a nonempty finite integer Nx4 LTRB annotation"});continue
        rows[key]=a;hashes[str(p)]=digest(p)
    if errors:raise ValueError("raw annotations malformed: "+json.dumps(errors[:3]))
    return rows,hashes


def probe_header(path):
    import av
    with av.open(str(path)) as container:
        s=container.streams.video[0];ctx=s.codec_context
        rate=s.average_rate or s.guessed_rate
        rotation=float(s.metadata.get("rotate",0) or 0)
        return {"width":s.width,"height":s.height,"fps":float(rate) if rate else None,
            "fps_fraction":str(rate) if rate else None,"container_nb_frames":s.frames or None,
            "container_duration_s":float(s.duration*s.time_base) if s.duration is not None else (container.duration/av.time_base if container.duration is not None else None),
            "time_base":str(s.time_base),"rotation":rotation,"codec":ctx.name,
            "has_b_frames":bool(ctx.has_b_frames),"has_audio":bool(container.streams.audio),
            "metadata_source":"PyAV headers, no media decode and no ffprobe dependency"}


class DecodeTrace:
    """Constant-memory pixels, O(T) PTS; observe the Adapter's sole decode."""
    def __init__(self,expected_frames):
        self.pts=[];self.shapes=Counter();self.probes={};self.time_base=None
        self.want={0,expected_frames//2,max(0,expected_frames-1)}

    def __call__(self,index,frame,time_base):
        if index!=len(self.pts):raise ValueError("decode ordinal was reordered/skipped")
        if self.time_base is None:self.time_base=Fraction(time_base)
        if Fraction(time_base)!=self.time_base:raise ValueError("stream time base changed")
        self.pts.append(None if frame.pts is None else int(frame.pts));self.shapes[(frame.width,frame.height)]+=1
        if index in self.want:
            rgb=frame.to_ndarray(format="rgb24")
            self.probes[str(index)]={"rgb_sha256":hashlib.sha256(rgb.tobytes()).hexdigest(),"width":frame.width,"height":frame.height,
                                     "pts":None if frame.pts is None else int(frame.pts)}

    def finish(self):
        if not self.pts:raise ValueError("no decoded video frames")
        if any(p is None for p in self.pts):raise ValueError("missing real PTS; refuse fabricated frame clock")
        pts=np.asarray(self.pts,np.int64);seconds=(pts-pts[0]).astype(np.float64)*float(self.time_base)
        dt=np.diff(seconds)
        if len(dt) and not (dt>0).all():raise ValueError("non-increasing decoded presentation PTS; no timestamp sorting allowed")
        if len(self.shapes)!=1:raise ValueError("coded frame resolution changes")
        return {"pts":pts,"time_s":seconds,"time_base":str(self.time_base),"method":"decode",
            "frame_count":len(pts),"duration_s":float(seconds[-1]+(np.median(dt) if len(dt) else 0)),
            "fps_measured":float(1/np.median(dt)) if len(dt) else None,"non_increasing":int((dt<=0).sum()),
            "first_time_zero":float(seconds[0])==0.,"missing_pts":0,"frame_dimensions":[list(s) for s in self.shapes],"frame_probes":self.probes,
            "order":"0-based enumerate(PyAV container.decode(video_stream)); not packet order and never sorted"}


def decode_contract():
    import av
    return canonical({"version":"adapter_one_full_decode_v1","av":av.__version__,
        "code":[inspect.getsource(x) for x in (DecodeTrace,read_frames,load_video_frames,DenseCropAdapter.__getitem__)]})


def snapshot_working_registry(registry,output):
    result={}
    for ds in ("DHF1K","RetargetVid"):
        for name in ("media.jsonl","annotations.jsonl","dataset_card.json"):
            p=Path(registry)/ds/"processed"/name
            if p.exists():
                data=p.read_bytes();h=hashlib.sha256(data).hexdigest()
                backup=Path(output)/"registry_before"/ds/(name+"."+h)
                backup.parent.mkdir(parents=True,exist_ok=True)
                if not backup.exists():backup.write_bytes(data)
                result[str(p)]=h
    return result


def merge_records(path,rows):
    previous={r["uid"]:r for r in read_jsonl(path)}
    for r in rows:
        previous[r["uid"]]={**previous.get(r["uid"],{}),**r}
    write_jsonl(path,[previous[k] for k in sorted(previous)])


def media_rows(inventory,by_group):
    dhf=[];rv=[]
    for n,item in sorted(inventory.items()):
        uid=dhf1k_id(n);official="train" if n<=600 else "val"
        split,reason=project_split(official,"publisher DHF1K split inherited; exposure has priority",by_group[uid])
        if split not in {"holdout_dev_exposed","holdout_confirm_exposed"}:raise ValueError("exposed source would enter a fresh/train partition")
        m=media_record(uid=uid,dataset="DHF1K",release="existing author video.rar, selected RetargetVid 200 sources",
            item_id=f"{n:03d}",source_video_id=uid,source_platform="dataset_native",group_id=uid,
            official_split=official,aic_split=official,aic_split_reason="publisher split only, not fresh project split",
            project_split=split,project_split_reason=reason,exposure_roles=sorted({e["role"] for e in by_group[uid]}),
            media_kind="video_file",media_path=item["path"],file_size=item["file_size"],sha256=item["sha256"],
            download_status="verified" if item.get("decode_verified") else "referenced_existing",
            preprocess_status="timeline_ok" if item.get("decode_verified") else "probed",
            width=item["width"],height=item["height"],fps=item["fps"],frame_count=item["frame_count"],duration_s=item.get("duration_s"),
            rotation=item["rotation"],codec=item["codec"],has_audio=item["has_audio"],timeline_path=item.get("timeline_path"),
            timeline_method="decode" if item.get("decode_verified") else None,
            notes="Shared physical AVI with RetargetVid; no saliency/fixation annotations fabricated")
        m.update(release_split="exposed_eval",training_eligible=False,fresh_validation_eligible=False)
        dhf.append(m)
        r=dict(m,uid=f"retargetvid:{n:03d}",dataset="RetargetVid",release="original 6-rater RetargetVid annotation archive, media-only rebind",
               media_reused_from=uid,annotation_types=["crop_box_dense"])
        rv.append(r)
    return dhf,rv


def annotation_rows(files,raw,inventory):
    result=[]
    for (n,ratio,ann),p in sorted(files.items()):
        result.append(annotation_record(uid=f"retargetvid:{n:03d}:{ratio}:a{ann}",dataset="RetargetVid",media_uid=f"retargetvid:{n:03d}",
            annotation_type="crop_box_dense",annotation_source="human",annotator=f"annotator_{ann}",variant=ratio,
            annotation_path=str(p),annotation_format="original integer left,top,right,bottom, one line per decoded frame",
            coverage={"annotated_frames":len(raw[n,ratio,ann]),"video_frames":inventory[n]["frame_count"],"fraction":len(raw[n,ratio,ann])/inventory[n]["frame_count"]},
            label_semantics="independent human crop, never average-box pseudo-GT",coord_format="raw zero-based LTRB; retained byte-identically",
            coord_target="xywh_halfopen_v1: [l,t,r-l,b-t], no clipping and no +1",
            time_reference="original line i+1 -> displayed decoded frame i, 0-based; real PTS timeline",
            valid_mask="line exists and audit confirms finite positive crop; failures block full readiness"))
    return result


def write_registry(registry,files,raw,inventory,by_group):
    d,r=media_rows(inventory,by_group)
    merge_records(Path(registry)/"DHF1K/processed/media.jsonl",d)
    merge_records(Path(registry)/"RetargetVid/processed/media.jsonl",r)
    merge_records(Path(registry)/"RetargetVid/processed/annotations.jsonl",annotation_rows(files,raw,inventory))


def graph_modes(iou,centers,active_span,threshold=.5,min_gap=.5):
    """Conservative six-rater multimodality proxy, not a six-point density test.

    Connect IoU>=threshold; require two disconnected components, each with >=2
    raters, separated by >=min_gap * median active crop span. Singleton outliers
    do not create a second mode. Definitions are fixed before this audit.
    """
    pairs=[(a,b) for a in range(6) for b in range(a+1,6)]
    patterns=np.zeros(iou.shape[-1],np.uint16)
    for k,(a,b) in enumerate(pairs):patterns|=(iou[a,b]>=threshold).astype(np.uint16)<<k
    result=np.zeros(len(patterns),bool)
    for pattern in np.unique(patterns):
        parent=list(range(6))
        def find(i):
            while parent[i]!=i:i=parent[i]
            return i
        for k,(a,b) in enumerate(pairs):
            if int(pattern)&(1<<k):parent[find(a)]=find(b)
        groups=defaultdict(list)
        for a in range(6):groups[find(a)].append(a)
        groups=[v for v in groups.values() if len(v)>=2]
        if len(groups)<2:continue
        frames=np.flatnonzero(patterns==pattern)
        positions=[np.median(centers[g][:,frames],axis=0) for g in groups]
        for i,a in enumerate(positions):
            for b in positions[i+1:]:result[frames]|=np.abs(a-b)>=min_gap*active_span[frames]
    return result


def agreement_one(boxes,width,height,ratio):
    """All 15 rater pairs, all frames; no GT box averaging."""
    n=boxes.shape[1];a=boxes[:,None,:,:];b=boxes[None,:,:,:]
    low=np.maximum(a[...,:2],b[...,:2]);high=np.minimum(a[...,:2]+a[...,2:],b[...,:2]+b[...,2:])
    inter=np.maximum(0,high-low).prod(-1);area=boxes[...,2:].prod(-1)
    iou=inter/(area[:,None]+area[None,:]-inter)
    pairs=np.asarray([(i,j) for i in range(6) for j in range(i+1,6)])
    pair=iou[pairs[:,0],pairs[:,1]]
    centers=boxes[...,:2]+boxes[...,2:]/2
    normalized=centers/[width,height]
    norm_difference=np.linalg.norm(normalized[pairs[:,0]]-normalized[pairs[:,1]],axis=-1)
    pixel_difference=np.linalg.norm(centers[pairs[:,0]]-centers[pairs[:,1]],axis=-1)
    mw,mh,axis=ws.max_window(width,height,ratio);active=0 if axis is None else axis
    scale=np.sqrt(area/(mw*mh));span=np.median(boxes[...,2+active],axis=0)
    modes={str(t):graph_modes(iou,centers[...,active],span,t,.5) for t in (.3,.5,.7)}
    result={"frames":n,"pairwise_IoU":summary(pair),"pairwise_center_L2_pixels":summary(pixel_difference),
        "pairwise_center_L2_image_normalized":summary(norm_difference),
        "center_range_x_image_normalized":summary(np.ptp(normalized[...,0],axis=0)),
        "center_range_y_image_normalized":summary(np.ptp(normalized[...,1],axis=0)),
        "scale_across_raters_std":summary(scale.std(0)),"scale_across_raters_range":summary(np.ptp(scale,axis=0)),
        "multimodal_proxy_fraction":float(modes['0.5'].mean()),
        "multimodal_sensitivity_IoU_threshold":{k:float(v.mean()) for k,v in modes.items()},
        "rater_pair_mean_IoU_matrix":iou.mean(-1).tolist(),"per_annotator":{}}
    for ann in range(6):
        others=[a for a in range(6) if a!=ann]
        offset=normalized[ann]-np.median(normalized[others],axis=0)
        size_offset=scale[ann]-np.median(scale[others],axis=0)
        result["per_annotator"][f"annotator_{ann+1}"]={
            "signed_center_x_peer_median_image_norm":float(offset[:,0].mean()),
            "signed_center_y_peer_median_image_norm":float(offset[:,1].mean()),
            "abs_center_x_peer_median_image_norm":float(np.abs(offset[:,0]).mean()),
            "abs_center_y_peer_median_image_norm":float(np.abs(offset[:,1]).mean()),
            "relative_scale_peer_median":float(size_offset.mean()),
            "IoU_with_other_five":float(iou[ann,others].mean())}
    return result


def coordinate_one(raw_boxes,width,height,ratio):
    converted=np.stack([ltrb_to_xywh(a) for a in raw_boxes])
    recovered=np.concatenate((converted[...,:2],converted[...,:2]+converted[...,2:]),axis=-1)
    flat=converted.reshape(-1,4);checks=ws.check_candidates(width,height,ratio,flat,px_tol=1.)
    exact_inside=(flat[:,:2]>=0).all(1)&((flat[:,:2]+flat[:,2:])<=[width,height]).all(1)
    negative=(raw_boxes[...,:2]<0).any(-1)
    positive=(converted[...,2:]>0).all(-1)
    # Compatibility comparison on all rater pairs and frames. The default
    # half-open contract is not silently replaced by upstream's +1-and-clamp.
    l=np.maximum(np.rint(raw_boxes),0);a=l[:,None];b=l[None,:]
    inter=np.maximum(0,np.minimum(a[...,2:],b[...,2:])-np.maximum(a[...,:2],b[...,:2])+1).prod(-1)
    area=(l[...,2:]-l[...,:2]+1).prod(-1)
    inc=inter/(area[:,None]+area[None,:]-inter)
    x=converted[:,None];y=converted[None,:]
    inter0=np.maximum(0,np.minimum(x[...,:2]+x[...,2:],y[...,:2]+y[...,2:])-np.maximum(x[...,:2],y[...,:2])).prod(-1)
    area0=converted[...,2:].prod(-1);half=inter0/(area0[:,None]+area0[None,:]-inter0)
    pairs=np.asarray([(a,b) for a in range(6) for b in range(a+1,6)])
    plus_difference=inc[pairs[:,0],pairs[:,1]]-half[pairs[:,0],pairs[:,1]]
    return converted,{"raw_boxes":raw_boxes.shape[0]*raw_boxes.shape[1],"roundtrip_ltrb_maxabs":float(np.abs(recovered-raw_boxes).max()),
        "raw_negative_left_top":int(negative.sum()),"nonpositive_windows":int((~positive).sum()),
        "outside_image_exact":int((~exact_inside).sum()),"ratio_mismatch_gt_rounding_tolerance_1px":int((~checks['ratio_ok']).sum()),
        "not_max_window_tolerance_1px":int((~checks['is_max_window']).sum()),
        "crop_widths":summary(flat[:,2]),"crop_heights":summary(flat[:,3]),
        "upstream_inclusive_plus1_minus_halfopen_rater_IoU":summary(plus_difference),
        "default_gt_boundary":"none_raw_unclipped","converted_height_kept_from_raw_annotation":True}


def aggregate_agreement(units):
    out={}
    for ratio in RATIOS:
        rows=[r for r in units if r["ratio"]==ratio]
        metrics={"source_macro_pairwise_IoU":"pairwise_IoU","source_macro_center_L2_pixels":"pairwise_center_L2_pixels",
                 "source_macro_center_L2_image_normalized":"pairwise_center_L2_image_normalized",
                 "source_macro_scale_std":"scale_across_raters_std","source_macro_scale_range":"scale_across_raters_range"}
        result={name:bootstrap_sources([r[key]["mean"] for r in rows]) for name,key in metrics.items()}
        result["source_macro_multimodal_proxy_fraction"]=bootstrap_sources([r["multimodal_proxy_fraction"] for r in rows])
        result["multimodal_sensitivity"]={str(t):bootstrap_sources([r["multimodal_sensitivity_IoU_threshold"][str(t)] for r in rows]) for t in (.3,.5,.7)}
        result["rater_pair_source_macro_IoU_matrix"]=np.mean([r["rater_pair_mean_IoU_matrix"] for r in rows],axis=0).tolist() if rows else None
        result["annotator_systematic_preferences"]={}
        for ann in RATERS:
            key=f"annotator_{ann}"
            result["annotator_systematic_preferences"][key]={field:bootstrap_sources([r["per_annotator"][key][field] for r in rows])
                for field in (rows[0]["per_annotator"][key] if rows else [])}
        result.update(source_videos=len(rows),annotated_frame_observations=sum(r["frames"] for r in rows),raters=6,
                      interpretations="peer-relative disagreement and systematic offsets, not annotator accuracy against unique GT")
        out[ratio]=result
    return {"by_ratio":out,"per_source_ratio":units,"GT_boxes_averaged_for_training":False,
        "multimodality_definition":{"proxy":True,"IoU_edge_threshold":.5,"sensitivity_IoU_thresholds":[.3,.5,.7],
            "two_disconnected_components_min_raters_each":2,"minimum_component_median_center_gap_in_active_crop_spans":.5,
            "limitation":"six raters cannot certify population preference modes; singleton outliers are not second modes"},
        "statistical_unit":"source video; all six raters, all frames and both ratios remain linked",
        "analysis_role":"exploratory exposed-data description, not fresh model confirmation"}


def adapter_annotation_test(registry,raw,inventory):
    ds=DenseCropAdapter(root=Path(registry),split=["holdout_dev_exposed","holdout_confirm_exposed"],fps=None,decode=False)
    checked=[]
    for i in range(len(ds)):
        item=ds[i];n=int(item["uid"].split(":")[1]);ratio=item["ratio"];expected=inventory[n]["frame_count"]
        if not np.array_equal(item["frame_idx"],np.arange(expected)) or not item["valid"].all():
            raise ValueError("Adapter frame mapping/mask does not cover every real frame")
        truth=np.stack([ltrb_to_xywh(raw[n,ratio,a]) for a in RATERS])
        if item["boxes_xywh"].shape!=truth.shape or not np.array_equal(item["boxes_xywh"],truth):
            raise ValueError("real Adapter boxes differ from raw LTRB conversion")
        if item["group_id"]!=dhf1k_id(n):raise ValueError("Adapter group split across raters/ratios")
        clock=load_timeline(inventory[n]["timeline_path"])
        if not np.array_equal(item["time_s"],clock["time_s"]):raise ValueError("Adapter is not using actual decoded PTS")
        checked.append({"uid":item["uid"],"ratio":ratio,"frames":expected,"first_frame_id":int(item["frame_idx"][0]),
                        "last_frame_id":int(item["frame_idx"][-1]),"rater_count":len(item["annotators"]),"all_values_equal_raw_conversion":True})
    train=len(DenseCropAdapter(root=Path(registry),split="train",decode=False))
    fresh=len(DenseCropAdapter(root=Path(registry),split=["dev","confirmation","holdout_confirm_reserved"],decode=False))
    if len(checked)!=400 or train or fresh:raise ValueError("exposure/Adapter unit count invariant violated")
    return {"passed":True,"annotation_units_read":len(checked),"raw_annotation_files_read":len(checked)*6,
        "all_frames_and_all_raters_compared":True,"Train_units":train,"fresh_validation_units":fresh,"units":checked}


def alignment_rows(raw,n,ratio,media):
    """Keep per-file failures explicit; never truncate raters to common length."""
    decoded=media["frame_count"] if media["decode_verified"] else None
    return [{"annotator":f"annotator_{ann}","ratio":ratio,"annotation_rows":len(raw[n,ratio,ann]),
             "decoded_frames":decoded,"aligned":decoded is not None and len(raw[n,ratio,ann])==decoded}
            for ann in RATERS]


def run_full_audit(media_root,registry_root,retargetvid_root,evidence_root,output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    media_root=Path(media_root).resolve();registry=Path(registry_root).resolve();rvroot=Path(retargetvid_root).resolve()
    protected,hashes,by_group=protected_sources(evidence_root)
    write_json(output/"run_config.json",{"audit_version":AUDIT_VERSION,"media_root":str(media_root),"registry_root":str(registry),
        "retargetvid_root":str(rvroot),"evidence_root":str(Path(evidence_root).resolve()),"protected_sha256":hashes,
        "GPU_used":False,"training":False,"source_ids":list(RETARGET_IDS),"rater_ids":list(RATERS),"ratios":list(RATIOS)})
    physical=ensure_selected_media(media_root,output/"media_reuse_registration.json")
    files,archives=prepare_annotations(rvroot);raw,annotation_hashes=read_annotations(files)
    write_json(output/"annotation_source_hashes.json",{"archives":archives,"original_files":annotation_hashes,"count":len(files)})
    before=snapshot_working_registry(registry,output)
    inventory={}
    for n,path in physical.items():
        header=probe_header(path);st=path.stat()
        count=header["container_nb_frames"] or len(raw[n,"1-3",1])
        if not count or not header["fps"]:raise ValueError("unusable video header")
        inventory[n]={**header,"video_id":f"{n:03d}","path":str(path),"file_size":st.st_size,"mtime_ns":st.st_mtime_ns,
            "sha256":digest(path),"frame_count":count,"duration_s":header["container_duration_s"],"decode_verified":False}
    write_registry(registry,files,raw,inventory,by_group)
    ds=DenseCropAdapter(root=registry,ratios=("1-3",),fps=None,max_frames=1,decode=True,size=None,decode_through_eof=True)
    if len(ds)!=200:raise ValueError("real RGB Adapter must address exactly 200 sources")
    contract=decode_contract();decode_count=0;cache_reused=0;decode_errors=[]
    for i in range(len(ds)):
        uid=ds.items[i][0][0];n=int(uid.split(":")[1]);m=inventory[n]
        record_path=output/"decoded_sources"/(f"{n:03d}.json")
        tlpath=registry/"DHF1K/processed/timelines"/(f"dhf1k_{n:03d}.npz")
        try:
            if record_path.exists():
                saved=json.loads(record_path.read_text())
                if saved["source_sha256"]!=m["sha256"] or saved["decode_contract_sha256"]!=contract or digest(tlpath)!=saved["timeline_sha256"]:
                    raise ValueError("changed/uncertain cached source; preserve existing evidence, do not blindly decode again")
                trace=saved["trace"];cache_reused+=1
            else:
                observer=DecodeTrace(len(raw[n,"1-3",1]));ds.decode_observer=observer
                item=ds[i];trace=observer.finish();decode_count+=1
                if not np.array_equal(item["frame_idx"],[0]) or item["frames"].shape!=(1,m["height"],m["width"],3):
                    raise ValueError("real Adapter RGB/frame-0 contract failed")
                firsthash=hashlib.sha256(item["frames"][0].tobytes()).hexdigest()
                if firsthash!=trace["frame_probes"]["0"]["rgb_sha256"]:raise ValueError("Adapter frame 0 differs from first displayed frame")
                save_timeline(trace,tlpath)
                serial={k:v for k,v in trace.items() if k not in {"pts","time_s"}}
                saved={"source_sha256":m["sha256"],"decode_contract_sha256":contract,"timeline_sha256":digest(tlpath),
                       "actual_DenseCropAdapter_RGB_read":True,"first_RGB_hash_matches_actual_first_decoded_frame":True,"trace":serial}
                write_json(record_path,saved);trace=serial
            m.update(frame_count=trace["frame_count"],duration_s=trace["duration_s"],fps_measured=trace["fps_measured"],
                     timeline_path=str(tlpath),timeline_sha256=digest(tlpath),decode_verified=True,frame_probes=trace["frame_probes"],
                     missing_pts=trace["missing_pts"],non_increasing=trace["non_increasing"],first_time_zero=trace["first_time_zero"],
                     decode_order=trace["order"],actual_DenseCropAdapter_RGB_read=True)
        except Exception as exc:
            decode_errors.append({"video_id":f"{n:03d}","error":type(exc).__name__+": "+str(exc)})
        write_json(output/"progress.json",{"processed_source_videos":i+1,"expected":200,"full_decodes_this_run":decode_count,
            "verified_prior_decodes_reused":cache_reused,"errors":decode_errors,"GPU_used":False})
        if (i+1)%10==0:print(json.dumps({"decoded_or_reused":i+1,"full_decodes_this_run":decode_count,"errors":len(decode_errors)}),flush=True)
    write_registry(registry,files,raw,inventory,by_group)
    alignment=[];coordinates=[];units=[];alignment_failed=[]
    for n in RETARGET_IDS:
        m=inventory[n];per_file=[]
        for ratio in RATIOS:
            lengths=[len(raw[n,ratio,a]) for a in RATERS]
            rows=alignment_rows(raw,n,ratio,m);aligned=all(r["aligned"] for r in rows)
            per_file.extend(rows)
            if aligned:
                boxes,check=coordinate_one(np.stack([raw[n,ratio,a] for a in RATERS]),m["width"],m["height"],ratio)
                coordinates.append(dict(video_id=f"{n:03d}",ratio=ratio,**check))
                if check["nonpositive_windows"]==0:
                    units.append(dict(video_id=f"{n:03d}",group_id=dhf1k_id(n),ratio=ratio,**agreement_one(boxes,m["width"],m["height"],ratio)))
            else:alignment_failed.append({"video_id":f"{n:03d}","ratio":ratio,"annotation_counts":lengths,"decoded_frames":m["frame_count"] if m["decode_verified"] else None})
        alignment.append({"video_id":f"{n:03d}","group_id":dhf1k_id(n),"files":per_file,
            "container_frames":m["container_nb_frames"],"decoded_frames":m["frame_count"] if m["decode_verified"] else None,
            "first_frame_id":0,"last_frame_id":m["frame_count"]-1 if m["decode_verified"] else None,
            "first_time_zero":m.get("first_time_zero"),"PTS_strictly_increasing":m.get("non_increasing")==0 if m["decode_verified"] else None,
            "actual_RGB_Adapter_test":m.get("actual_DenseCropAdapter_RGB_read",False)})
    after={k:digest(v) for k,v in protected.items()}
    if after!=hashes:raise ValueError("protected ledger/release was modified")
    annotation_unchanged=all(digest(path)==h for path,h in annotation_hashes.items())
    source_unchanged=all(Path(m["path"]).stat().st_size==m["file_size"] and Path(m["path"]).stat().st_mtime_ns==m["mtime_ns"] for m in inventory.values())
    if not annotation_unchanged or not source_unchanged:raise ValueError("raw source changed while auditing")
    verified=sum(m["decode_verified"] for m in inventory.values());goodfiles=sum(f["aligned"] for row in alignment for f in row["files"])
    media_result={"audit_version":AUDIT_VERSION,"physical_AVI_found":len(physical),"decoded_verified":verified,
        "expected_sources":200,"unique_video_sha256":len({m["sha256"] for m in inventory.values()}),
        "registered_DHF1K_subset_records":200,"registered_RetargetVid_records":200,"video_copies_created_for_RetargetVid":0,
        "full_decode_passes_this_run":decode_count,"verified_decode_cache_reused":cache_reused,
        "dataset_total_DHF1K_1000_claimed":False,"raw_media_changed":not source_unchanged,"source_videos":[inventory[n] for n in RETARGET_IDS],"errors":decode_errors}
    frame_result={"expected_annotation_files":2400,"actual_annotation_files":len(files),"aligned_annotation_files":goodfiles,
        "decoded_source_frames":sum(m["frame_count"] for m in inventory.values() if m["decode_verified"]),
        "annotated_box_rows":sum(len(v) for v in raw.values()),"all_files_aligned":goodfiles==2400,
        "mapping":"raw file line i+1 maps to actual displayed decoded frame i; ordinal 0..N-1; no packet-index mapping or PTS sorting",
        "failures":alignment_failed,"sources":alignment}
    coordinate_result={"coord_version":ws.COORD_VERSION,"formula":"x=left,y=top,w=right-left,h=bottom-top",
        "raw_annotations_unchanged":annotation_unchanged,"raw_annotation_sha256_manifest":canonical(annotation_hashes),
        "raw_boxes_checked":sum(r["raw_boxes"] for r in coordinates),"roundtrip_ltrb_maxabs":max((r["roundtrip_ltrb_maxabs"] for r in coordinates),default=None),
        "outside_image_raw_boxes":sum(r["outside_image_exact"] for r in coordinates),"nonpositive_windows":sum(r["nonpositive_windows"] for r in coordinates),
        "not_max_window_1px":sum(r["not_max_window_tolerance_1px"] for r in coordinates),
        "ratio_mismatch_1px":sum(r["ratio_mismatch_gt_rounding_tolerance_1px"] for r in coordinates),
        "clipping_applied":False,"plus1_added_to_default_GT":False,"per_source_ratio":coordinates}
    split_result={"source_groups":200,"ratio_units":400,"annotation_files":2400,"exposed_eval_sources":200,
        "fresh_validation_sources":0,"training_sources_created":0,"protected_original_sha256_before":hashes,
        "protected_original_sha256_after":after,"original_holdout_ledger_modified":False,"original_frozen_release_modified":False,
        "one_physical_video_group_for_both_datasets_all_ratios_all_raters":True,"new_training_experiment_version_created":False,
        "future_training_requires":"explicit new experiment version, mark these groups previously exposed and never use them as independent fresh validation",
        "registry_before_sha256":before,"source_groups_and_exposures":[{"group_id":dhf1k_id(n),"release_split":"exposed_eval", "roles":sorted({r['role'] for r in by_group[dhf1k_id(n)]})} for n in RETARGET_IDS]}
    reader=adapter_annotation_test(registry,raw,inventory) if verified==200 and goodfiles==2400 else {"passed":False,"reason":"decode/alignment failure; no false adapter success claimed"}
    safe=(verified==200 and goodfiles==2400 and not coordinate_result["nonpositive_windows"]
          and not coordinate_result["outside_image_raw_boxes"] and not coordinate_result["ratio_mismatch_1px"]
          and coordinate_result["roundtrip_ltrb_maxabs"]==0 and reader["passed"])
    agreement=aggregate_agreement(units)
    for name,value in (("media_inventory",media_result),("frame_annotation_alignment",frame_result),("coordinate_audit",coordinate_result),
                       ("split_leakage_audit",split_result),("annotator_agreement",agreement),("adapter_read_test",reader)):
        write_json(output/(name+".json"),value)
    for ds in ("DHF1K","RetargetVid"):
        card=Path(registry)/ds/"processed/dataset_card.json"
        previous=json.loads(card.read_text()) if card.exists() else {}
        previous.update(dataset=ds,media_registration_scope="RetargetVid 001-100,601-700 subset only",registered_sources=200,
            readiness="media_available_exposed_diagnostic_only" if safe else "audit_incomplete_or_failed",default_training_eligible=False,
            fresh_validation_eligible=False,audit_path=str(output),holdout_ledger_sha256=hashes["ledger"])
        write_json(card,previous)
    decision={"all_200_media_available":verified==200,"all_2400_annotations_aligned":goodfiles==2400,
        "safe_for_exposed_spatial_diagnostics":bool(safe),"fresh_validation_eligible":False,"training_started":False,"GPU_used":False,
        "P1_P2_weights_changed":False,"historical_release_media_SHA_match":None,
        "historical_SHA_limitation":"immutable release summary and ledger are present, original per-media release manifest is not in this clone; do not claim byte-for-byte comparison to historical AVI"}
    write_json(output/"decision.json",decision);write_report(output,decision,media_result,frame_result,coordinate_result,agreement,split_result)
    print(json.dumps(decision,indent=2),flush=True)
    if not safe:raise RuntimeError("RetargetVid audit did not clear diagnostic readiness; inspect delivered failure evidence")
    return decision


def write_report(output,decision,media,frames,coords,agreement,split):
    lines=["# RetargetVid 原始媒体接入与人工构图监督审计", "",
        f"200个AVI全部可用：{decision['all_200_media_available']}；2400份标注全部对齐：{decision['all_2400_annotations_aligned']}。",
        "", "## 媒体与实际读取", "",
        f"原地引用DHF1K的200个AVI，两数据集共用同一物理路径。此次完整顺序解码{media['full_decode_passes_this_run']}个源，复用已验证完整解码{media['verified_decode_cache_reused']}个源。没有为6名标注者/两画幅重复全量解码，也没有复制视频到RetargetVid目录。",
        f"实际解码源帧合计{frames['decoded_source_frames']:,}；原始人工框合计{frames['annotated_box_rows']:,}。SHA、大小、分辨率、FPS、容器/实解帧数、时长逐视频保留。",
        "实际DenseCropAdapter RGB读取与完整PTS捕获发生在同一个解码循环；随后400个视频×画幅单元的全部6名标注者/全部帧通过真实Adapter读取并逐值核对。",
        "", "## 坐标与时间合同", "",
        f"采用xywh_halfopen_v1：x=l,y=t,w=r-l,h=b-t，不加1、不裁边。LTRB往返最大差{coords['roundtrip_ltrb_maxabs']}；原始越界框{coords['outside_image_raw_boxes']}，非正尺寸框{coords['nonpositive_windows']}。原始ZIP与2400个TXT均未改写。",
        "原始第i+1行映射PyAV实际显示解码序号i；frame ID从0开始，保存真实PTS与time_base，不使用包序号，不按时间戳重排序。上游inclusive+1评分单独作为兼容对照，不覆盖默认坐标合同。",
        "", "## 六名标注者的构图分歧", "",
        "| 目标画幅 | Source-Macro两两IoU及95%CI | 多簇构图代理比例及95%CI | Scale跨标注者标准差 |", "|---|---|---|---|"]
    for ratio in RATIOS:
        a=agreement['by_ratio'][ratio];q=a['source_macro_pairwise_IoU'];m=a['source_macro_multimodal_proxy_fraction'];s=a['source_macro_scale_std']
        fmt=lambda v:"null" if v is None else f"{v:.6f}"
        lines.append(f"| {ratio.replace('-',':')} | {fmt(q['mean'])} {q['bootstrap_ci95']} | {fmt(m['mean'])} {m['bootstrap_ci95']} | {fmt(s['mean'])} |")
    lines.extend(["", f"Scale定义为sqrt(原始框面积/该画幅最大合法窗面积)。检查{coords['raw_boxes_checked']:,}个框，最大窗尺寸不一致数为{coords['not_max_window_1px']}（允许1px发布舍入误差）。若同帧Scale分歧为0，则标注者只提供固定窗大小下的构图位置差异，不能据此构造可变Scale的人工偏好。",
        "", "| 目标画幅 | 两两中心L2像素 Source-Macro均值 | 95%源级Bootstrap CI |", "|---|---|---|"])
    for ratio in RATIOS:
        q=agreement['by_ratio'][ratio]['source_macro_center_L2_pixels']
        lines.append(f"| {ratio.replace('-',':')} | {fmt(q['mean'])} | {q['bootstrap_ci95']} |")
    lines.extend(["", "以下为各标注者相对另外五人同帧中位中心的有符号活动轴偏移，再对源视频做等权Macro。正值为向右（1:3）/向下（3:1），单位是源图像活动轴长度的百分比；置信区间仅作已暴露数据的探索性描述。",
        "", "| 画幅 | 标注者 | 中心偏移(%) | 95%源级Bootstrap CI(%) | Scale偏移 |", "|---|---|---|---|---|"])
    for ratio in RATIOS:
        field='signed_center_x_peer_median_image_norm' if ratio=='1-3' else 'signed_center_y_peer_median_image_norm'
        for ann in RATERS:
            r=agreement['by_ratio'][ratio]['annotator_systematic_preferences'][f'annotator_{ann}'];q=r[field]
            ci=None if q['bootstrap_ci95'] is None else [round(v*100,4) for v in q['bootstrap_ci95']]
            offset=None if q['mean'] is None else q['mean']*100
            lines.append(f"| {ratio.replace('-',':')} | {ann} | {fmt(offset)} | {ci} | {fmt(r['relative_scale_peer_median']['mean'])} |")
    lines.extend(["", "每视频、每画幅均有IoU、中心绝对/归一化分歧、scale分歧、6×6标注者一致性矩阵和逐标注者偏好偏移。系统偏好相对另外五人的同帧中位位置/尺度度量，不是某人的准确率。没有把6个框的平均当唯一GT。",
        "多峰采用预定义探索性代理：IoU≥.5连接标注者，至少两个互不连通的簇、每簇至少2人，簇中位中心间隔至少半个活动轴窗口。另报IoU阈值.3/.7敏感性；6个标注者不能证明总体偏好分布存在统计学多峰。Bootstrap以独立源视频为单位，不把帧/15对标注者当独立样本。",
        "", "## 曝光与后续使用边界", "",
        "全部200个源、400个画幅单元、2400份标注继续为exposed_eval。DHF1K官方train/val标记只保留原发布含义；共享source group的工作project_split全部是holdout_dev_exposed或holdout_confirm_exposed。训练与fresh validation选择都返回0单元。",
        f"原始ledger SHA256：`{split['protected_original_sha256_before']['ledger']}`，审计前后完全一致。原始release和quality文件亦未改写；新路径只注册到工作manifest。",
        "可以用于已暴露构图机制诊断、多人标注分歧研究、坐标/解码/部署回归。不能作为fresh validation、独立泛化确认，不能由此清除P2研究的独立域晋级门。若未来训练，必须单独建立明确标为已暴露的新实验版本，且这些源不再承担独立泛化评价；本次未创建训练版本。",
        "这些画幅是1:3与3:1，不是AIC原生9:16/16:9的新验证域。最大窗尺寸一致性只说明数据合同，不证明可以从RetargetVid得到可变Scale偏好。",
        "", "## 证据与限制", "",
        "media_inventory.json、frame_annotation_alignment.json、coordinate_audit.json、annotator_agreement.json、split_leakage_audit.json、adapter_read_test.json与原始文件哈希清单是本报告证据。",
        "旧release的逐媒体原始manifest未在克隆中恢复，故当前文件SHA不能逐一与历史release媒体SHA重对；该项保留null。当前实际媒体身份、显示解码顺序、尺寸、2400份标注长度与Adapter转换均作独立实检。",
        "GPU使用0；没有训练，没有改变P1/P2权重。审计遵循原始数据只读、按源视频分层的探索性分析原则；工作流方法参考Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026), Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents, DOI 10.48550/arXiv.2609.00065（2026-11-08核验arXiv元数据）。"])
    (Path(output)/"final_report.md").write_text("\n".join(lines)+"\n")
