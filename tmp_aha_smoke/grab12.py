import re
def show(f,a=0,b=None,tag=""):
    print("="*90); print("FILE:",f,"|",tag,"lines",a+1,"-",b); print("="*90)
    L=open(f).read().splitlines(); b=b or len(L)
    for i,l in enumerate(L[a:b],a+1): print("%4d| %s"%(i,l))

V="/root/autodl-tmp/aha_repo/models/live_llava/video_head_live_llava_qwen.py"
print("### imports of video_head_live_llava_qwen.py")
for i,l in enumerate(open(V).read().splitlines()[:40],1):
    if re.match(r"\s*(import|from)\s", l): print("%4d| %s"%(i,l))
SE="/root/autodl-tmp/aha_repo/LLaVA_NeXT/llava/model/multimodal_encoder/siglip_encoder.py"
print("### where SigLipVisionModel / SigLipVisionConfig come from")
for i,l in enumerate(open(SE).read().splitlines(),1):
    if "SigLipVisionModel" in l or "SigLipVisionConfig" in l: print("%4d| %s"%(i,l.strip()[:160]))
show(SE,575,625,tag="SigLipVisionTower.forward")
A="/root/autodl-tmp/aha_repo/models/arguments_live.py"
show(A,30,70,tag="arguments_live system_prompt")
