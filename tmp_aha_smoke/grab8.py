import re
p="/root/autodl-tmp/aha_repo/test/inference.py"
L=open(p).read().splitlines()
pat=re.compile(r"def |frame_v_placeholder|v_placeholder|read_video|ImageOps|process_image|image_processor|score_heads|informative_logits|relevance_logits|uncertainty|joint_embed|frame_resolution")
for i,l in enumerate(L,1):
    if pat.search(l):
        print("%4d| %s"%(i,l))
