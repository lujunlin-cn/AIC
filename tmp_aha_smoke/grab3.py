def show(f, a=0, b=None):
    print("="*100); print("FILE:", f, "lines", a, "-", b); print("="*100)
    try:
        L = open(f).read().splitlines()
        b = b or len(L)
        for i, l in enumerate(L[a:b], a+1):
            print("%4d| %s" % (i, l))
    except Exception as e:
        print("ERR", e)

show("/root/autodl-tmp/aha_repo/models/live_llava/video_head_live_llava_qwen.py", 40, 62)
show("/root/autodl-tmp/aha_repo/models/live_llava/video_head_live_llava_qwen.py", 300, 360)
