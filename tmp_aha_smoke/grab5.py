def show(f, a=0, b=None):
    print("="*90); print("FILE:", f, "lines", a, "-", b); print("="*90)
    try:
        L = open(f).read().splitlines(); b = b or len(L)
        for i, l in enumerate(L[a:b], a+1): print("%4d| %s" % (i, l))
    except Exception as e: print("ERR", e)
show("/root/autodl-tmp/aha_repo/models/modeling_live.py", 90, 175)
show("/root/autodl-tmp/aha_repo/models/vision_live.py", 50, 95)
