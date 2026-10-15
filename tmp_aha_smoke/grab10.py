def show(f,a=0,b=None):
    print("="*90); print("FILE:",f,"lines",a+1,"-",b); print("="*90)
    L=open(f).read().splitlines(); b=b or len(L)
    for i,l in enumerate(L[a:b],a+1): print("%4d| %s"%(i,l))
show("/root/autodl-tmp/aha_repo/models/tokenization_live.py",110,175)
show("/root/autodl-tmp/aha_repo/models/configuration_live.py")
