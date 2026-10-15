import os
for f in ["_probe1.py","_probe2.py","_probe3.py","_probe4.py","_probe5.py"]:
    p="/root/autodl-tmp/"+f
    print("="*90); print("FILE:",p); print("="*90)
    try: print(open(p).read())
    except Exception as e: print("ERR",e)
print("="*90); print("install_aha.log TAIL"); print("="*90)
try:
    L=open("/root/autodl-tmp/install_aha.log").read().splitlines()
    print("\n".join(L[-40:]))
except Exception as e: print("ERR",e)
