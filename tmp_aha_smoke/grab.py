import subprocess, sys
files = [
 "/root/autodl-tmp/aha_repo/scripts/inference/tvsum.sh",
 "/root/autodl-tmp/aha_repo/test/inference.py",
]
for f in files:
    print("="*100)
    print("FILE:", f)
    print("="*100)
    try:
        print(open(f).read())
    except Exception as e:
        print("ERR", e)
