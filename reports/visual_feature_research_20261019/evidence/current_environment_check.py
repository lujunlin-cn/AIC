import json,platform,sys,shutil,glob,os,datetime
from pathlib import Path
out={'observed_at_shanghai':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'python':sys.version,'platform':platform.platform(),'machine':platform.machine(),'nvidia_smi':shutil.which('nvidia-smi'),'npu_smi':shutil.which('npu-smi'),'gpu_device_nodes':glob.glob('/dev/nvidia[0-9]*'),'ascend_device_nodes':glob.glob('/dev/davinci[0-9]*'),'project_data_root_exists':os.path.exists('/data/aic'),'scope':'Current execution container only; historical remote hosts were not probed.'}
print(json.dumps(out,ensure_ascii=False,indent=2))
