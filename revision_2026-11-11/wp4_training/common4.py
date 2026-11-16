from pathlib import Path
import sys,os,json,hashlib
HERE4=Path(__file__).resolve().parent
ROOT4=HERE4.parents[1];PROJECT4=ROOT4/'Dota2_matches';WP3=HERE4.parent/'wp3_models'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
sys.path.insert(0,str(WP3));sys.dont_write_bytecode=True
from runtime import sha
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(p)
