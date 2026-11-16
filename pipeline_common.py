from pathlib import Path
import sys,json,csv,hashlib,time,os
CODE=Path(__file__).resolve().parent
for folder in ('wp3_models','wp4_training','wp6_evaluation'):sys.path.insert(0,str(CODE/'revision_2026-11-11'/folder))
sys.path.insert(0,str(CODE/'Dota2_matches'))
HERE6=CODE/'outputs';ROOT6=HERE6
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
def check_lock(p):
 for n,h in read(p)['sources'].items():
  if sha(CODE/n)!=h:raise ValueError('Code identity changed: '+n)
def csvout(name,rows):
 rows=list(rows);p=HERE6/'reports'/name;p.parent.mkdir(parents=True,exist_ok=True)
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
