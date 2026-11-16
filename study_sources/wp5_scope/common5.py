from pathlib import Path
import sys,csv
HERE5=Path(__file__).resolve().parent
WP4=HERE5.parent/'wp4_training'
sys.path.insert(0,str(WP4))
from common4 import *
def csvout(name,rows):
    rows=list(rows)
    if not rows:return
    p=HERE5/'reports'/name;p.parent.mkdir(exist_ok=True,parents=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
