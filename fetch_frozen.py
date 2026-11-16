"""Fetch requested frozen IDs; never substitute matches or log API credentials."""
from pathlib import Path
import argparse,json,os,hashlib,time,urllib.request,urllib.parse,urllib.error
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',choices=['development','holdout'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--limit',type=int,default=3);p.add_argument('--download',action='store_true');a=p.parse_args()
    if a.limit<1:raise ValueError('Set a positive explicit limit')
    if a.cohort=='development':
        j=json.loads((ROOT/'manifests/development_raw.json').read_text());rows={int(r['match_id']):r['sha256'] for rs in j.values() for r in rs}
    else:
        j=json.loads((ROOT/'revision_2026-11-11/wp6_evaluation/raw_manifest.json').read_text());rows={int(Path(n).stem):h for n,h in j['hashes'].items()}
    selected=list(rows.items())[:a.limit]
    if not a.download:print(json.dumps({'frozen_ids':len(rows),'requested':len(selected),'action':'dry_run; pass --download to retrieve'}));return
    a.out.mkdir(parents=True,exist_ok=True)
    for mid,expected in selected:
        dest=a.out/f'{mid}.json';status='existing'
        if not dest.exists():
            key=os.environ.get('OPENDOTA_API_KEY');url=f'https://api.opendota.com/api/matches/{mid}'
            if key:url+='?'+urllib.parse.urlencode({'api_key':key})
            try:
                with urllib.request.urlopen(url,timeout=45) as r:raw=r.read()
                data=json.loads(raw)
                if data.get('match_id')!=mid:raise ValueError('Unexpected response ID')
                dest.write_bytes(raw);status='downloaded'
            except urllib.error.HTTPError as e:
                print(json.dumps({'match_id':mid,'http_status':e.code}));return
            except Exception as e:print(json.dumps({'match_id':mid,'error_type':type(e).__name__}));return
            time.sleep(2.1)
        actual=hashlib.sha256(dest.read_bytes()).hexdigest()
        with (a.out/'retrieval.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'match_id':mid,'status':status,'sha256':actual,'expected_sha256':expected,'exact':actual==expected})+'\n')
if __name__=='__main__':main()
