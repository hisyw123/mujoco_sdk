#!/usr/bin/env python3
"""Write dated hash manifests and unified diffs; never change the reference or source."""
import argparse
from datetime import datetime
import difflib
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''): h.update(part)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='/home/dongju/actuator-net-gym-articulated-main')
    p.add_argument('--original', default='/home/dongju/Downloads/actuator-net-gym-articulated-main')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    stamp=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%dT%H%M%S_%f')
    out=root/'audit'/stamp
    out.mkdir(parents=True,exist_ok=False)
    paths=['README.md','CMakeLists.txt','src/augmented_sim.cpp',
           'scripts/common.py','scripts/train.py','scripts/evaluate.py','scripts/pt_to_jit.py']
    report={'original':a.original,'source':a.source,'port':str(root),'files':[]}
    for relative in paths:
        original=Path(a.original)/relative
        for label,base in [('source',Path(a.source)),('port',root),('reference',root/'reference/raisim')]:
            current=base/relative
            entry=dict(area=label,path=relative,original_sha256=digest(original),
                       current_sha256=digest(current) if current.is_file() else None)
            entry['status']=('missing' if not current.is_file() else
                             'unchanged' if entry['current_sha256']==entry['original_sha256'] else 'modified')
            report['files'].append(entry)
            if entry['status']=='modified':
                diff=''.join(difflib.unified_diff(original.read_text().splitlines(True),
                            current.read_text().splitlines(True),fromfile=str(original),tofile=str(current)))
                (out/(label+'__'+relative.replace('/','__')+'.diff')).write_text(diff)
    report['port_inventory']={str(f.relative_to(root)):digest(f) for folder in ['scripts','tests','src']
                              for f in (root/folder).rglob('*')
                              if f.is_file() and '__pycache__' not in f.parts}
    for name in ['run.py','CMakeLists.txt','README.md','NATIVE_CPP.md','patch note.txt']:
        if (root/name).is_file(): report['port_inventory'][name]=digest(root/name)
    (out/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(out)
    for entry in report['files']:
        print(entry['area'],entry['status'],entry['path'])


if __name__=='__main__': main()
