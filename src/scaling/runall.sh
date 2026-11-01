#!/bin/bash
cd /home/claude/sc
python3 - <<'PY' > joblist.txt
import json
for j in json.load(open('jobs.json')): print(j['d'], j['L'], j['D'], f"runs/C{j['C']:.0e}_{j['name']}.json")
PY
cat joblist.txt | xargs -P 2 -L 1 sh -c 'python3 train.py --d $0 --L $1 --tokens $2 --out $3 >> runs/progress.log 2>&1'
echo ALLDONE >> runs/progress.log
