#!/bin/bash
cd /home/claude/sc
cat joblist2.txt | xargs -P 2 -L 1 sh -c 'python3 train.py --d $0 --L $1 --tokens $2 --out $3 >> runs/progress2.log 2>&1'
echo ALLDONE >> runs/progress2.log
