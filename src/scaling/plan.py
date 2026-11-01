import json, itertools
cfgs={"d32L2":(32,2),"d48L3":(48,3),"d64L4":(64,4),"d96L4":(96,4),"d128L5":(128,5),"d160L6":(160,6),"d192L6":(192,6)}
def N(d,L): return L*(12*d*d+13*d)+2*d
plan={1e12:["d32L2","d48L3","d64L4","d96L4","d128L5"],3e12:["d32L2","d48L3","d64L4","d96L4","d128L5","d160L6"],
      1e13:["d48L3","d64L4","d96L4","d128L5","d160L6","d192L6"],3e13:["d64L4","d96L4","d128L5","d160L6","d192L6"]}
jobs=[]
for C,names in plan.items():
    for n in names:
        d,L=cfgs[n]; per=6*N(d,L)+6*L*128*d; D=C/per
        jobs.append(dict(C=C,name=n,d=d,L=L,N=N(d,L),D=int(D)))
jobs.sort(key=lambda j:-j['C']*1.0)
json.dump(jobs,open('jobs.json','w'),indent=0)
for j in jobs: print(j)
