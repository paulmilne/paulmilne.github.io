import numpy as np, json
from curve import equiv2
fps=np.round(np.arange(0.70,1.16,0.05),3)
res=[(float(fp),)+equiv2(fp,1.03,0.01,'abs',T=150,nph=8) for fp in fps]
json.dump(res,open('fpdep.json','w')); print([(r[0],round(r[1])) for r in res])
