import numpy as np, json
from gensim.models import Word2Vec
m=Word2Vec.load('w2v_text8.model'); wv=m.wv
toks=open('dl/text8').read().split()
K=3000
vocab=wv.index_to_key[:K]; idx={w:i for i,w in enumerate(vocab)}
counts_all=np.array([wv.get_vecattr(w,'count') for w in wv.index_to_key],dtype=float)
total=counts_all.sum()
# gensim keep prob (sample=1e-3)
s=1e-3*total
keep_all=(np.sqrt(counts_all/s)+1)*s/counts_all; keep_all=np.minimum(keep_all,1.0)
full_idx={w:i for i,w in enumerate(wv.index_to_key)}
ids=np.array([full_idx.get(w,-1) for w in toks])
ids=ids[ids>=0]   # gensim drops min_count words before windowing
keep=keep_all[ids]
top=np.where(ids<K, ids, -1)
C=np.zeros(K*K)
W=5
for d in range(1,W+1):
    a=top[:-d]; b=top[d:]; ok=(a>=0)&(b>=0)
    w=((W-d+1)/W)*keep[:-d][ok]*keep[d:][ok]
    C+=np.bincount(a[ok]*K+b[ok],weights=w,minlength=K*K)
    C+=np.bincount(b[ok]*K+a[ok],weights=w,minlength=K*K)
C=C.reshape(K,K)
# pairs: centre w (row), context c (col)
# total pairs including contexts outside top-K: approximate #(w) via expected counts over all vocab
D_all=None
row=C.sum(1)
pn=counts_all**0.75; pn/=pn.sum()
Wv=wv.vectors[:K]; Cv=m.syn1neg[:K]
dots=Wv@Cv.T
# Use full-vocab row sums for #(w): compute total weighted pairs per centre word over all vocab
rowfull=np.zeros(K)
for d in range(1,W+1):
    a=ids[:-d]; b=ids[d:]
    w=((W-d+1)/W)*keep[:-d]*keep[d:]
    rowfull+=np.bincount(np.where(a<K,a,K),weights=w,minlength=K+1)[:K]
    rowfull+=np.bincount(np.where(b<K,b,K),weights=w,minlength=K+1)[:K]
k=5
mask=C>=20
pmi=np.log(C[mask]) - np.log(rowfull[:,None].repeat(K,1)[mask]) - np.log(pn[:K][None,:].repeat(K,0)[mask]) - np.log(k)
x=dots[mask]
r=np.corrcoef(pmi,x)[0,1]; sl=np.polyfit(pmi,x,1)
print('pairs',mask.sum(),'corr',round(r,3),'fit slope,int',sl, 'mean pmi',pmi.mean(),'mean dot',x.mean())
# binned for plot
bins=np.linspace(np.percentile(pmi,1),np.percentile(pmi,99),25)
bi=np.digitize(pmi,bins)
pts=[[round(float(pmi[bi==i].mean()),3),round(float(x[bi==i].mean()),3),int((bi==i).sum())] for i in range(1,len(bins)) if (bi==i).sum()>30]
rng=np.random.default_rng(0); sel=rng.choice(len(pmi),1500,replace=False)
json.dump(dict(corr=round(float(r),3),slope=round(float(sl[0]),3),intercept=round(float(sl[1]),3),n=int(mask.sum()),K=K,binned=pts,sample=[[round(float(pmi[i]),2),round(float(x[i]),2)] for i in sel]),open('pmi_check.json','w'))
print(pts[:5])
