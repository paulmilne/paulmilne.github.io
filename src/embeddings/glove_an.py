import gzip, numpy as np, os, gensim, json, time
t=time.time()
f=gzip.open('dl/glove100.gz','rt'); f.readline()
words=[];vecs=[]
for i,l in enumerate(f):
    if i>=100000: break
    p=l.rstrip().split(' '); words.append(p[0]); vecs.append(np.array(p[1:],dtype=np.float32))
E=np.vstack(vecs); idx={w:i for i,w in enumerate(words)}
U=E/np.linalg.norm(E,axis=1,keepdims=True)
np.save('glove100k.npy',E); json.dump(words,open('glove100k_words.json','w'))
qf=os.path.join(os.path.dirname(gensim.__file__),'test','test_data','questions-words.txt')
secs={};cur=None
for l in open(qf):
    if l.startswith(':'): cur=l[2:].strip(); secs[cur]=[]; continue
    q=l.lower().split()
    if all(w in idx for w in q): secs[cur].append([idx[w] for w in q])
res={}
tot=dict(ex=0,inc=0,n=0); sem=dict(ex=0,inc=0,n=0); syn=dict(ex=0,inc=0,n=0)
for s,qs in secs.items():
    Q=np.array(qs)
    T=U[Q[:,1]]-U[Q[:,0]]+U[Q[:,2]]
    S=T@U.T
    best_inc=S.argmax(1)
    S2=S.copy(); r=np.arange(len(Q))
    for j in range(3): S2[r,Q[:,j]]=-9
    best_ex=S2.argmax(1)
    ex=(best_ex==Q[:,3]).sum(); inc=(best_inc==Q[:,3]).sum()
    # what does inc return when wrong? fraction that is an input word
    inp=np.isin(best_inc,Q[:,:3]).sum() if False else sum(best_inc[i] in Q[i,:3] for i in range(len(Q)))
    res[s]=dict(n=len(Q),excl=int(ex),incl=int(inc),incl_input=int(inp))
    g=syn if s.startswith('gram') else sem
    for d in (g,tot): d['ex']+=int(ex); d['inc']+=int(inc); d['n']+=len(Q)
print(json.dumps(res,indent=0)[:3000])
print('sem',sem,'syn',syn,'tot',tot, {k:round(v['ex']/v['n'],3) for k,v in dict(sem=sem,syn=syn,tot=tot).items()},{k:round(v['inc']/v['n'],3) for k,v in dict(sem=sem,syn=syn,tot=tot).items()})
json.dump(dict(sections=res,sem=sem,syn=syn,tot=tot),open('glove_analogy.json','w'))
def nn(w,k=8):
    s=U@U[idx[w]]; o=np.argsort(-s)[1:k+1]; return [(words[i],round(float(s[i]),3)) for i in o]
for w in ['king','queen','paris','computer','bank','apple']: print(w,nn(w))
v=U[idx['king']]-U[idx['man']]+U[idx['woman']]; s=U@(v/np.linalg.norm(v)); o=np.argsort(-s)[:6]; print('k-m+w',[(words[i],round(float(s[i]),3)) for i in o])
print(time.time()-t)
