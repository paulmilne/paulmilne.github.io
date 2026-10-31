import re, math, json, collections
STOP=set("a an the of to in and or is are was were be been it its on for with as by at from that this which what how why where when who whose into than then there their they them can does do did not no".split())
def toks(s): return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP]
class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        self.D=[toks(d) for d in docs]; self.N=len(docs); self.avg=sum(map(len,self.D))/self.N
        self.df=collections.Counter(w for d in self.D for w in set(d)); self.k1=k1; self.b=b
        self.tf=[collections.Counter(d) for d in self.D]
    def idf(self,w): n=self.df.get(w,0); return math.log(1+(self.N-n+.5)/(n+.5))
    def scores(self,q):
        qs=toks(q); out=[]
        for tf,d in zip(self.tf,self.D):
            s=0
            for w in qs:
                f=tf.get(w,0)
                if f: s+=self.idf(w)*f*(self.k1+1)/(f+self.k1*(1-self.b+self.b*len(d)/self.avg))
            out.append(s)
        return out
if __name__=="__main__":
    d=json.load(open('blog_chunks.json')); Q=json.load(open('queries.json'))
    docs=[d['posts'][c['post']]['title']+'. '+c['heading']+'. '+c['text'] for c in d['chunks']]
    bm=BM25(docs); posts=[c['post'] for c in d['chunks']]
    def rank_posts(sc):
        best={}
        for s,p in zip(sc,posts): best[p]=max(best.get(p,-1e9),s)
        return sorted(best,key=lambda p:-best[p])
    hits=0; mrr=0
    for q,rel in Q:
        r=rank_posts(bm.scores(q)); 
        ranks=[r.index(p)+1 for p in rel if p in r]; first=min(ranks)
        hits+= first<=3; mrr+=1/first
        print(first, q, [d['posts'][p]['title'][:30] for p in r[:3]])
    print('BM25 hit@3',hits/len(Q),'MRR',mrr/len(Q))
