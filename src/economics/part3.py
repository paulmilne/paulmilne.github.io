"""Wealth concentration, Part 3: a model of the feedback loops.
Uses wealth_model.py (a fixed-size model of the top 1,000 fortunes driven by
the empirical growth, exit and entry distributions). Writes figures, a data
file for the browser model, and part3_results.json."""
import json, warnings, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from wealth_model import load, Model, stats, LABS
warnings.filterwarnings('ignore')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'axes.grid':True,'grid.alpha':.25,'legend.frameon':False})
C1,C2,C3,GR,BR,PU='#1f5fa8','#d9622b','#2a9d6f','#888888','#b08d3c','#7b4fa3'
out='../../economics/wealth-3-the-model/'; R={}
W=load(); M=Model(W); YR=list(range(2011,2027)); act={y:stats(M.start[y]) for y in YR}
R['actual_t10']={y:round(act[y]['t10'],1) for y in YR}; R['actual_t100']={y:round(act[y]['t100'],1) for y in YR}; R['actual_a100']={y:round(act[y]['a100'],2) for y in YR}; R['actual_a1000']={y:round(act[y]['a1000'],2) for y in YR}
q=lambda o,k='t10': np.percentile(np.array([[p[k] for p in path] for path in o]),[10,50,90],axis=0)

# ---- F1: multiplicative growth alone, and with turnover (parametric)
new_all=np.concatenate([M.new[y] for y in range(2011,2020)])
def para(sigma=.25,df=4,turn=156,exitp=.055,adv=0.0,years=150,seed=0,n=1000,equal=True):
    rng=np.random.default_rng(seed); x=np.zeros(n) if equal else np.sort(rng.exponential(1/1.3,n))[::-1]; sc=sigma/np.sqrt(df/(df-2)); path=[1.0 if equal else np.nan]
    for t in range(years):
        g=sc*rng.standard_t(df,len(x)); o=np.argsort(-x); g[o[:10]]+=adv; x=x+g
        if turn:
            x=np.sort(x[rng.random(len(x))>=exitp])[::-1]; ref=x[min(n-turn,len(x))-1]      # where the new threshold will fall
            x=np.sort(np.concatenate([x,ref+rng.choice(new_all,turn)]))[::-1][:n]
        x=x-x.min(); w=np.sort(np.exp(x-x.max()))[::-1]; path.append(w[:10].sum()/w.sum()*100)
    return np.array(path)
A=np.array([para(turn=0,seed=s) for s in range(40)]); B=np.array([para(seed=s) for s in range(40)])
fig,ax=plt.subplots(figsize=(8.2,4)); t=np.arange(A.shape[1])
for P,c,l in ((A,C2,'random growth only'),(B,C1,'random growth, with 15% of members replaced each year')):
    ax.fill_between(t,np.percentile(P,10,0),np.percentile(P,90,0),color=c,alpha=.18); ax.plot(t,np.median(P,0),color=c,lw=1.8,label=l)
ax.set_xlabel('years'); ax.set_ylabel('share held by the ten richest of 1,000 (%)'); ax.set_ylim(0,100); ax.legend(fontsize=8.5,loc='upper left'); ax.set_title('A thousand equal fortunes, each growing by a random percentage every year',fontsize=9.5)
fig.tight_layout(); fig.savefig(out+'fig_gibrat.png'); plt.close(fig)
R['gibrat']={'no_turnover':{int(k):round(float(np.median(A[:,k])),1) for k in (10,25,50,100,150)},'with_turnover':{int(k):round(float(np.median(B[:,k])),1) for k in (10,25,50,100,150)}}

# ---- F2: the measured inputs
sd={y:float(np.std(M.g[y][3])) for y in M.years}; turn={y:len(M.new[y]) for y in M.years}
PER={'2011–19':range(2011,2020),'2020–22':range(2020,2023),'2023–25':range(2023,2026)}; edge={}
for n,ys in PER.items():
    top=np.concatenate([M.g[y][0]-M.g[y][3].mean() for y in ys]); rest=np.concatenate([np.concatenate([M.g[y][1],M.g[y][2]])-M.g[y][3].mean() for y in ys])
    edge[n]=dict(top10=float(top.mean()),se=float(top.std(ddof=1)/np.sqrt(len(top))),rest=float(rest.mean()),n=len(top),sd_all=float(np.std(np.concatenate([M.g[y][3]-M.g[y][3].mean() for y in ys]))),turnover=float(np.mean([turn[y] for y in ys])))
R['inputs']={k:{a:round(b,3) for a,b in v.items()} for k,v in edge.items()}; R['sd_by_year']={y:round(v,3) for y,v in sd.items()}; R['turnover_by_year']=turn
fig,ax=plt.subplots(1,3,figsize=(12,3.7))
ax[0].bar(list(sd),list(sd.values()),color=[GR if y<2020 else C2 if y<2023 else C1 for y in sd]); ax[0].set_title('Spread of one-year growth rates\n(standard deviation of log growth)',fontsize=9.5); ax[0].set_ylim(0,.42); ax[0].set_xticks(range(2011,2026,2))
ax[1].bar(list(turn),list(turn.values()),color=[GR if y<2020 else C2 if y<2023 else C1 for y in turn]); ax[1].set_title('Turnover: new members of the top 1,000\neach year',fontsize=9.5); ax[1].set_xticks(range(2011,2026,2))
xx=np.arange(3); ax[2].bar(xx,[(edge[n]['top10']-edge[n]['rest'])*100 for n in PER],yerr=[edge[n]['se']*100 for n in PER],color=[GR,C2,C1],capsize=4); ax[2].axhline(0,color='k',lw=.6)
ax[2].set_xticks(xx); ax[2].set_xticklabels(PER); ax[2].set_ylabel('% a year'); ax[2].set_title('Growth advantage of the ten richest\nover the rest of the top 1,000',fontsize=9.5)
fig.tight_layout(); fig.savefig(out+'fig_inputs.png'); plt.close(fig)

# persistence of growth
G=np.log(W).diff(axis=1); cs=[]
for y in range(2012,2025):
    a,b=G[y],G[y+1]; m=a.notna()&b.notna(); cs.append((y,float(np.corrcoef(a[m],b[m])[0,1])))
a=np.log(W[2023]/W[2020]); b=np.log(W[2026]/W[2023]); m=a.notna()&b.notna()
R['persistence']=dict(mean_1y=round(float(np.mean([c for _,c in cs])),3),corr_2020_23_vs_2023_26=round(float(np.corrcoef(a[m],b[m])[0,1]),3),n=int(m.sum()))

# ---- F3: validation
VAL=[(2011,2016),(2016,2020),(2020,2023),(2023,2026)]; val={}
fig,ax=plt.subplots(1,4,figsize=(12.5,3.9),sharey=True)
for k,(y0,y1) in enumerate(VAL):
    e=M.run(y0,years=range(y0,y1),nsim=800,seed=3); r=M.run(y0,years=range(y0,y1),by_rank=True,nsim=800,seed=4)
    ve=np.array([p[-1]['t10'] for p in e]); vr=np.array([p[-1]['t10'] for p in r]); a1=act[y1]['t10']
    bp=ax[k].boxplot([ve,vr],widths=.5,showfliers=False,whis=(5,95),patch_artist=True,tick_labels=['equal\nodds','by rank\ngroup'])
    for b_,c in zip(bp['boxes'],(C1,BR)): b_.set_facecolor(c); b_.set_alpha(.5)
    ax[k].axhline(a1,color=C2,lw=2); ax[k].axhline(act[y0]['t10'],color=GR,lw=1,ls='--'); ax[k].set_title(f'{y0} → {y1}',fontsize=10)
    val[f'{y0}-{y1}']=dict(start=round(act[y0]['t10'],1),actual=round(a1,1),equal_median=round(float(np.median(ve)),1),equal_pct=round(float((ve<a1).mean()*100)),rank_median=round(float(np.median(vr)),1),rank_pct=round(float((vr<a1).mean()*100)),
                          equal_t100_pct=round(float((np.array([p[-1]['t100'] for p in e])<act[y1]['t100']).mean()*100)))
ax[0].set_ylabel('top-10 share of the top 1,000 (%)'); ax[0].plot([],[],color=C2,lw=2,label='actual, end of period'); ax[0].plot([],[],color=GR,ls='--',label='actual, start'); ax[0].legend(fontsize=8,loc='upper left')
fig.tight_layout(); fig.savefig(out+'fig_validation.png'); plt.close(fig); R['validation']=val

# ---- F4: projections
SC={'2011–19 conditions return':(dict(pool=list(range(2011,2020)),by_rank=False),C1),'2020–22 conditions (wide spread, equal odds)':(dict(pool=[2020,2021,2022],by_rank=False),BR),
    '2023–25 conditions continue':(dict(pool=[2023,2024,2025],by_rank=True),C2),'ten richest lose half, then 2011–19 conditions':(dict(pool=list(range(2011,2020)),by_rank=False,shock=(10,.5)),C3)}
H=25; proj={}
fig,ax=plt.subplots(figsize=(9.5,4.8)); ax.plot(YR,[act[y]['t10'] for y in YR],color='k',lw=2,label='actual')
for nm,(kw,c) in SC.items():
    Q=q(M.run(2026,horizon=H,nsim=500,seed=5,**kw)); tt=2026+np.arange(H+1); ax.fill_between(tt,Q[0],Q[2],color=c,alpha=.13); ax.plot(tt,Q[1],color=c,lw=1.6,label=nm)
    proj[nm]={f'+{k}':[round(float(Q[1,k]),1),round(float(Q[0,k]),1),round(float(Q[2,k]),1)] for k in (1,5,10,25)}
ax.set_ylabel('top-10 share of the top 1,000 (%)'); ax.set_ylim(0,85); ax.legend(fontsize=8,loc='upper left'); ax.set_title('What each set of conditions implies if it persisted (median and 80% range)',fontsize=9.5)
fig.tight_layout(); fig.savefig(out+'fig_projection.png'); plt.close(fig); R['projection']=proj
ss={}
for nm,kw in {'2011–19':dict(pool=list(range(2011,2020)),by_rank=False),'2020–22':dict(pool=[2020,2021,2022],by_rank=False),'2023–25 by rank':dict(pool=[2023,2024,2025],by_rank=True)}.items():
    Q=q(M.run(2011,horizon=80,nsim=150,seed=6,**kw)); ss[nm]={'+40':[round(float(v),1) for v in Q[:,40]],'+80':[round(float(v),1) for v in Q[:,80]]}
R['steady_state']=ss

# thumbnail and browser data
fig,ax=plt.subplots(figsize=(6,4)); ax.plot(YR,[act[y]['t10'] for y in YR],color='k',lw=2.5)
for nm,(kw,c) in list(SC.items())[:3]:
    Q=q(M.run(2026,horizon=12,nsim=150,seed=7,**kw)); ax.plot(2026+np.arange(13),Q[1],color=c,lw=2)
ax.set_title('Share of the ten richest: three futures',fontsize=11); ax.set_ylabel('%'); fig.tight_layout(); fig.savefig(out+'thumbnail.png'); plt.close(fig)
json.dump({'years':YR,'actual_t10':[round(act[y]['t10'],2) for y in YR],'start2011':np.round(M.start[2011],3).tolist(),'start2026':np.round(M.start[2026],3).tolist(),
           'entrants':np.round(np.quantile(new_all,np.linspace(.0025,.9975,200)),3).tolist()},open(out+'model.json','w'),separators=(',',':'))
json.dump(R,open('part3_results.json','w'),indent=1,default=float)
for k in ('gibrat','inputs','persistence','validation','projection','steady_state','actual_a100','actual_a1000','actual_t100'): print(k,json.dumps(R[k],default=float,ensure_ascii=False))
