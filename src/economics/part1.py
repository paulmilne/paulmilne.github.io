"""Wealth concentration, Part 1: what the data shows.
Reads src/economics/data (see fetch_data.py) and writes figures and a results file."""
import json, warnings, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'axes.grid':True,'grid.alpha':.25,'legend.frameon':False})
C1,C2,C3,GR,BR,PU='#1f5fa8','#d9622b','#2a9d6f','#888888','#b08d3c','#7b4fa3'
D='data/'; out='../../economics/wealth-1-the-facts/'; R={}

# ------------------------------------------------------------------ US: Fed Distributional Financial Accounts
lv=pd.read_csv(D+'dfa/dfa-networth-levels.csv'); lv['t']=pd.PeriodIndex(lv.Date.str.replace(':',''),freq='Q')
G=['TopPt1','RemainingTop1','Next9','Next40','Bottom50']
GN={'TopPt1':'Top 0.1%','RemainingTop1':'Next 0.9%','Next9':'Next 9%','Next40':'Next 40%','Bottom50':'Bottom 50%'}
GC={'TopPt1':C2,'RemainingTop1':BR,'Next9':C1,'Next40':C3,'Bottom50':PU}
nw=lv.pivot(index='t',columns='Category',values='Net worth')[G]; sh=nw.div(nw.sum(axis=1),axis=0)*100
x=sh.index.to_timestamp()
LAST=str(sh.index[-1]); R['last_quarter']=LAST
P=['1996Q2','2006Q2','2016Q2','2026Q2']; P0='1989Q3'
R['shares']={q:{GN[g]:round(sh.loc[q,g],1) for g in G} for q in [P0]+P+['2019Q4','2021Q4','2022Q4']}
R['top1']={q:round(sh.loc[q,['TopPt1','RemainingTop1']].sum(),1) for q in [P0]+P}

fig,ax=plt.subplots(1,2,figsize=(11,4.2),gridspec_kw={'width_ratios':[1.5,1]})
for g in G: ax[0].plot(x,sh[g],color=GC[g],lw=1.8 if g=='TopPt1' else 1.2,label=GN[g])
for g in G: ax[0].text(x[-1]+pd.Timedelta(days=150),sh[g].iloc[-1],f'{sh[g].iloc[-1]:.1f}%',color=GC[g],fontsize=8,va='center')
ax[0].set_ylabel('share of US household net worth (%)'); ax[0].legend(fontsize=8,ncol=5,loc='upper center',bbox_to_anchor=(.5,1.12)); ax[0].set_xlim(x[0],x[-1]+pd.Timedelta(days=900))
spans=[(P0,'1996Q2'),('1996Q2','2006Q2'),('2006Q2','2016Q2'),('2016Q2','2026Q2')]
w=.16
for k,g in enumerate(G):
    d=[sh.loc[b,g]-sh.loc[a,g] for a,b in spans]; ax[1].bar(np.arange(4)+(k-2)*w,d,w,color=GC[g],label=GN[g])
ax[1].axhline(0,color='k',lw=.6); ax[1].set_xticks(range(4)); ax[1].set_xticklabels(['1989–96\n(7 yrs)','1996–2006','2006–16','2016–26'],fontsize=8.5)
ax[1].set_ylabel('change in share (percentage points)'); ax[1].set_title('Change in each group\'s share, by decade',fontsize=9.5)
fig.tight_layout(); fig.savefig(out+'fig_us_shares.png'); plt.close(fig)
R['decade_change']={f'{a}-{b}':{GN[g]:round(sh.loc[b,g]-sh.loc[a,g],2) for g in G} for a,b in spans}

# timing within the last decade, against the value of corporate equities relative to GDP
eq=pd.read_csv(D+'fred/NCBEILQ027S.csv',parse_dates=['observation_date']).set_index('observation_date').iloc[:,0]
gdp=pd.read_csv(D+'fred/GDP.csv',parse_dates=['observation_date']).set_index('observation_date').iloc[:,0]
ff=pd.read_csv(D+'fred/FEDFUNDS.csv',parse_dates=['observation_date']).set_index('observation_date').iloc[:,0]
eqg=(eq/1000/gdp).dropna()*100     # equities ($m) over GDP ($bn, annual rate)
s01=sh.TopPt1.copy(); s01.index=x
j=pd.concat([s01.rename('s'),eqg.rename('e')],axis=1).dropna()
R['corr_level']=round(j.s.corr(j.e),3); R['corr_diff']=round(j.s.diff().corr(j.e.pct_change()),3)
fig,ax=plt.subplots(2,1,figsize=(10,6),sharex=True,gridspec_kw={'height_ratios':[2,1]})
ax[0].plot(j.index,j.s,color=C2,lw=1.8,label='top 0.1% share of net worth (left)'); ax[0].set_ylabel('% of net worth',color=C2)
a2=ax[0].twinx(); a2.plot(j.index,j.e,color=C1,lw=1.2,label='market value of US corporate equities, % of GDP (right)'); a2.set_ylabel('% of GDP',color=C1); a2.spines['top'].set_visible(False); a2.grid(False)
for a,b,lab,yy in [('2008-09','2009-06','financial\ncrisis',8.9),('2020-02','2020-04','pandemic',9.3),('2022-01','2022-10','rate\nrises',8.6)]:
    ax[0].axvspan(pd.Timestamp(a),pd.Timestamp(b),color=GR,alpha=.18); ax[0].text(pd.Timestamp(b)+pd.Timedelta(days=40) if lab!='pandemic' else pd.Timestamp(a)-pd.Timedelta(days=40),yy,lab,fontsize=7.5,color='#555',ha='left' if lab!='pandemic' else 'right')
h1,l1=ax[0].get_legend_handles_labels(); h2,l2=a2.get_legend_handles_labels(); ax[0].legend(h1+h2,l1+l2,fontsize=8,loc='upper left')
ax[0].set_ylim(8.2,16)
ax[1].plot(ff.index,ff,color=GR,lw=1.2); ax[1].set_ylabel('federal funds rate (%)'); ax[1].set_xlim(j.index[0],j.index[-1]); ax[1].set_ylim(0,10)
fig.tight_layout(); fig.savefig(out+'fig_us_equities.png'); plt.close(fig)
R['equities_pct_gdp']={str(k.date()):round(v) for k,v in eqg[[pd.Timestamp('1989-07-01'),pd.Timestamp('2016-04-01'),eqg.index[-1]]].items()}

# shift-share: asset mix (between) and share within each asset class
A=['Real estate','Consumer durables','Corporate equities and mutual fund shares','DB pension entitlements','DC pension entitlements','Unincorporated businesses','Other assets']
AN=['Real estate','Durables','Shares and funds','DB pensions','DC pensions','Private business','Other assets']
def shiftshare(a,b,grp):
    r0=lv[lv.t==a].set_index('Category'); r1=lv[lv.t==b].set_index('Category')
    w0=r0[A].sum()/r0[A].sum().sum(); w1=r1[A].sum()/r1[A].sum().sum()
    s0=r0.loc[grp,A].sum()/r0[A].sum(); s1=r1.loc[grp,A].sum()/r1[A].sum()
    return ((w1-w0)*(s0+s1)/2*100).values, ((s1-s0)*(w0+w1)/2*100).values
fig,ax=plt.subplots(1,2,figsize=(11,4),sharey=True)
for k,(a,b) in enumerate([('2006Q2','2016Q2'),('2016Q2','2026Q2')]):
    be,wi=shiftshare(a,b,['TopPt1']); yy=np.arange(len(A))
    ax[k].barh(yy-.2,be,.4,color=C1,label='asset mix: the class grew or shrank as a share of all wealth'); ax[k].barh(yy+.2,wi,.4,color=C2,label='within: the top 0.1% hold more or less of the class')
    ax[k].set_yticks(yy); ax[k].set_yticklabels(AN); ax[k].invert_yaxis(); ax[k].axvline(0,color='k',lw=.6)
    ax[k].set_title(f'{a[:4]}–{b[:4]}: share of assets {be.sum()+wi.sum():+.1f} pts  =  mix {be.sum():+.1f}  +  within {wi.sum():+.1f}',fontsize=9.5); ax[k].set_xlabel('contribution (percentage points)')
    R[f'shiftshare_{a}_{b}']=dict(total=round(be.sum()+wi.sum(),2),mix=round(be.sum(),2),within=round(wi.sum(),2),mix_shares=round(be[2],2),within_shares=round(wi[2],2))
h,l=ax[0].get_legend_handles_labels(); fig.legend(h,l,fontsize=8.5,loc='upper center',ncol=2,bbox_to_anchor=(.5,1.06))
fig.tight_layout(); fig.savefig(out+'fig_shiftshare.png',bbox_inches='tight'); plt.close(fig)
r0=lv[lv.t=='2016Q2'].set_index('Category'); r1=lv[lv.t=='2026Q2'].set_index('Category')
gain=r1.loc['TopPt1',A]-r0.loc['TopPt1',A]; R['top01_gain_tn']=round(gain.sum()/1e6,1); R['top01_gain_by_asset_pct']=dict(zip(AN,(gain/gain.sum()*100).round(1)))
R['top01_share_of_asset_2026']=dict(zip(AN,(r1.loc['TopPt1',A]/r1[A].sum()*100).round(1))); R['top01_composition_2026']=dict(zip(AN,(r1.loc['TopPt1',A]/r1.loc['TopPt1','Assets']*100).round(1)))
R['top01_composition_2016']=dict(zip(AN,(r0.loc['TopPt1',A]/r0.loc['TopPt1','Assets']*100).round(1)))
R['bottom50_composition_2026']=dict(zip(AN,(r1.loc['Bottom50',A]/r1.loc['Bottom50','Assets']*100).round(1)))
R['total_networth_tn']={'2016Q2':round(nw.loc['2016Q2'].sum()/1e6,1),'2026Q2':round(nw.loc['2026Q2'].sum()/1e6,1)}

# real growth of each group's net worth
cpi=pd.read_csv(D+'fred/CPIAUCSL.csv',parse_dates=['observation_date']).set_index('observation_date').iloc[:,0]
c=lambda q: cpi[pd.Period(q,'Q').start_time:pd.Period(q,'Q').end_time].mean()
R['real_growth']={f'{a}-{b}':{GN[g]:round(nw.loc[b,g]/nw.loc[a,g]/(c(b)/c(a)),2) for g in G} for a,b in [('2006Q2','2016Q2'),('2016Q2','2026Q2')]}
R['inflation_2016_2026']=round(c('2026Q2')/c('2016Q2'),3)

# ------------------------------------------------------------------ Forbes
rows=[]
for y in range(2011,2027):
    for p in json.load(open(D+f'forbes/billionaires_{y}.json'))['personList']['personsLists']:
        rows.append(dict(year=y,name=p.get('personName'),worth=(p.get('finalWorth') or 0)/1000,country=p.get('countryOfCitizenship'),ind=(p.get('industries') or [None])[0],selfmade=p.get('selfMade')))
f=pd.DataFrame(rows); f=f[f.worth>=1]
f['ind']=f.ind.str.replace(' and ',' & ').replace({'Investments':'Finance & Investments','Finance':'Finance & Investments','Health care':'Healthcare','Media':'Media & Entertainment','Gaming':'Gambling & Casinos'})
wb=json.load(open(D+'worldbank/gdp_current_usd.json'))[1]
wg=pd.DataFrame([(r['countryiso3code'],int(r['date']),r['value']) for r in wb],columns=['c','y','v']).dropna(); wg=wg[wg.c=='WLD'].set_index('y').v/1e9
g=f.groupby('year').agg(n=('name','size'),total=('worth','sum'),top10=('worth',lambda s:s.nlargest(10).sum()),top1=('worth','max'))
g['pct_gdp']=[t/wg[y-1]*100 for y,t in zip(g.index,g.total)]; g['top10_share']=g.top10/g.total*100
R['forbes']={int(y):dict(n=int(r.n),total_tn=round(r.total/1000,2),pct_world_gdp=round(r.pct_gdp,1),top10_tn=round(r.top10/1000,2),top10_share=round(r.top10_share,1),richest_bn=round(r.top1)) for y,r in g.iterrows() if y in (2011,2016,2020,2021,2023,2026)}
fig,ax=plt.subplots(1,3,figsize=(12,3.8))
ax[0].bar(g.index,g.total/1000,color=C1); ax[0].set_ylabel('$ trillion'); ax[0].set_title('Total wealth of the world\'s billionaires',fontsize=9.5)
a2=ax[0].twinx(); a2.plot(g.index,g.n,'o-',color=C2,ms=3); a2.set_ylabel('number of billionaires',color=C2); a2.spines['top'].set_visible(False); a2.grid(False)
ax[1].plot(g.index,g.pct_gdp,'o-',color=C1,ms=3); ax[1].set_ylabel('% of previous year\'s world GDP'); ax[1].set_title('…relative to the world economy',fontsize=9.5); ax[1].set_ylim(0,18)
ax[2].plot(g.index,g.top10_share,'o-',color=C2,ms=3); ax[2].set_ylabel('% of all billionaire wealth'); ax[2].set_title('Share held by the ten richest',fontsize=9.5); ax[2].set_ylim(0,14)
fig.tight_layout(); fig.savefig(out+'fig_billionaires.png'); plt.close(fig)

def decomp(a,b):
    A_=f[f.year==a].drop_duplicates('name').set_index('name').worth; B_=f[f.year==b].drop_duplicates('name').set_index('name').worth
    st=A_.index.intersection(B_.index)
    return dict(start=A_.sum(),end=B_.sum(),stayers=len(st),stay_start=A_[st].sum(),stay_end=B_[st].sum(),entrants=len(B_)-len(st),entrant_wealth=B_.drop(st).sum(),exits=len(A_)-len(st),exit_wealth=A_.drop(st).sum())
d1,d0=decomp(2016,2026),decomp(2011,2016)
R['decomp_2016_2026']={k:round(float(v),1) for k,v in d1.items()}; R['decomp_2011_2016']={k:round(float(v),1) for k,v in d0.items()}
R['stayer_multiple']={'2011-2016':round(d0['stay_end']/d0['stay_start'],2),'2016-2026':round(d1['stay_end']/d1['stay_start'],2)}
# top-10 stayers' contribution
A_=f[f.year==2016].drop_duplicates('name').set_index('name').worth; B_=f[f.year==2026].drop_duplicates('name').set_index('name').worth; st=A_.index.intersection(B_.index)
gains=(B_[st]-A_[st]).sort_values(ascending=False); R['top10_gainers_share_of_stayer_gain']=round(gains.head(10).sum()/gains.sum()*100,1); R['top10_gainers']={k:round(v) for k,v in gains.head(10).items()}
R['median_stayer_multiple']=round(float((B_[st]/A_[st]).median()),2)
tot=lambda y,m: f[(f.year==y)&m].worth.sum()/g.total[y]*100
R['us_share']={y:round(tot(y,f.country=='United States'),1) for y in (2016,2026)}; R['tech_share']={y:round(tot(y,f.ind=='Technology'),1) for y in (2016,2026)}
R['selfmade_share']={y:round(tot(y,f.selfmade==True),1) for y in (2016,2026)}
ustot=lv.groupby('Date')['Net worth'].sum()/1e3
R['us_billionaires_pct_us_wealth']={y:round(f[(f.year==y)&(f.country=='United States')].worth.sum()/ustot[f'{y}:Q1']*100,2) for y in (2011,2016,2021,2026)}
R['us_billionaires_n']={y:int(((f.year==y)&(f.country=='United States')).sum()) for y in (2016,2026)}

fig,ax=plt.subplots(1,3,figsize=(12.5,4.4),gridspec_kw={'width_ratios':[1.1,1,1]})
vals=[d1['start'],-d1['exit_wealth'],d1['stay_end']-d1['stay_start'],d1['entrant_wealth'],d1['end']]; labs=['2016','left\nthe list','gains of\nthose who\nstayed','new\nnames','2026']
base=0
for i,(v,l) in enumerate(zip(vals,labs)):
    if i in (0,4): ax[0].bar(i,v/1000,color=GR); 
    else: ax[0].bar(i,v/1000,bottom=base/1000,color=C2 if v<0 else C1)
    ax[0].text(i,(max(base,base+v) if i not in (0,4) else v)/1000+.3,f'{v/1000:+.1f}' if i not in (0,4) else f'{v/1000:.1f}',ha='center',fontsize=8)
    base = v if i==0 else base+v if i<4 else base
ax[0].set_xticks(range(5)); ax[0].set_xticklabels(labs,fontsize=8); ax[0].set_ylabel('$ trillion'); ax[0].set_title('Where the growth came from, 2016–2026',fontsize=9.5)
for k,(col,title) in enumerate([('country','By country of citizenship'),('ind','By sector')]):
    t=f[f.year.isin([2016,2026])].pivot_table(index=col,columns='year',values='worth',aggfunc='sum').fillna(0)/1000
    t['gain']=t[2026]-t[2016]; t=t.sort_values('gain',ascending=False).head(8)[::-1]
    yy=np.arange(len(t)); ax[k+1].barh(yy-.2,t[2016],.4,color=GR,label='2016'); ax[k+1].barh(yy+.2,t[2026],.4,color=C1,label='2026')
    ax[k+1].set_yticks(yy); ax[k+1].set_yticklabels(t.index,fontsize=8); ax[k+1].set_xlabel('$ trillion'); ax[k+1].set_title(title,fontsize=9.5); ax[k+1].legend(fontsize=8,loc='lower right')
    R['gain_by_'+col]={i:round(r.gain,2) for i,r in t[::-1].iterrows()}
fig.tight_layout(); fig.savefig(out+'fig_billionaire_sources.png'); plt.close(fig)
R['total_gain_tn']=round((d1['end']-d1['start'])/1000,2)

# Pareto tail. Measured by rank (top 100, top 1,000), because a fixed $1bn threshold gives a misleading trend
# when the whole list grows: the distribution is not an exact Pareto, so the estimate depends on where it is cut.
def hill_k(wv,k): wv=np.sort(wv)[::-1][:k]; return (k-1)/np.log(wv[:-1]/wv[-1]).sum()
def hill(wv): wv=np.sort(wv)[::-1]; return len(wv)/np.log(wv/wv.min()).sum()
al={int(y):dict(all=hill(dd.worth.values),top1000=hill_k(dd.worth.values,1000),top100=hill_k(dd.worth.values,100)) for y,dd in f.groupby('year')}
R['pareto_alpha']={y:{k:round(v,2) for k,v in d.items()} for y,d in al.items()}
t1000=f.groupby('year').worth.apply(lambda s_: np.sort(s_.values)[::-1][:1000]); R['top10_of_top1000']={int(y):round(v[:10].sum()/v.sum()*100,1) for y,v in t1000.items()}; R['top100_of_top1000']={int(y):round(v[:100].sum()/v.sum()*100,1) for y,v in t1000.items()}
fig,ax=plt.subplots(1,2,figsize=(10.5,4))
for y,cc in ((2016,GR),(2020,C1),(2026,C2)):
    wv=t1000[y]; ax[0].loglog(wv/wv[-1],np.arange(1,1001),'.',color=cc,ms=3,label=str(y))
ax[0].set_xlabel('wealth, as a multiple of the 1,000th fortune'); ax[0].set_ylabel('rank'); ax[0].legend(fontsize=8.5); ax[0].set_title('The top 1,000: rank against relative wealth',fontsize=9.5)
ys=sorted(al); ax[1].plot(ys,[al[y]['top100'] for y in ys],'o-',color=C2,ms=3,label='top 100'); ax[1].plot(ys,[al[y]['top1000'] for y in ys],'s-',color=C1,ms=3,label='top 1,000')
ax[1].plot(ys,[al[y]['all'] for y in ys],':',color=GR,label='everyone above $1bn (fixed threshold)'); ax[1].legend(fontsize=8); ax[1].set_ylim(.8,2.2)
ax[1].set_ylabel('Pareto exponent α (lower = fatter tail)'); ax[1].set_title('The tail got fatter among the top 100',fontsize=9.5)
fig.tight_layout(); fig.savefig(out+'fig_pareto.png'); plt.close(fig)

# ------------------------------------------------------------------ cross-country (World Inequality Database, via Our World in Data)
o=pd.read_csv(D+'owid/wealth-share-richest-1-percent.csv'); o.columns=['entity','code','year','top1']
sel=['United States','United Kingdom','France','Germany','China','India','World']
p=o[o.entity.isin(sel)].pivot(index='year',columns='entity',values='top1')
R['wid_top1']={int(y):p.loc[y,sel].round(1).to_dict() for y in (1995,2005,2015,int(p.index.max()))}
fig,ax=plt.subplots(figsize=(8.5,4.3))
cols=dict(zip(sel,[C2,C1,C3,BR,'#c0392b',PU,'k']))
for e in sel: ax.plot(p.index,p[e],color=cols[e],lw=2 if e=='World' else 1.3,ls='--' if e=='World' else '-'); ax.text(p.index.max()+.4,p[e].iloc[-1],e,color=cols[e],fontsize=8,va='center')
ax.axvspan(2015,p.index.max(),color=GR,alpha=.12); ax.set_xlim(1980,p.index.max()+9); ax.set_ylabel('top 1% share of household wealth (%)')
ax.set_title('Top 1% wealth shares, World Inequality Database',fontsize=9.5)
fig.tight_layout(); fig.savefig(out+'fig_countries.png'); plt.close(fig)

# thumbnail and data for the interactive
fig,ax=plt.subplots(figsize=(6,4)); ax.plot(x,sh.TopPt1,color=C2,lw=2.5); ax.fill_between(x,8,sh.TopPt1,color=C2,alpha=.15)
ax.set_title('Share of US wealth held by the top 0.1%',fontsize=11); ax.set_ylim(8,16); ax.set_ylabel('%'); fig.tight_layout(); fig.savefig(out+'thumbnail.png'); plt.close(fig)
lvl=nw.copy(); defl=pd.Series([c(str(q)) for q in nw.index],index=nw.index); hh=133e6  # households, approximate, for per-household averages
J={'quarters':[str(q) for q in sh.index],'groups':[GN[g] for g in G],'share':{GN[g]:sh[g].round(2).tolist() for g in G},
   'real_index':{GN[g]:(nw[g]/defl/(nw[g]/defl).loc['1989Q3']*100).round(1).tolist() for g in G},
   'assets':AN,'asset_share':{GN[g]:{an:(lv[lv.Category==g].set_index('t')[a]/lv.groupby('t')[a].sum()*100).round(2).tolist() for a,an in zip(A,AN)} for g in G}}
json.dump(J,open(out+'dfa.json','w'),separators=(',',':'))
json.dump(R,open('part1_results.json','w'),indent=1,default=float)
print({k:R[k] for k in ['corr_level','corr_diff','equities_pct_gdp','real_growth','inflation_2016_2026','top01_gain_tn','top01_gain_by_asset_pct','top01_share_of_asset_2026','top01_composition_2016','top01_composition_2026','bottom50_composition_2026','total_networth_tn','shiftshare_2006Q2_2016Q2','shiftshare_2016Q2_2026Q2','decade_change','top1']})
