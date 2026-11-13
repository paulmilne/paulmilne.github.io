"""Wealth concentration, Part 2: testing the candidate causes.
Reads src/economics/data (see fetch_data.py); writes figures and part2_results.json."""
import json, warnings, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'axes.grid':True,'grid.alpha':.25,'legend.frameon':False})
C1,C2,C3,GR,BR,PU='#1f5fa8','#d9622b','#2a9d6f','#888888','#b08d3c','#7b4fa3'
D='data/'; out='../../economics/wealth-2-the-causes/'; R={}
fr=lambda s: pd.read_csv(D+f'fred/{s}.csv',parse_dates=['observation_date']).set_index('observation_date').iloc[:,0]

# ------------------------------------------------------------ 1. equities = profits x multiple; rates and QE
df=pd.concat([(fr('NCBEILQ027S')/1000).rename('eq'),fr('GDP').rename('gdp'),fr('CP').rename('cp'),fr('DFII10').resample('QS').mean().rename('ry'),
              fr('FEDFUNDS').resample('QS').mean().rename('ff'),(fr('WALCL').resample('QS').mean()/1e6).rename('fed'),fr('PRS85006173').rename('ls')],axis=1).dropna(subset=['eq','gdp'])
df['eg']=df['eq']/df['gdp']*100; df['pg']=df['cp']/df['gdp']*100; df['pe']=df['eq']/df['cp']
T=pd.Timestamp
periods=[('1989-07-01','2016-04-01','1989–2016'),('2007-07-01','2019-10-01','2007–19\n(peak to peak)'),('2019-10-01','2021-10-01','2019–21\nfirst burst'),('2021-10-01','2022-10-01','2022\nrates rise'),('2022-10-01','2026-04-01','2022–26\nsecond burst'),('2016-04-01','2026-04-01','2016–26\nwhole decade')]
R['equity_points']={k:dict(eq_gdp=round(df.loc[k,'eg']),profit_gdp=round(df.loc[k,'pg'],1),multiple=round(df.loc[k,'pe'],1),real_yield=None if pd.isna(df.loc[k,'ry']) else round(df.loc[k,'ry'],2),fed_tn=None if pd.isna(df.loc[k,'fed']) else round(df.loc[k,'fed'],1)) for k in ['1989-07-01','2007-07-01','2016-04-01','2019-10-01','2021-10-01','2022-10-01','2026-04-01']}
dec=[]
for a,b,n in periods:
    A,B=df.loc[a],df.loc[b]; dec.append(dict(n=n,total=np.log(B.eg/A.eg)*100,profit=np.log(B.pg/A.pg)*100,mult=np.log(B.pe/A.pe)*100,dry=B.ry-A.ry,dfed=B.fed-A.fed))
R['equity_decomp']=[{k:(v if isinstance(v,str) else (None if pd.isna(v) else round(float(v),1))) for k,v in d.items()} for d in dec]
fig,ax=plt.subplots(1,2,figsize=(11.5,4.3),gridspec_kw={'width_ratios':[1.15,1]})
x=np.arange(len(dec)); ax[0].bar(x-.2,[d['profit'] for d in dec],.4,color=C3,label='profits as a share of GDP'); ax[0].bar(x+.2,[d['mult'] for d in dec],.4,color=C2,label='valuation multiple (equity value ÷ profits)')
ax[0].axhline(0,color='k',lw=.6); ax[0].set_xticks(x); ax[0].set_xticklabels([d['n'] for d in dec],fontsize=7.5); ax[0].set_ylabel('contribution to change in equities/GDP (log points × 100)'); ax[0].legend(fontsize=8)
ax[0].set_title('What moved the value of US equities relative to GDP',fontsize=9.5)
a2=ax[1]; a2.plot(df.index,df.pg,color=C3,label='after-tax corporate profits, % of GDP (left)'); a2.set_ylabel('% of GDP',color=C3); a3=a2.twinx(); a3.plot(df.index,df.ls,color=PU,label='labour share index, nonfarm business (right)'); a3.set_ylabel('index, 2017 = 100',color=PU); a3.spines['top'].set_visible(False); a3.grid(False)
h1,l1=a2.get_legend_handles_labels(); h2,l2=a3.get_legend_handles_labels(); a2.legend(h1+h2,l1+l2,fontsize=8,loc='upper center'); a2.set_xlim(T('1989-01-01'),df.index[-1]); a2.set_title('Profits up, labour share down',fontsize=9.5)
fig.tight_layout(); fig.savefig(out+'fig_equity_decomp.png'); plt.close(fig)
R['labour_share_index']={k:round(df.loc[k,'ls'],1) for k in ['1989-07-01','2000-01-01','2016-04-01','2026-04-01']}

lv=pd.read_csv(D+'dfa/dfa-networth-levels.csv'); lv['t']=pd.PeriodIndex(lv.Date.str.replace(':',''),freq='Q').to_timestamp()
nw=lv.pivot(index='t',columns='Category',values='Net worth'); s01=(nw.TopPt1/nw.sum(axis=1)*100)
x_=df.dropna(subset=['ry']); pre=x_[:'2021-12-31']; post=x_['2022-01-01':]
R['corr_yield_multiple']={'2003-2021':round(pre.ry.corr(pre.pe),2),'2022-2026':round(post.ry.corr(post.pe),2)}
el=np.log(df.loc['2022-10-01','pe']/df.loc['2021-10-01','pe'])/(df.loc['2022-10-01','ry']-df.loc['2021-10-01','ry'])
pred=el*(df.loc['2021-10-01','ry']-df.loc['2007-07-01','ry']); act=np.log(df.loc['2021-10-01','pe']/df.loc['2007-07-01','pe'])
R['rates_backofenvelope']=dict(pct_per_point=round((np.exp(el)-1)*100,1),d_yield_2007_2021=round(df.loc['2021-10-01','ry']-df.loc['2007-07-01','ry'],2),predicted_rise_pct=round((np.exp(pred)-1)*100),actual_rise_pct=round((np.exp(act)-1)*100),share_explained_pct=round(pred/act*100))
fig,ax=plt.subplots(1,2,figsize=(11.5,4.2))
ax[0].scatter(pre.ry,pre.pe,s=14,color=C1,label=f'2003–2021 (correlation {pre.ry.corr(pre.pe):+.2f})'); ax[0].scatter(post.ry,post.pe,s=22,color=C2,marker='s',label=f'2022–2026 (correlation {post.ry.corr(post.pe):+.2f})')
ax[0].plot(post.ry,post.pe,color=C2,lw=.6,alpha=.6); ax[0].annotate('2026 Q2',(post.ry.iloc[-1],post.pe.iloc[-1]),xytext=(6,4),textcoords='offset points',fontsize=8,color=C2); ax[0].annotate('2021 Q4',(df.loc['2021-10-01','ry'],df.loc['2021-10-01','pe']),xytext=(6,4),textcoords='offset points',fontsize=8,color=C1)
ax[0].set_xlabel('10-year real interest rate (%)'); ax[0].set_ylabel('valuation multiple (equity value ÷ profits)'); ax[0].legend(fontsize=8); ax[0].set_title('Low real rates went with high valuations, until 2022',fontsize=9.5)
ax[1].plot(df.index,df.fed,color=C1,label='Federal Reserve assets, $ trillion (left)'); ax[1].set_ylabel('$ trillion',color=C1); a3=ax[1].twinx(); a3.plot(s01.index,s01,color=C2,lw=1.6,label='top 0.1% share of net worth, % (right)'); a3.set_ylabel('%',color=C2); a3.spines['top'].set_visible(False); a3.grid(False)
h1,l1=ax[1].get_legend_handles_labels(); h2,l2=a3.get_legend_handles_labels(); ax[1].legend(h1+h2,l1+l2,fontsize=8,loc='upper left'); ax[1].set_xlim(T('2003-01-01'),df.index[-1]); ax[1].set_title('The Fed shrank its balance sheet; the top share kept rising',fontsize=9.5)
fig.tight_layout(); fig.savefig(out+'fig_rates.png'); plt.close(fig)
R['top01_share']={k:round(s01[T(k)],1) for k in ['2007-10-01','2009-01-01','2019-10-01','2021-10-01','2022-10-01','2026-04-01','2000-01-01','2003-01-01']}

# ------------------------------------------------------------ Forbes panel
rows=[]
for y in range(2011,2027):
    for p in json.load(open(D+f'forbes/billionaires_{y}.json'))['personList']['personsLists']:
        rows.append(dict(year=y,name=p.get('personName'),worth=(p.get('finalWorth') or 0)/1000,country=p.get('countryOfCitizenship'),ind=(p.get('industries') or [None])[0],selfmade=p.get('selfMade'),src=(p.get('source') or '').lower(),birth=p.get('birthDate')))
f=pd.DataFrame(rows); f=f[f.worth>=1].drop_duplicates(['year','name'])
f['ind']=f.ind.str.replace(' and ',' & ').replace({'Investments':'Finance & Investments','Finance':'Finance & Investments','Health care':'Healthcare','Media':'Media & Entertainment','Gaming':'Gambling & Casinos'})
f['age']=f.year-pd.to_datetime(f.birth,unit='ms',errors='coerce').dt.year
tot=f.groupby('year').worth.sum()

# 2. concentration: published S&P 500 figures (RBC Wealth Management) and the Forbes list
SP={'1990':19,'2000 (peak)':27,'2015':19,'2025':40.7}; R['sp500_top10_weight']=SP; R['sp500_top10_earnings_2025']=32
tech=f[f.ind=='Technology'].groupby('year').worth.sum()/tot*100; top10=f.groupby('year').worth.apply(lambda s:s.nlargest(10).sum())/tot*100
fig,ax=plt.subplots(1,2,figsize=(10.5,3.9))
ax[0].bar(range(4),list(SP.values()),color=[GR,GR,GR,C2]); ax[0].bar(3.0,32,width=.35,color=C3,label='their share of index earnings, 2025'); ax[0].set_xticks(range(4)); ax[0].set_xticklabels(SP.keys()); ax[0].set_ylabel('% of S&P 500 market value'); ax[0].legend(fontsize=8,loc='upper left')
for i,v in enumerate(SP.values()): ax[0].text(i,v+.8,f'{v:g}%',ha='center',fontsize=8.5)
ax[0].set_title('Weight of the ten largest companies in the S&P 500',fontsize=9.5)
ax[1].plot(tech.index,tech,'o-',color=C1,ms=3,label='technology fortunes'); ax[1].plot(top10.index,top10,'s-',color=C2,ms=3,label='the ten richest people'); ax[1].set_ylabel('% of all billionaire wealth'); ax[1].legend(fontsize=8); ax[1].set_ylim(0,26); ax[1].set_title('Shares of world billionaire wealth',fontsize=9.5)
fig.tight_layout(); fig.savefig(out+'fig_concentration.png'); plt.close(fig)
R['tech_share']={int(y):round(v,1) for y,v in tech.items() if y in (2011,2016,2021,2026)}

# 3. scale dependence: one-year growth by size of fortune (people on the list in both years)
w=f.pivot(index='name',columns='year',values='worth'); recs=[]
for y in range(2011,2026):
    a=w[y].dropna(); recs.append(pd.DataFrame(dict(year=y,w0=a,w1=w[y+1].reindex(a.index))))
p=pd.concat(recs); p['stay']=p.w1.notna(); bins=[1,2,5,10,20,50,1e4]; lab=['1–2','2–5','5–10','10–20','20–50','50+']; p['b']=pd.cut(p.w0,bins,labels=lab,right=False)
PER={'2011–15':(2011,2015),'2016–19':(2016,2019),'2020–25':(2020,2025)}; sc={}
for n,(lo,hi) in PER.items():
    q=p[(p.year>=lo)&(p.year<=hi)&p.stay]
    g=q.groupby('b').apply(lambda d:(d.w1.sum()/d.w0.sum()-1)*100); med=q.groupby('b').apply(lambda d:(d.w1/d.w0-1).median()*100)
    sc[n]=dict(pooled=g.round(1).to_dict(),median=med.round(1).to_dict(),n=q.groupby('b').size().to_dict())
R['scale']=sc; R['exit_rate']={n:round((1-p[(p.year>=lo)&(p.year<=hi)].groupby('b').stay.mean()).mul(100).round(1).to_dict()['2–5'],1) for n,(lo,hi) in PER.items()}
fig,ax=plt.subplots(1,2,figsize=(10.5,4)); xx=np.arange(1,6)
for k,(n,cc) in enumerate(zip(PER,(GR,C1,C2))):
    ax[0].bar(xx+(k-1)*.27,[sc[n]['pooled'][l] for l in lab[1:]],.27,color=cc,label=n); ax[1].bar(xx+(k-1)*.27,[sc[n]['median'][l] for l in lab[1:]],.27,color=cc,label=n)
for a,t in zip(ax,('Growth of each bracket\'s combined wealth','Growth of the median member')):
    a.axhline(0,color='k',lw=.6); a.set_xticks(xx); a.set_xticklabels(lab[1:]); a.set_xlabel('wealth at the start of the year ($ billion)'); a.set_ylabel('one-year growth (%)'); a.set_title(t,fontsize=9.5); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(out+'fig_scale.png'); plt.close(fig)

# 4. mobility
mob={}
for a,b in [(2011,2016),(2016,2021),(2021,2026),(2011,2021),(2016,2026)]:
    A=set(f[f.year==a].nlargest(100,'worth').name); B=set(f[f.year==b].nlargest(100,'worth').name); L=set(f[f.year==b].name); AA=set(f[f.year==a].name)
    mob[f'{a}-{b}']=dict(top100_still_top100=len(A&B),top100_still_listed=len(A&L),all_still_listed_pct=round(len(AA&L)/len(AA)*100,1))
R['mobility']=mob; ex=p.groupby('year').stay.apply(lambda s:(1-s.mean())*100)
R['annual_exit_rate']={int(y):round(v,1) for y,v in ex.items()}
R['age_median']={int(y):float(v) for y,v in f.groupby('year').age.median().items() if y in (2011,2016,2021,2026)}
R['selfmade_share']={int(y):round(f[(f.year==y)&(f.selfmade==True)].worth.sum()/tot[y]*100,1) for y in (2011,2016,2021,2026)}
# who are the ten richest, then and now
R['top10']={int(y):f[f.year==y].nlargest(10,'worth')[['name','worth','src']].values.tolist() for y in (2016,2026)}
R['top10_overlap']=len(set(f[f.year==2016].nlargest(10,'worth').name)&set(f[f.year==2026].nlargest(10,'worth').name))

# 5. sources of fortunes
share=lambda m: {int(y):round(f[(f.year==y)&m].worth.sum()/tot[y]*100,1) for y in (2011,2016,2021,2026)}
R['source_share']={'private equity':share(f.src.str.contains('private equity|leveraged buyout|buyouts')),'hedge funds':share(f.src.str.contains('hedge fund')),'venture capital':share(f.src.str.contains('venture capital')),
                   'real estate':share(f.ind=='Real Estate'),'technology':share(f.ind=='Technology'),'finance & investments':share(f.ind=='Finance & Investments')}
inds=['Technology','Finance & Investments','Fashion & Retail','Manufacturing','Real Estate','Energy','Healthcare','Food & Beverage']
ps=f.pivot_table(index='year',columns='ind',values='worth',aggfunc='sum').div(tot,axis=0)*100
fig,ax=plt.subplots(1,2,figsize=(11,4.1))
cols=[C1,C2,BR,C3,PU,'#555','#c0392b','#7fb3d5']
for i,c in zip(inds,cols): ax[0].plot(ps.index,ps[i],color=c,lw=2 if i=='Technology' else 1.2,label=i)
ax[0].set_ylabel('% of all billionaire wealth'); ax[0].legend(fontsize=7.5,ncol=2,loc='upper left'); ax[0].set_ylim(0,30); ax[0].set_title('Billionaire wealth by sector',fontsize=9.5)
# 6. by country, relative to own GDP
wb=json.load(open(D+'worldbank/gdp_current_usd.json'))[1]; Gd=pd.DataFrame([(r['country']['value'],int(r['date']),r['value']) for r in wb],columns=['c','y','v']).dropna()
cs=['United States','India','Germany','France','China','United Kingdom','Japan']; cg={}
for c in cs: cg[c]={y:round(f[(f.year==y)&(f.country==c)].worth.sum()*1e9/Gd[(Gd.c==c)&(Gd.y==y-1)].v.iloc[0]*100,1) for y in range(2011,2027)}
R['country_pct_gdp']={c:{y:cg[c][y] for y in (2011,2016,2021,2026)} for c in cs}
cc=dict(zip(cs,[C2,PU,BR,C3,'#c0392b',C1,'#555']))
for c in cs: ax[1].plot(list(cg[c].keys()),list(cg[c].values()),color=cc[c],lw=1.4); ax[1].text(2026.2,cg[c][2026],c,color=cc[c],fontsize=8,va='center')
ax[1].set_xlim(2011,2030.5); ax[1].set_xticks(range(2012,2027,2)); ax[1].set_ylabel("% of the country's own GDP"); ax[1].set_title("Billionaires' wealth relative to their country's economy",fontsize=9.5)
fig.tight_layout(); fig.savefig(out+'fig_sources.png'); plt.close(fig)

# thumbnail
fig,ax=plt.subplots(figsize=(6,4)); x=np.arange(len(dec)); ax.bar(x-.2,[d['profit'] for d in dec],.4,color=C3); ax.bar(x+.2,[d['mult'] for d in dec],.4,color=C2); ax.axhline(0,color='k',lw=.6); ax.set_xticks([]); ax.set_title('Profits or valuations?',fontsize=12); fig.tight_layout(); fig.savefig(out+'thumbnail.png'); plt.close(fig)
json.dump(R,open('part2_results.json','w'),indent=1,default=float)
for k in ['equity_points','equity_decomp','rates_backofenvelope','corr_yield_multiple','labour_share_index','top01_share','scale','exit_rate','mobility','age_median','selfmade_share','source_share','country_pct_gdp','top10_overlap','tech_share']: print(k,json.dumps(R[k],default=float))
print(R['top10'])
