import re, json, glob, yaml, sys
root=sys.argv[1]; out=sys.argv[2]
files=sorted(glob.glob(root+'/*/*/index.qmd'))
files=[f for f in files if '/docs/' not in f and '/_' not in f]
chunks=[]; posts=[]
def clean(t):
    t=re.sub(r'```.*?```','',t,flags=re.S)
    t=re.sub(r'<style>.*?</style>','',t,flags=re.S)
    t=re.sub(r'<[^>]+>','',t)
    t=re.sub(r'\$\$.*?\$\$',' [equation] ',t,flags=re.S)
    t=re.sub(r'\$([^$\n]{1,80})\$',r'\1',t)
    t=re.sub(r'!\[[^\]]*\]\([^)]*\)','',t)
    t=re.sub(r'\[([^\]]+)\]\([^)]*\)',r'\1',t)
    t=re.sub(r'\{[.#][^}]*\}','',t)
    t=re.sub(r'^:::.*$','',t,flags=re.M)
    t=re.sub(r'^\|.*\|$','',t,flags=re.M)
    t=re.sub(r'`r [^`]*`','',t)
    t=re.sub(r'[*_`]','',t)
    t=re.sub(r'\\[a-zA-Z]+','',t)
    return re.sub(r'[ \t]+',' ',t)
for f in files:
    s=open(f).read()
    m=re.match(r'---\n(.*?)\n---\n(.*)',s,flags=re.S)
    if not m: continue
    fm=yaml.safe_load(m.group(1)); body=m.group(2)
    if fm.get('draft') or 'embeddings-part-2' in f: continue   # Part 2 holds the test questions
    url='/'+f[len(root)+1:].replace('index.qmd','')
    pid=len(posts); posts.append(dict(title=fm.get('title',''),url=url,section=url.split('/')[1],date=str(fm.get('date',''))))
    secs=re.split(r'\n(#{2,3} .*)\n',body)
    head='Introduction'; parts=[(head,secs[0])]
    for i in range(1,len(secs),2): parts.append((secs[i].lstrip('# ').strip(), secs[i+1] if i+1<len(secs) else ''))
    for h,txt in parts:
        paras=[p.strip() for p in clean(txt).split('\n\n') if len(p.strip().split())>8]
        buf=[]
        for p in paras:
            buf.append(p)
            if sum(len(x.split()) for x in buf)>=120:
                chunks.append(dict(post=pid,heading=h,text=' '.join(buf))); buf=[]
        if buf:
            if chunks and chunks[-1]['post']==pid and chunks[-1]['heading']==h and sum(len(x.split()) for x in buf)<50:
                chunks[-1]['text']+=' '+' '.join(buf)
            else: chunks.append(dict(post=pid,heading=h,text=' '.join(buf)))
for c in chunks: c['text']=re.sub(r'\s+',' ',c['text']).strip()
def split_long(c, maxw=110):
    words=c['text'].split()
    if len(words)<=maxw+30: return [c]
    sents=re.split(r'(?<=[.!?])\s+',c['text']); out=[]; buf=[]
    for st in sents:
        buf.append(st)
        if sum(len(x.split()) for x in buf)>=maxw*0.8:
            out.append(dict(c,text=' '.join(buf))); buf=[]
    if buf:
        if out and sum(len(x.split()) for x in buf)<40: out[-1]['text']+=' '+' '.join(buf)
        else: out.append(dict(c,text=' '.join(buf)))
    return out
chunks=[d for c in chunks for d in split_long(c)]
json.dump(dict(posts=posts,chunks=chunks),open(out,'w'),ensure_ascii=False)
import statistics
L=[len(c['text'].split()) for c in chunks]
print(len(posts),'posts',len(chunks),'chunks','median words',statistics.median(L),max(L),min(L))
