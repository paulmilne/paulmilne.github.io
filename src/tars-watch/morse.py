import numpy as np, string
from scipy.optimize import brentq
M={'A':'.-','B':'-...','C':'-.-.','D':'-..','E':'.','F':'..-.','G':'--.','H':'....','I':'..','J':'.---','K':'-.-','L':'.-..','M':'--','N':'-.','O':'---','P':'.--.','Q':'--.-','R':'.-.','S':'...','T':'-','U':'..-','V':'...-','W':'.--','X':'-..-','Y':'-.--','Z':'--..',
   '0':'-----','1':'.----','2':'..---','3':'...--','4':'....-','5':'.....','6':'-....','7':'--...','8':'---..','9':'----.'}
R={v:k for k,v in M.items()}
def units(code): return sum(1 if c=='.' else 3 for c in code)+len(code)-1
def capacity():
    def det(W):
        a=lambda ts: sum(W**(-t) for t in ts)
        return np.linalg.det(np.array([[a([2,4]),a([3,6])],[a([2,4]),0.0]])-np.eye(2))
    return np.log2(brentq(det,1.01,3))
# English letter frequencies (approx., %)
EN=dict(zip('ETAOINSHRDLCUMWFGYPBVKJXQZ',[12.7,9.1,8.2,7.5,7.0,6.7,6.3,6.1,6.0,4.3,4.0,2.8,2.8,2.4,2.4,2.2,2.0,2.0,1.9,1.5,1.0,0.8,0.15,0.15,0.10,0.07]))
def rate(probs):
    p=np.array(list(probs.values()),float); p/=p.sum()
    H=-(p*np.log2(p)).sum(); L=sum(pi*(units(M[k])+3) for k,pi in zip(probs,p))
    return H/L, H, L
def encode(text):
    """list of (on, duration_units) segments"""
    seg=[]
    words=text.upper().split()
    for wi,w in enumerate(words):
        for li,ch in enumerate(w):
            code=M[ch]
            for ei,e in enumerate(code):
                seg.append((1,1 if e=='.' else 3))
                if ei<len(code)-1: seg.append((0,1))
            if li<len(w)-1: seg.append((0,3))
        if wi<len(words)-1: seg.append((0,7))
    return seg
def decode(seg):
    out=[];cur='';word=''
    for on,d in seg:
        if on: cur+='.' if d<2 else '-'
        else:
            if d<2: continue
            word+=R.get(cur,'?'); cur=''
            if d>=5: out.append(word); word=''
    if cur: word+=R.get(cur,'?')
    out.append(word); return ' '.join(out)
def noisy(seg,sigma,rng):
    return [(on,max(0.05,d+sigma*rng.standard_normal()*np.sqrt(d))) for on,d in seg]
if __name__=='__main__':
    C=capacity(); print('capacity',C)
    print('English',rate(EN)); print('random letters+digits',rate({k:1 for k in M})); print('random digits',rate({k:1 for k in '0123456789'})); print('hex',rate({k:1 for k in '0123456789ABCDEF'}))
    print(decode(encode('STAY')), sum(d for _,d in encode('STAY')))
