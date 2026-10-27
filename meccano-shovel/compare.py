import imageio.v3 as iio, numpy as np, sys
from PIL import Image
pairs=[('out/photo_side.png','out/shovel_photo_side.png'),('out/photo_front.png','out/shovel_photo_front.png')]
rows=[]
for a,b in pairs:
    A=Image.open(a).convert('RGB'); B=Image.open(b).convert('RGB')
    h=520
    A=A.resize((int(A.width*h/A.height),h)); B=B.resize((int(B.width*h/B.height),h))
    c=Image.new('RGB',(A.width+B.width+10,h),'white'); c.paste(A,(0,0)); c.paste(B,(A.width+10,0)); rows.append(c)
W=max(r.width for r in rows); out=Image.new('RGB',(W,sum(r.height for r in rows)+10),'white')
y=0
for r in rows: out.paste(r,(0,y)); y+=r.height+10
out.save('out/compare.png')
