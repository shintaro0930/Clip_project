import os
from PIL import Image

filenames = os.listdir('./used_pictures/')
image_base_dir = '/work/project/used_pictures/'
imgl=[]
ww=[]
hh=[]
for fname in sorted(filenames):
    path, ext = os.path.splitext( os.path.basename(image_base_dir + fname) )
    if ext=='.JPG' and path[0:2]!='._':
        pic=path+ext
        im=Image.open(pic)
        w=im.size[0]
        h=im.size[1]
        print(pic, w, h)
        imgl=imgl+[pic]
        ww=ww+[w]
        hh=hh+[h]

f=open('maggie.html','w')
print('<html>',file=f)
print('<body>',file=f)
print('<table>',file=f)
n=len(imgl)
m=int(n/5)+1
k=-1
for i in range(0,m):
    print('<tr>',file=f)
    for j in range(0,5):
        k=k+1
        if k<=n-1:
            pic=imgl[k]
            w1=200
            h1=int(hh[k]/ww[k]*200)
            print('<td align="center"><img src="'+pic+'" alt="pic" width="'+str(w1)+'", height="'+str(h1)+'"><br><a href="'+pic+'">I'+pic+'<a></td>',file=f)
        else:
            print('<td></td>',file=f)
    print('</tr>',file=f)
print('</table>',file=f)
print('</body>',file=f)
print('</html>',file=f)
f.close()