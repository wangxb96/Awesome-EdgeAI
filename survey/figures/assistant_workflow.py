"""Prospective component roles for a private mobile tool assistant.

This illustration instantiates task, model, permission and accounting boundaries.
No panel or path represents a measured implementation, task result or speedup.
The deployment_reasoning module supplies the shared editable vector icon language.
"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import deployment_reasoning as dr
OUT=Path(__file__).resolve().parent
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'font.size':7.0,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axis=plt.subplots(figsize=(6.27,2.65));dr.setup(fig,axis,160,67.5)
dr.text(80,64.5,'One goal: a confirmed reminder from a private source constraint',8.0,weight='bold')
for x,w,title,draw,c,fill,detail in [
 (3,34,'Local evidence',dr.sheet,dr.BLUE,dr.PALE_BLUE,'C2: retrieve / retain fact'),
 (43,34,'Text / call models',dr.chip,dr.TEAL,dr.PALE_TEAL,'C1/C3: reason / propose'),
 (83,34,'Host-enforced gate',dr.shield,dr.AMBER,dr.PALE_AMBER,'C4: authorize action'),
 (123,34,'Tool + receipt',dr.agent,dr.TEAL,dr.PALE_TEAL,'C3/C4: commit / verify')]:
 dr.rounded(x,30,w,28,fill,r=1.3)
 dr.text(x+w/2,54,title,7.0,weight='bold')
 draw(x+w/2,43.5,.85,c)
 dr.text(x+w/2,33.5,detail,6.5)
for start,end in [(37.4,42.5),(77.4,82.5),(117.4,122.5)]:dr.arrow((start,44),(end,44),dr.TEAL,.7,scale=5)
dr.text(20,23,'Private notes and state',6.7)
dr.text(20,18.5,'follow a declared boundary',6.7)
dr.rounded(43,3,34,17,dr.PALE_BLUE,r=1.2)
dr.server(49,12,.62)
dr.text(65,13,'Eligible',6.7,weight='bold');dr.text(65,8,'remote help',6.7)
dr.arrow((60,29.5),(60,20.5),dr.BLUE,.7,both=True)
dr.text(66,25,'Conditional',6.4)
dr.rounded(83,3,34,17,dr.PALE_AMBER,r=1.2)
dr.stop(91,12,.8,dr.AMBER);dr.text(104,12,'Defer / fail',6.7,weight='bold')
dr.arrow((100,29.5),(100,20.5),dr.AMBER,.7)
dr.text(113.5,25,'Gate / link failure',6.4)
dr.rounded(123,3,34,17,dr.LIGHT,r=1.2)
dr.text(140,15,'Record episode e',6.8,weight='bold')
dr.text(140,10,'Success / deadline',6.6);dr.text(140,5.5,'+ total task energy',6.6)
dr.arrow((140,29.5),(140,20.5),dr.TEAL,.7)
dr.arrow((117.4,11.5),(122.5,11.5),dr.AMBER,.7,scale=5)
fig.savefig(OUT/'assistant_workflow.pdf',bbox_inches='tight',pad_inches=.025)
fig.savefig(OUT/'assistant_workflow.svg',bbox_inches='tight',pad_inches=.025)
fig.savefig(OUT/'assistant_workflow.tiff',dpi=600,bbox_inches='tight',pad_inches=.025,pil_kwargs={'compression':'tiff_lzw'})
fig.savefig(OUT/'assistant_workflow.png',dpi=300,bbox_inches='tight',pad_inches=.025)
p=OUT/'assistant_workflow.svg';p.write_text('\n'.join(v.rstrip() for v in p.read_text().splitlines())+'\n')
plt.close(fig)
