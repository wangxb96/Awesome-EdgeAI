"""Conceptual deployment and evaluation diagrams.

These diagrams expose feasibility gates, shared resources, and measurement
boundaries. Box widths and arrows encode no measurements, calibrated policy,
relative speed, or experimentally established additive gain.
"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Rectangle
OUT=Path(__file__).resolve().parent
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],"font.size":7.2,"pdf.fonttype":42,"svg.fonttype":"none"})
BLUE,TEAL,AMBER='#497BA6','#318D85','#B9823C'
PB,PT,PA,GRAY='#EDF3F8','#EDF6F3','#FAF3E9','#F4F6F8'
LINE='#A7B8C2'

def setup(fig,ax,xmax,ymax):
 fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
 ax.set(xlim=(0,xmax),ylim=(0,ymax));ax.axis('off')

def txt(ax,x,y,s,size=7.2,weight='normal',ha='center',**kw):
 return ax.text(x,y,s,fontsize=size,color='black',ha=ha,va='center',weight=weight,zorder=8,**kw)

def box(ax,x,y,w,h,s,fill=GRAY,edge='none',size=7.2,weight='normal',r=1.5):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',fc=fill,ec=edge,lw=.7,zorder=2))
 if s:txt(ax,x+w/2,y+h/2,s,size,weight,linespacing=1.15)

def line(ax,xs,ys,c=LINE,lw=.8,ls='-'):
 ax.plot(xs,ys,c=c,lw=lw,ls=ls,solid_capstyle='round',zorder=3)

def arrow(ax,a,b,c=LINE,lw=.8,ms=6):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=ms,lw=lw,color=c,shrinkA=.5,shrinkB=.5,zorder=4))

def placement():
 fig,ax=plt.subplots(figsize=(3.48,2.85));setup(fig,ax,100,100)
 box(ax,4,91,92,7,'Task + resource + trust requirements',PB,size=7.3,weight='bold')
 arrow(ax,(56,90.7),(56,84),BLUE)
 box(ax,22,68,69,15,'Local path feasible?\nQuality · memory · deadline',PB,BLUE,size=7.1)
 line(ax,[21.8,17,17],[75.5,75.5,53.5],TEAL)
 arrow(ax,(17,58),(17,53),TEAL);txt(ax,9,65,'Yes',6.8)
 box(ax,4,39,26,14,'Local\nexecution',PT,TEAL,size=7.3,weight='bold')
 arrow(ax,(56.5,67.7),(56.5,58.6),AMBER);txt(ax,63,63.5,'No',6.8)
 box(ax,42,40,54,18,'Delegation feasible?\nQuality · deadline\nTransfer · queue · trust',PB,BLUE,size=6.8)
 arrow(ax,(55.5,39.7),(55.5,28.7),TEAL);txt(ax,48,34,'Yes',6.8)
 arrow(ax,(85,39.7),(85,28.7),AMBER);txt(ax,92,34,'No',6.8)
 box(ax,42,16,27,12,'Delegated\nexecution',PT,TEAL,size=7.0,weight='bold')
 box(ax,75,16,21,12,'Defer /\ndegrade',PA,AMBER,size=7.0,weight='bold')
 box(ax,4,2,92,10,'Report success, deadline misses, energy\nand the adequacy of the fallback.',GRAY,size=6.8)
 fig.savefig(OUT/'placement_decision.pdf',bbox_inches='tight',pad_inches=.025)
 fig.savefig(OUT/'placement_decision.svg',bbox_inches='tight',pad_inches=.025)
 fig.savefig(OUT/'placement_decision.tiff',dpi=600,bbox_inches='tight',pad_inches=.025,pil_kwargs={'compression':'tiff_lzw'})
 fig.savefig(OUT/'placement_decision.png',dpi=300,bbox_inches='tight',pad_inches=.025);plt.close(fig)

def couplings():
 fig,ax=plt.subplots(figsize=(7.12,2.64));setup(fig,ax,180,70)
 for x in (60,120):line(ax,[x,x],[3,62],'#D9E2E7',.6)
 txt(ax,3,67,'a  Shared memory ceiling',8.3,'bold',ha='left')
 line(ax,[8,52],[54.5,54.5],BLUE,.8,'--');txt(ax,30,59,'Available RAM',7.1)
 for y,s,fill in [(45,'Weights',PB),(36.5,'Retained KV',PT),(28,'Workspace',PA)]:box(ax,8,y,44,7.5,s,fill,size=7.6,r=.5)
 txt(ax,30,18,'Weight bits ↓: capacity released\nLonger context: KV traffic grows',7.1,linespacing=1.3)
 txt(ax,30,5.5,'Test quality + peak RAM + latency',7.0,weight='bold')
 txt(ax,63,67,'b  Contended execution resources',8.3,'bold',ha='left')
 for x,s,fill in [(64,'Draft',PB),(83,'Verify',PT),(102,'Offload',PA)]:box(ax,x,45,16,10,s,fill,size=7.2)
 for x,e in [(72,80),(91,91),(110,102)]:arrow(ax,(x,44.6),(e,37.8),TEAL,.7,5)
 box(ax,67,28,49,9,'Compute + memory + link',PT,TEAL,size=7.4)
 txt(ax,91.5,18,'Acceptance, switching and transfers\nchange the net benefit.',7.1,linespacing=1.3)
 txt(ax,91,5.5,'Test the combined critical path',7.0,weight='bold')
 txt(ax,123,67,'c  Workflow amplification',8.3,'bold',ha='left')
 for x,s,fill in [(124,'Model',PB),(143,'Tool',PT),(162,'State',PA)]:box(ax,x,43,15,11,s,fill,size=7.2)
 arrow(ax,(139.5,48.5),(142.5,48.5),TEAL,.7,5)
 arrow(ax,(158.5,48.5),(161.5,48.5),TEAL,.7,5)
 line(ax,[169.5,169.5,131.5],[42.7,36.8,36.8],TEAL,.7)
 arrow(ax,(131.5,36.8),(131.5,42.5),TEAL,.7,5)
 box(ax,137,26,29,8,'Depth / retries',PA,size=7.2)
 txt(ax,151,17,'More calls add model, tool\nand communication work.',7.1,linespacing=1.3)
 txt(ax,151,5.5,'Test success + failed actions',7.0,weight='bold')
 fig.savefig(OUT/'cross_layer_couplings.pdf',bbox_inches='tight',pad_inches=.025)
 fig.savefig(OUT/'cross_layer_couplings.svg',bbox_inches='tight',pad_inches=.025)
 fig.savefig(OUT/'cross_layer_couplings.tiff',dpi=600,bbox_inches='tight',pad_inches=.025,pil_kwargs={'compression':'tiff_lzw'})
 fig.savefig(OUT/'cross_layer_couplings.png',dpi=300,bbox_inches='tight',pad_inches=.025);plt.close(fig)

def workflow():
 fig,ax=plt.subplots(figsize=(7.12,2.56));setup(fig,ax,180,68)
 line(ax,[117,117],[2,61],'#D9E2E7',.6)
 txt(ax,3,64,'a  Define timing and energy boundaries',8.4,'bold',ha='left')
 for x,w,label,fill in [(4,14,'Queue',GRAY),(20,19,'Prefill',PB),(41,20,'Decode',PT),(63,20,'Tool I/O',PA),(85,23,'Resume /\nfinish',PT)]:box(ax,x,39,w,10,label,fill,size=7.2,r=.7)
 for x1,x2 in [(18.3,19.7),(39.3,40.7),(61.3,62.7),(83.3,84.7)]:arrow(ax,(x1,44),(x2,44),LINE,.65,4)
 for x in (4,44,108):line(ax,[x,x],[49.8,53],BLUE,.6)
 txt(ax,12,57,'Request start',7.0);txt(ax,48,57,'First token',7.0);txt(ax,99,57,'Terminal event',7.0)
 line(ax,[4,4,44,44],[34,30,30,34],BLUE,.8);txt(ax,24,26,'TTFT',7.5,weight='bold')
 line(ax,[4,4,108,108],[21,18,18,21],TEAL,.8);txt(ax,56,14,'Task completion / termination',7.4,weight='bold')
 box(ax,4,3,104,7,'Declare devices, energy window and idle policy',PA,size=7.1)
 txt(ax,121,64,'b  Keep failure denominators',8.4,'bold',ha='left')
 box(ax,123,46,53,9,'All task episodes A',PB,size=7.7,weight='bold')
 arrow(ax,(141,45.6),(134.5,40.3),TEAL,.8,6);arrow(ax,(158,45.6),(164,40.3),AMBER,.8,6)
 box(ax,123,29,23,11,'Complete',PT,TEAL,size=7.2)
 box(ax,151,29,25,11,'Fail / timeout',PA,AMBER,size=7.2)
 txt(ax,134.5,24,'Success subset S',6.8);txt(ax,163.5,24,'Keep in A',6.8)
 box(ax,122,7,55,12,'Report S/A + failure counts\nLabel conditional efficiency',GRAY,size=7.0)
 txt(ax,149.5,2.5,'Separate request and task tails',6.8)
 fig.savefig(OUT/'workflow_evaluation.pdf',bbox_inches='tight',pad_inches=.025)
 fig.savefig(OUT/'workflow_evaluation.svg',bbox_inches='tight',pad_inches=.025)
 fig.savefig(OUT/'workflow_evaluation.tiff',dpi=600,bbox_inches='tight',pad_inches=.025,pil_kwargs={'compression':'tiff_lzw'})
 fig.savefig(OUT/'workflow_evaluation.png',dpi=300,bbox_inches='tight',pad_inches=.025);plt.close(fig)

if __name__=='__main__':
 placement();couplings();workflow()

# Normalize serializer whitespace without changing vector geometry.
for stem in ("placement_decision", "cross_layer_couplings", "workflow_evaluation"):
    svg_path = OUT / (stem + ".svg")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
