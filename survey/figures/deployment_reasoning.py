"""Code-native scientific illustrations using the overview's vector language.

The conceptual panels distinguish feasibility, shared resources, episode-level
accounting and communication patterns. Icons denote roles, not tested products.
No geometry represents a measurement, threshold or comparative speed.
"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, PathPatch
from matplotlib.path import Path as MPath
OUT=Path(__file__).resolve().parent
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],"font.size":7.0,"pdf.fonttype":42,"svg.fonttype":"none"})
INK=MUTED="#000000"
BLUE,TEAL,AMBER="#497BA6","#318D85","#B9823C"
PALE_BLUE,PALE_TEAL,PALE_AMBER="#EDF3F8","#EDF6F3","#FAF3E9"
STROKE,LIGHT="#BCCBD3","#F5F7F9"
def line(xs,ys,color=STROKE,lw=.8,z=3,style="-"):
    ax.plot(xs,ys,color=color,lw=lw,solid_capstyle="round",zorder=z,linestyle=style)

def text(x,y,s,size=6.4,color=INK,weight="normal",ha="center",**kw):
    return ax.text(x,y,s,fontsize=size,color=color,weight=weight,ha=ha,va="center",zorder=10,**kw)

def rounded(x,y,w,h,face,edge="none",r=1.6,lw=.7,z=1):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0,rounding_size={r}",
                    facecolor=face,edgecolor=edge,linewidth=lw,zorder=z)
    ax.add_patch(p)
    return p

def arrow(a,b,c=STROKE,lw=.8,rad=0,scale=6,both=False):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="<->" if both else "-|>",mutation_scale=scale,
        connectionstyle=f"arc3,rad={rad}",color=c,lw=lw,shrinkA=.4,shrinkB=.4,zorder=4))

def chip(x,y,s=1,c=BLUE):
    # An illustrated package with a small model graph in its silicon area.
    rounded(x-4.1*s,y-3.0*s,8.2*s,6*s,"#D7E5F0",c,r=.8,lw=.75,z=5)
    rounded(x-3*s,y-2.1*s,6*s,4.2*s,"white",c,r=.4,lw=.55,z=6)
    for q in (-2.4,-.8,.8,2.4):
        line([x+q*s,x+q*s],[y+3*s,y+4*s],c,.65,5)
        line([x+q*s,x+q*s],[y-3*s,y-4*s],c,.65,5)
    for q in (-1.4,0,1.4):
        line([x-4.1*s,x-5.1*s],[y+q*s,y+q*s],c,.65,5)
        line([x+4.1*s,x+5.1*s],[y+q*s,y+q*s],c,.65,5)
    for yy in (-1.1,1.1):
        for xx in (-1.8,0,1.8):
            if yy<0:
                for dx in (-1.8,0,1.8): line([x+xx*s,x+dx*s],[y+yy*s,y+1.1*s],c,.3,7)
            ax.add_patch(Circle((x+xx*s,y+yy*s),.35*s,fc=c,ec="none",zorder=8))

def cache(x,y,s=1,c=TEAL):
    # Stacked KV-state pages and a scheduling dial.
    for dx,dy,fill in ((1.2,1.3,"#D4E9E4"),(.5,.55,"#E2F0EC"),(0,0,"white")):
        rounded(x-4*s+dx*s,y-2.7*s+dy*s,7*s,5.1*s,fill,c,r=.5,lw=.65,z=5)
    for ix in range(3):
        for iy in range(2): rounded(x+(-3.1+1.65*ix)*s,y+(-1.7+1.7*iy)*s,1.1*s,.9*s,c,r=.15,z=6)
    ax.add_patch(Circle((x+3*s,y-1.8*s),2.4*s,fc="white",ec=c,lw=.7,zorder=7))
    line([x+3*s,x+3*s,x+4.2*s],[y-.35*s,y-1.8*s,y-2.4*s],c,.7,8)

def agent(x,y,s=1,c=AMBER):
    pts=[(x-3.4*s,y-1.4*s),(x,y+2.8*s),(x+3.4*s,y-1.4*s)]
    for i in range(3):
        a,b=pts[i],pts[(i+1)%3]
        arrow(a,b,c,.65,scale=4)
    for i,(xx,yy) in enumerate(pts):
        rounded(xx-1.5*s,yy-1.3*s,3*s,2.6*s,"white",c,r=.7,lw=.7,z=6)
        if i==0:
            line([xx-.7*s,xx+.7*s],[yy+.45*s]*2,c,.6,7)
            line([xx-.7*s,xx+.4*s],[yy-.35*s]*2,c,.6,7)
        elif i==1:
            ax.add_patch(Circle((xx,yy),.55*s,fc=c,ec="none",zorder=7))
        else:
            line([xx-.7*s,xx-.1*s,xx+.8*s],[yy,yy-.55*s,yy+.6*s],c,.7,7)

def phone(x,y,s=1):
    # Screen, speech/context cards, and an embedded local model chip.
    rounded(x-3.7*s,y-5.8*s,7.4*s,11.6*s,"#DCE8F1",BLUE,r=1.2,lw=.8,z=6)
    rounded(x-2.9*s,y-4.6*s,5.8*s,8.5*s,"white",r=.45,z=7)
    line([x-.9*s,x+.9*s],[y+4.8*s]*2,BLUE,.6,8)
    rounded(x-2.1*s,y+.4*s,4.2*s,2.2*s,PALE_BLUE,r=.5,z=8)
    line([x-1.4*s,x+1.4*s],[y+1.75*s]*2,BLUE,.5,9)
    line([x-1.4*s,x+.7*s],[y+1*s]*2,BLUE,.5,9)
    chip(x,y-2.0*s,.36,BLUE)
    ax.add_patch(Circle((x,y-5.1*s),.28*s,fc=BLUE,ec="none",zorder=8))
    # Adjacent sensor/embedded device, not a tested hardware model.
    rounded(x+3.2*s,y-5*s,4.6*s,3.2*s,"#ECF3F8",BLUE,r=.5,lw=.6,z=8)
    for dx in (4.2,5.5,6.8): ax.add_patch(Circle((x+dx*s,y-3.4*s),.25*s,fc=BLUE,ec="none",zorder=9))

def server(x,y,s=1):
    # A shallow isometric edge rack with three service drawers.
    ax.add_patch(Polygon([(x-4.8*s,y+4.7*s),(x+3.6*s,y+4.7*s),(x+5.5*s,y+6*s),(x-2.9*s,y+6*s)],fc="#D8E9E5",ec=TEAL,lw=.65,zorder=5))
    ax.add_patch(Polygon([(x+3.6*s,y-5*s),(x+5.5*s,y-3.7*s),(x+5.5*s,y+6*s),(x+3.6*s,y+4.7*s)],fc="#C0DCD5",ec=TEAL,lw=.65,zorder=5))
    rounded(x-4.8*s,y-5*s,8.4*s,9.7*s,"white",TEAL,r=.5,lw=.75,z=6)
    for dy in (-3.1,-.1,2.9):
        rounded(x-4*s,y+(dy-.9)*s,6.8*s,1.8*s,PALE_TEAL,TEAL,r=.3,lw=.4,z=7)
        ax.add_patch(Circle((x-2.8*s,y+dy*s),.25*s,fc=TEAL,ec="none",zorder=8))
        line([x-1.6*s,x+1.8*s],[y+dy*s]*2,TEAL,.5,8)

def cloud(x,y,s=1):
    verts=[(-6,-2),(-9,-2),(-9,3),(-5.5,3),
           (-5,7),(.8,8),(3.2,4.7),
           (6,5.5),(8,2.8),(7,.5),
           (10,-1),(9,-4),(6.2,-4),
           (-5.8,-4),(-8,-4),(-9,-2),(-6,-2),(-6,-2)]
    verts=[(x+xx*s,y+yy*s) for xx,yy in verts]
    codes=([MPath.MOVETO]+[MPath.CURVE4]*12+[MPath.LINETO]
           +[MPath.CURVE4]*3+[MPath.CLOSEPOLY])
    ax.add_patch(PathPatch(MPath(verts,codes),fc="#E4EDF5",ec=BLUE,lw=.75,zorder=6))
    for dy in (-1.7,.4):
        rounded(x-2.7*s,y+dy*s,5.6*s,1.6*s,"white",BLUE,r=.35,lw=.5,z=7)
        ax.add_patch(Circle((x-1.7*s,y+(dy+.8)*s),.2*s,fc=BLUE,ec="none",zorder=8))
        line([x-.8*s,x+1.9*s],[y+(dy+.8)*s]*2,BLUE,.45,8)

def budget_icon(kind,x,y,c):
    if kind=="memory":
        rounded(x-1.7,y-1.1,3.4,2.2,"white",c,r=.25,lw=.55,z=5)
        for dx in (-.9,0,.9): line([x+dx,x+dx],[y-1.1,y-1.7],c,.5,6)
    elif kind=="energy":
        ax.add_patch(Polygon([(x+.3,y+1.8),(x-1.2,y-.1),(x-.1,y-.1),(x-.35,y-1.7),(x+1.35,y+.4),(x+.2,y+.4)],fc=c,ec="none",zorder=5))
    elif kind=="deadline":
        ax.add_patch(Circle((x,y),1.65,fc="white",ec=c,lw=.6,zorder=5))
        line([x,x,x+.9],[y+1.1,y,y-.45],c,.6,6)
    else:
        for xx,yy in ((x-1.5,y-.8),(x,y+1),(x+1.5,y-.8)):
            line([x,xx],[y+1,yy],c,.5,5)
            ax.add_patch(Circle((xx,yy),.45,fc=c,ec="none",zorder=6))


def setup(fig, axis, xmax, ymax):
    global ax
    ax=axis
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.set(xlim=(0,xmax),ylim=(0,ymax));ax.axis('off')

def export(fig, stem):
    fig.savefig(OUT/(stem+'.pdf'),bbox_inches='tight',pad_inches=.025)
    fig.savefig(OUT/(stem+'.svg'),bbox_inches='tight',pad_inches=.025)
    fig.savefig(OUT/(stem+'.tiff'),dpi=600,bbox_inches='tight',pad_inches=.025,pil_kwargs={'compression':'tiff_lzw'})
    fig.savefig(OUT/(stem+'.png'),dpi=300,bbox_inches='tight',pad_inches=.025)
    p=OUT/(stem+'.svg');p.write_text('\n'.join(v.rstrip() for v in p.read_text().splitlines())+'\n')
    plt.close(fig)

def shield(x,y,s=1,c=AMBER):
    ax.add_patch(Polygon([(x-2.2*s,y+2.2*s),(x,y+3*s),(x+2.2*s,y+2.2*s),(x+1.8*s,y-.5*s),(x,y-2.5*s),(x-1.8*s,y-.5*s)],fc='white',ec=c,lw=.7,zorder=5))
    line([x-1.1*s,x-.2*s,x+1.25*s],[y+.2*s,y-.7*s,y+1*s],c,.8,6)

def tool(x,y,s=1,c=AMBER):
    # A tool panel with a slider and a verified return value.
    rounded(x-3.7*s,y-2.5*s,7.4*s,5*s,'white',c,r=.6,lw=.7,z=5)
    line([x-2.6*s,x+2.5*s],[y+1.1*s]*2,c,.5,6)
    ax.add_patch(Circle((x-.7*s,y+1.1*s),.5*s,fc=c,ec='none',zorder=7))
    line([x-.9*s,x-.1*s,x+1.4*s],[y-1.0*s,y-1.6*s,y-.2*s],c,.7,6)

def sheet(x,y,s=1,c=TEAL):
    rounded(x-2.8*s,y-3.4*s,5.6*s,6.8*s,'white',c,r=.4,lw=.7,z=5)
    for dy,w in [(1.7,3.7),(.1,3.7),(-1.5,2.7)]:line([x-1.8*s,x+(-1.8+w)*s],[y+dy*s]*2,c,.55,6)

def check(x,y,s=1,c=TEAL):
    ax.add_patch(Circle((x,y),2.1*s,fc='white',ec=c,lw=.7,zorder=5))
    line([x-1.1*s,x-.2*s,x+1.1*s],[y,y-.9*s,y+1*s],c,.8,6)

def stop(x,y,s=1,c=AMBER):
    ax.add_patch(Circle((x,y),2.1*s,fc='white',ec=c,lw=.7,zorder=5))
    line([x-1*s,x+1*s],[y,y],c,.9,6)

def placement():
    fig,axis=plt.subplots(figsize=(3.48,2.85));setup(fig,axis,100,82)
    text(50,79,'Preserve task adequacy within the budget',7.3,weight='bold')
    rounded(3,67.5,94,8,LIGHT,r=1.3)
    for x,label,kind,c in [(12,'Memory','memory',BLUE),(36,'Energy','energy',AMBER),(60,'Deadline','deadline',BLUE)]:
        budget_icon(kind,x-6,71.5,c);text(x-2,71.5,label,6.5,ha='left')
    shield(83,71.5,.55);text(87,71.5,'Trust',6.5,ha='left')
    rounded(23,51,74,12,PALE_BLUE,BLUE,r=1.3)
    text(60,59,'Local execution feasible?',7.0,weight='bold')
    text(60,54,'Quality · RAM · deadline · energy',6.4)
    phone(12,57,.65)
    line([22.6,13,13],[57,57,43.5],TEAL,.75)
    arrow((13,46),(13,43),TEAL,.75);text(7,48,'Yes',6.4)
    rounded(3,26,30,16,PALE_TEAL,r=1.5)
    chip(18,36,.72,TEAL);text(18,28.5,'Execute locally',6.5,weight='bold')
    arrow((62,50.5),(62,45),AMBER,.75);text(68,47.5,'No',6.4)
    rounded(39,30,58,14,PALE_BLUE,BLUE,r=1.3)
    server(47,38,.40)
    text(72,40,'Delegation feasible?',6.6,weight='bold')
    text(69,34,'Quality · delay · energy · trust',6.4)
    arrow((53,29.6),(53,23),TEAL,.75);text(46,26,'Yes',6.4)
    arrow((85,29.6),(85,23),AMBER,.75);text(92,26,'No',6.4)
    rounded(37,9,32,13,PALE_TEAL,r=1.4)
    server(44,16.5,.45);text(58,16,'Delegate',6.5,weight='bold')
    rounded(74,9,23,13,PALE_AMBER,r=1.4)
    stop(85,18,.65);text(85.5,12,'Defer / degrade',6.4,weight='bold')
    text(50,3,'Verify success, deadline, energy and fallback safety',6.4)
    export(fig,'placement_decision')

def couplings():
    fig,axis=plt.subplots(figsize=(7.12,2.64));setup(fig,axis,180,67)
    for x in (60,120):line([x,x],[3,61],STROKE,.55)
    text(3,64,'a  Memory capacity',8.1,weight='bold',ha='left')
    phone(12,44,.95)
    rounded(23,30,32,25,LIGHT,r=1.2)
    line([26,52],[52,52],BLUE,.7,style='--')
    text(39,57,'Available RAM',7.0)
    for y,label,c,fill in [(44,'Weights',BLUE,PALE_BLUE),(37.5,'KV state',TEAL,PALE_TEAL),(31,'Workspace',AMBER,PALE_AMBER)]:
        rounded(26,y,26,5,fill,c,r=.5,lw=.45);text(39,y+2.5,label,6.8)
    text(30,22,'Lower weight bits release capacity;',7.0)
    text(30,17,'longer context adds KV traffic.',7.0)
    text(30,6,'Check quality + peak RAM + latency',7.0,weight='bold')
    text(63,64,'b  Execution contention',8.1,weight='bold',ha='left')
    for x,name,draw,c in [(71,'Draft',chip,BLUE),(91,'Verify',cache,TEAL),(111,'Offload',None,AMBER)]:
        rounded(x-8,35,16,21,{BLUE:PALE_BLUE,TEAL:PALE_TEAL,AMBER:PALE_AMBER}[c],r=1.2)
        if draw:draw(x,47,.74,c)
        else:server(x,47,.68)
        text(x,38.5,name,7.0,weight='bold')
        arrow((x,34.6),(x,29.2),c,.7)
    rounded(65,22,51,7,LIGHT,r=1.1);text(90.5,25.5,'Compute · memory · communication',6.8)
    text(91,15,'Acceptance, transfers and switching',7.0)
    text(91,10,'change the shared critical path.',7.0)
    text(91,3,'Measure the composed configuration',7.0,weight='bold')
    text(123,64,'c  Workflow amplification',8.1,weight='bold',ha='left')
    for x,label,draw,c in [(131,'Model',chip,BLUE),(151,'Tool',tool,AMBER),(171,'State',sheet,TEAL)]:
        rounded(x-8,34,16,22,{BLUE:PALE_BLUE,TEAL:PALE_TEAL,AMBER:PALE_AMBER}[c],r=1.2)
        draw(x,46,.72,c);text(x,37,label,7.0,weight='bold')
    arrow((139.2,45),(142.7,45),TEAL,.65,scale=5)
    arrow((159.2,45),(162.7,45),TEAL,.65,scale=5)
    line([171,171,131,131],[33.7,28,28,33.7],TEAL,.7)
    arrow((131,30),(131,33.5),TEAL,.7,scale=5)
    text(151,22,'Reasoning depth and retries',7.0)
    text(151,15,'amplify model, tool and link work.',7.0)
    text(151,6,'Check success + failed/unsafe actions',7.0,weight='bold')
    export(fig,'cross_layer_couplings')

def workflow():
    fig,axis=plt.subplots(figsize=(7.12,2.56));setup(fig,axis,180,65)
    line([116,116],[3,59],STROKE,.55)
    text(3,62,'a  Instrument the complete task',8.1,weight='bold',ha='left')
    entries=[(13,'Queue',sheet,BLUE,PALE_BLUE),(35,'Prefill',chip,BLUE,PALE_BLUE),(57,'Decode',cache,TEAL,PALE_TEAL),(79,'Tool I/O',tool,AMBER,PALE_AMBER),(101,'Resume',agent,TEAL,PALE_TEAL)]
    for x,label,draw,c,fill in entries:
        rounded(x-9,31,18,23,fill,r=1.2);draw(x,44,.73,c);text(x,34,label,7.0,weight='bold')
    for x in (22.3,44.3,66.3,88.3):arrow((x,43),(x+3.1,43),TEAL,.65,scale=5)
    text(5,57,'Start',6.7,ha='left');text(39,57,'First token',6.7,ha='left');text(109,57,'Terminal event',6.7,ha='right')
    line([4,4,48,48],[28,25,25,28],BLUE,.8);text(26,21,'Request TTFT',7.0,weight='bold')
    line([4,4,110,110],[16,13,13,16],TEAL,.8);text(57,9,'Task completion / termination',7.0,weight='bold')
    text(57,3,'Declare devices, energy window and idle policy',6.7)
    text(120,62,'b  Retain every task episode',8.1,weight='bold',ha='left')
    rounded(121,43,57,13,PALE_BLUE,r=1.2)
    sheet(128,49.5,.8,BLUE);text(153,49.5,'All attempted episodes A',7.1,weight='bold')
    arrow((138,42.5),(134,36),TEAL,.75)
    arrow((163,42.5),(168,36),AMBER,.75)
    rounded(121,20,26,15,PALE_TEAL,r=1.2);check(134,29,.85,TEAL);text(134,22.5,'Complete',7.0,weight='bold')
    rounded(152,20,26,15,PALE_AMBER,r=1.2);stop(165,29,.85,AMBER);text(165,22.5,'Fail / timeout',6.8,weight='bold')
    text(134,15,'Success subset S',6.7);text(165,15,'Remain in A',6.7)
    rounded(121,3,57,8,LIGHT,r=1.1);text(149.5,7,'S/A + failures; label conditional efficiency',6.7)
    export(fig,'workflow_evaluation')

def patterns():
    fig,axis=plt.subplots(figsize=(6.27,4.40));setup(fig,axis,160,112)
    text(80,108,'What crosses the device boundary?',8.3,weight='bold')
    # Full-request delegation sends input/output rather than model-layer state.
    rounded(3,72,154,29,LIGHT,r=1.5)
    text(7,97,'a  Whole-request delegation',7.6,weight='bold',ha='left')
    phone(24,84,.9);server(137,84,.9)
    text(24,74,'Client',6.9,weight='bold');text(137,74,'Edge / cloud',6.9,weight='bold')
    arrow((36,88),(125,88),BLUE,.8);text(80,91.5,'Prompt / selected evidence',7.0)
    arrow((125,80),(36,80),TEAL,.8);text(80,76,'Generated output',7.0)
    # A single model is physically sharded across participants.
    rounded(3,38,154,29,LIGHT,r=1.5)
    text(7,63,'b  Layer partitioning of one model',7.6,weight='bold',ha='left')
    for x,label,c,fill in [(25,'Client shard',BLUE,PALE_BLUE),(80,'Edge shard',TEAL,PALE_TEAL),(135,'Remote shard',AMBER,PALE_AMBER)]:
        chip(x,52,.85,c);text(x,41,label,6.9,weight='bold')
    arrow((33,52),(70,52),BLUE,.8);text(52,56,'Activations',7.0)
    arrow((88,52),(125,52),TEAL,.8);text(107,56,'Activations',7.0)
    # A gate chooses between complete models, not consecutive model layers.
    rounded(3,3,154,30,LIGHT,r=1.5)
    text(7,29,'c  Large–small model cascade',7.6,weight='bold',ha='left')
    phone(22,16,.9);text(22,6,'Local SLM',6.9,weight='bold')
    rounded(62,12,36,10,PALE_BLUE,BLUE,r=1.1)
    text(80,17,'Validated routing gate',6.9,weight='bold')
    arrow((32,17),(61,17),BLUE,.8);text(47,21,'Local result',6.7)
    server(136,16,.9);text(136,6,'Larger model',6.9,weight='bold')
    arrow((98.5,17),(125,17),AMBER,.8);text(113,22,'Escalate',6.8)
    line([80,80],[11.7,6.5],TEAL,.7);check(80,5.5,.55,TEAL);text(97,6,'Accept locally',6.7)
    export(fig,'collaboration_patterns')

if __name__=='__main__':
    placement();couplings();workflow();patterns()
