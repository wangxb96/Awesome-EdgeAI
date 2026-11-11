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
    fig,axis=plt.subplots(figsize=(7.12,3.00));setup(fig,axis,180,76)
    line([116,116],[2,71],STROKE,.55)
    text(3,73,'a  Link calls to one complete task episode',8.0,weight='bold',ha='left')
    text(57,67.5,'Episode e: one user goal, including internal retries',7.0)
    # This serial trace is an illustrative dependency pattern, not a time scale.
    line([5,5,58,58],[60,62,62,60],BLUE,.65)
    text(31.5,64.5,'Model request r1',6.7)
    text(71,61,'Tool t1',6.7)
    text(97,61,'Request r2',6.7)
    nodes=[(5,15,'Queue',sheet,BLUE,PALE_BLUE),
           (22,17,'Prefill',chip,BLUE,PALE_BLUE),
           (41,17,'Decode',cache,TEAL,PALE_TEAL),
           (62,18,'Tool I/O',tool,AMBER,PALE_AMBER),
           (84,26,'Resume',agent,TEAL,PALE_TEAL)]
    for x,w,label,draw,c,fill in nodes:
        rounded(x,44,w,13,fill,r=1.0)
        draw(x+w/2,52.5,.40,c);text(x+w/2,46.7,label,6.6,weight='bold')
    for x1,x2 in [(20.3,21.7),(39.3,40.7),(58.3,61.7),(80.3,83.7)]:
        arrow((x1,50.5),(x2,50.5),TEAL,.6,scale=4.5)
    text(50,39.7,'First token of r1',6.6)
    line([5,5,41,41],[40,36.5,36.5,40],BLUE,.75)
    text(23,32.5,'TTFT(r1)',7.1,weight='bold')
    line([5,5,110,110],[28,24.5,24.5,28],TEAL,.75)
    text(57.5,20.5,'Task completion / termination',7.2,weight='bold')
    rounded(5,8,105,8,PALE_AMBER,r=1.1)
    budget_icon('energy',11,12,AMBER)
    text(61,12,'Integrate power over declared devices and task window',6.8)
    text(57.5,3.3,'Link episode_id, request_id and parent_request_id',6.7)

    text(120,73,'b  Fix outcomes and denominators',8.0,weight='bold',ha='left')
    rounded(120,59,58,9,PALE_BLUE,r=1.1)
    text(149,63.5,'A = all attempted task episodes',7.1,weight='bold')
    for x,w,label,fill,c in [(120,18,'Success\nS',PALE_TEAL,TEAL),
                            (140,18,'Completed,\nunsuccessful',LIGHT,BLUE),
                            (160,18,'Fail /\ntimeout',PALE_AMBER,AMBER)]:
        rounded(x,41,w,12,fill,r=1.0)
        text(x+w/2,47,label,6.7,weight='bold',linespacing=1.15)
        arrow((149,58.5),(x+w/2,53.5),c,.65,scale=5)
    text(149,36.3,'Task success = S/A, including all failures in A',6.7)
    rounded(120,22,58,10,PALE_TEAL,r=1.1)
    text(149,28.5,'Deadline rate = (successful by D) / A',6.8,weight='bold')
    text(149,24.3,'Early failure is not fast successful completion',6.6)
    rounded(120,9,58,10,LIGHT,r=1.1)
    text(149,15.5,'Report request and task tails separately',6.8,weight='bold')
    text(149,11.3,'Timeout duration is a censoring bound',6.6)
    text(149,3.3,'Keep failed-task energy; label success-only metrics',6.6)
    export(fig,'workflow_evaluation')

def patterns():
    fig,axis=plt.subplots(figsize=(3.48,3.34));setup(fig,axis,100,96)
    # Each compact row shows payload, ownership and a distinguishing driver.
    rounded(2,64,96,30,LIGHT,r=1.4)
    text(5,91,'a  Whole-request delegation',7.4,weight='bold',ha='left')
    phone(12,80,.54);server(87,80,.60)
    arrow((23,81.5),(77,81.5),BLUE,.7,scale=5)
    arrow((77,78.5),(23,78.5),TEAL,.7,scale=5)
    text(50,85.5,'Prompt / evidence',6.6)
    text(50,74.8,'Generated output',6.6)
    text(50,69.5,'KV: remote; disclose input/output residency',6.5)
    text(50,65.8,'Driver: transfers + queue + generation',6.5)

    rounded(2,32,96,30,LIGHT,r=1.4)
    text(5,59,'b  Layer partitioning of one model',7.4,weight='bold',ha='left')
    for x,label,c in [(13,'Client',BLUE),(50,'Edge',TEAL),(87,'Remote',AMBER)]:
        chip(x,49.5,.60,c);text(x,43.5,label,6.4,weight='bold')
    arrow((20,49.5),(42,49.5),BLUE,.7,scale=5)
    arrow((57,49.5),(79,49.5),TEAL,.7,scale=5)
    text(31,54,'Act.',6.5);text(68,54,'Act.',6.5)
    text(50,38,'KV: node-local; payload: activations',6.5)
    text(50,34,'Driver: link delay + decode synchronization',6.5)

    rounded(2,0,96,30,LIGHT,r=1.4)
    text(5,27,'c  Large-small model cascade',7.4,weight='bold',ha='left')
    phone(12,18.5,.54);server(87,18.5,.60)
    rounded(36,15,31,7,PALE_BLUE,BLUE,r=1.0)
    text(51.5,18.5,'Validated gate',6.5,weight='bold')
    arrow((23,18.5),(35.5,18.5),BLUE,.7,scale=5)
    arrow((67.5,18.5),(77,18.5),AMBER,.7,scale=5)
    text(29,23,'Result',6.4);text(74,23,'Escalate',6.4)
    line([51.5,51.5],[14.7,11.2],TEAL,.65)
    check(51.5,10.3,.50,TEAL);text(67.5,11,'Accept locally',6.4)
    text(50,6,'KV: model-local; payload: selected context',6.5)
    text(50,2,'Risk: misrouting and wasted local work',6.5)
    export(fig,'collaboration_patterns')

if __name__=='__main__':
    placement();couplings();workflow();patterns()
