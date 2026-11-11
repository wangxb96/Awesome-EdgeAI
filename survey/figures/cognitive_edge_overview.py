"""Editable vector illustration of budget-constrained cognitive edge computing.

Control flow runs from supplied task/resource budgets through joint model,
runtime and agent policies to cross-tier execution. Observations update the
policy, not the externally supplied budgets. The drawing is conceptual and
contains no measured data or claims of tested device coverage.
"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, PathPatch
from matplotlib.path import Path as MPath

OUT = Path(__file__).resolve().parent
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.4, "pdf.fonttype": 42, "svg.fonttype": "none",
})
INK = "#000000"
MUTED = "#000000"
BLUE, TEAL, AMBER = "#497BA6", "#318D85", "#B9823C"
PALE_BLUE, PALE_TEAL, PALE_AMBER = "#EDF3F8", "#EDF6F3", "#FAF3E9"
STROKE, LIGHT = "#BCCBD3", "#F5F7F9"
fig, ax = plt.subplots(figsize=(3.48, 2.70))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set(xlim=(0,100), ylim=(0,78))
ax.axis("off")

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

# Supplied constraints enter the controller independently of feedback.
text(48,74.4,"Reasoning capability within a resource budget",7.6,weight="bold")
rounded(3,64.2,89,6.6,LIGHT,r=1.4)
for x,label,kind,c in [(13,"Memory","memory",BLUE),(35,"Energy","energy",AMBER),(57,"Deadline","deadline",BLUE),(79,"Link","link",TEAL)]:
    budget_icon(kind,x-6,67.5,c)
    text(x-2,67.5,label,6.6,ha="left")
arrow((47.5,63.9),(47.5,61.0),INK,.8)

# Three policy levers form one controller, not three exclusive deployment paths.
text(47.5,59.3,"Joint model, runtime and agent adaptation",7.1,weight="bold")
for x,w,c,fill,name,detail,draw in [
    (3,27.5,BLUE,PALE_BLUE,"Model","Precision · size\nExpert activation",chip),
    (33.5,27.5,TEAL,PALE_TEAL,"Runtime","KV state · scheduling\nOffload policy",cache),
    (64,28,AMBER,PALE_AMBER,"Agent","Context · tool use\nReasoning depth",agent)]:
    rounded(x,37.0,w,19.7,fill,r=1.6)
    draw(x+w/2,50.6,.79,c)
    text(x+w/2,44.4,name,7.0,weight="bold")
    text(x+w/2,40.0,detail,6.6,color=MUTED,linespacing=1.18)
line([16.75,16.75,78,78],[36.8,34.8,34.8,36.8],STROKE,.65)
line([47.25,47.25],[37.0,34.8],STROKE,.65)
arrow((47.5,34.8),(47.5,32.5),INK,.8)

# Recognisable flat vector illustrations make the physical execution tiers clear.
rounded(3,12.2,89,19.5,LIGHT,r=2)
phone(16,23.0,.80)
server(47,23.0,.80)
cloud(78,22.7,.78)
arrow((25.0,23.1),(37.5,23.1),TEAL,.9,both=True)
arrow((58.0,23.1),(68.5,23.1),TEAL,.9,both=True)
text(17,15.4,"Device",6.8,weight="bold")
text(47.3,15.4,"Edge",6.8,weight="bold")
text(78,15.4,"Cloud",6.8,weight="bold")
arrow((47.5,12.1),(47.5,9.9),INK,.8)

# Metrics are observations. Their return path updates the controller.
rounded(3,2.6,89,7.0,PALE_TEAL,r=1.4)
text(47.5,7.5,"Observe quality, cost and safety",6.8,weight="bold")
text(47.5,4.4,"Reasoning fidelity · p95 latency · J/token · Safety",6.4,color=MUTED)
line([92.3,96.7,96.7],[6.2,6.2,59.3],TEAL,.85)
arrow((96.7,59.3),(92.2,59.3),TEAL,.85)

fig.savefig(OUT/"cognitive_edge_overview.pdf",bbox_inches="tight",pad_inches=.025)
fig.savefig(OUT/"cognitive_edge_overview.svg",bbox_inches="tight",pad_inches=.025)
fig.savefig(OUT/"cognitive_edge_overview.tiff",dpi=600,bbox_inches="tight",pad_inches=.025,pil_kwargs={"compression":"tiff_lzw"})
fig.savefig(OUT/"cognitive_edge_overview.png",dpi=300,bbox_inches="tight",pad_inches=.025)
svg=OUT/"cognitive_edge_overview.svg"
svg.write_text("\n".join(s.rstrip() for s in svg.read_text().splitlines())+"\n")
plt.close(fig)
