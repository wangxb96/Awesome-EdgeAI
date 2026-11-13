"""Editable conceptual resource map; geometry encodes no measured performance.

The four stages connect task acceptance, compatible controls, placement and
complete-episode verification. The evidence band qualifies every stage;
feedback means retesting configuration choices, not an online quality oracle.
"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import deployment_reasoning as draw

OUT = Path(__file__).resolve().parent
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
                     'font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})

def main():
    fig, ax = plt.subplots(figsize=(7.12,3.50))
    draw.setup(fig,ax,180,88)
    draw.text(90,84,'Preserve cognitive capability within edge budgets',11,weight='bold')
    draw.rounded(5,71,170,8,draw.LIGHT,r=1.5)
    draw.text(90,75,'Memory  ·  Latency  ·  Energy  ·  Connectivity  ·  Trust',8.4)
    cards=[(5,'Define the task',draw.PALE_BLUE),(49,'Choose controls',draw.PALE_TEAL),
           (93,'Place execution',draw.PALE_BLUE),(137,'Verify outcomes',draw.PALE_AMBER)]
    for x,title,fill in cards:
        draw.rounded(x,30,38,34,fill,r=2.2)
        draw.text(x+19,59,title,8.6,weight='bold')
    draw.cache(24,47,.95,draw.BLUE)
    draw.chip(68,47,.92,draw.TEAL)
    draw.phone(106,47,.61);draw.server(119,47,.61)
    draw.tool(156,47,1.1,draw.AMBER)
    for x,s in [(24,'Context · reasoning\nTools · acceptance'),(68,'Context · precision\nDepth · scheduling'),
                (112,'Device · edge · cloud\nState · links · fallback'),(156,'Success · deadline\nTask energy · safety')]:
        draw.text(x,35,s,7.5,linespacing=1.35)
    for x in [43,87,131]:draw.arrow((x+.5,47),(x+5,47),draw.BLUE,.85,scale=6)
    draw.arrow((68,70.6),(68,64.5),draw.TEAL,.8,scale=6)
    draw.rounded(5,3,170,17,draw.LIGHT,r=2)
    draw.sheet(16,11.5,.87,draw.TEAL)
    draw.text(96,14.5,'Evidence and trust',8.8,weight='bold')
    draw.text(96,7.4,'Task outcomes · source-specific measurements · reproducibility limits',7.7)
    draw.arrow((156,29.5),(156,20.5),draw.AMBER,.8,scale=6)
    draw.arrow((68,20.5),(68,29.5),draw.TEAL,.8,scale=6)
    draw.text(89,25,'Retest the configuration',7.2,ha='left')
    fig.savefig(OUT/'resource_map.pdf',bbox_inches='tight',pad_inches=.03)
    fig.savefig(OUT/'resource_map.svg',bbox_inches='tight',pad_inches=.03)
    fig.savefig(OUT/'resource_map.png',dpi=300,bbox_inches='tight',pad_inches=.03)
    fig.savefig(OUT/'resource_map.tiff',dpi=600,bbox_inches='tight',pad_inches=.03,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)
    p=OUT/'resource_map.svg';p.write_text('\n'.join(v.rstrip() for v in p.read_text().splitlines())+'\n')

if __name__=='__main__':main()
