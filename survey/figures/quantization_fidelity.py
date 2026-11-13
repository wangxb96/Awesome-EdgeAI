"""Replot recorded EM scores from QSpec (EMNLP 2025), Table 1.

Primary source: https://aclanthology.org/2025.emnlp-main.240/
Llama-3-8B-Instruct with Atom quantization. This is a literature replot,
not a new measurement; the source table does not supply uncertainty estimates.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],"font.size":7,"pdf.fonttype":42,"svg.fonttype":"none","axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.6})
names=['W16A16','W4A16','W4A4']
with (OUT/'quantization_fidelity_source.csv').open(newline='') as source:
    rows=list(csv.DictReader(source))
if len(rows)!=3 or len({row['benchmark'] for row in rows})!=3:
    raise ValueError('Expected three distinct benchmark rows in the source CSV')
labels=[f"{row['benchmark']}\n{int(row['shots'])}-shot" for row in rows]
values=np.array([[float(row[name+'_EM_pct']) for name in names] for row in rows])
if not np.isfinite(values).all() or ((values<0)|(values>100)).any():
    raise ValueError('Exact-match percentages must be finite and in [0,100]')
colors=['#5B7FA0','#4F9C91','#D0A269']
fig,ax=plt.subplots(figsize=(3.48,2.10))
fig.subplots_adjust(left=.15,right=.98,bottom=.22,top=.80)
x=np.arange(3);width=.23
for j,(name,c) in enumerate(zip(names,colors)):
    bars=ax.bar(x+(j-1)*width,values[:,j],width,color=c,edgecolor='white',linewidth=.4,label=name,zorder=3)
    for rect,v in zip(bars,values[:,j]):
        ax.text(rect.get_x()+rect.get_width()/2,v+2,f'{v:.1f}',ha='center',va='bottom',fontsize=6.0,color='black')
ax.set(xticks=x,xticklabels=labels,ylim=(0,100),yticks=[0,25,50,75,100],ylabel='Exact match (%)')
ax.tick_params(labelsize=6.2,length=2.5,width=.6)
ax.legend(loc='lower center',bbox_to_anchor=(.50,1.12),ncol=3,fontsize=6.1,frameon=False,handlelength=1.15,columnspacing=1.1)
fig.savefig(OUT/'quantization_fidelity.pdf',bbox_inches='tight',pad_inches=.025)
fig.savefig(OUT/'quantization_fidelity.svg',bbox_inches='tight',pad_inches=.025)
fig.savefig(OUT/'quantization_fidelity.tiff',dpi=600,bbox_inches='tight',pad_inches=.025,pil_kwargs={'compression':'tiff_lzw'})
fig.savefig(OUT/'quantization_fidelity.png',dpi=300,bbox_inches='tight',pad_inches=.025)
plt.close(fig)
svg = OUT/'quantization_fidelity.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
