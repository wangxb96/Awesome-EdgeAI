"""Recovered server logs: descriptive results and measurement-boundary diagnostics.

No device is executed. All logged observations in the declared two campaigns
are retained. Request SD is dispersion, not an independence-based interval.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT=Path(__file__).resolve().parent
DATA=OUT.parent/'server'
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
 'font.size':7.2,'axes.titlesize':8.2,'axes.labelsize':7.2,'xtick.labelsize':6.8,
 'ytick.labelsize':7.0,'pdf.fonttype':42,'svg.fonttype':'none',
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'legend.frameon':False})
BLUE='#5B7FA0';TEAL='#4F9C91';AMBER='#D0A269';GRAY='#D4DCE2';INK='#283744'
rows=list(csv.DictReader((DATA/'reanalysis_by_run.csv').open()))
labels=['DeepSeek-R1 14B','Gemma3 4B','GPT-OSS 20B','Llama2 7B','Qwen3 8B: main','Qwen3 8B: follow-up']
colors=[BLUE]*5+[AMBER]
y=np.arange(len(rows))

def save(fig,stem):
    fig.savefig(OUT/f'{stem}.pdf',bbox_inches='tight',pad_inches=.04)
    fig.savefig(OUT/f'{stem}.svg',bbox_inches='tight',pad_inches=.04)
    fig.savefig(OUT/f'{stem}.png',dpi=300,bbox_inches='tight',pad_inches=.04)
    fig.savefig(OUT/f'{stem}.tiff',dpi=600,bbox_inches='tight',pad_inches=.04,pil_kwargs={'compression':'tiff_lzw'})
    p=OUT/f'{stem}.svg';p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    plt.close(fig)

def frame(ax,title,xlabel):
    ax.set_title(title,loc='left',pad=9,fontweight='bold')
    ax.set_yticks(y,labels);ax.set_ylim(5.55,-.55);ax.set_xlabel(xlabel,labelpad=5)
    ax.grid(axis='x',color='#E8EDF1',lw=.5,zorder=0);ax.tick_params(length=2.5,width=.5)

fig,axs=plt.subplots(2,2,figsize=(7.0,5.9))
fig.subplots_adjust(left=.18,right=.98,top=.90,bottom=.09,wspace=.72,hspace=.68)
ax=axs[0,0];frame(ax,'a  Score definition changes the observation','Recorded / audited match (%)')
for i,r in enumerate(rows):
    a=float(r['answer_containment'])*100;b=float(r['strict_string_audit'])*100
    ax.plot([a,b],[i,i],color=GRAY,lw=2,zorder=2)
    ax.scatter([a],[i],s=23,marker='o',color=colors[i],zorder=3)
    ax.scatter([b],[i],s=24,marker='D',facecolor='white',edgecolor=TEAL,lw=.9,zorder=4)
ax.set_xlim(-4,104);ax.set_xticks([0,25,50,75,100])
handles=[plt.Line2D([],[],marker='o',ls='',color=BLUE,label='Archived answer containment'),
         plt.Line2D([],[],marker='D',ls='',mfc='white',mec=TEAL,label='Post-hoc exact-string audit')]
fig.legend(handles=handles,loc='upper center',ncol=2,bbox_to_anchor=(.54,1.005),fontsize=7.2,columnspacing=2)
ax=axs[0,1];frame(ax,'b  Request timing differs from decoding','Wall-request latency (ms)')
for i,r in enumerate(rows):
    med=float(r['latency_median_ms']);tail=float(r['latency_p95_linear_ms'])
    ax.plot([med,tail],[i,i],color=colors[i],lw=1.8,zorder=2)
    ax.scatter([med],[i],s=21,color=colors[i],zorder=3)
    ax.scatter([tail],[i],s=22,marker='>',color=colors[i],zorder=3)
ax.set_xlim(0,2600);ax.set_xticks([0,1000,2000])
ax=axs[1,0];frame(ax,'c  Output-token rate is a different boundary','Decode tokens/s: request mean ± SD')
for i,r in enumerate(rows):
    ax.errorbar(float(r['decode_tps_mean']),i,xerr=float(r['decode_tps_request_sd']),
                fmt='o',ms=4,color=colors[i],elinewidth=1,capsize=2,zorder=3)
ax.set_xlim(0,315);ax.set_xticks([0,100,200,300])
ax=axs[1,1];frame(ax,'d  Energy values remain endpoint proxies','Endpoint GPU proxy (J/output token)')
for i,r in enumerate(rows):ax.scatter([float(r['endpoint_gpu_energy_proxy_j_per_token'])],[i],s=25,color=colors[i],zorder=3)
ax.set_xlim(0,15);ax.set_xticks([0,5,10,15])
save(fig,'server_results_reanalysis')

raw=list(csv.DictReader((DATA/'archived/server_suite_raw_1777121713.csv').open()))
fig,axs=plt.subplots(1,3,figsize=(7.0,3.0),gridspec_kw={'width_ratios':[1.15,1,1.15]})
fig.subplots_adjust(left=.08,right=.985,top=.78,bottom=.22,wspace=.55)
ax=axs[0];ax.set_title('a  Context budget is not\nobserved input length',loc='left',fontweight='bold',pad=13)
for i,context in enumerate([512,1024,2048,4096]):
    actual=[int(r['prompt_tokens']) for r in raw if int(r['context_len'])==context]
    assert min(actual)>0 and context>0, 'Log-scale token counts must be positive.'
    ax.plot([min(actual),max(actual)],[i,i],color=TEAL,lw=3)
    ax.scatter([np.median(actual)],[i],color=TEAL,s=16,zorder=3)
    ax.scatter([context],[i],marker='D',facecolor='white',edgecolor=BLUE,s=24,zorder=3)
ax.set_xscale('log');ax.set_xlim(10,5500);ax.set_xticks([16,95,512,4096],['16','95','512','4096'])
ax.set_yticks(range(4),['512','1024','2048','4096']);ax.set_ylim(3.6,-.6)
ax.set_xlabel('Tokens (log scale)');ax.set_ylabel('Configured context ceiling')
ax.text(.0,-.36,'Teal: actual range; diamond: ceiling',transform=ax.transAxes,fontsize=6.5)
ax.tick_params(length=2.5,width=.5)
ax=axs[1];ax.set_title('b  A returned response\nis not guaranteed',loc='left',fontweight='bold',pad=13)
empty=[100*int(r['empty_responses'])/int(r['n_logged']) for r in [rows[4],rows[5]]]
ax.bar([0,1],empty,color=GRAY,width=.6,label='Empty stored response')
ax.bar([0,1],100-np.array(empty),bottom=empty,color=AMBER,width=.6,label='Nonempty stored response')
ax.set_ylim(0,100);ax.set_yticks([0,50,100]);ax.set_xticks([0,1],['Main\nn = 240','Follow-up\nn = 200'])
ax.set_ylabel('Qwen3 8B stored responses (%)')
ax.text(.0,-.36,'Grey: empty; amber: nonempty',transform=ax.transAxes,fontsize=6.5)
ax.tick_params(length=2.5,width=.5)
ax=axs[2];ax.set_axis_off();ax.set_title('c  Two endpoint samples\ndo not integrate power',loc='left',fontweight='bold',pad=13)
for y0,title,detail in [(.77,'Before request','Sum of visible GPU power'),(.37,'After request','Sum of visible GPU power')]:
    ax.add_patch(FancyBboxPatch((.02,y0-.15),.96,.28,boxstyle='round,pad=.01,rounding_size=.02',
                              fc='#EEF3F7',ec=BLUE,lw=.7,transform=ax.transAxes))
    ax.text(.5,y0+.025,title,ha='center',va='center',fontsize=7.5,weight='bold',transform=ax.transAxes)
    ax.text(.5,y0-.075,detail,ha='center',va='center',fontsize=6.5,transform=ax.transAxes)
ax.add_patch(FancyArrowPatch((.5,.61),(.5,.53),arrowstyle='-|>',mutation_scale=7,color=BLUE,lw=.8,transform=ax.transAxes))
ax.text(.5,.02,'Proxy = endpoint mean × wall time',ha='center',fontsize=6.5,transform=ax.transAxes)
ax.text(.0,-.36,'No sampled trace or board attribution',transform=ax.transAxes,fontsize=6.5)
save(fig,'server_measurement_boundaries')
