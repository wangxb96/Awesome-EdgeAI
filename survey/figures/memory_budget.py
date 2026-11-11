"""Analytical memory illustration, not measured hardware performance.

A hypothetical decoder separates fixed weight storage from linear persistent
KV state. Every curve uses the complete 0--32768-token range. Runtime buffers,
activations, allocator overhead and OS memory are deliberately not modelled.
"""
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],"font.size":7,"pdf.fonttype":42,"svg.fonttype":"none","axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.6})
# Explicit illustrative assumptions, not a named model configuration.
params=7e9; layers=32; kv_heads=8; head_dim=128; batch=1; kv_bits=16
context=np.linspace(0,32768,129)
gib=2**30
kv=2*layers*batch*context*kv_heads*head_dim*kv_bits/8/gib
w4=params*4/8/gib; w16=params*16/8/gib
fig,ax=plt.subplots(figsize=(3.48,2.12))
fig.subplots_adjust(left=.15,right=.97,bottom=.21,top=.94)
ax.fill_between(context/1024,w4,w4+kv,color="#D9EBE5",zorder=1)
ax.plot(context/1024,w16+kv,color="#497BA6",lw=1.25,zorder=3)
ax.plot(context/1024,w4+kv,color="#318D85",lw=1.25,zorder=3)
ax.axhline(w4,color="#99B8AF",lw=.65,ls=":",zorder=2)
ax.axhline(8,color="#7A7D84",lw=.8,ls=(0,(4,3)),zorder=2)
ax.text(.7,8.55,"Illustrative 8 GiB budget",fontsize=6.1,color="black")
ax.text(31.5,17.75,"16-bit weights + KV",ha="right",fontsize=6.2,color="black")
ax.text(31.5,4.6,"4-bit weights + KV",ha="right",fontsize=6.2,color="black")
ax.text(20,4.3,"KV state",ha="center",fontsize=6.0,color="black")
ax.set(xlim=(0,32),ylim=(0,19.2),xticks=[0,8,16,24,32],yticks=[0,4,8,12,16],xlabel="Retained context (×1024 tokens)",ylabel="Weights + KV memory (GiB)")
ax.tick_params(labelsize=6.3,length=2.5,width=.6)
ax.yaxis.grid(False)
ax.set_axisbelow(True)
fig.savefig(OUT/"memory_budget.pdf",bbox_inches="tight",pad_inches=.025)
fig.savefig(OUT/"memory_budget.svg",bbox_inches="tight",pad_inches=.025)
fig.savefig(OUT/"memory_budget.tiff",dpi=600,bbox_inches="tight",pad_inches=.025,pil_kwargs={"compression":"tiff_lzw"})
fig.savefig(OUT/"memory_budget.png",dpi=300,bbox_inches="tight",pad_inches=.025)
np.savetxt(OUT/"memory_budget_analytical.csv",np.column_stack([context,kv,np.full_like(kv,w4),w4+kv,np.full_like(kv,w16),w16+kv]),delimiter=",",header="context_tokens,kv_GiB,weights_4bit_GiB,total_4bit_GiB,weights_16bit_GiB,total_16bit_GiB",comments="")
plt.close(fig)
