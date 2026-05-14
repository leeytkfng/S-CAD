import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(0, 14); ax.set_ylim(0, 5)
ax.axis('off')
fig.patch.set_facecolor('white')

C = dict(
    input='#E8EAF6', gate='#FFCDD2', sep='#FFF9C4',
    enc='#BBDEFB', acou='#C8E6C9', feat='#E1BEE7',
    lgbm='#FFF3E0', out_r='#A5D6A7', out_s='#EF9A9A',
    border='#455A64', arr='#37474F'
)

def box(x,y,w,h,txt,col,fs=8.5,bold=False,sub=None):
    p = FancyBboxPatch((x-w/2,y-h/2),w,h,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        facecolor=col, edgecolor=C['border'], lw=1.2, zorder=3)
    ax.add_patch(p)
    weight='bold' if bold else 'normal'
    if sub:
        ax.text(x,y+0.13,txt,ha='center',va='center',fontsize=fs,
                fontweight=weight,color='#212121',zorder=4)
        ax.text(x,y-0.2,sub,ha='center',va='center',fontsize=6.5,
                color='#546E7A',style='italic',zorder=4)
    else:
        ax.text(x,y,txt,ha='center',va='center',fontsize=fs,
                fontweight=weight,color='#212121',zorder=4)

def arr(x1,y1,x2,y2,lbl=None):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),
        arrowprops=dict(arrowstyle='->',color=C['arr'],lw=1.4),zorder=2)
    if lbl:
        mx,my=(x1+x2)/2,(y1+y2)/2
        ax.text(mx+0.05,my+0.08,lbl,fontsize=6.5,color='#546E7A',zorder=5)

def line(x1,y1,x2,y2,style='solid',lw=1.2):
    ax.plot([x1,x2],[y1,y2],color=C['arr'],lw=lw,linestyle=style,zorder=2)

# ── 입력 ──────────────────────────────────────────────────
box(1,2.5,1.4,0.7,'Input\nAudio',C['input'],fs=9,bold=True)

# ── Gate ──────────────────────────────────────────────────
arr(1.7,2.5,2.5,2.5)
box(3.0,2.5,1.0,0.65,'Gate\n(LCNN-SE)',C['gate'],fs=8,sub='gate_score')

# gate_score 위로
line(3.0,2.83,3.0,4.0)
ax.text(3.05,3.6,'gate_score',fontsize=6.5,color='#546E7A',rotation=90,va='center')

# ── SepFormer ─────────────────────────────────────────────
arr(3.5,2.5,4.5,2.5)
box(5.2,2.5,1.4,0.7,'SepFormer\n(Fine-tuned)',C['sep'],fs=8,bold=True)

# 두 스트림
line(5.2,2.85,5.2,3.8); ax.text(4.6,3.3,'Speech\nStream',fontsize=6.5,color='#1565C0',ha='center')
arr(5.2,3.8,4.2,3.8)
box(3.6,3.8,1.0,0.55,'Speech\nEncoder',C['enc'],fs=7.5,sub='speech_score')
arr(3.1,3.8,2.5,3.8)

line(5.2,2.15,5.2,1.2); ax.text(4.6,1.7,'Env\nStream',fontsize=6.5,color='#2E7D32',ha='center')
arr(5.2,1.2,4.2,1.2)
box(3.6,1.2,1.0,0.55,'Env\nEncoder',C['enc'],fs=7.5,sub='env_score')
arr(3.1,1.2,2.5,1.2)

# ── 음향 분석 ──────────────────────────────────────────────
arr(5.9,2.5,6.7,2.5)
box(7.3,2.5,1.1,1.6,'Acoustic\nAnalysis',C['acou'],fs=8)
ax.text(7.3,3.1,'noise_dist',fontsize=6,color='#333',ha='center')
ax.text(7.3,2.7,'slope_diff',fontsize=6,color='#333',ha='center')
ax.text(7.3,2.35,'msc / xcorr',fontsize=6,color='#333',ha='center')
ax.text(7.3,2.0,'energy_ratio',fontsize=6,color='#333',ha='center')

# 세 피처 합류 수평선
line(2.5,4.0,9.5,4.0,style='dashed',lw=0.9)  # 상단 수집선
for xp in [2.5,3.0,7.8]:
    line(xp,4.0,xp,3.8,lw=0.9)
line(2.5,1.0,7.8,1.0,style='dashed',lw=0.9)  # 하단 수집선
for xp in [2.5,3.0,7.8]:
    line(xp,1.2,xp,1.0,lw=0.9)

arr(7.85,2.5,8.7,2.5)

# ── 피처 엔지니어링 ────────────────────────────────────────
box(9.15,2.5,0.9,0.65,'Feature\nEngineering',C['feat'],fs=7.5,
    sub='13-dim vector')
arr(9.6,2.5,10.3,2.5)

# ── LightGBM ──────────────────────────────────────────────
box(10.9,2.5,1.0,0.65,'LightGBM\nClassifier',C['lgbm'],fs=8,bold=True)

# ── 출력 ──────────────────────────────────────────────────
arr(11.4,2.5,12.2,3.1)
arr(11.4,2.5,12.2,1.9)
box(12.7,3.1,1.0,0.5,'Authentic',C['out_r'],fs=8,bold=True)
box(12.7,1.9,1.0,0.5,'Spoof',C['out_s'],fs=8,bold=True)

# 범례
legend = [
    mpatches.Patch(fc=C['gate'],  ec=C['border'], label='Anti-Spoofing (LCNN-SE)'),
    mpatches.Patch(fc=C['sep'],   ec=C['border'], label='Source Separation (SepFormer)'),
    mpatches.Patch(fc=C['enc'],   ec=C['border'], label='Stream Encoder'),
    mpatches.Patch(fc=C['acou'],  ec=C['border'], label='Acoustic Analysis'),
    mpatches.Patch(fc=C['feat'],  ec=C['border'], label='Feature Engineering'),
    mpatches.Patch(fc=C['lgbm'],  ec=C['border'], label='LightGBM'),
]
ax.legend(handles=legend,loc='lower center',ncol=6,fontsize=6.5,
          framealpha=0.9,edgecolor='#90A4AE',bbox_to_anchor=(0.5,-0.02))

ax.set_title('Figure 1. S-CAD Pipeline Architecture',
             fontsize=10, fontweight='bold', pad=8)

plt.tight_layout()
plt.savefig('/root/figure1_scad.png', dpi=180, bbox_inches='tight',
            facecolor='white')
print("저장: figure1_scad.png")
