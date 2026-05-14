import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

fig, ax = plt.subplots(figsize=(22, 14))
ax.set_xlim(0, 22)
ax.set_ylim(0, 14)
ax.axis('off')
fig.patch.set_facecolor('#FAFAFA')

C_INPUT  = '#E8EAF6'
C_GATE   = '#FFCDD2'
C_SEP    = '#FFF9C4'
C_ENC    = '#BBDEFB'
C_ACOU   = '#C8E6C9'
C_ENGI   = '#E1BEE7'
C_VEC    = '#F3E5F5'
C_LGBM   = '#FFF3E0'
C_REAL   = '#A5D6A7'
C_SS     = '#FFAB91'
C_SE     = '#81D4FA'
C_FAKE   = '#EF9A9A'
BORDER   = '#455A64'
ARROW    = '#37474F'


def box(ax, x, y, w, h, title, sub=None, color='white',
        fontsize=9.5, bold=True, tc='#212121', r=0.25):
    p = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle=f"round,pad=0.05,rounding_size={r}",
                       facecolor=color, edgecolor=BORDER, linewidth=1.4, zorder=3)
    ax.add_patch(p)
    if sub:
        ax.text(x, y + 0.18, title, ha='center', va='center',
                fontsize=fontsize, fontweight='bold' if bold else 'normal',
                color=tc, zorder=4)
        ax.text(x, y - 0.22, sub, ha='center', va='center',
                fontsize=7.5, color='#546E7A', style='italic', zorder=4)
    else:
        ax.text(x, y, title, ha='center', va='center',
                fontsize=fontsize, fontweight='bold' if bold else 'normal',
                color=tc, zorder=4)


def arr(ax, x1, y1, x2, y2, label=None, lc=ARROW, lw=1.6, lpos='right'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=lc, lw=lw,
                                mutation_scale=14), zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        dx = 0.12 if lpos == 'right' else -0.12
        ax.text(mx + dx, my, label, fontsize=7.5, color='#455A64',
                ha='left' if lpos == 'right' else 'right', va='center', zorder=5)


def hline(ax, x1, x2, y, color=ARROW, lw=1.3, style='solid'):
    ax.plot([x1, x2], [y, y], color=color, lw=lw, linestyle=style, zorder=2)


def vline(ax, x, y1, y2, color=ARROW, lw=1.3, style='solid'):
    ax.plot([x, x], [y1, y2], color=color, lw=lw, linestyle=style, zorder=2)


# ── 제목 ──────────────────────────────────────────────────
ax.text(11, 13.4, 'CompSpoofV2 Audio Spoofing Detection System',
        ha='center', va='center', fontsize=15, fontweight='bold', color='#1A237E',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8EAF6',
                  edgecolor='#3949AB', lw=1.8))

# ══ ROW 1: Input ══════════════════════════════════════════
box(ax, 11, 12.3, 2.6, 0.75, 'Input Audio', color=C_INPUT, fontsize=11)
arr(ax, 11, 11.92, 11, 11.5)

# ══ ROW 2: Gatekeeper ═════════════════════════════════════
box(ax, 11, 11.0, 3.0, 0.85, 'Step 0  |  Gatekeeper', 'LCNN-SE  →  gate_score',
    color=C_GATE)

# gate score 가로로 피처벡터 쪽으로
arr(ax, 12.5, 11.0, 15.8, 11.0, label='gate_score', lpos='right')

# ══ ROW 3: SepFormer ══════════════════════════════════════
arr(ax, 11, 10.57, 11, 10.1)
box(ax, 11, 9.65, 3.2, 0.8, 'Step 2  |  SepFormer',
    'Fine-tuned v2 (all 4 compound labels)', color=C_SEP)

# Speech / Env 스트림 분기
#  왼쪽
vline(ax, 7.5, 9.65, 8.7)
hline(ax, 7.5, 9.4, 9.65)
arr(ax, 7.5, 8.7, 7.5, 8.25)
ax.text(7.0, 9.15, 'Speech\nStream', fontsize=8, color='#1565C0',
        ha='center', va='center')

#  오른쪽
vline(ax, 14.5, 9.65, 8.7)
hline(ax, 9.4, 14.5, 9.65)
arr(ax, 14.5, 8.7, 14.5, 8.25)
ax.text(15.0, 9.15, 'Env\nStream', fontsize=8, color='#2E7D32',
        ha='center', va='center')

# ══ ROW 4: Encoders + Acoustic ════════════════════════════
box(ax, 7.5, 7.85, 2.6, 0.75, 'Speech Encoder', '(LCNN-SE)', color=C_ENC)
box(ax, 14.5, 7.85, 2.6, 0.75, 'Env Encoder', '(LCNN-SE)', color=C_ENC)

# center acoustic steps
box(ax, 9.5, 7.85, 2.2, 0.75, 'Step 6', 'Noise Floor', color=C_ACOU)
box(ax, 11.0, 7.85, 2.2, 0.75, 'Step 7.5', 'EDC / RT60 Slope', color=C_ACOU)
box(ax, 12.8, 7.85, 2.0, 0.75, 'MSC + XCorr', '', color=C_ACOU)

# stream lines to acoustic
hline(ax, 8.8, 8.4, 7.85, color='#90A4AE', lw=1.0, style='dashed')
hline(ax, 13.8, 11.9, 7.85, color='#90A4AE', lw=1.0, style='dashed')

# encoder outputs → collector
arr(ax, 7.5, 7.47, 7.5, 7.0, label='speech_score')
arr(ax, 14.5, 7.47, 14.5, 7.0, label='env_score', lpos='left')
arr(ax, 9.5, 7.47, 9.5, 7.0, label='noise_dist')
arr(ax, 11.0, 7.47, 11.0, 7.0, label='slope×3', lpos='left')
arr(ax, 12.8, 7.47, 12.8, 7.0, label='msc, xcorr', lpos='left')

# ══ ROW 5: Feature collector line ═════════════════════════
hline(ax, 7.0, 15.0, 6.9, color='#78909C', lw=1.0, style='dashed')
# collector → engineering
arr(ax, 11.0, 6.9, 11.0, 6.45)

# ══ ROW 5b: Feature Engineering ══════════════════════════
box(ax, 11.0, 6.1, 4.0, 0.65, 'Feature Engineering',
    'speech×env  |  max_stream  |  |speech - env|', color=C_ENGI)
arr(ax, 11.0, 5.77, 11.0, 5.25)

# ══ ROW 6: 12-Dim Feature Vector ══════════════════════════
feat_box = FancyBboxPatch((7.5, 4.1), 7.0, 1.0,
                          boxstyle='round,pad=0.05,rounding_size=0.2',
                          facecolor=C_VEC, edgecolor='#7986CB',
                          linewidth=1.5, linestyle='dashed', zorder=1)
ax.add_patch(feat_box)
ax.text(11.0, 5.25, '12-Dimensional Feature Vector', ha='center', va='center',
        fontsize=9, fontweight='bold', color='#4527A0', zorder=4)

feats = ['gate_score', 'speech_score', 'env_score', 'noise_dist',
         'slope_diff', 'slope_s', 'slope_e', 'msc', 'xcorr',
         'speech×env', 'max_stream', '|Δstream|']
feat_colors = ['#FFCDD2', '#BBDEFB', '#C8E6C9', '#F3E5F5',
               '#F3E5F5', '#F3E5F5', '#F3E5F5', '#DCEDC8', '#DCEDC8',
               '#E1BEE7', '#E1BEE7', '#E1BEE7']
ncols = 6
for i, (f, fc) in enumerate(zip(feats, feat_colors)):
    row = i // ncols
    col = i % ncols
    fx = 8.05 + col * 1.15
    fy = 4.82 - row * 0.47
    fp = FancyBboxPatch((fx - 0.52, fy - 0.18), 1.04, 0.35,
                        boxstyle='round,pad=0.02,rounding_size=0.08',
                        facecolor=fc, edgecolor='#90A4AE', linewidth=0.9, zorder=4)
    ax.add_patch(fp)
    ax.text(fx, fy + 0.0, f, ha='center', va='center', fontsize=7.2, zorder=5)

arr(ax, 11.0, 4.1, 11.0, 3.55)

# ══ ROW 7: LightGBM ═══════════════════════════════════════
box(ax, 11.0, 3.15, 4.0, 0.75, 'Step 8  |  LightGBM Classifier',
    '5-Fold CV  |  class_weight: {REAL×2.9, SS×3.3, SE×1.0, FAKE×1.7}',
    color=C_LGBM, fontsize=9.5)

# 4 outputs
out_data = [
    (8.2,  'REAL',         C_REAL, '#1B5E20'),
    (9.8,  'SPOOF\nSPEECH', C_SS,  '#BF360C'),
    (11.4, 'SPOOF\nENV',    C_SE,  '#01579B'),
    (13.0, 'FAKE',          C_FAKE, '#B71C1C'),
]
for xo, lbl, col, tc in out_data:
    arr(ax, 11.0, 2.77, xo, 2.25, lc='#546E7A', lw=1.3)
    box(ax, xo, 1.9, 1.3, 0.6, lbl, color=col, fontsize=9.5, tc=tc)

ax.text(11.0, 1.25, 'Classification Output', ha='center', fontsize=8.5,
        color='#37474F', style='italic')

# ══ 범례 ══════════════════════════════════════════════════
legend_items = [
    mpatches.Patch(facecolor=C_GATE, edgecolor=BORDER, label='Gatekeeper / Anti-Spoofing (LCNN-SE)'),
    mpatches.Patch(facecolor=C_SEP,  edgecolor=BORDER, label='Source Separation (SepFormer v2)'),
    mpatches.Patch(facecolor=C_ENC,  edgecolor=BORDER, label='Stream Encoder (LCNN-SE)'),
    mpatches.Patch(facecolor=C_ACOU, edgecolor=BORDER, label='Acoustic Analysis'),
    mpatches.Patch(facecolor=C_ENGI, edgecolor=BORDER, label='Feature Engineering'),
    mpatches.Patch(facecolor=C_LGBM, edgecolor=BORDER, label='LightGBM Classifier'),
]
ax.legend(handles=legend_items, loc='lower left', fontsize=8.5,
          framealpha=0.95, edgecolor='#90A4AE', bbox_to_anchor=(0.01, 0.01))

plt.tight_layout(pad=0.5)
plt.savefig('/root/architecture.png', dpi=160, bbox_inches='tight',
            facecolor='#FAFAFA', edgecolor='none')
print("저장: /root/architecture.png")
