"""
SepFormer Fine-tuning Report Generator  (English, matplotlib PdfPages)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

OUTPUT = "sepformer_finetune_report.pdf"

C_BLUE   = '#2C6FAC'
C_GREEN  = '#2E8B57'
C_RED    = '#C0392B'
C_ORANGE = '#E67E22'
C_GRAY   = '#7F8C8D'
C_DARK   = '#2C3E50'
C_YELLOW = '#F39C12'

def new_fig(w=11, h=8.5):
    return plt.figure(figsize=(w, h), facecolor='white')

def footer(fig, page_n):
    fig.text(0.5, 0.02,
             f"S-CAD Pipeline  |  SepFormer Fine-tuning Report  |  Page {page_n}",
             ha='center', fontsize=8, color=C_GRAY)
    fig.add_artist(plt.Line2D([0.05, 0.95], [0.04, 0.04],
                               transform=fig.transFigure,
                               color=C_GRAY, linewidth=0.5))

def rounded_box(ax, x, y, w, h, fc, ec, lw=1.5, radius=0.03):
    rect = mpatches.FancyBboxPatch((x, y), w, h,
                                    boxstyle=f"round,pad={radius}",
                                    facecolor=fc, edgecolor=ec, linewidth=lw,
                                    transform=ax.transAxes, zorder=2)
    ax.add_patch(rect)

with PdfPages(OUTPUT) as pdf:

    # ── PAGE 1: Cover ───────────────────────────────────────────
    fig = new_fig()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(C_DARK); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    for y in np.linspace(0.05, 0.95, 10):
        ax.axhline(y, color='white', alpha=0.03, linewidth=18)

    fig.text(0.5, 0.74, "SepFormer Fine-tuning", ha='center',
             fontsize=32, fontweight='bold', color='white')
    fig.text(0.5, 0.65, "Analysis Report", ha='center', fontsize=22, color='#AED6F1')
    fig.text(0.5, 0.57, "S-CAD Pipeline  —  CompSpoofV2 Domain Adaptation",
             ha='center', fontsize=13, color='#85C1E9')
    fig.add_artist(plt.Line2D([0.2, 0.8], [0.52, 0.52],
                               transform=fig.transFigure, color='#AED6F1', linewidth=1.5))

    toc = [
        "1.  Problem Definition: Domain Mismatch",
        "2.  Fine-tuning Design & Loss Functions",
        "3.  Training Convergence Results",
        "4.  Feature Comparison: Before vs After",
        "5.  Conclusion & Implications",
    ]
    for i, t in enumerate(toc):
        fig.text(0.5, 0.46 - i*0.056, t, ha='center', fontsize=11.5, color='#D5E8F3')

    fig.text(0.5, 0.08, "2026", ha='center', fontsize=10, color=C_GRAY)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ── PAGE 2: Problem Definition ─────────────────────────────
    fig = new_fig()
    fig.text(0.5, 0.96, "1. Problem Definition: Domain Mismatch",
             ha='center', fontsize=17, fontweight='bold', color=C_DARK)
    fig.text(0.5, 0.92, "Why was SepFormer fine-tuning necessary?",
             ha='center', fontsize=10.5, color=C_GRAY, style='italic')
    footer(fig, 2)

    # Left box: pretrained domain
    ax1 = fig.add_axes([0.05, 0.60, 0.40, 0.28])
    ax1.set_xlim(0,1); ax1.set_ylim(0,1); ax1.axis('off')
    rounded_box(ax1, 0.01, 0.01, 0.98, 0.98, '#EBF5FB', C_BLUE)
    ax1.set_title("Pretrained Domain  (WSJ0-2Mix)", fontsize=10.5,
                  fontweight='bold', color=C_BLUE, pad=7)
    for y, txt in [(0.76, "Input: Speech + Speech mixture"),
                   (0.55, "Stream 1  ->  Speaker A"),
                   (0.36, "Stream 2  ->  Speaker B"),
                   (0.14, "Both streams have speech characteristics")]:
        ax1.text(0.5, y, txt, ha='center', fontsize=9.5, color=C_DARK)

    # Right box: actual domain
    ax2 = fig.add_axes([0.55, 0.60, 0.40, 0.28])
    ax2.set_xlim(0,1); ax2.set_ylim(0,1); ax2.axis('off')
    rounded_box(ax2, 0.01, 0.01, 0.98, 0.98, '#FDEDEC', C_RED)
    ax2.set_title("Actual Domain  (CompSpoofV2)", fontsize=10.5,
                  fontweight='bold', color=C_RED, pad=7)
    for y, txt, c in [(0.76, "Input: Speech + Environment mixture", C_DARK),
                      (0.55, "Stream 1  ->  Human speech", C_BLUE),
                      (0.36, "Stream 2  ->  Env. sound  <- completely different!", C_RED),
                      (0.14, "Model has never seen env. sound during training", C_RED)]:
        ax2.text(0.5, y, txt, ha='center', fontsize=9.5, color=c)

    # Arrow between boxes (use overlay axes)
    ax_arr = fig.add_axes([0.44, 0.68, 0.12, 0.10])
    ax_arr.set_xlim(0,1); ax_arr.set_ylim(0,1); ax_arr.axis('off')
    ax_arr.annotate("", xy=(0.85, 0.5), xytext=(0.15, 0.5),
                    arrowprops=dict(arrowstyle='->', color=C_ORANGE, lw=2.5))
    ax_arr.text(0.5, 0.18, "Applied\nto", ha='center', fontsize=8, color=C_ORANGE)

    # Consequence bar chart
    ax3 = fig.add_axes([0.05, 0.17, 0.88, 0.36])
    ax3.set_facecolor('#FAFAFA')
    ax3.set_title(
        "Consequence of Domain Mismatch — env stream score by compound label (before fine-tuning)",
        fontsize=10, fontweight='bold', color=C_DARK, pad=7)

    labels = ['bonafide_bonafide\n(REAL)', 'spoof_bonafide\n(MANIP)',
              'bonafide_spoof\n(MANIP)', 'spoof_spoof\n(FAKE)']
    env_before = [0.172, 0.433, 0.414, 0.472]
    colors_b   = [C_GREEN, C_ORANGE, C_ORANGE, C_RED]
    x = np.arange(4)
    bars = ax3.bar(x, env_before, color=colors_b, alpha=0.75, width=0.5, edgecolor='white')
    ax3.axhline(0.05, color=C_RED, linestyle='--', linewidth=1.5,
                label='Ideal REAL baseline')
    ax3.set_xticks(x); ax3.set_xticklabels(labels, fontsize=9)
    ax3.set_ylabel('env_score  (LCNN-SE output)', fontsize=9)
    ax3.set_ylim(0, 0.60)
    for bar, val in zip(bars, env_before):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    ax3.text(1.5, 0.52,
             "REAL env_score too low (0.172)\n"
             "-> SepFormer assigns near-zero\n"
             "   energy to env stream",
             ha='center', fontsize=8.5, color=C_RED,
             bbox=dict(boxstyle='round', facecolor='#FDEDEC', alpha=0.85))
    ax3.legend(fontsize=8.5); ax3.grid(axis='y', alpha=0.3)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ── PAGE 3: Fine-tuning Design ─────────────────────────────
    fig = new_fig()
    fig.text(0.5, 0.96, "2. Fine-tuning Design & Loss Functions",
             ha='center', fontsize=17, fontweight='bold', color=C_DARK)
    fig.text(0.5, 0.92, "What was trained and how?",
             ha='center', fontsize=10.5, color=C_GRAY, style='italic')
    footer(fig, 3)

    # Config box
    ax_cfg = fig.add_axes([0.05, 0.74, 0.90, 0.15])
    ax_cfg.set_xlim(0,1); ax_cfg.set_ylim(0,1); ax_cfg.axis('off')
    rounded_box(ax_cfg, 0.01, 0.01, 0.98, 0.98, '#EBF5FB', C_BLUE)
    cfg_items = [
        ("Data",    "CompSpoofV2 bonafide_bonafide  25,189 pairs  (mix, speech_gt, env_gt)"),
        ("Frozen",  "Encoder frozen (preserve learned filterbank)  |  MaskNet + Decoder trained (25.7M params)"),
        ("Config",  "Batch=8   LR=1e-5   Epochs=5   CosineAnnealingLR   GradClip max_norm=5.0   AMP enabled"),
    ]
    for i, (lbl, txt) in enumerate(cfg_items):
        ax_cfg.text(0.01, 0.84 - i*0.32, f"[{lbl}]", fontsize=10, fontweight='bold',
                    color=C_BLUE, va='top')
        ax_cfg.text(0.09, 0.84 - i*0.32, txt, fontsize=9, color=C_DARK, va='top')

    # SI-SNR PIT Loss box
    ax_l1 = fig.add_axes([0.05, 0.41, 0.42, 0.28])
    ax_l1.set_xlim(0,1); ax_l1.set_ylim(0,1); ax_l1.axis('off')
    rounded_box(ax_l1, 0.01, 0.01, 0.98, 0.98, '#EBF5FB', C_BLUE)
    ax_l1.set_title("SI-SNR PIT Loss", fontsize=11, fontweight='bold', color=C_BLUE, pad=7)
    pit_lines = [
        "Optimizes separation quality directly.",
        "PIT: permutation-invariant assignment",
        "  -> best stream pairing selected.",
        "",
        "SI-SNR(s_hat, s) =",
        "  10*log10( ||alpha*s||^2 /",
        "            ||s_hat - alpha*s||^2 )",
        "",
        "L_PIT = min_pi [ -SI-SNR(s1,s_pi1)",
        "                 -SI-SNR(s2,s_pi2) ]",
    ]
    for i, line in enumerate(pit_lines):
        ax_l1.text(0.05, 0.90 - i*0.093, line, fontsize=8.5, color=C_DARK, va='top',
                   fontfamily='monospace' if any(c in line for c in ['=','||','*','[','-']) else 'sans-serif')

    # Energy Ratio Loss box
    ax_l2 = fig.add_axes([0.53, 0.41, 0.42, 0.28])
    ax_l2.set_xlim(0,1); ax_l2.set_ylim(0,1); ax_l2.axis('off')
    rounded_box(ax_l2, 0.01, 0.01, 0.98, 0.98, '#FDEDEC', C_RED)
    ax_l2.set_title("Energy Ratio Loss  <-- Core Contribution",
                    fontsize=11, fontweight='bold', color=C_RED, pad=7)
    er_lines = [
        "Corrects env stream under-estimation.",
        "MSE between predicted & GT energy ratio.",
        "",
        "r_gt  = E(env_gt) /",
        "        (E(speech_gt) + E(env_gt))",
        "",
        "r_est = E(s_hat_env) /",
        "        (E(s_hat_sp) + E(s_hat_env))",
        "",
        "L_energy = E[ (r_est - r_gt)^2 ]",
    ]
    for i, line in enumerate(er_lines):
        ax_l2.text(0.05, 0.90 - i*0.093, line, fontsize=8.5, color=C_DARK, va='top',
                   fontfamily='monospace' if any(c in line for c in ['=','(','[','^']) else 'sans-serif')

    # Total loss box
    ax_tot = fig.add_axes([0.05, 0.10, 0.90, 0.26])
    ax_tot.set_xlim(0,1); ax_tot.set_ylim(0,1); ax_tot.axis('off')
    rounded_box(ax_tot, 0.01, 0.01, 0.98, 0.98, '#FAFAFA', C_DARK)
    ax_tot.text(0.5, 0.82, "Total Loss Function", ha='center', fontsize=12,
                fontweight='bold', color=C_DARK)
    ax_tot.text(0.5, 0.52,
                "L_total  =  L_PIT  +  lambda * L_energy          (lambda = 0.5)",
                ha='center', fontsize=13, color=C_DARK, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor=C_DARK, linewidth=1.5))
    ax_tot.text(0.5, 0.18,
                "lambda=0.5: equal weight for separation quality & energy distribution  |  "
                "Encoder frozen to preserve pretrained filterbank",
                ha='center', fontsize=9, color=C_GRAY, style='italic')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ── PAGE 4: Training Convergence ───────────────────────────
    fig = new_fig()
    fig.text(0.5, 0.96, "3. Training Convergence Results",
             ha='center', fontsize=17, fontweight='bold', color=C_DARK)
    fig.text(0.5, 0.92, "Loss evolution over 5 epochs  (3,149 batches/epoch, batch=8, 25,189 samples)",
             ha='center', fontsize=10.5, color=C_GRAY, style='italic')
    footer(fig, 4)

    # Loss curve
    # Approximate curve from actual log data
    ep_size = 3149
    checkpoints = [
        (100,   9.28),  (200,   0.00),  (300,  -4.25),  (400,  -6.83),  (500,  -8.69),
        (700,  -10.5),  (1000, -12.8),  (1500, -14.5),  (2000, -16.1),  (3000, -18.2),
        (ep_size, -19.5),
        (ep_size+500, -20.1), (ep_size+1500, -20.6), (ep_size+3000, -21.1),
        (ep_size*2, -21.1),
        (ep_size*2+500, -21.4), (ep_size*2+2000, -21.7), (ep_size*2+3000, -21.9),
        (ep_size*3, -21.9),
        (ep_size*3+500, -22.1), (ep_size*3+2000, -22.2), (ep_size*3+3000, -22.3),
        (ep_size*4, -22.2),
        (ep_size*4+500, -22.3), (ep_size*4+1000, -22.35),
        (ep_size*4+2000, -22.45), (ep_size*4+2900, -22.51),
        (ep_size*4+3000, -22.53), (ep_size*5, -22.54),
    ]
    bx, ly = zip(*checkpoints)

    ax = fig.add_axes([0.08, 0.42, 0.88, 0.44])
    ax.plot(bx, ly, color=C_BLUE, linewidth=2.2, zorder=3)
    ax.fill_between(bx, ly, -25, alpha=0.10, color=C_BLUE)
    ax.set_facecolor('#FAFAFA')
    ax.grid(True, alpha=0.3)

    for i in range(1, 6):
        ax.axvline(ep_size*i, color='gray', linestyle='--', alpha=0.4, linewidth=1)
        ax.text(ep_size*i - ep_size*0.55, -24.2, f'Epoch {i}', fontsize=8, color=C_GRAY)

    ax.set_xlabel('Cumulative Batch', fontsize=10)
    ax.set_ylabel('Loss  (lower is better)', fontsize=10)
    ax.set_title('Full Training Loss Curve  (SI-SNR PIT + 0.5 x Energy Ratio Loss)',
                 fontsize=11, fontweight='bold')
    ax.set_ylim(-25, 12)

    kp = [(100, 9.28, '+9.28\n(start)'), (500, -8.69, '-8.69'),
          (ep_size*5, -22.54, '-22.54\n(final)')]
    for bxi, lyi, lbl in kp:
        offset = 4 if lyi > 0 else -3.5
        ax.annotate(lbl, xy=(bxi, lyi), xytext=(bxi, lyi + offset),
                    fontsize=8, ha='center', color=C_RED,
                    arrowprops=dict(arrowstyle='->', color=C_RED, lw=1),
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor=C_RED))

    # Summary table
    ax2 = fig.add_axes([0.05, 0.10, 0.88, 0.26])
    ax2.set_xlim(0,1); ax2.set_ylim(0,1); ax2.axis('off')
    ax2.set_title("Per-epoch Loss Summary", fontsize=11, fontweight='bold',
                  color=C_DARK, pad=6)

    cols     = ['Epoch', 'Batches', 'avg_loss', 'SI-SNR', 'E-Ratio MSE', 'Improvement']
    rows     = [
        ['1', '3,149', '-19.50', '-19.58', '0.169', '+28.78  (initial -> Ep1)'],
        ['2', '3,149', '-21.10', '-21.18', '0.163', '+1.60'],
        ['3', '3,149', '-21.90', '-21.97', '0.158', '+0.80'],
        ['4', '3,149', '-22.20', '-22.27', '0.155', '+0.30'],
        ['5', '3,149', '-22.54', '-22.62', '0.151', '+0.34   <- Best checkpoint'],
    ]
    col_xs = [0.01, 0.09, 0.19, 0.29, 0.39, 0.52]

    hdr = mpatches.FancyBboxPatch((0, 0.82), 1, 0.18, boxstyle="square,pad=0",
                                   facecolor=C_DARK, transform=ax2.transAxes)
    ax2.add_patch(hdr)
    for cx, col in zip(col_xs, cols):
        ax2.text(cx+0.01, 0.89, col, fontsize=8.5, color='white',
                 fontweight='bold', va='center')

    for i, row in enumerate(rows):
        y = 0.72 - i * 0.148
        bg = '#F2F3F4' if i % 2 == 0 else 'white'
        ax2.add_patch(mpatches.FancyBboxPatch((0, y-0.055), 1, 0.135,
                       boxstyle="square,pad=0", facecolor=bg, transform=ax2.transAxes))
        for cx, val in zip(col_xs, row):
            c = C_GREEN if 'Best' in val else C_DARK
            ax2.text(cx+0.01, y, val, fontsize=8, color=c, va='center')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ── PAGE 5: Before vs After Feature Comparison ─────────────
    fig = new_fig()
    fig.text(0.5, 0.96, "4. Feature Comparison: Before vs After Fine-tuning",
             ha='center', fontsize=17, fontweight='bold', color=C_DARK)
    fig.text(0.5, 0.92, "Based on probe_features.py  (10 samples per compound label)",
             ha='center', fontsize=10.5, color=C_GRAY, style='italic')
    footer(fig, 5)

    compounds_short = ['bb\n(REAL)', 'sb\n(MANIP)', 'bsp\n(MANIP)', 'ss\n(FAKE)']
    x = np.arange(4); bw = 0.35

    before_s = [0.032, 0.513, 0.233, 0.425]
    after_s  = [0.019, 0.271, 0.338, 0.668]
    before_e = [0.172, 0.433, 0.414, 0.472]
    after_e  = [0.320, 0.237, 0.291, 0.461]

    # speech_score bar
    ax1 = fig.add_axes([0.05, 0.60, 0.40, 0.28])
    ax1.bar(x-bw/2, before_s, bw, label='Before', color=C_GRAY,   alpha=0.75, edgecolor='white')
    ax1.bar(x+bw/2, after_s,  bw, label='After',  color=C_BLUE,   alpha=0.85, edgecolor='white')
    ax1.set_title('[B] speech_score  Comparison', fontsize=10, fontweight='bold')
    ax1.set_xticks(x); ax1.set_xticklabels(compounds_short, fontsize=9)
    ax1.set_ylim(0, 0.85); ax1.legend(fontsize=8); ax1.grid(axis='y', alpha=0.3)
    ax1.set_facecolor('#FAFAFA')
    ax1.text(3.45, 0.72, 'FAKE up  ->\nimproved', ha='right', fontsize=8, color=C_GREEN,
             bbox=dict(boxstyle='round', facecolor='#E9F7EF', alpha=0.85))
    for xi, (bv, av) in enumerate(zip(before_s, after_s)):
        ax1.text(xi-bw/2, bv+0.01, f'{bv:.3f}', ha='center', fontsize=7, color=C_GRAY)
        ax1.text(xi+bw/2, av+0.01, f'{av:.3f}', ha='center', fontsize=7, color=C_BLUE, fontweight='bold')

    # env_score bar
    ax2 = fig.add_axes([0.55, 0.60, 0.40, 0.28])
    ax2.bar(x-bw/2, before_e, bw, label='Before', color=C_GRAY, alpha=0.75, edgecolor='white')
    ax2.bar(x+bw/2, after_e,  bw, label='After',  color=C_RED,  alpha=0.85, edgecolor='white')
    ax2.set_title('[C] env_score  Comparison', fontsize=10, fontweight='bold')
    ax2.set_xticks(x); ax2.set_xticklabels(compounds_short, fontsize=9)
    ax2.set_ylim(0, 0.60); ax2.legend(fontsize=8); ax2.grid(axis='y', alpha=0.3)
    ax2.set_facecolor('#FAFAFA')
    ax2.text(0.45, 0.50, 'REAL rose (0.172->0.320)\nmonotonicity broken!',
             ha='center', fontsize=8, color=C_RED,
             bbox=dict(boxstyle='round', facecolor='#FDEDEC', alpha=0.85))
    for xi, (bv, av) in enumerate(zip(before_e, after_e)):
        ax2.text(xi-bw/2, bv+0.01, f'{bv:.3f}', ha='center', fontsize=7, color=C_GRAY)
        ax2.text(xi+bw/2, av+0.01, f'{av:.3f}', ha='center', fontsize=7, color=C_RED, fontweight='bold')

    # 2D scatter before
    ax3 = fig.add_axes([0.06, 0.13, 0.37, 0.38])
    cols_2d = [C_GREEN, C_ORANGE, C_ORANGE, C_RED]
    lbls_2d = ['REAL(bb)', 'MANIP(sb)', 'MANIP(bsp)', 'FAKE(ss)']
    mks_2d  = ['o', 's', '^', 'D']
    for (s,e), c, l, m in zip(zip(before_s, before_e), cols_2d, lbls_2d, mks_2d):
        ax3.scatter(s, e, color=c, s=130, label=l, marker=m, zorder=5, edgecolor='white', linewidth=0.5)
    ax3.set_xlabel('speech_score', fontsize=9); ax3.set_ylabel('env_score', fontsize=9)
    ax3.set_title('2D Distribution  (Before)', fontsize=9.5, fontweight='bold')
    ax3.set_xlim(-0.05, 0.75); ax3.set_ylim(-0.05, 0.60)
    ax3.legend(fontsize=7.5, loc='upper left'); ax3.grid(alpha=0.3)
    ax3.set_facecolor('#FAFAFA')
    ax3.axhline(0.35, color=C_ORANGE, linestyle=':', alpha=0.5)
    ax3.axvline(0.40, color=C_ORANGE, linestyle=':', alpha=0.5)

    # Arrow axes
    ax_mid = fig.add_axes([0.44, 0.27, 0.12, 0.10])
    ax_mid.set_xlim(0,1); ax_mid.set_ylim(0,1); ax_mid.axis('off')
    ax_mid.annotate("", xy=(0.85,0.5), xytext=(0.15,0.5),
                    arrowprops=dict(arrowstyle='->', color=C_BLUE, lw=2.5))
    ax_mid.text(0.5, 0.12, "Fine-tuning", ha='center', fontsize=8.5,
                fontweight='bold', color=C_BLUE)

    # 2D scatter after
    ax4 = fig.add_axes([0.57, 0.13, 0.37, 0.38])
    for (s,e), c, l, m in zip(zip(after_s, after_e), cols_2d, lbls_2d, mks_2d):
        ax4.scatter(s, e, color=c, s=130, label=l, marker=m, zorder=5, edgecolor='white', linewidth=0.5)
    ax4.set_xlabel('speech_score', fontsize=9); ax4.set_ylabel('env_score', fontsize=9)
    ax4.set_title('2D Distribution  (After)', fontsize=9.5, fontweight='bold')
    ax4.set_xlim(-0.05, 0.85); ax4.set_ylim(-0.05, 0.55)
    ax4.legend(fontsize=7.5, loc='upper left'); ax4.grid(alpha=0.3)
    ax4.set_facecolor('#FAFAFA')
    ax4.axhline(0.35, color=C_ORANGE, linestyle=':', alpha=0.5)
    ax4.axvline(0.40, color=C_ORANGE, linestyle=':', alpha=0.5)
    ax4.text(0.70, 0.48, 'Better\n2D sep.', ha='center', fontsize=8, color=C_GREEN,
             bbox=dict(boxstyle='round', facecolor='#E9F7EF', alpha=0.85))

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ── PAGE 6: Monotonicity & Conclusion ──────────────────────
    fig = new_fig()
    fig.text(0.5, 0.96, "5. Conclusion & Implications",
             ha='center', fontsize=17, fontweight='bold', color=C_DARK)
    fig.text(0.5, 0.92, "Comprehensive evaluation of fine-tuning effects",
             ha='center', fontsize=10.5, color=C_GRAY, style='italic')
    footer(fig, 6)

    # Monotonicity comparison table
    ax_t = fig.add_axes([0.04, 0.64, 0.92, 0.25])
    ax_t.set_xlim(0,1); ax_t.set_ylim(0,1); ax_t.axis('off')
    ax_t.set_title("Monotonicity Comparison  (REAL -> MANIP -> FAKE direction consistency)",
                   fontsize=10.5, fontweight='bold', color=C_DARK, pad=6)

    t_cols = ['Feature', 'Before fine-tuning', 'Mono?', 'After fine-tuning', 'Mono?', 'Status']
    t_data = [
        ['[A] gate',       '0.204->0.784->0.934', 'OK', '0.204->0.784->0.934', 'OK', 'Maintained'],
        ['[B] speech',     '0.032->0.373->0.425', 'OK', '0.019->0.305->0.668', 'OK', 'Improved'],
        ['[C] env',        '0.172->0.424->0.472', 'OK', '0.320->0.264->0.461', 'BROKEN', 'Reversed'],
        ['[G] slope_diff', '(unstable)',           '--', '8.3->9.5->3.9',       '--', 'Unstable'],
        ['[H] MSC',        '0.054->0.084->0.072', 'OK', '0.054->0.084->0.072', 'OK', 'Maintained'],
        ['[I] xcorr',      '0.059->0.090->0.084', 'OK', '0.059->0.090->0.084', 'OK', 'Maintained'],
    ]
    t_col_xs = [0.01, 0.13, 0.38, 0.45, 0.70, 0.78]
    status_colors = {
        'Maintained': C_GRAY, 'Improved': C_GREEN, 'Reversed': C_RED, 'Unstable': C_ORANGE
    }

    hdr_bg = mpatches.FancyBboxPatch((0, 0.86), 1, 0.14, boxstyle="square,pad=0",
                                      facecolor=C_DARK, transform=ax_t.transAxes)
    ax_t.add_patch(hdr_bg)
    for cx, col in zip(t_col_xs, t_cols):
        ax_t.text(cx+0.01, 0.91, col, fontsize=8.5, color='white',
                  fontweight='bold', va='center')

    for i, row in enumerate(t_data):
        y = 0.78 - i * 0.135
        bg = '#F2F3F4' if i % 2 == 0 else 'white'
        ax_t.add_patch(mpatches.FancyBboxPatch((0, y-0.055), 1, 0.130,
                        boxstyle="square,pad=0", facecolor=bg, transform=ax_t.transAxes))
        for j, (cx, val) in enumerate(zip(t_col_xs, row)):
            c = status_colors.get(val, C_DARK) if j == 5 else \
                (C_RED if val == 'BROKEN' else (C_GREEN if val == 'OK' else C_DARK))
            fw = 'bold' if j in (0, 5) else 'normal'
            ax_t.text(cx+0.01, y, val, fontsize=8, color=c, va='center', fontweight=fw)

    # Three conclusion boxes
    box_data = [
        (C_GREEN, "Fine-tuning Succeeded",
         ["SI-SNR: +9.28 -> -22.54 dB",
          "  (31.8 dB total improvement)",
          "speech_score gap widened:",
          "  REAL=0.019 vs FAKE=0.668",
          "Energy Ratio Loss worked:",
          "  REAL env energy 0.172->0.320"]),
        (C_ORANGE, "Unexpected Side Effect",
         ["env_score monotonicity broken:",
          "  REAL(0.320) > MANIP(0.237)",
          "Root cause: better separation",
          "  -> REAL env stream also rises",
          "Separation quality improvement",
          "  != Discriminability improvement"]),
        (C_BLUE, "Next Steps",
         ["Retrain LightGBM with new",
          "  feature distribution",
          "2D space (speech x env) shows",
          "  better class separation",
          "env_score treated as reversed",
          "  feature automatically"]),
    ]
    for i, (color, title, items) in enumerate(box_data):
        ax_b = fig.add_axes([0.03 + i*0.325, 0.07, 0.295, 0.52])
        ax_b.set_xlim(0,1); ax_b.set_ylim(0,1); ax_b.axis('off')
        rounded_box(ax_b, 0.02, 0.02, 0.96, 0.96, color+'18', color, lw=1.8)
        ax_b.text(0.5, 0.90, title, ha='center', fontsize=10, fontweight='bold',
                  color=color, va='top')
        ax_b.axhline(0.82, color=color, linewidth=0.8, alpha=0.5,
                     xmin=0.05, xmax=0.95)
        for j, item in enumerate(items):
            ax_b.text(0.06, 0.75 - j*0.123, item, fontsize=8.5, color=C_DARK, va='top')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

print(f"Done: {OUTPUT}")
