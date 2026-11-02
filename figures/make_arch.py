import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyArrowPatch, Rectangle, Polygon, FancyBboxPatch
import numpy as np, os

D = "/tmp/v5/mutcase"
tiles = {m: mpimg.imread(os.path.join(D, f"seq_{m}.png")) for m in ["T1", "T1ce", "T2", "FLAIR"]}
voxel = mpimg.imread(os.path.join(D, "voxelmap.png"))
RED, GREEN, INK = "#E53935", "#2FA84F", "#222222"

W, H = 11.4, 5.0
fig = plt.figure(figsize=(W, H), dpi=220)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 114); ax.set_ylim(0, 50); ax.axis("off")

def arrow(x0, y0, x1, y1, lw=1.5, dashed=False):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                 lw=lw, color=INK, linestyle="--" if dashed else "-", shrinkA=1, shrinkB=1, zorder=6))

def img_tile(x, ytop, w, img, label=None):
    h = w * img.shape[0] / img.shape[1]
    ia = ax.inset_axes([x/114, (ytop - h)/50, w/114, h/50])
    ia.imshow(img); ia.axis("off")
    for s in ia.spines.values(): s.set_visible(True); s.set_color("#999999"); s.set_linewidth(0.7)
    if label: ax.text(x + w/2, ytop - h - 0.7, label, ha="center", va="top", fontsize=7.4, color=INK)
    return h

def cuboid(x, yc, w, h, d=1.2, fc="#DCE7F5", ec="#5B7BA8", lw=1.1):
    y = yc - h/2
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=lw, zorder=3))
    ax.add_patch(Polygon([(x+w, y), (x+w+d, y+d), (x+w+d, y+h+d), (x+w, y+h)], closed=True, facecolor="#C3D4EA", edgecolor=ec, lw=lw, zorder=3))
    ax.add_patch(Polygon([(x, y+h), (x+d, y+h+d), (x+w+d, y+h+d), (x+w, y+h)], closed=True, facecolor="#EAF1FA", edgecolor=ec, lw=lw, zorder=3))

def lane(seq_y_top, yc, title, badge_xy, badge_wh, badge_text, badge_fc, badge_ec, badge_tc):
    ax.text(0.5, title[1], title[0], fontsize=10.5, fontweight="bold", va="center")
    bb = FancyBboxPatch(badge_xy, badge_wh[0], badge_wh[1], boxstyle="round,pad=0.22", fc=badge_fc, ec=badge_ec, lw=1.1)
    ax.add_patch(bb)
    ax.text(badge_xy[0] + badge_wh[0]/2, badge_xy[1] + badge_wh[1]/2, badge_text, ha="center", va="center", fontsize=8, color=badge_tc)
    x = 0.5
    for m in ["T1", "T1ce", "T2", "FLAIR"]:
        img_tile(x, seq_y_top, 7.6, tiles[m], m)
        x += 8.6
    return x + 0.4

# ---------- Lane A ----------
nextx = lane(45.0, 37.0,
    ("(a)  Model A - mask-guided 2.5D classifier (EfficientNet-B0, 4.18M params)", 47.2),
    (76.5, 46.2), (22.5, 2.5), "needs a tumor mask from source data", "#FBEEEE", "#C07070", "#8A2E2E")
yA = 36.2
arrow(nextx, yA, nextx + 2.2, yA)
gx = nextx + 2.6
for i, (dx, dy) in enumerate([(0, 0), (1.0, 0.8), (2.0, 1.6)]):
    ax.add_patch(Rectangle((gx+dx, yA-2.1+dy), 4.6, 4.0, facecolor="#FFFFFF", edgecolor="#666666", lw=1.0, zorder=3+i))
ax.text(gx + 3.3, yA - 3.0, "2.5D sampling:\n3 planes x 3 sites x 3 slices\n= 9 views x 12 channels", ha="center", va="top", fontsize=7.0)
arrow(gx + 7.6, yA, gx + 9.8, yA)
ex = gx + 10.2
for w, h, lab in [(1.9, 5.6, "56"), (1.6, 4.4, "28"), (1.3, 3.2, "14"), (1.0, 2.2, "7")]:
    cuboid(ex, yA, w, h)
    ax.text(ex + w/2, yA - h/2 - 1.0, lab, ha="center", fontsize=6.8, color="#444444")
    ex += w + 2.1
ax.text(gx + 16.2, yA - 4.5, "shared pretrained\nEfficientNet-B0 encoder", ha="center", va="top", fontsize=6.8)
arrow(ex + 0.1, yA, ex + 2.2, yA)
axg = ex + 2.6
for i in range(6):
    ax.add_patch(Rectangle((axg, yA - 2.5 + i*0.9), 1.5, 0.6, fc="#9FC2E8" if i in (1, 4) else "#E3ECF7", ec="#5B7BA8", lw=0.7, zorder=3))
ax.text(axg + 0.75, yA - 3.5, "attention over\n9 views", ha="center", va="top", fontsize=6.8)
arrow(axg + 2.3, yA, axg + 5.9, yA)
px = axg + 6.2
ax.add_patch(Rectangle((px, yA - 0.75), 11.5, 1.5, fc="#FFF3D6", ec="#B08A3C", lw=1.2))
ax.plot([px + 11.5*0.81]*2, [yA - 1.15, yA + 1.15], color="#B08A3C", lw=1.6)
ax.text(px + 11.5*0.81, yA + 1.6, "threshold 0.81", fontsize=6.8, ha="center", color="#8A6A24")
ax.text(px, yA - 1.5, "0", fontsize=6.8, ha="center", va="top")
ax.text(px + 11.5, yA - 1.5, "1", fontsize=6.8, ha="center", va="top")
ax.text(px + 5.75, yA - 3.0, "P(IDH-mutant),\nvalidation-fitted operating point", ha="center", va="top", fontsize=6.8)

# ---------- Lane B ----------
nextx = lane(24.5, 16.5,
    ("(b)  Model B - self-segmenting multicontrast network (MC-net; nnU-Net v2, 2D)", 26.7),
    (76.5, 25.7), (22.5, 2.5), "no tumor mask needed at inference", "#EAF6EC", "#5E9A68", "#2A6B35")
yB = 15.7
arrow(nextx, yB, nextx + 2.2, yB)
ux = nextx + 2.6
enc = [(1.9, 6.2), (1.6, 4.4), (1.3, 2.8)]
pos = []
xs = ux
for w, h in enc:
    cuboid(xs, yB, w, h, fc="#DFF0E4", ec="#4E8460")
    pos.append((xs, w, h)); xs += w + 1.5
cuboid(xs, yB, 1.1, 1.7, fc="#C9E4D2", ec="#4E8460"); xs += 1.1 + 1.5
dpos = []
for w, h in reversed(enc):
    cuboid(xs, yB, w, h, fc="#EAF6EE", ec="#4E8460")
    dpos.append((xs, w, h)); xs += w + 1.5
for (ex0, ew, eh), (dx0, dw, dh) in zip(pos, dpos):
    arrow(ex0 + ew/2 + 0.5, yB + eh/2 + 0.9, dx0 + dw/2 + 0.5, yB + dh/2 + 0.9, lw=0.9, dashed=True)
ax.text((ux + xs - 1.5)/2, yB - 4.0, "auto-configured U-Net (schematic),\ndashed = skip connections", ha="center", va="top", fontsize=7.0)
arrow(xs + 0.1, yB, xs + 2.2, yB)
vx = xs + 2.6
img_tile(vx, yB + 6.4, 8.8, voxel)
ax.text(vx + 4.4, yB - 6.9, "actual dual-class voxel output\n(this case; one slice shown)", ha="center", va="top", fontsize=6.8)
arrow(vx + 9.3, yB, vx + 12.2, yB)
lx = vx + 12.6
ax.add_patch(plt.Circle((lx + 1.9, yB + 0.8), 1.8, fc="#F2D8D8", ec=RED, lw=1.4))
ax.add_patch(plt.Circle((lx + 4.9, yB - 1.0), 0.75, fc="#DDDDDD", ec="#999999", lw=1.1))
ax.plot([lx + 4.3, lx + 5.5], [yB - 0.35, yB - 1.65], color="#999999", lw=1.3)
ax.text(lx + 3.0, yB - 4.1, "largest 3D connected\ncomponent kept", ha="center", va="top", fontsize=6.8)
arrow(lx + 6.4, yB, lx + 8.6, yB)
bx = lx + 9.0
ax.add_patch(Rectangle((bx, yB - 0.75), 10.5, 1.5, fc="#F4F4F4", ec="#666666", lw=1.1))
ax.add_patch(Rectangle((bx, yB - 0.75), 10.5*0.587, 1.5, fc=RED, ec="none"))
ax.text(bx, yB - 1.5, "0", fontsize=6.8, ha="center", va="top")
ax.text(bx + 10.5, yB - 1.5, "1", fontsize=6.8, ha="center", va="top")
ax.text(bx + 5.25, yB + 1.6, "mutant voxel fraction 0.59", fontsize=7.0, ha="center", color=RED)
ax.text(bx + 5.25, yB - 2.6, "majority vote -> IDH-mutant\n(this case: correct)", ha="center", va="top", fontsize=7.0, fontweight="bold")

# ---------- legend (bottom strip) ----------
ly = 3.2
ax.add_patch(Rectangle((0.5, ly - 2.6), 52, 4.4, fc="#FAFAFA", ec="#AAAAAA", lw=0.9))
ax.add_patch(Rectangle((1.5, ly - 0.4), 2.0, 1.2, fc=RED, ec="none")); ax.text(4.1, ly + 0.2, "mutant voxels", fontsize=7.4, va="center")
ax.add_patch(Rectangle((13.5, ly - 0.4), 2.0, 1.2, fc=GREEN, ec="none")); ax.text(16.1, ly + 0.2, "wildtype voxels", fontsize=7.4, va="center")
ax.text(28.0, ly + 0.2, "MRI: one real UCSF-PDGM case (IDH-mutant)", fontsize=7.4, va="center")
ax.plot([1.5, 3.5], [ly - 1.8]*2, color=INK, lw=1.5); ax.text(4.1, ly - 1.8, "forward", fontsize=7.4, va="center")
ax.plot([13.5, 15.5], [ly - 1.8]*2, color=INK, lw=1.2, linestyle="--"); ax.text(16.1, ly - 1.8, "skip connection", fontsize=7.4, va="center")

fig.savefig("/tmp/v5/fig_arch_v5.pdf")
fig.savefig("/tmp/v5/fig_arch_v5_preview.png", dpi=110)
print("ARCH-V2-DONE")
