import os, numpy as np, SimpleITK as sitk
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

ROOT = os.path.expanduser("~/Repos/idh-mcnet")
import sys
pid = sys.argv[1] if len(sys.argv) > 1 else "UCSF-PDGM-XXXX"
OUT = os.path.join(ROOT, "error_analysis_20261101/arch_tiles_" + pid)
os.makedirs(OUT, exist_ok=True)
base = os.path.join(ROOT, "data/ucsf_core350", pid)

def load(p): return sitk.GetArrayFromImage(sitk.ReadImage(p)).astype(np.float32)

ref = load(f"{base}/{pid}_tumor_segmentation.nii.gz") > 0
pred = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(ROOT, "mcnet_eval/test60_preds", pid + ".nii.gz")))
pred_full = pred
zz = np.where(ref.any(axis=(1, 2)))[0]
# choose the in-tumor slice with the most mutant-class predicted voxels (display choice only)
best_z, best_n = int(zz[len(zz)//2]), -1
for cand in zz:
    n1 = int((pred_full[cand] == 1).sum())
    if n1 > best_n: best_z, best_n = int(cand), n1
z = best_z

def crop(a):
    g = a if a.ndim == 2 else a.mean(axis=2)
    fg = g > g.max()*0.02 if g.max() > 0 else np.zeros(g.shape, bool)
    ys, xs = np.where(fg)
    if not len(ys): return a
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    p = 6
    return a[max(0,y0-p):min(a.shape[0],y1+p), max(0,x0-p):min(a.shape[1],x1+p)]

def save_gray(img, name):
    v = img[img > 0]; lo, hi = np.percentile(v, [1, 99])
    plt.imsave(os.path.join(OUT, name), crop(img), cmap="gray", vmin=lo, vmax=hi)

for mod, fn in [("T1","T1"),("T1ce","T1c"),("T2","T2"),("FLAIR","FLAIR")]:
    save_gray(load(f"{base}/{pid}_{fn}.nii.gz")[z], f"seq_{mod}.png")

flair = load(f"{base}/{pid}_FLAIR.nii.gz")[z]
pz = pred[z]
fg = pz > 0
v = flair[flair > 0]; lo, hi = np.percentile(v, [1, 99])
rgb = plt.cm.gray(np.clip((flair - lo) / (hi - lo + 1e-9), 0, 1))[..., :3]
alpha = 0.55
rgb[pz == 2] = (1-alpha)*rgb[pz == 2] + alpha*np.array([0.15, 0.75, 0.25])   # wildtype green
rgb[pz == 1] = (1-alpha)*rgb[pz == 1] + alpha*np.array([0.90, 0.15, 0.15])   # mutant red
plt.imsave(os.path.join(OUT, "voxelmap.png"), crop(rgb))
n_mut, n_wt = int((pz==1).sum()), int((pz==2).sum())
print("slice_frac", n_mut/max(1,n_mut+n_wt), "mut_vox", n_mut, "wt_vox", n_wt)
print("TILES-DONE", os.listdir(OUT))
