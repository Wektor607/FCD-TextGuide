import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Tuple

import h5py
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.utils.data import Sampler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import meld_graph.mesh_tools as mt
from meld_graph.meld_cohort import MeldCohort
from meld_graph.paths import (BASE_PATH, DEFAULT_HDF5_FILE_ROOT, MELD_DATA_PATH,
                              MELD_PARAMS_PATH, SURFACE_PARTIAL)
from scripts.manage_results.plot_prediction_report import create_surface_plots
from utils.config import SUBJECTS_DIR
from utils.converter_mgh_to_nifti import (convert_prediction_mgh_to_nii,
                                          get_combat_feature_path, save_mgh)

SEED = 42

class LesionOversampleSampler(Sampler):
    """
        A sampler that takes ALL the healthy examples exactly once,
        and lesion examples are with replacement to fill the entire epoch.
    """

    def __init__(self, labels, seed=42):
        self.labels = labels
        random.seed(seed)
        
        self.hc_idx = [i for i, label in enumerate(labels) if label == 0]
        self.les_idx = [i for i, label in enumerate(labels) if label == 1]
        
        self.epoch_size = len(labels)

    def __iter__(self):
        
        idxs = self.hc_idx.copy()
        
        n_les_to_sample = self.epoch_size - len(idxs)
        
        idxs += random.choices(self.les_idx, k=n_les_to_sample)
        
        random.shuffle(idxs)
        return iter(idxs)

    def __len__(self):
        return self.epoch_size


def summarize_ci(scores, B=10_000, alpha=0.05, seed=42):
    x = np.asarray(scores, dtype=float)
    x = x[~np.isnan(x)]
    N = x.size
    if N == 0:
        return np.nan, np.nan, np.nan
    if N == 1:
        return float(x[0]), float(x[0]), float(x[0])

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, N, size=(B, N))  # 10k resamples
    boot_meds = np.median(x[idx], axis=1)  # median in each resample
    lo, hi = np.percentile(boot_meds, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return float(np.median(x)), float(lo), float(hi)

def threshold_surface_prediction(pred, percentile=99.5):
    pred = pred.copy()
    nz = pred[pred > 0]
    if nz.size == 0:
        return np.zeros_like(pred)
    thr = np.percentile(nz, percentile)
    pred[pred < thr] = 0
    return pred


def convert_preds_to_nifti(ckpt_path, 
                           subject_ids, 
                           probs_bin=None, 
                           gt_bin=None,
                           c=None, 
                           mode="test"):
    subjects_fs_dir = Path(MELD_DATA_PATH) / "input"
    predictions_output_root = Path(MELD_DATA_PATH) / "output" / "predictions_reports" / ckpt_path
    os.makedirs(predictions_output_root, exist_ok=True)

    results = {}

    gt_iter = gt_bin if gt_bin is not None else [None] * len(subject_ids)
    for (sid, pred, gt) in zip(subject_ids, probs_bin, gt_iter):
        # Convert prediction tensor → numpy
        predictions = pred.detach().cpu().numpy() if hasattr(pred, "detach") else np.asarray(pred)
        gt_cortex = gt.detach().cpu().numpy() if (gt is not None and hasattr(gt, "detach")) else (np.asarray(gt) if gt is not None else None)

        classifier_dir = subjects_fs_dir / sid / "xhemi" / "classifier"
        predictions_dir = predictions_output_root / sid / "predictions"
        os.makedirs(classifier_dir, exist_ok=True)
        os.makedirs(predictions_dir, exist_ok=True)

        # ============================================================
        # (1) Save each hemisphere prediction as MGH → NIfTI
        # ============================================================
        for idx, hemi in enumerate(["lh", "rh"]):

            pred_surface = np.zeros_like(c.cortex_mask, dtype=np.float32)
            pred_surface[c.cortex_mask] = predictions[idx]

            if gt_cortex is not None:
                gt_surface = np.zeros_like(c.cortex_mask, dtype=np.uint8)
                gt_surface[c.cortex_mask] = gt_cortex[idx]
            else:
                gt_surface = None

            combat_file = get_combat_feature_path(BASE_PATH, sid)

            with h5py.File(combat_file, "r") as f:
                key = ".combat.on_lh.thickness.sm3.mgh"
                if key not in f[hemi]:
                    raise KeyError(f"No dataset {key} in HDF5 for {hemi}")
                base_arr = f[hemi][key][:]

            # MGH template
            affine = nib.load(
                SUBJECTS_DIR / "fsaverage_sym" / "mri" / "T1.mgz"
            ).affine

            mgh_img = nib.MGHImage(base_arr[np.newaxis, :, np.newaxis], affine)

            out_mgh_pred = classifier_dir / f"{hemi}.prediction.mgh"
            save_mgh(out_mgh_pred, pred_surface, mgh_img)

            convert_prediction_mgh_to_nii(
                subjects_fs_dir,
                out_mgh_pred,
                hemi,
                predictions_dir,
                verbose=True,
            )

            surf_vis_path = predictions_dir / f"{hemi}_surface_visualisation.png"
            surf_pred = threshold_surface_prediction(pred_surface, percentile=99.5)

            volume_3d_visualisation(
                prediction_surf=surf_pred,
                hemi_name=hemi,
                save_path=surf_vis_path,
                gt_mask=gt_surface
            )
            print(f"✓ Saved surface visualisation: {surf_vis_path}")


        # ============================================================
        # (2) Combine LH + RH into final_nii
        # ============================================================
        lh_nii = predictions_dir / "lh.prediction.nii.gz"
        rh_nii = predictions_dir / "rh.prediction.nii.gz"
        final_nii = predictions_dir / f"prediction_{sid}.nii.gz"

        lh_p = lh_nii if lh_nii.exists() else None
        rh_p = rh_nii if rh_nii.exists() else None
        if lh_p and rh_p:
            lh_img = nib.load(str(lh_p))
            rh_img = nib.load(str(rh_p))
            combined = np.maximum(lh_img.get_fdata(), rh_img.get_fdata())
            
            nib.save(nib.Nifti1Image(combined, lh_img.affine, lh_img.header), str(final_nii))
            print(f"🎉 Final combined PRED NIfTI: {final_nii}")
        else:
            src = lh_p or rh_p
            if src:
                shutil.copy2(str(src), str(final_nii))
            else:
                raise FileNotFoundError("No hemi predictions found")

        # =============================================
        # (3) Combine LH + RH visualisations
        # =============================================

        lh_png = predictions_dir / "lh_surface_visualisation.png"
        rh_png = predictions_dir / "rh_surface_visualisation.png"
        combined_png = predictions_dir / f"{sid}_surface_combined.png"

        if lh_png.exists() and rh_png.exists():
            concat_side_by_side(lh_png, rh_png, combined_png)
        else:
            print(f"[WARN] Missing hemisphere PNGs for subject {sid}")

        results[sid] = final_nii   # <---- save result for this subject

    # ============================================================
    # RETURN ONLY AFTER PROCESSING ALL SUBJECTS
    # ============================================================
    return results

def compute_surface_boundary(mask, neighbours):
    """
    mask: (N,) binary GT mask
    neighbours: (N, K) vertex neighbours
    """
    boundary = np.zeros_like(mask, dtype=np.uint8)
    for v in np.where(mask > 0)[0]:
        if np.any(mask[neighbours[v]] == 0):
            boundary[v] = 1
    return boundary


def volume_3d_visualisation(prediction_surf, 
                            hemi_name, 
                            save_path,
                            gt_mask=None):
    """
    Creates MELD-style lateral + medial hemisphere render and saves as PNG.
    """
    if not hasattr(np, "float"):
        np.float = float
    if not hasattr(np, "int"):
        np.int = int
    if not hasattr(np, "bool"):
        np.bool = bool

    c = MeldCohort(hdf5_file_root=DEFAULT_HDF5_FILE_ROOT, dataset=None)
    surf = mt.load_mesh_geometry(os.path.join(MELD_PARAMS_PATH, SURFACE_PARTIAL))

    # Use MELD's native renderer (from plot_prediction_report)
    if gt_mask is not None:
        if torch.is_tensor(gt_mask):
            gt_mask = gt_mask.detach().cpu().numpy()

        gt_mask = (gt_mask > 0).astype(np.uint8)

    if gt_mask is not None:
        neighbours = np.load(
            os.path.join(MELD_DATA_PATH, "icospheres", "ico7.neighbours.npy"),
            allow_pickle=True
        )
        gt_boundary = compute_surface_boundary(gt_mask, neighbours)
    else:
        gt_boundary = None

    im_lat, im_med = create_surface_plots(
        surf,
        prediction=prediction_surf,
        c=c,
        boundary=gt_boundary,
    )

    fig = plt.figure(figsize=(10, 4))
    plt.suptitle(f"{hemi_name.upper()} hemisphere", fontsize=16)

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(im_lat)
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(im_med)
    ax2.axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

def concat_side_by_side(img1_path, img2_path, save_path):
    im1 = Image.open(img1_path)
    im2 = Image.open(img2_path)

    # align by height
    h = im1.height + im2.height
    w = max(im1.width, im2.width)

    new_im = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    new_im.paste(im1, (0, 0))
    new_im.paste(im2, (0, im1.height))

    new_im.save(save_path)
    print(f"✓ Saved combined: {save_path}")

def summarize_clusters(cluster_mask, hemi_names=["left", "right"]):
    summary = []
    for h, hemi in enumerate(hemi_names):
        hemi_mask = cluster_mask[h]  # binary mask for this hemisphere
        labels, num = ndimage.label(hemi_mask)  # find connected clusters
        for cluster_id in range(1, num + 1):
            coords = np.argwhere(labels == cluster_id)
            volume = coords.shape[0]  # number of voxels
            center = coords.mean(axis=0).astype(int).tolist()  # center of mass
            summary.append({
                "hemi": hemi,
                "volume_voxels": int(volume),
                "center": center
            })
    return summary

def get_device() -> Tuple[torch.device, bool]:
    # Probe CUDA availability but be defensive: some builds report CUDA available
    # but initializing CUDA fails if drivers are missing. Try a lightweight probe
    # and fall back to CPU on any exception.
    try:
        if torch.cuda.is_available():
            try:
                # This may raise if CUDA driver isn't present or not initialized
                _ = torch.cuda.current_device()
                device = torch.device("cuda")
            except Exception:
                # CUDA not usable at runtime — fall back to CPU
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")
    except Exception:
        device = torch.device("cpu")

    return device

def worker_init_fn(worker_id: int) -> None:
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)

def move_to_device(obj, device: torch.device):
    """Recursively move tensors in nested structures to device."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        seq = [move_to_device(v, device) for v in obj]
        return type(obj)(seq)
    return obj

def random_from_distribution(dist: dict):
    keys = list(dist.keys())
    probs = list(dist.values())
    return random.choices(keys, weights=probs, k=1)[0]

def generate_random_text(text_probs: json):
    """
    Generate: <hemisphere> + <lobe>
    Example: "Left Hemisphere; Temporal lobe"
    """

    hemi = random_from_distribution(text_probs.get("hemisphere_text", {}))
    lobe = random_from_distribution(text_probs.get("lobe_text", {}))

    return f"{hemi}; {lobe}"