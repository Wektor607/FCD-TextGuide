import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold
import pytorch_lightning as pl
import torch
import torch.multiprocessing
from engine.loss_meld import dice_coeff, tp_fp_fn_tn
from engine.wrapper import LanGuideMedSegWrapper
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

import utils.config as config
from meld_graph.meld_cohort import MeldCohort
from utils.data import EpilepDataset
from utils.utils import (get_device, move_to_device, summarize_ci,
                         worker_init_fn)
from utils.utils import convert_preds_to_nifti, summarize_ci

# Keep reproducibility settings at top
SEED = 42
pl.seed_everything(SEED, workers=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Multiprocessing strategy for dataloaders
torch.multiprocessing.set_sharing_strategy("file_system")

def get_cfg() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Language-guide Medical Image Segmentation"
    )
    parser.add_argument("--config", default="./config/training.yaml", type=str, help="config file")
    parser.add_argument("--meld_check", action="store_true", help="enable MELD test check mode")
    parser.add_argument("--ckpt_prefix", default=None, type=str, help="optional checkpoint prefix to load")
    parser.add_argument("--ckpt_path", default=None, type=str, help="comma-separated list of checkpoints for ensemble")

    cli = parser.parse_args()
    if cli.config is None:
        parser.error("--config is required")

    cfg = config.load_cfg_from_cfg_file(cli.config)
    cfg.meld_check = cli.meld_check
    cfg.ckpt_path = cli.ckpt_path
    cfg.ckpt_prefix = cli.ckpt_prefix
    return cfg

def prepare_dataloader(args, tokenizer, cohort, text_emb) -> DataLoader:
    df = pd.read_csv(args.split_path, sep=",")
    # test_ids = df[(df["split"] == "test") & (df["subject_id"].str.contains("FCD"))]["subject_id"].tolist()
    test_ids = df[(df["split"] == "test")]["subject_id"].tolist()
    print("CSV and SPLIT path: ", args.csv_path, args.split_path)
    ds_test = EpilepDataset(
        csv_path=args.csv_path,
        tokenizer=tokenizer,
        feature_path=args.feature_path,
        subject_ids=test_ids,
        cohort=cohort,
        max_length=args.max_len,
        text_emb=text_emb,
        atlas_type=args.atlas_type,
    )

    dl_test = DataLoader(
        ds_test,
        batch_size=args.valid_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
    )
    return dl_test


def load_ensemble_models(ckpt_prefix: str, args, eva, exp_flags, device: torch.device) -> List[torch.nn.Module]:
    ckpt_prefix = Path(ckpt_prefix)

    ckpt_paths = [
        ckpt_prefix.parent / f"{ckpt_prefix.name}_fold{i+1}.ckpt"
        for i in range(5)
    ]
    att_mechanism = False
    text_emb = False
    for exp, flags in exp_flags.items():
        if exp in (args.ckpt_prefix or ""):
            att_mechanism = flags.get("self_att_mechanism", False)
            text_emb = flags.get("text_emb", False)
            print(f"[INFO] Experiment '{exp}' flags: self_att_mechanism={att_mechanism}, text_emb={text_emb}")
            break
    # ######################## DELETE AFTER EXPERIMENT WITHOUT TEXT EMBEDDING
    # text_emb = False
    # print(f"[INFO] Experiment '{exp}' flags: self_att_mechanism={att_mechanism}, text_emb={text_emb}")
    ######################## DELETE AFTER EXPERIMENT WITHOUT TEXT EMBEDDING
    tokenizer = AutoTokenizer.from_pretrained(args.bert_type, trust_remote_code=True) if text_emb else None
    print(f"[INFO] Using ensemble of {len(ckpt_paths)} models:", ckpt_paths)
    models = []
    for i, ckpt_path in enumerate(ckpt_paths):
        model = LanGuideMedSegWrapper.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            args=args,
            eva=eva,
            fold_number=i,
            att_mechanism=att_mechanism,
            text_emb=text_emb,
            weights_only=False,
        )
        model.eval()
        model.to(device)
        models.append(model)

    return models, tokenizer


def run_ensemble_inference(dl_test: DataLoader, models: List[torch.nn.Module], eva, device: torch.device, cohort: MeldCohort = None, prefix: str = "") -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cortex_mask = torch.from_numpy(cohort.cortex_mask).to(device)
    all_subject_ids = []
    all_labels = []
    all_probs = []
    all_dist_maps = []
    all_pred_dists = []

    with torch.no_grad():
        for batch in tqdm(dl_test, desc="Ensemble inference"):
            subject_ids = batch["subject_id"]

            batch_on_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            y = batch_on_device["roi"]
            text = batch.get("text", batch_on_device.get("text"))

            B, H, V7 = y.shape
            y_mask = y[:, :, cortex_mask]
            target = y_mask.view(B, -1).long().view(B, H, -1)

            dist_maps = batch_on_device["dist_maps"].reshape(B, H, V7)
            dist_maps_cortex = dist_maps[:, :, cortex_mask].view(B, -1)

            # collect model predictions for this batch
            model_probs = []
            model_pred_dists = []
            for model in models:
                text_on_device = move_to_device(text, device)

                outputs = model([subject_ids, text_on_device])  # [B * H * V7, 2]
                # move logits to device early and reshape safely (reshape handles non-contiguous tensors)
                logp = outputs["log_softmax"].to(device)

                # infer the V7 dimension automatically using -1 and reorder to (B, 2, H, V7)
                logp = logp.reshape(B, H, V7, 2).permute(0, 3, 1, 2)

                # select cortex vertices and collapse dims to (B, 2, H * n_cortex)
                logp = logp[..., cortex_mask].reshape(B, 2, -1)

                # get positive-class probabilities and restore (B, H, n_cortex)
                probs = logp[:, 1, :].exp()
                pprobs = probs.view(B, H, -1).contiguous().detach()
                model_probs.append(pprobs)

                # distance head predictions
                if "non_lesion_logits" in outputs:
                    pred_dist = outputs["non_lesion_logits"].to(device)
                    pred_dist = pred_dist.reshape(B, H, V7)[:, :, cortex_mask].reshape(B, -1).detach()
                    model_pred_dists.append(pred_dist)

            probs_stack = torch.stack(model_probs, dim=0)
            probs_mean = probs_stack.mean(dim=0)

            if model_pred_dists:
                pred_dist_mean = torch.stack(model_pred_dists, dim=0).mean(dim=0)
            else:
                pred_dist_mean = torch.zeros(B, dist_maps_cortex.shape[-1], device=device)

            all_subject_ids.extend(subject_ids)
            all_labels.append(target.cpu())
            all_dist_maps.append(dist_maps_cortex.cpu())
            all_probs.append(probs_mean.detach())
            all_pred_dists.append(pred_dist_mean.cpu())

    all_labels = torch.cat(all_labels, dim=0)
    all_dist_maps = torch.cat(all_dist_maps, dim=0)
    all_preds = torch.cat(all_probs, dim=0)
    all_pred_dists = torch.cat(all_pred_dists, dim=0)

    return all_subject_ids, all_labels, all_dist_maps, all_preds, all_pred_dists


def find_optimal_threshold_on_val(
    args, models, eva, device, cohort, tokenizer, text_emb,
    aggregations=("mean", "median", "min"),
):
    """Ensemble inference on trainval to find optimal distance filter threshold.

    Uses full ensemble on trainval subjects, computes per-cluster distance stats,
    then sweeps thresholds using per-PATIENT sensitivity/specificity (not per-cluster).

    Returns dict: {aggregation: (best_threshold, best_j, sensitivity, specificity)}
    """
    df = pd.read_csv(args.split_path, sep=",")
    trainval_ids = df[df.split == "trainval"]["subject_id"].tolist()

    if len(trainval_ids) == 0:
        print("[WARN] No trainval subjects found in split CSV. Skipping threshold finding.")
        return {}

    print(f"\n[VAL] Running ensemble on {len(trainval_ids)} trainval subjects for threshold finding...")

    ds_val = EpilepDataset(
        csv_path=args.csv_path, tokenizer=tokenizer,
        feature_path=args.feature_path, subject_ids=trainval_ids,
        cohort=cohort, max_length=args.max_len,
        text_emb=text_emb, atlas_type=args.atlas_type,
    )
    dl_val = DataLoader(
        ds_val, batch_size=args.valid_batch_size, shuffle=False,
        num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn,
    )

    val_sids, val_labels, val_dist_maps, val_probs, val_pred_dists = \
        run_ensemble_inference(dl_val, models, eva, device, cohort, prefix="val")

    # Collect per-subject cluster stats
    subject_clusters = []  # list of (sid, is_control, [{label, agg_values}])
    for i in range(len(val_sids)):
        sid = val_sids[i]
        pred = val_probs[i]
        tgt = val_labels[i]
        dist_map_subj = val_dist_maps[i]
        is_control = ("FCD" not in sid) or (tgt.sum() == 0)

        pred_left = pred[0].detach().cpu().numpy().astype("float32")
        pred_right = pred[1].detach().cpu().numpy().astype("float32")
        pred_np = np.concatenate([pred_left, pred_right], axis=-1)

        mini = {sid: {"result": pred_np}}
        out = eva.threshold_and_cluster(data_dictionary=mini, save_prediction=False)
        probs_flat = out[sid]["cluster_thresholded"]
        if isinstance(probs_flat, torch.Tensor):
            probs_flat = probs_flat.detach().cpu().numpy()

        boundary_zone = dist_map_subj.cpu().numpy() if isinstance(dist_map_subj, torch.Tensor) else np.array(dist_map_subj)
        boundary_zone = boundary_zone < 20

        cluster_ids = np.unique(probs_flat)
        cluster_ids = cluster_ids[cluster_ids > 0]
        tp_set = set(np.unique(probs_flat[boundary_zone])[np.unique(probs_flat[boundary_zone]) > 0].tolist())

        pred_dist_subj = val_pred_dists[i].numpy()
        clusters = []
        for cid in cluster_ids:
            cmask = probs_flat == cid
            cdists = pred_dist_subj[cmask]
            label = "TP" if cid in tp_set else "FP"
            clusters.append({
                "label": label,
                "pred_dist_min": float(cdists.min()),
                "pred_dist_mean": float(cdists.mean()),
                "pred_dist_median": float(np.median(cdists)),
            })
        subject_clusters.append({"sid": sid, "is_control": is_control, "clusters": clusters})

    total_patients = sum(1 for s in subject_clusters if not s["is_control"])
    total_ctrl = sum(1 for s in subject_clusters if s["is_control"])
    print(f"[VAL] Patients: {total_patients}, Controls: {total_ctrl}")

    # Per-patient sweep: for each threshold, compute sensitivity & specificity
    thresholds = np.arange(-0.2, 1.01, 0.01)
    results = {}

    print(f"\n{'='*65}")
    print(f"OPTIMAL THRESHOLDS ON VALIDATION (per-patient Youden's J)")
    print(f"{'='*65}")
    print(f" {'Aggregation':>11} | {'Threshold':>9} | {'Sensitivity':>11} | {'Specificity':>11} | {'Youden J':>8}")
    print("-" * 65)

    for agg in aggregations:
        col = f"pred_dist_{agg}"
        best_j, best_thresh, best_sens, best_spec = -1, 0, 0, 0

        for thresh in thresholds:
            patients_with_tp = 0
            ctrl_with_no_fp = 0

            for s in subject_clusters:
                surviving = [c for c in s["clusters"] if c[col] < thresh]
                has_tp = any(c["label"] == "TP" for c in surviving)
                has_fp = any(c["label"] == "FP" for c in surviving)

                if not s["is_control"]:
                    if has_tp:
                        patients_with_tp += 1
                else:
                    if not has_fp:
                        ctrl_with_no_fp += 1

            sens = patients_with_tp / total_patients if total_patients > 0 else 0.0
            spec = ctrl_with_no_fp / total_ctrl if total_ctrl > 0 else 0.0
            j = sens + spec - 1.0

            if j > best_j:
                best_j, best_thresh, best_sens, best_spec = j, thresh, sens, spec

        results[agg] = (best_thresh, best_j, best_sens, best_spec)
        print(f" {agg:>11} | {best_thresh:>9.2f} | {best_sens:>10.1%} | {best_spec:>10.1%} | {best_j:>8.3f}")

    print(f"{'='*65}")
    return results


def postprocess_and_save(args, all_subject_ids, all_labels, all_dist_maps, all_preds, eva, ckpt_prefix: str, cohort: MeldCohort, dataset_type: str, dataset_name: str, text_emb: bool, all_pred_dists: torch.Tensor = None, val_threshold: float = None, val_aggregation: str = "mean"):
    dice_scores, iou_scores, ppv_scores, ppv_clusters_scores = [], [], [], []
    results = []
    cluster_dist_analysis = []  # for distance-based filtering analysis

    fp_control_clusters = []
    for i in range(all_labels.shape[0]):
        sid = all_subject_ids[i]
        pred = all_preds[i]
        tgt = all_labels[i]
        dist_map_subj = all_dist_maps[i]

        is_control = ("FCD" not in sid) or (tgt.sum() == 0)

        # pred_np = torch.cat([pred[0], pred[1]], dim=0).detach().cpu().numpy().astype("float32")
        # mini = {sid: {"result": pred_np}}
        pred_left = pred[0].detach().cpu().numpy().astype("float32")
        pred_right = pred[1].detach().cpu().numpy().astype("float32")

        # concatenate along vertex axis, not channel axis
        pred_np = np.concatenate([pred_left, pred_right], axis=-1)
        mini = {sid: {"result": pred_np}}

        out = eva.threshold_and_cluster(data_dictionary=mini, save_prediction=False)
        probs_flat = out[sid]["cluster_thresholded"]

        # --- Distance-based cluster filtering (threshold from validation) ---
        if val_threshold is not None and all_pred_dists is not None:
            agg_fn = {"mean": np.mean, "median": np.median, "min": np.min}[val_aggregation]
            pred_dist_subj_filter = all_pred_dists[i].numpy()
            probs_flat_np = probs_flat.detach().cpu().numpy() if isinstance(probs_flat, torch.Tensor) else probs_flat.copy()
            cluster_ids_filter = np.unique(probs_flat_np)
            cluster_ids_filter = cluster_ids_filter[cluster_ids_filter > 0]
            for cid in cluster_ids_filter:
                cmask = probs_flat_np == cid
                if agg_fn(pred_dist_subj_filter[cmask]) >= val_threshold:
                    probs_flat_np[cmask] = 0
            probs_flat = probs_flat_np

        boundary_zone = dist_map_subj < 20

        if isinstance(probs_flat, torch.Tensor):
            probs_flat_cpu = probs_flat.detach().cpu().numpy()
        else:
            probs_flat_cpu = probs_flat
        
        ###############################
        cluster_ids = probs_flat_cpu[probs_flat_cpu > 0].astype(np.int32)
        ids, counts = np.unique(cluster_ids, return_counts=True)
        # sort by cluster size descending
        order = np.argsort(-counts)
        ids, counts = ids[order], counts[order]

        max_cluster = int(counts[0]) if len(counts) else 0
        # log for controls to monitor FP clusters
        if is_control:
            print(sid, "n_clusters=", len(counts), "max_cluster=", max_cluster, "top_sizes=", counts[:5])
        ###############################
        # if "_C_" in sid:
        #     convert_preds_to_nifti(ckpt_prefix, [sid], [probs_flat_cpu.reshape(2, -1)], [tgt], cohort)

        boundary_zone_cpu = boundary_zone.detach().cpu().numpy() if isinstance(boundary_zone, torch.Tensor) else np.array(boundary_zone)


        difference = np.setdiff1d(np.unique(probs_flat_cpu), np.unique(probs_flat_cpu[boundary_zone_cpu]))
        difference = difference[difference > 0]
        n_fp_clusters = len(difference)
        correct_values = np.unique(probs_flat_cpu[boundary_zone_cpu])
        correct_values = correct_values[correct_values > 0]
        n_tp_clusters = len(correct_values)

        # --- Distance head analysis per cluster ---
        if all_pred_dists is not None:
            pred_dist_subj = all_pred_dists[i].numpy()  # [2*N_cortex], normalized (/ 300)
            all_cluster_ids = np.unique(probs_flat_cpu)
            all_cluster_ids = all_cluster_ids[all_cluster_ids > 0]
            tp_set = set(correct_values.tolist())
            fp_set = set(difference.tolist())

            for cid in all_cluster_ids:
                cmask = probs_flat_cpu == cid
                cluster_pred_dists = pred_dist_subj[cmask]
                label = "TP" if cid in tp_set else "FP"
                cluster_dist_analysis.append({
                    "subject_id": sid,
                    "cluster_id": int(cid),
                    "label": label,
                    "is_control": is_control,
                    "n_vertices": int(cmask.sum()),
                    "pred_dist_min": float(cluster_pred_dists.min()),
                    "pred_dist_mean": float(cluster_pred_dists.mean()),
                    "pred_dist_median": float(np.median(cluster_pred_dists)),
                    "pred_dist_max": float(cluster_pred_dists.max()),
                })

        gt_flat = tgt.reshape(-1)
        mask_np = (probs_flat_cpu > 0).astype(int)
        mask = torch.from_numpy(mask_np).long().to(gt_flat.device)
        labels = gt_flat.bool().long()

        dices = dice_coeff(torch.nn.functional.one_hot(mask, num_classes=2), labels)
        tp, fp, fn, tn = tp_fp_fn_tn(mask, labels)
        iou = tp / (tp + fp + fn + 1e-8)
        ppv = tp / (tp + fp + 1e-8)
        ppv_clusters = n_tp_clusters / (n_tp_clusters + n_fp_clusters + 1e-8)
        
        if not is_control:
            dice_scores.append(float(dices[1].detach().cpu()))
            ppv_scores.append(float(ppv))
            iou_scores.append(float(iou))
            ppv_clusters_scores.append(float(ppv_clusters))

            print(f"[{sid}] Dice lesional={dices[1]:.3f}, IoU={iou:.3f}, PPV={ppv:.3f}, TP={tp}, FP={fp}, FN={fn}, TN={tn}")
            results.append({
                "subject_id": sid,
                "number FP clusters": n_fp_clusters,
                "number TP clusters": n_tp_clusters,
                "dice": float(dices[1]),
                "iou": float(iou),
                "ppv_voxel": float(ppv),
                "ppv_clusters": float(ppv_clusters),
            })
        else:
            fp_control_clusters.append(n_fp_clusters)
            results.append({
                "subject_id": sid,
                "number FP clusters": n_fp_clusters,
                "number TP clusters": 0,
                "dice": None,
                "iou": None,
                "ppv_voxel": None,
                "ppv_clusters": None,
            })

    d_med, d_lo, d_hi = summarize_ci(dice_scores)
    p_med, p_lo, p_hi = summarize_ci(ppv_scores)
    i_med, i_lo, i_hi = summarize_ci(iou_scores)
    ppv_clusters_med, ppv_clusters_lo, ppv_clusters_hi = summarize_ci(ppv_clusters_scores)
    n_tp_clusters = sum(r["number TP clusters"] for r in results)
    n_fp_clusters = sum(r["number FP clusters"] for r in results)
    ppv_clusters = n_tp_clusters / (n_tp_clusters + n_fp_clusters + 1e-8)

    # Sensitivity
    tp_clusters_list = [r["number TP clusters"] for r in results if "FCD" in r["subject_id"]]
    total_tp = len(tp_clusters_list)
    found_tp = sum(1 for t in tp_clusters_list if t > 0)
    pct_tp = found_tp / total_tp if total_tp > 0 else 0.0

    # Specificity
    total_ctrl = len(fp_control_clusters)
    no_fp_ctrl = sum(1 for t in fp_control_clusters if t == 0)
    pct_spec = no_fp_ctrl / total_ctrl if total_ctrl > 0 else 0.0


    fp_fcd_clusters = [r["number FP clusters"] for r in results if "FCD" in r["subject_id"]]
    avg_fp_ctrl = float(np.mean(fp_control_clusters)) if fp_control_clusters else 0.0
    avg_fp_fcd  = float(np.mean(fp_fcd_clusters))     if fp_fcd_clusters     else 0.0

    print("\n=== ENSEMBLE OVERALL TEST METRICS ===")
    print(f"Dice : {d_med:.3f} (95% CI {d_lo:.3f}-{d_hi:.3f})")
    print(f"PPV_pixels  : {p_med:.3f} (95% CI {p_lo:.3f}-{p_hi:.3f})")
    print(f"PPV_clusters_mean  : {ppv_clusters:.3f}")
    print(f"PPV_clusters_median  : {ppv_clusters_med:.3f} (95% CI {ppv_clusters_lo:.3f}-{ppv_clusters_hi:.3f})")
    print(f"IoU  : {i_med:.3f} (95% CI {i_lo:.3f}-{i_hi:.3f})")
    print(f"Sensitivity (patients only): {found_tp} / {total_tp} FCDs ({pct_tp:.1%})")
    print(f"Specificity (controls only): {no_fp_ctrl} / {total_ctrl} scans with no FP ({pct_spec:.1%})")
    print(f"Avg FP clusters per control : {avg_fp_ctrl:.2f}")
    print(f"Avg FP clusters per FCD     : {avg_fp_fcd:.2f}")

    data_type = dataset_type.split("/")[-1].split("_")[0]
    data_name = dataset_name.split("/")[-1].split(".")[0]
    df = pd.DataFrame(results)
    if not text_emb:
        data_name = "no_text"
    
    df.to_csv(f"{ckpt_prefix}_{data_type}_{args.atlas_type}_{data_name}_{val_aggregation}_results.csv", index=False)

    # --- Distance head cluster analysis ---
    if cluster_dist_analysis:
        df_dist = pd.DataFrame(cluster_dist_analysis)
        dist_csv = f"{ckpt_prefix}_{data_type}_{data_name}_cluster_dist_analysis_{val_aggregation}.csv"
        df_dist.to_csv(dist_csv, index=False)
        print(f"\n[INFO] Saved cluster distance analysis to {dist_csv}")

        tp_rows = df_dist[df_dist["label"] == "TP"]
        fp_rows = df_dist[df_dist["label"] == "FP"]
        fp_ctrl = df_dist[(df_dist["label"] == "FP") & (df_dist["is_control"])]

        print("\n=== DISTANCE HEAD CLUSTER ANALYSIS ===")
        print(f"Total clusters: {len(df_dist)} (TP={len(tp_rows)}, FP={len(fp_rows)}, FP on controls={len(fp_ctrl)})")
        if len(tp_rows) > 0:
            print(f"  TP pred_dist_min:  mean={tp_rows['pred_dist_min'].mean():.4f}, median={tp_rows['pred_dist_min'].median():.4f}, "
                  f"std={tp_rows['pred_dist_min'].std():.4f}, range=[{tp_rows['pred_dist_min'].min():.4f}, {tp_rows['pred_dist_min'].max():.4f}]")
        if len(fp_rows) > 0:
            print(f"  FP pred_dist_min:  mean={fp_rows['pred_dist_min'].mean():.4f}, median={fp_rows['pred_dist_min'].median():.4f}, "
                  f"std={fp_rows['pred_dist_min'].std():.4f}, range=[{fp_rows['pred_dist_min'].min():.4f}, {fp_rows['pred_dist_min'].max():.4f}]")
        if len(fp_ctrl) > 0:
            print(f"  FP(ctrl) pred_dist_min: mean={fp_ctrl['pred_dist_min'].mean():.4f}, median={fp_ctrl['pred_dist_min'].median():.4f}")

        # --- Sweep distance filter thresholds ---
        print("\n=== DISTANCE FILTER SWEEP ===")
        print(f"{'threshold':>10} | {'TP_kept':>7} / {'TP_total':>8} | {'FP_kept':>7} / {'FP_total':>8} | {'FP_ctrl_kept':>12} / {'FP_ctrl_total':>13} | {'Sensitivity':>11} | {'Specificity':>11}")
        print("-" * 110)

        for thresh in [-0.1, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
            tp_kept = len(tp_rows[tp_rows["pred_dist_min"] < thresh]) if len(tp_rows) > 0 else 0
            fp_kept = len(fp_rows[fp_rows["pred_dist_min"] < thresh]) if len(fp_rows) > 0 else 0
            fp_ctrl_kept = len(fp_ctrl[fp_ctrl["pred_dist_min"] < thresh]) if len(fp_ctrl) > 0 else 0

            # Recompute sensitivity: how many FCD patients still have at least one TP cluster
            tp_surviving = tp_rows[tp_rows["pred_dist_min"] < thresh]
            patients_with_tp = tp_surviving["subject_id"].nunique() if len(tp_surviving) > 0 else 0
            sens = patients_with_tp / total_tp if total_tp > 0 else 0.0

            # Recompute specificity: controls with zero surviving FP clusters
            fp_ctrl_surviving = fp_ctrl[fp_ctrl["pred_dist_min"] < thresh]
            ctrl_with_fp = fp_ctrl_surviving["subject_id"].nunique() if len(fp_ctrl_surviving) > 0 else 0
            spec = (total_ctrl - ctrl_with_fp) / total_ctrl if total_ctrl > 0 else 0.0

            print(f"{thresh:>10.2f} | {tp_kept:>7} / {len(tp_rows):>8} | {fp_kept:>7} / {len(fp_rows):>8} | {fp_ctrl_kept:>12} / {len(fp_ctrl):>13} | {sens:>10.1%} | {spec:>10.1%}")


# def pixelwise_gating_sweep(
#     all_subject_ids, all_labels, all_dist_maps, all_preds, all_pred_dists,
#     eva, ckpt_prefix, cohort,
# ):
#     """Sweep pixel-wise gating parameters (threshold, temperature).

#     For each (thresh, temp) combination, applies gate = sigmoid(-(dist - thresh) / temp)
#     to raw segmentation probs BEFORE clustering, then computes sensitivity/specificity.
#     """
#     from scipy.special import expit  # sigmoid

#     thresholds = np.arange(0.1, 0.85, 0.05)
#     temperatures = [0.01, 0.05, 0.1, 0.2, 0.5]

#     n_subjects = all_labels.shape[0]

#     # Pre-compute per-subject data that doesn't change across sweep
#     subject_data = []
#     for i in range(n_subjects):
#         sid = all_subject_ids[i]
#         pred = all_preds[i]
#         tgt = all_labels[i]
#         dist_map_subj = all_dist_maps[i]
#         is_control = ("FCD" not in sid) or (tgt.sum() == 0)

#         pred_left = pred[0].detach().cpu().numpy().astype("float32")
#         pred_right = pred[1].detach().cpu().numpy().astype("float32")
#         pred_np = np.concatenate([pred_left, pred_right], axis=-1)
#         pred_dist = all_pred_dists[i].numpy()

#         boundary_zone = dist_map_subj.detach().cpu().numpy() if isinstance(dist_map_subj, torch.Tensor) else np.array(dist_map_subj)
#         boundary_zone = boundary_zone < 20

#         subject_data.append({
#             "sid": sid,
#             "pred_np": pred_np,
#             "pred_dist": pred_dist,
#             "boundary_zone": boundary_zone,
#             "is_control": is_control,
#         })

#     total_patients = sum(1 for s in subject_data if not s["is_control"])
#     total_ctrl = sum(1 for s in subject_data if s["is_control"])

#     print(f"\n{'='*80}")
#     print(f"PIXEL-WISE GATING SWEEP ({len(thresholds)} thresholds x {len(temperatures)} temperatures)")
#     print(f"Patients: {total_patients}, Controls: {total_ctrl}")
#     print(f"{'='*80}")

#     sweep_results = []

#     for temp in temperatures:
#         for thresh in thresholds:
#             gate_params = (thresh, temp)
#             patients_with_tp = 0
#             ctrl_with_no_fp = 0

#             for s in subject_data:
#                 gate = expit(-(s["pred_dist"] - thresh) / temp)
#                 gated_pred = s["pred_np"] * gate.astype("float32")

#                 mini = {s["sid"]: {"result": gated_pred}}
#                 out = eva.threshold_and_cluster(data_dictionary=mini, save_prediction=False)
#                 probs_flat = out[s["sid"]]["cluster_thresholded"]

#                 if isinstance(probs_flat, torch.Tensor):
#                     probs_flat = probs_flat.detach().cpu().numpy()

#                 bz = s["boundary_zone"]
#                 difference = np.setdiff1d(np.unique(probs_flat), np.unique(probs_flat[bz]))
#                 difference = difference[difference > 0]
#                 n_fp = len(difference)
#                 correct = np.unique(probs_flat[bz])
#                 correct = correct[correct > 0]
#                 n_tp = len(correct)

#                 if not s["is_control"]:
#                     if n_tp > 0:
#                         patients_with_tp += 1
#                 else:
#                     if n_fp == 0:
#                         ctrl_with_no_fp += 1

#             sens = patients_with_tp / total_patients if total_patients > 0 else 0.0
#             spec = ctrl_with_no_fp / total_ctrl if total_ctrl > 0 else 0.0
#             j_index = sens + spec - 1.0

#             sweep_results.append({
#                 "threshold": float(thresh),
#                 "temperature": float(temp),
#                 "sensitivity": sens,
#                 "specificity": spec,
#                 "youden_j": j_index,
#             })

#             print(f"  thresh={thresh:.2f}, temp={temp:.2f} | Sens={sens:.1%} | Spec={spec:.1%} | J={j_index:.3f}")

#     df_sweep = pd.DataFrame(sweep_results)
#     sweep_csv = f"{ckpt_prefix}_pixelwise_gating_sweep.csv"
#     df_sweep.to_csv(sweep_csv, index=False)
#     print(f"\n[INFO] Saved pixel-wise gating sweep to {sweep_csv}")

#     # Print best result per temperature
#     print("\n=== BEST PIXEL-WISE GATING PER TEMPERATURE ===")
#     print(f"{'temp':>6} | {'threshold':>9} | {'Sensitivity':>11} | {'Specificity':>11} | {'Youden J':>8}")
#     print("-" * 55)
#     for temp in temperatures:
#         df_temp = df_sweep[df_sweep["temperature"] == temp]
#         best = df_temp.loc[df_temp["youden_j"].idxmax()]
#         print(f"{best['temperature']:>6.2f} | {best['threshold']:>9.2f} | {best['sensitivity']:>10.1%} | {best['specificity']:>10.1%} | {best['youden_j']:>8.3f}")

#     # Overall best
#     best_overall = df_sweep.loc[df_sweep["youden_j"].idxmax()]
#     print(f"\nBEST OVERALL: thresh={best_overall['threshold']:.2f}, temp={best_overall['temperature']:.2f}, "
#           f"Sens={best_overall['sensitivity']:.1%}, Spec={best_overall['specificity']:.1%}, J={best_overall['youden_j']:.3f}")


def main():
    args = get_cfg()
    eva, cohort, exp_flags = config.inference_config()

    device = get_device()
    print("start testing on device:", device)

    models, tokenizer = load_ensemble_models(args.ckpt_prefix, args, eva, exp_flags, device)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # CHANGED from False on True
    text_emb = True # True
    print(f"[INFO] Text_emb={text_emb}")
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # --- Step 1: Find optimal threshold on validation (out-of-fold) ---
    # VAL_AGGREGATION = "mean"  # "mean", "median", or "min"
    # val_results = find_optimal_threshold_on_val(
    #     args, models, eva, device, cohort, tokenizer, text_emb,
    #     aggregations=(VAL_AGGREGATION, "median", "min"),
    # )
    # val_threshold = None
    # if val_results and VAL_AGGREGATION in val_results:
    #     val_threshold = val_results[VAL_AGGREGATION][0]
    #     print(f"\n[INFO] Using {VAL_AGGREGATION} threshold from validation: {val_threshold:.3f}")

    # --- Step 2: Run test with the threshold from validation ---
    dl_test = prepare_dataloader(args, tokenizer, cohort, text_emb)
    all_subject_ids, all_labels, all_dist_maps, all_probs, all_pred_dists = run_ensemble_inference(dl_test, models, eva, device, cohort, prefix=args.ckpt_prefix)

    VAL_AGGREGATION = "median"  # "mean", "median", or "min"
    val_threshold = 0.61 # median: 0.61, min: 0.22, mean: 0.63  
    postprocess_and_save(args, all_subject_ids, all_labels, all_dist_maps, all_probs, eva, args.ckpt_prefix, cohort, args.split_path, args.csv_path, text_emb, all_pred_dists=all_pred_dists, val_threshold=val_threshold, val_aggregation=VAL_AGGREGATION)
    print(f"{VAL_AGGREGATION} distance threshold applied: {val_threshold:.3f}")

if __name__ == "__main__":
    main()
