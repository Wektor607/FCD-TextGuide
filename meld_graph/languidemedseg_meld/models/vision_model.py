from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GraphNorm, SAGEConv

from meld_graph.icospheres import IcoSpheres
from meld_graph.paths import FEATURE_PATH
from utils.config import DATA_DIR, REPO_ROOT


# TODO: Conduct experiments with different GNN layers (GAT, GCN, etc.)
class ResidualBlock(nn.Module):
    # def __init__(self, dim, dropout=0.1, aggr="mean", layerscale=0.1):
    #     super().__init__()
    #     self.norm = nn.LayerNorm(dim)              # стабильнее GraphNorm
    #     self.conv = SAGEConv(dim, dim, aggr=aggr)  # попробуй aggr="max" для очагов
    #     self.act  = nn.GELU()
    #     self.drop = nn.Dropout(dropout)
    #     self.alpha = nn.Parameter(torch.tensor(layerscale))  # скейл резидуала

    # def forward(self, x, edge_index, batch=None):
    #     h = self.norm(x)                # pre-norm
    #     h = self.conv(h, edge_index)
    #     h = self.act(h)
    #     h = self.drop(h)
    #     return x + self.alpha * h       # без пост-нормы: сохраняем identity

    def __init__(self, dim, dropout=0.1):
        super().__init__()

        self.conv = SAGEConv(dim, dim, aggr="max")
        self.norm1 = GraphNorm(dim)  # BatchNorm doesn't work here
        self.norm2 = GraphNorm(dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        h = self.norm1(x, batch)  # <- give higher perfomance
        h = self.conv(h, edge_index)
        # residual connection
        h = x + h
        h = self.relu(h)
        # h = self.dropout(h) # <- test it
        h = self.norm2(h, batch)  # <- necessary second normalization
        return h

class VisionModel(nn.Module):
    def __init__(
        self,
        feature_dim: List[int],
        device: str | torch.device,
        gnn_min_verts: int = 642,
        fold_number: int = 0
    ) -> None:
        super().__init__()

        ico_path = Path("data/icospheres")

        self.fold_number = fold_number
        self.device = (
            torch.device(device) if not isinstance(device, torch.device) else device
        )

        self.icos = IcoSpheres(icosphere_path=str(ico_path))
        self._nverts_to_level = {
            len(self.icos.icospheres[level]["coords"]): level
            for level in self.icos.icospheres
        }
        # GNN layers for each of the first five stages
        self.gnn_min_verts = gnn_min_verts
        self.gnn_layers = nn.ModuleList(
            [
                ResidualBlock(feat_dim, dropout=0.1)  # MAKE HYPERPARAMETERS
                for feat_dim in feature_dim
            ]
        )

    def forward(self, subject_ids: List[str]) -> Dict[str, List[Batch]]:
        ref_subject = subject_ids[0]
        ref_npz = Path(FEATURE_PATH) / ref_subject / "features" / "feature_maps.npz"

        with np.load(ref_npz, allow_pickle=False) as ref_features:
            stage_keys = sorted(
                ref_features.files,
                key=lambda k: int(k.replace("stage", ""))
            )

        graph_list_per_stage: List[List[Data]] = [
            [] for _ in stage_keys
        ]
                
        for subject_id in subject_ids:
            # Step 1: ensure MELD features exist for this subject
            npz_path = Path(FEATURE_PATH) / subject_id / "features" / "feature_maps.npz"

            if not npz_path.is_file():
                raise FileNotFoundError(
                    f"feature_maps.npz not found for subject '{subject_id}': {npz_path}"
                )

            # Step 2: load subject’s NPZ
            with np.load(npz_path, allow_pickle=False) as features:
                # Step 3: build graphs for each used stage
                for i, stage in enumerate(stage_keys):
                    feat_torch = torch.from_numpy(features[stage])                    
                    feat = feat_torch[self.fold_number]  # shape = (H, N_i, C_i)
                    feat = feat.to(self.device)
                    H, N, C = feat.shape
                    feat_tensor = feat.view(H * N, C)

                    if N not in self._nverts_to_level:
                        raise ValueError(f"No icosphere level for N={N}")

                    level = self._nverts_to_level[N]
                    edge_lh = self.icos.icospheres[level]["t_edges"]

                    edge_rh = edge_lh.clone()
                    edge_rh[0] += N
                    edge_rh[1] += N
                    edge_index = torch.cat([edge_lh, edge_rh], dim=1)
                    
                    data = Data(x=feat_tensor, edge_index=edge_index, num_nodes=H * N)
                    graph_list_per_stage[i].append(data)

        # Batch each stage’s list of Data
        batched_per_stage: List[Batch] = []
        for i, data_list in enumerate(graph_list_per_stage):
            batch = Batch.from_data_list(data_list)
            batch = batch.to(self.device)

            V_total, _ = batch.x.size()
            N = V_total // (2 * batch.num_graphs)  # H = 2

            if N >= self.gnn_min_verts:
                batch.x = self.gnn_layers[i](batch.x, batch.edge_index, batch.batch)

            batched_per_stage.append(batch)

        return {"feature": batched_per_stage}
