import warnings
import torch
from openfgl.flcore.base import BaseClient
from openfgl.flcore.hcfl_plus.adapter import HCFLTaskAdapter
from openfgl.flcore.hcfl_plus.utils import infer_phi_theta_indices


class HCFLLUSClient(BaseClient):
    """
    Client for the HCFL-PLUS algorithm with phi/theta split and feature prototypes.
    """

    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super().__init__(args, client_id, data, data_dir, message_pool, device, personalized=True)
        self.current_cluster = 0
        self.cluster_weights_i = None

        self.phi_indices, self.theta_indices = infer_phi_theta_indices(self.task.model, self.args)
        if not self.theta_indices:
            raise ValueError("HCFL+ could not determine predictor head indices.")
        self.args.num_head_layers = len(self.theta_indices)
        self.task = HCFLTaskAdapter(self.task, self.phi_indices, self.theta_indices, self.args)

        self._prev_phi_params = None
        self._warned_stale_params = False
        self.delta_theta_per_cluster = None

    def execute(self):
        self.delta_theta_per_cluster = None
        server_payload = self.message_pool.get("server")
        if server_payload is None:
            self.task.train()
            return

        phi_weights = server_payload.get("phi_weights")
        theta_models = server_payload.get("theta_weights")
        cluster_label_dists = server_payload.get("cluster_label_dists")
        if cluster_label_dists is not None:
            cluster_label_dists = torch.tensor(cluster_label_dists, dtype=torch.float32, device=self.device)
        if phi_weights is None or theta_models is None:
            warnings.warn(
                "Server payload missing phi/theta weights. Running local train without synchronization.", RuntimeWarning
            )
            self.task.train()
            return

        membership = server_payload["client_membership"][self.client_id]
        self.cluster_weights_i = torch.as_tensor(membership, dtype=torch.float32, device=self.device)

        K = len(theta_models)
        if self.cluster_weights_i.numel() < K:
            pad = torch.zeros(K - self.cluster_weights_i.numel(), device=self.device, dtype=self.cluster_weights_i.dtype)
            self.cluster_weights_i = torch.cat([self.cluster_weights_i, pad], dim=0)
        elif self.cluster_weights_i.numel() > K:
            self.cluster_weights_i = self.cluster_weights_i[:K]
        all_params = list(self.task.model.parameters())

        with torch.no_grad():
            for local_param_idx, global_param in enumerate(phi_weights):
                param_idx = self.phi_indices[local_param_idx]
                all_params[param_idx].data.copy_(global_param.data)

        weighted_theta = [torch.zeros_like(p) for p in theta_models[0]]
        with torch.no_grad():
            for k in range(K):
                omega = self.cluster_weights_i[k]
                theta_k = theta_models[k]
                for pid, param in enumerate(weighted_theta):
                    param.data.add_(omega * theta_k[pid].to(param.device).data)

            for local_param_idx, global_param in enumerate(weighted_theta):
                param_idx = self.theta_indices[local_param_idx]
                all_params[param_idx].data.copy_(global_param.data)

        self._prev_phi_params = [all_params[i].detach().clone() for i in self.phi_indices]

        # Adapter needs client instance and raw cluster heads for EM updates.
        self.task.train(self, theta_models, cluster_label_dists=cluster_label_dists)

    def send_message(self):
        prototypes, counts = self._compute_prototypes()

        all_params = list(self.task.model.parameters())
        current_phi = [all_params[i] for i in self.phi_indices]
        if self._prev_phi_params is None:
            delta_phi = [torch.zeros_like(p) for p in current_phi]
            delta_theta_per_cluster = self._zero_theta_delta_payload()

            current_round = self.message_pool.get("round", 0)
            if current_round > 0 and not self._warned_stale_params:
                warnings.warn(
                    f"Client {self.client_id} sending zero deltas due to missing execute() state.", RuntimeWarning
                )
                self._warned_stale_params = True
        else:
            delta_phi = []
            with torch.no_grad():
                for new_p, old_p in zip(current_phi, self._prev_phi_params):
                    delta_phi.append(new_p.detach().clone() - old_p)
            delta_theta_per_cluster = self._prepare_theta_delta_payload()

        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "delta_phi": delta_phi,
            "delta_theta_per_cluster": delta_theta_per_cluster,
            "membership": self.cluster_weights_i.cpu() if self.cluster_weights_i is not None else None,
            "prototypes": prototypes,
            "label_counts": counts,
        }

    def _compute_prototypes(self):
        self.task.model.eval()
        data = self.task.splitted_data["data"].to(self.device)

        try:
            model_output = self.task.model(data)
            if isinstance(model_output, tuple) and len(model_output) >= 2:
                features = model_output[0].detach().to(torch.float32)
                logits = model_output[1].detach()
            else:
                logits = model_output.detach()
                warnings.warn(
                    "Model forward returned logits only; skipping feature-based prototypes.", RuntimeWarning
                )
                return None, None
        except Exception as exc:
            warnings.warn(f"HCFL+ prototype extraction failed: {exc}", RuntimeWarning)
            return None, None

        if logits.dim() == 1:
            logits = logits.unsqueeze(-1)

        labels, mask = self._get_train_labels_and_mask()
        labels = labels.to(features.device).long()
        mask = mask.to(features.device).bool()

        selected_features = features[mask]
        selected_labels = labels[mask]
        num_classes = self.task.num_global_classes

        proto_c = torch.zeros(num_classes, features.size(1), device=features.device, dtype=features.dtype)
        counts = torch.zeros(num_classes, device=features.device, dtype=features.dtype)

        if selected_features.numel() > 0:
            ones = torch.ones(selected_labels.shape[0], device=features.device, dtype=features.dtype)
            proto_c.index_add_(0, selected_labels, selected_features)
            counts.index_add_(0, selected_labels, ones)
            proto_c = proto_c / counts.clamp_min(1.0).unsqueeze(1)
            proto_lf = selected_features.mean(dim=0)
        else:
            proto_lf = torch.zeros(features.size(1), device=features.device, dtype=features.dtype)

        prototypes = {"P_c": proto_c.cpu(), "P_lf": proto_lf.cpu()}
        return prototypes, counts.cpu()

    def _get_train_labels_and_mask(self):
        splitted = self.task.splitted_data
        if "merged_edge_label" in splitted:
            labels = splitted["merged_edge_label"]
            mask = splitted["merged_edge_train_mask"]
        else:
            labels = splitted["data"].y
            mask = splitted["train_mask"]
        return labels.clone().detach(), mask.clone().detach()

    def _prepare_theta_delta_payload(self):
        if self.delta_theta_per_cluster is None:
            return self._zero_theta_delta_payload()

        payload = []
        for deltas in self.delta_theta_per_cluster:
            cluster_payload = [tensor.detach().clone().cpu() for tensor in deltas]
            payload.append(cluster_payload)

        if not payload:
            return self._zero_theta_delta_payload()

        expected_clusters = self.cluster_weights_i.numel() if self.cluster_weights_i is not None else len(payload)
        while len(payload) < expected_clusters:
            payload.append([tensor.detach().clone().zero_() for tensor in payload[0]])
        return payload

    def _zero_theta_delta_payload(self):
        all_params = list(self.task.model.parameters())
        theta_template = [all_params[i].detach().clone().zero_().cpu() for i in self.theta_indices]
        num_clusters = self.cluster_weights_i.numel() if self.cluster_weights_i is not None else getattr(self.args, "num_clusters", 1)
        return [[tensor.detach().clone() for tensor in theta_template] for _ in range(num_clusters)]
