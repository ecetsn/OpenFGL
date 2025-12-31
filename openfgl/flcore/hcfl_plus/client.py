import warnings
import torch
from openfgl.utils.privacy_utils import gaussian_eps, compose_eps
from openfgl.utils.secure_aggregation import mask_tensor
from openfgl.flcore.base import BaseClient
from openfgl.flcore.hcfl_plus.adapter import HCFLTaskAdapter
from openfgl.flcore.hcfl_plus.graph_adapter import HCFLGraphTaskAdapter
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
        if getattr(self.args, "task", "") == "graph_cls":
            self.task = HCFLGraphTaskAdapter(self.task, self.phi_indices, self.theta_indices, self.args)
            self._is_graph_task = True
        else:
            self.task = HCFLTaskAdapter(self.task, self.phi_indices, self.theta_indices, self.args)
            self._is_graph_task = False

        self._prev_phi_params = None
        self._warned_stale_params = False
        self.delta_theta_per_cluster = None
        self._dp_stats_logged = False
        self._dp_membership_logged = False
        self._dp_accounting_counts = {"stats": 0, "membership": 0, "sample": 0}
        self._dp_accounting_logged = False
        self._secure_agg_logged = False

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

        membership = self.cluster_weights_i
        if membership is not None and getattr(self.args, "hcfl_dp_membership", False):
            if getattr(self.args, "debug", False) and not self._dp_membership_logged:
                print(f"[hcfl_plus][client {self.client_id}] DP on membership enabled")
                self._dp_membership_logged = True
            membership = self._apply_dp_to_membership(membership)
            if getattr(self.args, "hcfl_dp_accounting", False):
                self._dp_accounting_counts["membership"] += 1

        secure_masks = None
        if getattr(self.args, "secure_agg_stats", False) and prototypes is not None and counts is not None:
            if getattr(self.args, "debug", False) and not self._secure_agg_logged:
                print(f"[hcfl_plus][client {self.client_id}] secure_agg_stats placeholder enabled")
                self._secure_agg_logged = True
            mask_scale = getattr(self.args, "secure_agg_mask_scale", 1.0)
            masked_pc, mask_pc = mask_tensor(prototypes["P_c"], mask_scale=mask_scale)
            masked_plf, mask_plf = mask_tensor(prototypes["P_lf"], mask_scale=mask_scale)
            masked_counts, mask_counts = mask_tensor(counts, mask_scale=mask_scale)
            prototypes = {"P_c": masked_pc, "P_lf": masked_plf}
            counts = masked_counts
            secure_masks = {"P_c": mask_pc, "P_lf": mask_plf, "counts": mask_counts}

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
            "membership": membership.cpu() if membership is not None else None,
            "prototypes": prototypes,
            "label_counts": counts,
            "secure_masks": secure_masks,
        }
        self._log_dp_accounting()

    def _compute_prototypes(self):
        if self._is_graph_task:
            return self._compute_graph_prototypes()
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

        if getattr(self.args, "hcfl_dp_stats", False):
            if getattr(self.args, "debug", False) and not self._dp_stats_logged:
                print(f"[hcfl_plus][client {self.client_id}] DP on prototypes/counts enabled")
                self._dp_stats_logged = True
            proto_c, proto_lf, counts = self._apply_dp_to_stats(proto_c, proto_lf, counts)
            if getattr(self.args, "hcfl_dp_accounting", False):
                self._dp_accounting_counts["stats"] += 1

        prototypes = {"P_c": proto_c.cpu(), "P_lf": proto_lf.cpu()}
        return prototypes, counts.cpu()

    def _compute_graph_prototypes(self):
        self.task.model.eval()
        train_loader = self.task.splitted_data["train_dataloader"]
        num_classes = self.task.num_global_classes

        proto_c = None
        proto_lf_sum = None
        counts = torch.zeros(num_classes, device=self.device, dtype=torch.float32)
        total = 0

        with torch.no_grad():
            for batch in train_loader:
                batch = batch.to(self.device)
                features, _ = self.task.model(batch)
                if features.dim() == 1:
                    features = features.unsqueeze(-1)
                labels = batch.y.long()
                if proto_c is None:
                    proto_c = torch.zeros(num_classes, features.size(1), device=self.device, dtype=features.dtype)
                    proto_lf_sum = torch.zeros(features.size(1), device=self.device, dtype=features.dtype)

                ones = torch.ones(labels.shape[0], device=self.device, dtype=features.dtype)
                proto_c.index_add_(0, labels, features)
                counts.index_add_(0, labels, ones)
                proto_lf_sum += features.sum(dim=0)
                total += labels.shape[0]

        if proto_c is None:
            return None, None

        proto_c = proto_c / counts.clamp_min(1.0).unsqueeze(1)
        proto_lf = proto_lf_sum / max(1, total)

        if getattr(self.args, "hcfl_dp_stats", False):
            if getattr(self.args, "debug", False) and not self._dp_stats_logged:
                print(f"[hcfl_plus][client {self.client_id}] DP on prototypes/counts enabled")
                self._dp_stats_logged = True
            proto_c, proto_lf, counts = self._apply_dp_to_stats(proto_c, proto_lf, counts)
            if getattr(self.args, "hcfl_dp_accounting", False):
                self._dp_accounting_counts["stats"] += 1

        prototypes = {"P_c": proto_c.cpu(), "P_lf": proto_lf.cpu()}
        return prototypes, counts.cpu()

    def _get_train_labels_and_mask(self):
        """Get training labels and mask for node-level tasks.

        Note: For graph classification tasks, labels are handled directly
        in _compute_graph_prototypes() and HCFLGraphTaskAdapter.train().
        """
        splitted = self.task.splitted_data
        if "merged_edge_label" in splitted:
            labels = splitted["merged_edge_label"]
            mask = splitted["merged_edge_train_mask"]
        else:
            labels = splitted["data"].y
            mask = splitted["train_mask"]
        return labels.clone().detach(), mask.clone().detach()

    def _apply_dp_to_stats(self, proto_c, proto_lf, counts):
        proto_clip = getattr(self.args, "hcfl_proto_clip", 0.0)
        proto_noise = getattr(self.args, "hcfl_proto_noise", 0.0)
        count_clip = getattr(self.args, "hcfl_count_clip", 0.0)
        count_noise = getattr(self.args, "hcfl_count_noise", 0.0)

        if proto_clip > 0:
            norms = proto_c.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            scale = torch.clamp(proto_clip / norms, max=1.0)
            proto_c = proto_c * scale
            lf_norm = proto_lf.norm(p=2).clamp_min(1e-12)
            lf_scale = min(1.0, proto_clip / float(lf_norm))
            proto_lf = proto_lf * lf_scale

        if proto_noise > 0:
            proto_c = proto_c + torch.randn_like(proto_c) * proto_noise
            proto_lf = proto_lf + torch.randn_like(proto_lf) * proto_noise

        if count_clip > 0:
            count_norm = counts.norm(p=2).clamp_min(1e-12)
            count_scale = min(1.0, count_clip / float(count_norm))
            counts = counts * count_scale

        if count_noise > 0:
            counts = counts + torch.randn_like(counts) * count_noise
            counts = counts.clamp_min(0.0)

        return proto_c, proto_lf, counts

    def _apply_dp_to_membership(self, membership):
        mem_clip = getattr(self.args, "hcfl_membership_clip", 0.0)
        mem_noise = getattr(self.args, "hcfl_membership_noise", 0.0)

        membership = membership.detach().clone()
        if mem_clip > 0:
            mem_norm = membership.norm(p=2).clamp_min(1e-12)
            mem_scale = min(1.0, mem_clip / float(mem_norm))
            membership = membership * mem_scale

        if mem_noise > 0:
            membership = membership + torch.randn_like(membership) * mem_noise

        membership = membership.clamp_min(0.0)
        total = membership.sum().clamp_min(1e-12)
        membership = membership / total
        return membership

    def _log_dp_accounting(self):
        if not getattr(self.args, "hcfl_dp_accounting", False):
            return
        if not getattr(self.args, "debug", False) or self._dp_accounting_logged:
            return
        delta = getattr(self.args, "dp_delta", 0.0)
        info = []

        if self._dp_accounting_counts["stats"] > 0:
            proto_clip = getattr(self.args, "hcfl_proto_clip", 0.0)
            proto_noise = getattr(self.args, "hcfl_proto_noise", 0.0)
            count_clip = getattr(self.args, "hcfl_count_clip", 0.0)
            count_noise = getattr(self.args, "hcfl_count_noise", 0.0)
            eps_proto = compose_eps(gaussian_eps(proto_clip, proto_noise, delta), self._dp_accounting_counts["stats"])
            eps_count = compose_eps(gaussian_eps(count_clip, count_noise, delta), self._dp_accounting_counts["stats"])
            if eps_proto is not None:
                info.append(f"eps_proto~{eps_proto:.4f}")
            if eps_count is not None:
                info.append(f"eps_count~{eps_count:.4f}")

        if self._dp_accounting_counts["membership"] > 0:
            mem_clip = getattr(self.args, "hcfl_membership_clip", 0.0)
            mem_noise = getattr(self.args, "hcfl_membership_noise", 0.0)
            eps_mem = compose_eps(gaussian_eps(mem_clip, mem_noise, delta), self._dp_accounting_counts["membership"])
            if eps_mem is not None:
                info.append(f"eps_membership~{eps_mem:.4f}")

        if self._dp_accounting_counts["sample"] > 0:
            sample_clip = getattr(self.args, "hcfl_sample_clip", 0.0)
            sample_noise = getattr(self.args, "hcfl_sample_noise", 0.0)
            eps_sample = compose_eps(gaussian_eps(sample_clip, sample_noise, delta), self._dp_accounting_counts["sample"])
            if eps_sample is not None:
                info.append(f"eps_sample~{eps_sample:.4f}")

        if info:
            print(f"[hcfl_plus][client {self.client_id}] DP accounting " + " ".join(info))
            self._dp_accounting_logged = True
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
