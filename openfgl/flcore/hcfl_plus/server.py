import torch
from openfgl.flcore.base import BaseServer
from openfgl.flcore.hcfl_plus.metrics import get_distance_metric
from openfgl.flcore.hcfl_plus.utils import infer_phi_theta_indices
from openfgl.utils.secure_aggregation import unmask_tensor


class HCFLLUSServer(BaseServer):
    """
    Server for the HCFL-PLUS algorithm, managing the phi/theta split and adaptive soft clustering.
    """

    def __init__(self, args, global_data, data_dir, message_pool, device):
        super().__init__(args, global_data, data_dir, message_pool, device, personalized=True)
        self.cluster_assignments = [0 for _ in range(self.args.num_clients)]
        self.client_membership = [torch.tensor([1.0], dtype=torch.float32) for _ in range(self.args.num_clients)]

        # --- HCFL+ Model Split: Dynamic Index Calculation (Tier 1) ---
        all_params = list(self.task.model.parameters())
        self.phi_indices, self.theta_indices = infer_phi_theta_indices(self.task.model, self.args)
        if not self.theta_indices:
            raise ValueError("HCFL+ could not determine predictor head indices on server.")
        self.args.num_head_layers = len(self.theta_indices)
        self.feature_extractor = [all_params[i].detach().clone() for i in self.phi_indices]
        initial_predictor_params = [all_params[i].detach().clone() for i in self.theta_indices]

        initial_k = getattr(self.args, "num_clusters", 1)
        self.cluster_weights = [
            self._clone_param_list(initial_predictor_params) for _ in range(initial_k)
        ]

        self.prototype_cache = {cid: None for cid in range(self.args.num_clients)}
        self.mean_feature_cache = {cid: None for cid in range(self.args.num_clients)}
        self.label_hist_cache = {cid: None for cid in range(self.args.num_clients)}

        self.split_threshold = getattr(self.args, "hcfl_split_tol", 0.3)
        self.merge_threshold = getattr(self.args, "hcfl_merge_tol", 0.15)
        self.min_cluster_size = getattr(self.args, "hcfl_min_cluster_size", 2)
        self.warmup_rounds = getattr(self.args, "hcfl_warmup_rounds", 5)
        self.max_clusters = getattr(self.args, "hcfl_max_clusters", 8)
        self.prototype_momentum = getattr(self.args, "hcfl_proto_momentum", 0.5)
        self.distance_metric = get_distance_metric(self.args)
        self.num_global_classes = self.task.num_global_classes

    def execute(self):
        sampled_clients = self.message_pool["sampled_clients"]

        for cid in sampled_clients:
            msg = self.message_pool[f"client_{cid}"]
            if "membership" in msg:
                m = msg["membership"]
                if not isinstance(m, torch.Tensor):
                    m = torch.tensor(m, dtype=torch.float32)

                K = len(self.cluster_weights)
                if m.numel() < K:
                    pad = torch.zeros(K - m.numel(), dtype=torch.float32)
                    m = torch.cat([m, pad], dim=0)
                elif m.numel() > K:
                    m = m[:K]

                self.client_membership[cid] = m

            if getattr(self.args, "debug", False):
                proto_info = msg.get("prototypes")
                counts_info = msg.get("label_counts")
                if proto_info is not None and counts_info is not None:
                    self._log_dp_stats(cid, proto_info, counts_info)

        for cid in range(self.args.num_clients):
            membership = self.client_membership[cid]
            if isinstance(membership, torch.Tensor) and membership.numel() > 0:
                self.cluster_assignments[cid] = int(membership.argmax().item())

        self._aggregate_feature_extractor(sampled_clients)
        for cluster_id in range(len(self.cluster_weights)):
            self.cluster_weights[cluster_id] = self._aggregate_cluster(cluster_id, sampled_clients)

        for cid in sampled_clients:
            self._update_statistics(cid, self.message_pool[f"client_{cid}"])

        self._maybe_split_clusters()
        self._maybe_remove_clusters()
        self._maybe_merge_clusters()
        self._synchronize_server_model()

        if getattr(self.args, "debug", False):
            self._log_membership_stats(sampled_clients)

    def _aggregate_feature_extractor(self, client_ids):
        total_samples = sum(self.message_pool[f"client_{cid}"]["num_samples"] for cid in client_ids)
        if total_samples == 0:
            return

        new_phi = [p.detach().clone() for p in self.feature_extractor]
        eta_g = getattr(self.args, "hcfl_global_lr", 1.0)

        for cid in client_ids:
            msg = self.message_pool[f"client_{cid}"]
            deltas_phi = msg.get("delta_phi")
            if deltas_phi is None:
                continue

            n_i = msg["num_samples"]
            coeff = n_i / total_samples

            for pid, param in enumerate(new_phi):
                param.data += eta_g * coeff * deltas_phi[pid].data

        self.feature_extractor = new_phi

    def send_message(self):
        phi_payload = [param.detach().clone() for param in self.feature_extractor]
        theta_payload = [
            [param.detach().clone() for param in weights] for weights in self.cluster_weights
        ]

        membership_payload = [m.clone().cpu().tolist() for m in self.client_membership]

        num_clusters = len(self.cluster_weights)
        num_classes = self.num_global_classes
        cluster_label_dists = torch.zeros(num_clusters, num_classes, dtype=torch.float32)
        for cid, counts in self.label_hist_cache.items():
            if counts is None:
                continue
            cluster_id = self.cluster_assignments[cid]
            if cluster_id < num_clusters:
                cluster_label_dists[cluster_id] += counts.detach().clone().cpu()
        cluster_label_dists = cluster_label_dists / (cluster_label_dists.sum(dim=1, keepdim=True) + 1e-9)
        cluster_label_payload = cluster_label_dists.tolist()

        self.message_pool["server"] = {
            "phi_weights": phi_payload,
            "theta_weights": theta_payload,
            "client_membership": membership_payload,
            "cluster_assignments": self.cluster_assignments.copy(),
            "cluster_label_dists": cluster_label_payload,
        }

    def _aggregate_cluster(self, cluster_id, client_ids):
        total_weight = 0.0
        for cid in client_ids:
            msg = self.message_pool[f"client_{cid}"]
            n_i = msg["num_samples"]
            w_i_k = float(self.client_membership[cid][cluster_id])
            if w_i_k <= 0:
                continue
            total_weight += n_i * w_i_k

        if total_weight == 0:
            return [p.detach().clone() for p in self.cluster_weights[cluster_id]]

        new_params = [p.detach().clone() for p in self.cluster_weights[cluster_id]]
        eta_g = getattr(self.args, "hcfl_global_lr", 1.0)

        for cid in client_ids:
            msg = self.message_pool[f"client_{cid}"]
            delta_theta_per_cluster = msg.get("delta_theta_per_cluster")
            if delta_theta_per_cluster is None or len(delta_theta_per_cluster) <= cluster_id:
                continue
            deltas_theta_weighted = delta_theta_per_cluster[cluster_id]

            n_i = msg["num_samples"]
            w_i_k = float(self.client_membership[cid][cluster_id])
            if w_i_k <= 0:
                continue

            coeff = (n_i * w_i_k) / total_weight
            for pid, param in enumerate(new_params):
                delta_param = deltas_theta_weighted[pid].to(param.device)
                param.data += eta_g * coeff * delta_param.data

        return [p.detach().clone() for p in new_params]

    def _synchronize_server_model(self):
        cluster_map = self._cluster_to_clients()
        total = sum(len(clients) for clients in cluster_map.values())
        if total == 0:
            return

        with torch.no_grad():
            all_server_params = list(self.task.model.parameters())

            for i, phi_param in enumerate(self.feature_extractor):
                all_server_params[self.phi_indices[i]].data.copy_(phi_param.data)

            avg_theta = [torch.zeros_like(p) for p in self.cluster_weights[0]]
            for cluster_id, theta_k in enumerate(self.cluster_weights):
                num_clients_in_cluster = len(cluster_map.get(cluster_id, []))
                if num_clients_in_cluster == 0:
                    continue

                weight = num_clients_in_cluster / total
                for pid, param in enumerate(avg_theta):
                    param.data += weight * theta_k[pid].data

            for i, theta_param in enumerate(avg_theta):
                all_server_params[self.theta_indices[i]].data.copy_(theta_param.data)

    def _update_statistics(self, client_id, message):
        prototypes_dict = message.get("prototypes")
        counts = message.get("label_counts")
        if getattr(self.args, "secure_agg_stats", False):
            prototypes_dict, counts = self._unmask_stats(client_id, prototypes_dict, counts, message.get("secure_masks"))
        if (
            prototypes_dict is None
            or counts is None
            or "P_c" not in prototypes_dict
            or "P_lf" not in prototypes_dict
        ):
            return

        proto_c = prototypes_dict["P_c"].to(torch.float32)
        proto_lf = prototypes_dict["P_lf"].to(torch.float32)
        counts = counts.to(torch.float32)

        cached_proto = self.prototype_cache.get(client_id)
        if cached_proto is None:
            self.prototype_cache[client_id] = proto_c
        else:
            self.prototype_cache[client_id] = (
                self.prototype_momentum * cached_proto + (1 - self.prototype_momentum) * proto_c
            )

        cached_mean_feature = self.mean_feature_cache.get(client_id)
        if cached_mean_feature is None:
            self.mean_feature_cache[client_id] = proto_lf
        else:
            self.mean_feature_cache[client_id] = (
                self.prototype_momentum * cached_mean_feature + (1 - self.prototype_momentum) * proto_lf
            )

        cached_counts = self.label_hist_cache.get(client_id)
        if cached_counts is None:
            self.label_hist_cache[client_id] = counts
        else:
            self.label_hist_cache[client_id] = (
                self.prototype_momentum * cached_counts + (1 - self.prototype_momentum) * counts
            )

    def _unmask_stats(self, client_id, prototypes_dict, counts, masks):
        if prototypes_dict is None or counts is None or masks is None:
            return prototypes_dict, counts
        try:
            proto_c = unmask_tensor(prototypes_dict.get("P_c"), masks.get("P_c"))
            proto_lf = unmask_tensor(prototypes_dict.get("P_lf"), masks.get("P_lf"))
            counts = unmask_tensor(counts, masks.get("counts"))
            return {"P_c": proto_c, "P_lf": proto_lf}, counts
        except Exception:
            if getattr(self.args, "debug", False):
                print(f"[hcfl_plus][server] secure_agg_stats unmask failed for client {client_id}")
            return prototypes_dict, counts

    def _log_dp_stats(self, client_id, proto_info, counts_info):
        proto_c = proto_info.get("P_c") if isinstance(proto_info, dict) else None
        proto_lf = proto_info.get("P_lf") if isinstance(proto_info, dict) else None
        counts = counts_info
        if isinstance(counts, torch.Tensor):
            counts = counts.to(torch.float32)
        else:
            counts = torch.tensor(counts, dtype=torch.float32)

        info = []
        if proto_c is not None:
            proto_c = proto_c.to(torch.float32)
            info.append(f"P_c_norm={proto_c.norm(p=2).item():.4f}")
        if proto_lf is not None:
            proto_lf = proto_lf.to(torch.float32)
            info.append(f"P_lf_norm={proto_lf.norm(p=2).item():.4f}")
        if counts is not None:
            info.append(f"counts_sum={counts.sum().item():.4f}")

        if info:
            print(f"[hcfl_plus][server] stats recv client={client_id} " + " ".join(info))

    def _log_membership_stats(self, sampled_clients):
        if not sampled_clients:
            return
        mems = []
        for cid in sampled_clients:
            m = self.client_membership[cid]
            if not isinstance(m, torch.Tensor):
                m = torch.tensor(m, dtype=torch.float32)
            mems.append(m)
        if not mems:
            return
        mems = torch.stack(mems, dim=0)
        avg_entropy = (-mems * (mems.clamp_min(1e-12).log())).sum(dim=1).mean().item()
        avg_max = mems.max(dim=1).values.mean().item()
        print(f"[hcfl_plus][server] membership avg_entropy={avg_entropy:.4f} avg_max={avg_max:.4f}")

    def _client_representation(self, client_id):
        proto_c = self.prototype_cache.get(client_id)
        proto_lf = self.mean_feature_cache.get(client_id)
        if proto_c is None or proto_lf is None:
            return None

        norm_lf = proto_lf.norm(p=2)
        if norm_lf.item() < 1e-12:
            return None

        msg = self.message_pool.get(f"client_{client_id}")
        deltas = None
        if msg is not None:
            delta_theta = msg.get("delta_theta_per_cluster")
            if delta_theta is not None:
                main_cluster = self.cluster_assignments[client_id]
                if main_cluster < len(delta_theta):
                    deltas = [tensor.detach().clone().to(self.device) for tensor in delta_theta[main_cluster]]

        return {
            "P_c": proto_c,
            "P_lf": proto_lf / norm_lf,
            "membership": self.client_membership[client_id].to(self.device),
            "grads": deltas,
        }

    def _calculate_fine_grained_distance(self, rep_a, rep_b, cluster_id):
        if self.distance_metric is None:
            self.distance_metric = get_distance_metric(self.args)
        return self.distance_metric.compute(rep_a, rep_b, cluster_id)

    def _maybe_split_clusters(self):
        if self.message_pool["round"] < self.warmup_rounds:
            return

        cluster_map = self._cluster_to_clients()
        best_spread_overall = -1.0
        cluster_to_split = None
        anchor_pair_to_split = None

        for cluster_id, client_ids in cluster_map.items():
            if len(client_ids) < self.min_cluster_size * 2:
                continue

            representations = {cid: self._client_representation(cid) for cid in client_ids}
            valid_clients = [cid for cid, rep in representations.items() if rep is not None]
            if len(valid_clients) < self.min_cluster_size * 2:
                continue

            anchor_pair, max_dist_k, mean_dist_k = self._find_farthest_pair_fine_grained(
                valid_clients, representations, cluster_id
            )
            if anchor_pair is None:
                continue

            spread_k = max_dist_k - mean_dist_k
            if spread_k > best_spread_overall:
                best_spread_overall = spread_k
                cluster_to_split = cluster_id
                anchor_pair_to_split = anchor_pair

        if cluster_to_split is None or best_spread_overall < self.split_threshold:
            return

        if anchor_pair_to_split is None:
            return

        client_ids = cluster_map[cluster_to_split]
        representations = {cid: self._client_representation(cid) for cid in client_ids}
        left_anchor, right_anchor = anchor_pair_to_split

        left_rep = representations[left_anchor]
        right_rep = representations[right_anchor]
        left_group, right_group = [], []

        for cid in client_ids:
            rep = representations.get(cid)
            if rep is None:
                if self.cluster_assignments[cid] == cluster_to_split:
                    if len(left_group) <= len(right_group):
                        left_group.append(cid)
                    else:
                        right_group.append(cid)
                continue

            dist_left = self._calculate_fine_grained_distance(rep, left_rep, cluster_to_split)
            dist_right = self._calculate_fine_grained_distance(rep, right_rep, cluster_to_split)

            if dist_left <= dist_right:
                left_group.append(cid)
            else:
                right_group.append(cid)

        if len(left_group) < self.min_cluster_size or len(right_group) < self.min_cluster_size:
            return
        if len(self.cluster_weights) >= self.max_clusters:
            return

        self._apply_split(cluster_to_split, left_group, right_group)

    def _find_farthest_pair_fine_grained(self, clients, representations, cluster_id):
        max_dist = -1.0
        pair = None
        all_distances = []
        for i in range(len(clients)):
            for j in range(i + 1, len(clients)):
                rep_i = representations[clients[i]]
                rep_j = representations[clients[j]]
                dist = self._calculate_fine_grained_distance(rep_i, rep_j, cluster_id)
                all_distances.append(dist)
                if dist > max_dist:
                    max_dist = dist
                    pair = (clients[i], clients[j])
        if not all_distances:
            mean_dist = 0.0
        else:
            mean_dist = sum(all_distances) / len(all_distances)
        return pair, max_dist, mean_dist

    def _apply_split(self, cluster_id, left_clients, right_clients):
        eta_split = getattr(self.args, "hcfl_split_lr", getattr(self.args, "hcfl_global_lr", 1.0))
        base_theta = self.cluster_weights[cluster_id]
        left_grad = self._compute_subcluster_gradient(cluster_id, left_clients)
        right_grad = self._compute_subcluster_gradient(cluster_id, right_clients)

        left_theta = [param.detach().clone() for param in base_theta]
        right_theta = [param.detach().clone() for param in base_theta]

        if left_grad is not None:
            for pid, param in enumerate(left_theta):
                grad = left_grad[pid].to(param.device)
                param.data.add_(eta_split * grad.data)
        if right_grad is not None:
            for pid, param in enumerate(right_theta):
                grad = right_grad[pid].to(param.device)
                param.data.add_(eta_split * grad.data)

        self.cluster_weights[cluster_id] = left_theta
        self.cluster_weights.append(right_theta)
        new_cluster_id = len(self.cluster_weights) - 1

        for cid in range(self.args.num_clients):
            membership = self.client_membership[cid]
            if membership.numel() < len(self.cluster_weights):
                zeros = torch.zeros(len(self.cluster_weights) - membership.numel(), dtype=membership.dtype, device=membership.device)
                membership = torch.cat([membership, zeros], dim=0)

            original_value = membership[cluster_id].clone()
            half_value = original_value / 2.0
            membership[cluster_id] = half_value
            membership[new_cluster_id] = half_value.clone()
            self.client_membership[cid] = membership

        left_set = set(left_clients)
        right_set = set(right_clients)
        for cid in right_clients:
            self.cluster_assignments[cid] = new_cluster_id
        for cid in left_clients:
            self.cluster_assignments[cid] = cluster_id

    def _compute_subcluster_gradient(self, cluster_id, client_ids):
        aggregated = None
        total_weight = 0.0
        for cid in client_ids:
            msg = self.message_pool.get(f"client_{cid}")
            if msg is None:
                continue
            delta_theta = msg.get("delta_theta_per_cluster")
            if delta_theta is None or len(delta_theta) <= cluster_id:
                continue
            client_delta = delta_theta[cluster_id]
            membership = self.client_membership[cid]
            if membership is None or membership.numel() <= cluster_id:
                weight = 0.0
            else:
                weight = float(membership[cluster_id].item())
            if weight <= 0:
                continue

            if aggregated is None:
                aggregated = [torch.zeros_like(param) for param in client_delta]
            for pid, param in enumerate(client_delta):
                aggregated[pid] += weight * param
            total_weight += weight

        if aggregated is None or total_weight <= 0:
            return None
        for tensor in aggregated:
            tensor.div_(total_weight)
        return aggregated

    def _maybe_remove_clusters(self):
        if self.message_pool["round"] < self.warmup_rounds:
            return
        if len(self.cluster_weights) <= 1:
            return

        active_clusters = set()
        for membership in self.client_membership:
            if membership is None or membership.numel() == 0:
                continue
            max_val = membership.max().item()
            if max_val <= 0:
                continue
            active = (membership == max_val).nonzero(as_tuple=False).flatten().tolist()
            for cluster_id in active:
                active_clusters.add(cluster_id)

        removable = [
            cluster_id for cluster_id in range(len(self.cluster_weights)) if cluster_id not in active_clusters
        ]
        for cluster_id in sorted(removable, reverse=True):
            self._remove_cluster(cluster_id)

    def _remove_cluster(self, cluster_id):
        del self.cluster_weights[cluster_id]

        for cid in range(self.args.num_clients):
            membership = self.client_membership[cid]
            if membership.numel() <= cluster_id:
                continue
            new_membership = torch.cat([membership[:cluster_id], membership[cluster_id + 1 :]], dim=0)
            total = new_membership.sum()
            if total.item() > 0:
                new_membership = new_membership / total
            else:
                new_membership = torch.ones_like(new_membership) / max(1, new_membership.numel())
            self.client_membership[cid] = new_membership

            assigned_cluster = self.cluster_assignments[cid]
            if assigned_cluster == cluster_id:
                self.cluster_assignments[cid] = int(new_membership.argmax().item())
            elif assigned_cluster > cluster_id:
                self.cluster_assignments[cid] = assigned_cluster - 1

    def _maybe_merge_clusters(self):
        if self.message_pool["round"] < self.warmup_rounds:
            return
        if len(self.cluster_weights) <= 1:
            return

        cluster_map = self._cluster_to_clients()
        cluster_reps = {}

        for cluster_id, client_ids in cluster_map.items():
            if not client_ids:
                continue

            total_P_c = None
            total_P_lf = None
            rep_count = 0
            grad_accumulator = None
            grad_clients = 0

            for cid in client_ids:
                rep = self._client_representation(cid)
                if rep is None:
                    continue
                proto_c = rep["P_c"]
                proto_lf = rep["P_lf"]
                if total_P_c is None:
                    total_P_c = proto_c.clone()
                    total_P_lf = proto_lf.clone()
                else:
                    total_P_c += proto_c
                    total_P_lf += proto_lf
                rep_count += 1

                if rep.get("grads") is not None:
                    grads = rep["grads"]
                    if grad_accumulator is None:
                        grad_accumulator = [g.clone() for g in grads]
                    else:
                        for g_acc, g in zip(grad_accumulator, grads):
                            g_acc += g
                    grad_clients += 1

            if rep_count == 0:
                continue

            cluster_rep = {
                "P_c": total_P_c / rep_count,
                "P_lf": total_P_lf / rep_count,
                "membership": torch.tensor([1.0], device=self.device),
            }
            if grad_accumulator is not None and grad_clients > 0:
                cluster_rep["grads"] = [g / grad_clients for g in grad_accumulator]
            cluster_reps[cluster_id] = cluster_rep

        if len(cluster_reps) <= 1:
            return

        min_dist = float("inf")
        merge_pair = None
        active_clusters = list(cluster_reps.keys())

        for i in range(len(active_clusters)):
            for j in range(i + 1, len(active_clusters)):
                c_a = active_clusters[i]
                c_b = active_clusters[j]
                dist = self._calculate_fine_grained_distance(cluster_reps[c_a], cluster_reps[c_b], 0)
                if dist < min_dist:
                    min_dist = dist
                    merge_pair = (c_a, c_b)

        if merge_pair is None or min_dist >= self.merge_threshold:
            return

        keep_idx, remove_idx = sorted(merge_pair)
        print(f"Merging cluster {remove_idx} into {keep_idx} (Dist: {min_dist:.4f})")
        self._apply_merge(keep_idx, remove_idx)

    def _apply_merge(self, keep_idx, remove_idx):
        if keep_idx == remove_idx or remove_idx >= len(self.cluster_weights):
            return

        target_params = self.cluster_weights[keep_idx]
        source_params = self.cluster_weights[remove_idx]

        total_keep = 0.0
        total_remove = 0.0
        for cid in range(self.args.num_clients):
            membership = self.client_membership[cid]
            if membership is None:
                continue
            if membership.numel() > keep_idx:
                total_keep += float(membership[keep_idx].item())
            if membership.numel() > remove_idx:
                total_remove += float(membership[remove_idx].item())

        denom = total_keep + total_remove
        if denom <= 0:
            denom = 1.0
        alpha_keep = total_keep / denom
        alpha_remove = total_remove / denom

        for p_t, p_s in zip(target_params, source_params):
            p_t.data = alpha_keep * p_t.data + alpha_remove * p_s.data

        for cid in range(self.args.num_clients):
            membership = self.client_membership[cid]
            if membership.numel() <= remove_idx:
                continue

            membership[keep_idx] += membership[remove_idx]
            new_membership = torch.cat([membership[:remove_idx], membership[remove_idx + 1 :]], dim=0)
            self.client_membership[cid] = new_membership

            assigned_cluster = self.cluster_assignments[cid]
            if assigned_cluster == remove_idx:
                self.cluster_assignments[cid] = keep_idx
            elif assigned_cluster > remove_idx:
                self.cluster_assignments[cid] = assigned_cluster - 1

        del self.cluster_weights[remove_idx]

    def _cluster_to_clients(self):
        mapping = {idx: [] for idx in range(len(self.cluster_weights))}
        for cid, cluster_id in enumerate(self.cluster_assignments):
            mapping.setdefault(cluster_id, []).append(cid)
        return mapping

    def _clone_param_list(self, params):
        return [param.detach().clone() for param in params]
