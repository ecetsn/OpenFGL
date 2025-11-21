import torch
from openfgl.flcore.base import BaseServer


class HCFLLUSServer(BaseServer):
    """
    Server for the HCFL-PLUS algorithm with shared encoder and per-cluster heads.
    """

    def __init__(self, args, global_data, data_dir, message_pool, device):
        super().__init__(args, global_data, data_dir, message_pool, device, personalized=True)
        self.cluster_assignments = [0 for _ in range(self.args.num_clients)]
        encoder, head = self._split_encoder_head(self.task.model)
        self.encoder_weights = encoder
        self.cluster_heads = [head]
        self.prototype_cache = {cid: None for cid in range(self.args.num_clients)}
        self.label_hist_cache = {cid: None for cid in range(self.args.num_clients)}
        self.resp_cache = {cid: None for cid in range(self.args.num_clients)}
        self.loss_cache = {cid: None for cid in range(self.args.num_clients)}

        self.split_threshold = getattr(self.args, "hcfl_split_tol", 0.3)
        self.min_cluster_size = getattr(self.args, "hcfl_min_cluster_size", 2)
        self.warmup_rounds = getattr(self.args, "hcfl_warmup_rounds", 5)
        self.max_clusters = getattr(self.args, "hcfl_max_clusters", 8)
        self.prototype_momentum = getattr(self.args, "hcfl_proto_momentum", 0.5)
        self.merge_threshold = getattr(self.args, "hcfl_merge_tol", 0.2)
        self.distance_mode = getattr(self.args, "hcfl_distance_mode", "ascp")
        self.rho = getattr(self.args, "hcfl_split_tol", 0.3)
        self.loss_importance = getattr(self.args, "hcfl_loss_importance", 0.0)
        self.verbose = getattr(self.args, "debug", False)

    def execute(self):
        """
        Aggregate soft client updates for encoder and cluster heads, refresh statistics,
        and adapt clusters.
        """
        sampled_clients = self.message_pool["sampled_clients"]
        num_clusters = len(self.cluster_heads)

        aggregated_heads = [self._zeros_like_param_list(w) for w in self.cluster_heads]
        head_totals = torch.zeros(num_clusters, dtype=torch.float32)

        enc_agg = self._zeros_like_param_list(self.encoder_weights)
        enc_total = 0.0

        for cid in sampled_clients:
            message = self.message_pool[f"client_{cid}"]
            responsibilities = message.get("cluster_responsibilities")
            if responsibilities is None:
                responsibilities = torch.zeros(num_clusters, dtype=torch.float32)
                responsibilities[self.cluster_assignments[cid]] = 1.0
            else:
                responsibilities = torch.as_tensor(responsibilities, dtype=torch.float32)
            self.resp_cache[cid] = responsibilities
            self.loss_cache[cid] = message.get("cluster_losses")

            head_updates = message.get("cluster_updates")
            encoder_update = message.get("encoder_update")
            num_samples = message.get("num_samples", 0)

            best_cluster = int(torch.argmax(responsibilities).item())
            self.cluster_assignments[cid] = best_cluster
            self._update_statistics(cid, message)

            if encoder_update is not None:
                for pid, param in enumerate(encoder_update):
                    enc_agg[pid].data += float(num_samples) * param.data
                enc_total += float(num_samples)

            if head_updates is None:
                continue

            for k in range(min(num_clusters, len(head_updates))):
                weight = float(responsibilities[k]) * num_samples
                if weight <= 0:
                    continue
                for pid, param in enumerate(head_updates[k]):
                    aggregated_heads[k][pid].data += weight * param.data
                head_totals[k] += weight

        if enc_total > 0:
            for pid, param in enumerate(enc_agg):
                param.data.mul_(1.0 / enc_total)
            self.encoder_weights = [p.detach().clone() for p in enc_agg]

        for k in range(num_clusters):
            if head_totals[k] == 0:
                continue
            scale = 1.0 / head_totals[k]
            for pid, param in enumerate(aggregated_heads[k]):
                param.data.mul_(scale)
            self.cluster_heads[k] = [p.detach().clone() for p in aggregated_heads[k]]

        self._maybe_split_clusters()
        self._maybe_merge_clusters()
        self._synchronize_server_model()
        if self.verbose:
            sizes = self._cluster_sizes()
            print(f"[hcfl_plus] round={self.message_pool.get('round', -1)} clusters={len(sizes)} sizes={sizes}")

    def send_message(self):
        """
        Distribute encoder and cluster-specific heads to all clients.
        """
        cluster_payload = [[param.detach().clone() for param in weights] for weights in self.cluster_heads]
        self.message_pool["server"] = {
            "encoder_weights": [p.detach().clone() for p in self.encoder_weights],
            "cluster_weights": cluster_payload,
            "cluster_assignments": self.cluster_assignments.copy(),
        }

    def _synchronize_server_model(self):
        """
        Keep the base server model roughly aligned with the average cluster to
        support any optional global evaluations.
        """
        cluster_sizes = self._cluster_sizes()
        total = sum(cluster_sizes)
        if total == 0:
            return

        self._load_encoder(self.task.model, self.encoder_weights)

        head_avg = self._zeros_like_param_list(self.cluster_heads[0])
        with torch.no_grad():
            for pid, _ in enumerate(head_avg):
                head_avg[pid].data.zero_()
            for cluster_id, size in enumerate(cluster_sizes):
                if size == 0:
                    continue
                weight = size / total
                for pid, param in enumerate(head_avg):
                    param.data += weight * self.cluster_heads[cluster_id][pid].data
            self._load_head(self.task.model, head_avg)

    def _update_statistics(self, client_id, message):
        proto = message.get("prototypes")
        counts = message.get("label_counts")
        if proto is None or counts is None:
            return

        proto = proto.to(torch.float32)
        counts = counts.to(torch.float32)

        cached_proto = self.prototype_cache.get(client_id)
        cached_counts = self.label_hist_cache.get(client_id)
        if cached_proto is None:
            self.prototype_cache[client_id] = proto
        else:
            self.prototype_cache[client_id] = (
                self.prototype_momentum * cached_proto + (1 - self.prototype_momentum) * proto
            )

        if cached_counts is None:
            self.label_hist_cache[client_id] = counts
        else:
            self.label_hist_cache[client_id] = (
                self.prototype_momentum * cached_counts + (1 - self.prototype_momentum) * counts
            )

    def _maybe_split_clusters(self):
        """
        Perform a simple dissimilarity-based split if a cluster drifts apart.
        """
        if self.message_pool["round"] < self.warmup_rounds:
            return

        cluster_map = self._cluster_to_clients()
        for cluster_id, client_ids in cluster_map.items():
            if len(client_ids) < self.min_cluster_size * 2:
                continue
            valid_clients = [cid for cid in client_ids if self.prototype_cache.get(cid) is not None]
            if len(valid_clients) < self.min_cluster_size * 2:
                continue

            pairwise = []
            far_pair = None
            far_dist = -1.0
            for i in range(len(valid_clients)):
                for j in range(i + 1, len(valid_clients)):
                    d = self._distance_clients(valid_clients[i], valid_clients[j], cluster_id)
                    pairwise.append(d)
                    if d > far_dist:
                        far_dist = d
                        far_pair = (valid_clients[i], valid_clients[j])
            if not pairwise:
                continue
            max_dist = max(pairwise)
            mean_dist = sum(pairwise) / len(pairwise)
            if (max_dist - mean_dist) < self.rho:
                continue
            anchor_pair = far_pair
            if anchor_pair is None:
                continue

            left_anchor, right_anchor = anchor_pair
            left_group, right_group = [], []
            for cid in client_ids:
                if self.prototype_cache.get(cid) is None:
                    if len(left_group) <= len(right_group):
                        left_group.append(cid)
                    else:
                        right_group.append(cid)
                    continue
                dist_left = self._distance_clients(cid, left_anchor, cluster_id)
                dist_right = self._distance_clients(cid, right_anchor, cluster_id)
                if dist_left <= dist_right:
                    left_group.append(cid)
                else:
                    right_group.append(cid)

            if len(left_group) < self.min_cluster_size or len(right_group) < self.min_cluster_size:
                continue
            if len(self.cluster_heads) >= self.max_clusters:
                break

            self._apply_split(cluster_id, left_group, right_group)
            self._renorm_after_split(cluster_id, len(self.cluster_heads) - 1)
            if self.verbose:
                print(f"[hcfl_plus] split cluster {cluster_id} -> {cluster_id},{len(self.cluster_heads)-1} (sizes {len(left_group)}/{len(right_group)})")

    def _cluster_to_clients(self):
        mapping = {idx: [] for idx in range(len(self.cluster_heads))}
        for cid, cluster_id in enumerate(self.cluster_assignments):
            mapping.setdefault(cluster_id, []).append(cid)
        return mapping

    def _cluster_sizes(self):
        mapping = self._cluster_to_clients()
        return [len(mapping.get(idx, [])) for idx in range(len(self.cluster_heads))]

    def _apply_split(self, cluster_id, left_clients, right_clients):
        new_weights = [param.detach().clone() for param in self.cluster_heads[cluster_id]]
        self.cluster_heads.append(new_weights)
        new_cluster_id = len(self.cluster_heads) - 1

        for cid in right_clients:
            self.cluster_assignments[cid] = new_cluster_id
        for cid in left_clients:
            self.cluster_assignments[cid] = cluster_id
        for cid in left_clients + right_clients:
            self._set_one_hot_resp(cid, self.cluster_assignments[cid])
            self.loss_cache[cid] = None

    def _client_representation(self, client_id):
        proto = self.prototype_cache.get(client_id)
        if proto is None:
            return None
        flat_proto = proto.reshape(-1)
        label_hist = self.label_hist_cache.get(client_id)
        if label_hist is None:
            label_hist = torch.zeros(self.task.num_global_classes, dtype=torch.float32)
        else:
            label_hist = label_hist.to(torch.float32)

        label_sum = label_hist.sum()
        if label_sum > 0:
            label_feat = label_hist / label_sum
        else:
            label_feat = torch.zeros_like(label_hist)

        proto_norm = flat_proto.norm(p=2)
        if proto_norm.item() < 1e-12:
            return None
        flat_proto = flat_proto / proto_norm

        label_norm = label_feat.norm(p=2)
        if label_norm.item() < 1e-12:
            label_feat = torch.zeros_like(label_feat)
        else:
            label_feat = label_feat / label_norm

        return (flat_proto, label_feat)

    def _find_farthest_pair(self, clients, representations):
        max_dist = -1.0
        pair = None
        for i in range(len(clients)):
            for j in range(i + 1, len(clients)):
                rep_i = representations[clients[i]]
                rep_j = representations[clients[j]]
                dist = self._distance(rep_i, rep_j)
                if dist > max_dist:
                    max_dist = dist
                    pair = (clients[i], clients[j])
        return pair, max_dist

    def _distance(self, rep_a, rep_b):
        proto_a, label_a = rep_a
        proto_b, label_b = rep_b

        dc = 1 - torch.dot(proto_a, proto_b).clamp(-1.0, 1.0)
        dlf = 1 - torch.dot(label_a, label_b).clamp(-1.0, 1.0)

        if self.distance_mode == "cscp":
            return dc.item()
        return max(dc.item(), dlf.item())

    def _set_one_hot_resp(self, cid, cluster_id):
        num_clusters = len(self.cluster_heads)
        resp = torch.zeros(num_clusters, dtype=torch.float32)
        if cluster_id < num_clusters:
            resp[cluster_id] = 1.0
        self.resp_cache[cid] = resp

    @staticmethod
    def _clone_param_list(params):
        return [param.detach().clone() for param in params]

    @staticmethod
    def _zeros_like_param_list(params):
        return [torch.zeros_like(param) for param in params]

    def _maybe_merge_clusters(self):
        """
        Merge the two most similar clusters when they are sufficiently close.
        """
        if len(self.cluster_heads) <= 1:
            return

        cluster_map = self._cluster_to_clients()
        # Compute pairwise cluster distances
        best_pair = None
        best_dist = float("inf")
        pairwise_stats = {}
        for i in range(len(self.cluster_heads)):
            for j in range(i + 1, len(self.cluster_heads)):
                dists = self._distance_clusters(i, j, cluster_map)
                if not dists:
                    continue
                min_dist = min(dists)
                dispersion = max(dists) - (sum(dists) / len(dists))
                pairwise_stats[(i, j)] = (min_dist, dispersion)
                if min_dist < best_dist:
                    best_dist = min_dist
                    best_pair = (i, j)

        if not pairwise_stats:
            return

        # Merge only if clusters are sufficiently similar (low dispersion and low min distance)
        min_dist, dispersion = pairwise_stats.get(best_pair, (float("inf"), float("inf")))
        if best_pair is None or min_dist >= self.merge_threshold or dispersion >= self.merge_threshold:
            return

        i, j = best_pair
        self._apply_merge(i, j, cluster_map)
        if self.verbose:
            print(f"[hcfl_plus] merge clusters {i} and {j} (dist={min_dist:.4f}, dispersion={dispersion:.4f})")

    def _cluster_representation(self, clients):
        protos = []
        labels = []
        for cid in clients:
            proto = self.prototype_cache.get(cid)
            counts = self.label_hist_cache.get(cid)
            if proto is None or counts is None:
                continue
            protos.append(proto)
            labels.append(counts)
        if not protos:
            return None
        proto_stack = torch.stack(protos, dim=0)  # [N, C, F]
        count_stack = torch.stack(labels, dim=0)  # [N, C]

        # Mean per-class prototype weighted by availability
        proto_mean = proto_stack.mean(dim=0).reshape(-1)
        label_mean = (count_stack.sum(dim=0) / count_stack.sum().clamp_min(1e-12))

        p_norm = proto_mean.norm(p=2)
        l_norm = label_mean.norm(p=2)
        if p_norm.item() < 1e-12:
            return None
        proto_mean = proto_mean / p_norm
        label_mean = label_mean / l_norm.clamp_min(1e-12)
        return (proto_mean, label_mean)

    def _distance_clusters(self, cluster_a, cluster_b, cluster_map):
        distances = []
        for cid_a in cluster_map.get(cluster_a, []):
            for cid_b in cluster_map.get(cluster_b, []):
                # Evaluate both weighted (cluster-aware) and unweighted distances to avoid trivial zeros.
                d_base = self._distance_clients(cid_a, cid_b, None)
                d_weighted = min(
                    self._distance_clients(cid_a, cid_b, cluster_a),
                    self._distance_clients(cid_a, cid_b, cluster_b),
                )
                d = max(d_base, d_weighted)
                if d != float("inf"):
                    distances.append(d)
        return distances

    def _apply_merge(self, cluster_a, cluster_b, cluster_map):
        clients_a = cluster_map.get(cluster_a, [])
        clients_b = cluster_map.get(cluster_b, [])
        size_a = max(len(clients_a), 1)
        size_b = max(len(clients_b), 1)

        merged_weights = []
        for pid in range(len(self.cluster_heads[cluster_a])):
            merged = (
                self.cluster_heads[cluster_a][pid] * float(size_a)
                + self.cluster_heads[cluster_b][pid] * float(size_b)
            ) / float(size_a + size_b)
            merged_weights.append(merged.detach().clone())

        self.cluster_heads[cluster_a] = merged_weights
        self.cluster_heads.pop(cluster_b)

        for cid, cluster_id in enumerate(self.cluster_assignments):
            if cluster_id == cluster_b:
                self.cluster_assignments[cid] = cluster_a
            elif cluster_id > cluster_b:
                self.cluster_assignments[cid] -= 1
        self._renorm_after_merge(cluster_a)
        for cid in clients_a + clients_b:
            self._set_one_hot_resp(cid, self.cluster_assignments[cid])
            self.loss_cache[cid] = None

    def _distance_clients(self, cid_a, cid_b, cluster_id=None):
        proto_a = self.prototype_cache.get(cid_a)
        proto_b = self.prototype_cache.get(cid_b)
        counts_a = self.label_hist_cache.get(cid_a)
        counts_b = self.label_hist_cache.get(cid_b)
        if proto_a is None or proto_b is None or counts_a is None or counts_b is None:
            return float("inf")

        proto_a = proto_a.to(torch.float32)
        proto_b = proto_b.to(torch.float32)
        counts_a = counts_a.to(torch.float32)
        counts_b = counts_b.to(torch.float32)

        # d_c: max over classes of cosine distance between per-class prototypes (where both have samples)
        common = (counts_a > 0) & (counts_b > 0)
        d_c = 0.0
        if common.any():
            protos_a = proto_a[common]
            protos_b = proto_b[common]
            # normalize per class
            protos_a = protos_a / protos_a.norm(dim=1, keepdim=True).clamp_min(1e-12)
            protos_b = protos_b / protos_b.norm(dim=1, keepdim=True).clamp_min(1e-12)
            cos = (protos_a * protos_b).sum(dim=1).clamp(-1.0, 1.0)
            d_c = float((1 - cos).max().item())

        # d_lf: cosine distance between mean feature prototypes
        mean_a = (proto_a * counts_a.unsqueeze(1)).sum(dim=0) / counts_a.sum().clamp_min(1e-12)
        mean_b = (proto_b * counts_b.unsqueeze(1)).sum(dim=0) / counts_b.sum().clamp_min(1e-12)
        mean_a = mean_a / mean_a.norm(p=2).clamp_min(1e-12)
        mean_b = mean_b / mean_b.norm(p=2).clamp_min(1e-12)
        cos_lf = float(torch.dot(mean_a, mean_b).clamp(-1.0, 1.0).item())
        d_lf = 1 - cos_lf

        base = d_c if self.distance_mode == "cscp" else max(d_c, d_lf)

        if cluster_id is None:
            return base

        weight = self._resp_weight(cid_a, cluster_id) * self._resp_weight(cid_b, cluster_id)
        if self.loss_importance > 0:
            loss_a = self._loss_value(cid_a, cluster_id)
            loss_b = self._loss_value(cid_b, cluster_id)
            if loss_a is not None and loss_b is not None:
                weight = weight * (1.0 + self.loss_importance * ((loss_a + loss_b) / 2.0))

        return base * weight

    def _resp_weight(self, cid, cluster_id):
        resp = self.resp_cache.get(cid)
        if resp is None or cluster_id >= resp.numel():
            return 1.0 if self.cluster_assignments[cid] == cluster_id else 0.0
        return float(resp[cluster_id])

    def _loss_value(self, cid, cluster_id):
        loss_vec = self.loss_cache.get(cid)
        if loss_vec is None or len(loss_vec) <= cluster_id:
            return None
        return float(loss_vec[cluster_id])

    def _renorm_after_split(self, old_cluster, new_cluster):
        # Placeholder: responsibilities are client-side; just ensure assignments stay consistent.
        return

    def _renorm_after_merge(self, kept_cluster):
        # Placeholder: responsibilities are client-side; nothing to renorm server-side.
        return

    def _split_encoder_head(self, model):
        # If a layered model exists, use head_layer_idx if provided; otherwise, last layer is head
        if hasattr(model, "layers") and len(model.layers) >= 1:
            encoder = []
            head = []
            head_idx = self.args.hcfl_head_layer_idx
            if head_idx < 0 or head_idx >= len(model.layers):
                head_idx = len(model.layers) - 1
            for idx, layer in enumerate(model.layers):
                params = list(layer.parameters())
                if idx < head_idx:
                    encoder.extend([p.detach().clone() for p in params])
                else:
                    head.extend([p.detach().clone() for p in params])
            return encoder, head
        params = [p.detach().clone() for p in model.parameters()]
        return [], params

    def _load_encoder(self, model, encoder_params):
        if not encoder_params:
            return
        if hasattr(model, "layers") and len(model.layers) > 1:
            enc_iter = iter(encoder_params)
            with torch.no_grad():
                for idx, layer in enumerate(model.layers):
                    head_idx = self.args.hcfl_head_layer_idx
                    if head_idx < 0 or head_idx >= len(model.layers):
                        head_idx = len(model.layers) - 1
                    if idx >= head_idx:
                        break
                    for p in layer.parameters():
                        try:
                            src = next(enc_iter)
                        except StopIteration:
                            return
                        p.data.copy_(src.data)
        else:
            # no distinct encoder; skip
            return

    def _load_head(self, model, head_params):
        if not head_params:
            return
        if hasattr(model, "layers") and len(model.layers) >= 1:
            head_iter = iter(head_params)
            with torch.no_grad():
                head_idx = self.args.hcfl_head_layer_idx
                if head_idx < 0 or head_idx >= len(model.layers):
                    head_idx = len(model.layers) - 1
                for p in model.layers[head_idx].parameters():
                    try:
                        src = next(head_iter)
                    except StopIteration:
                        return
                    p.data.copy_(src.data)
        else:
            head_iter = iter(head_params)
            with torch.no_grad():
                for p in model.parameters():
                    try:
                        src = next(head_iter)
                    except StopIteration:
                        return
                    p.data.copy_(src.data)
