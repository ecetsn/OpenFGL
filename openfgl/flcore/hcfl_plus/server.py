import torch
from openfgl.flcore.base import BaseServer


class HCFLLUSServer(BaseServer):
    """
    Server for the HCFL-PLUS algorithm. The server keeps track of clusters of
    clients, aggregates their updates separately and performs simple prototype
    based splitting when clients become too dissimilar.
    """

    def __init__(self, args, global_data, data_dir, message_pool, device):
        super().__init__(args, global_data, data_dir, message_pool, device, personalized=True)
        self.cluster_assignments = [0 for _ in range(self.args.num_clients)]
        ##added
        self.client_membership = [torch.tensor([1.0], dtype=torch.float32) for _ in range(self.args.num_clients)]

        self.cluster_weights = [self._clone_param_list(self.task.model.parameters())]
        self.prototype_cache = {cid: None for cid in range(self.args.num_clients)}
        self.label_hist_cache = {cid: None for cid in range(self.args.num_clients)}

        self.split_threshold = getattr(self.args, "hcfl_split_tol", 0.3)
        self.min_cluster_size = getattr(self.args, "hcfl_min_cluster_size", 2)
        self.warmup_rounds = getattr(self.args, "hcfl_warmup_rounds", 5)
        self.max_clusters = getattr(self.args, "hcfl_max_clusters", 8)
        self.prototype_momentum = getattr(self.args, "hcfl_proto_momentum", 0.5)

    def execute(self):
        """
        Aggregate updates from each cluster and refresh statistics used for
        similarity-driven clustering.
        """
        '''
        sampled_clients = self.message_pool["sampled_clients"]
        clustered_clients = {}
        for cid in sampled_clients:
            clustered_clients.setdefault(self.cluster_assignments[cid], []).append(cid)

        for cluster_id, client_ids in clustered_clients.items():
            self.cluster_weights[cluster_id] = self._aggregate_cluster(cluster_id, client_ids)

        for cid in sampled_clients:
            self._update_statistics(cid, self.message_pool[f"client_{cid}"])

        self._maybe_split_clusters()
        self._synchronize_server_model()
        '''
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

        for cluster_id in range(len(self.cluster_weights)):
            self.cluster_weights[cluster_id] = self._aggregate_cluster(cluster_id, sampled_clients)
        
        for cid in sampled_clients:
             self._update_statistics(cid, self.message_pool[f"client_{cid}"])
        
        self._maybe_split_clusters()

        self._synchronize_server_model()


    def send_message(self):
        """
        Distribute cluster-specific weights to all clients.
        """
        
        cluster_payload = [
        [param.detach().clone() for param in weights]
        for weights in self.cluster_weights
        ]

        membership_payload = [m.clone().cpu().tolist() for m in self.client_membership]
        
        self.message_pool["server"] = {
            "cluster_weights": cluster_payload,
            "client_membership": membership_payload,
            "cluster_assignments": self.cluster_assignments.copy(),
        }

    def _aggregate_cluster(self, cluster_id, client_ids):
        """
        Weighted averaging for a specific cluster.
        """
        '''
        total_samples = sum(self.message_pool[f"client_{cid}"]["num_samples"] for cid in client_ids)
        if total_samples == 0:
            return [param.detach().clone() for param in self.cluster_weights[cluster_id]]

        new_params = [torch.zeros_like(param) for param in self.cluster_weights[cluster_id]]
        for cid in client_ids:
            client_weight = self.message_pool[f"client_{cid}"]["weight"]
            weight = self.message_pool[f"client_{cid}"]["num_samples"] / total_samples
            for idx, param in enumerate(new_params):
                param.data += weight * client_weight[idx].data
        return [param.detach().clone() for param in new_params]
        '''
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
            deltas = msg["delta"]
            n_i = msg["num_samples"]
            w_i_k = float(self.client_membership[cid][cluster_id])
            if w_i_k <= 0:
                continue
            coeff = (n_i * w_i_k) / total_weight
            for pid, param in enumerate(new_params):
              param.data += eta_g * coeff * deltas[pid].data
        
        return [p.detach().clone() for p in new_params]

    def _synchronize_server_model(self):
        """
        Keep the base server model roughly aligned with the average cluster to
        support any optional global evaluations.
        """
        cluster_sizes = self._cluster_sizes()
        total = sum(cluster_sizes)
        if total == 0:
            return

        with torch.no_grad():
            for pid, param in enumerate(self.task.model.parameters()):
                param.data.zero_()
                for cluster_id, size in enumerate(cluster_sizes):
                    if size == 0:
                        continue
                    weight = size / total
                    param.data += weight * self.cluster_weights[cluster_id][pid].data

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
            representations = {cid: self._client_representation(cid) for cid in client_ids}
            valid_clients = [cid for cid, rep in representations.items() if rep is not None]
            if len(valid_clients) < self.min_cluster_size * 2:
                continue

            anchor_pair, max_dist = self._find_farthest_pair(valid_clients, representations)
            if anchor_pair is None or max_dist < self.split_threshold:
                continue

            left_anchor, right_anchor = anchor_pair
            left_rep = representations[left_anchor]
            right_rep = representations[right_anchor]
            left_group, right_group = [], []
            for cid in client_ids:
                rep = representations.get(cid)
                if rep is None:
                    if len(left_group) <= len(right_group):
                        left_group.append(cid)
                    else:
                        right_group.append(cid)
                    continue
                dist_left = self._distance(rep, left_rep)
                dist_right = self._distance(rep, right_rep)
                if dist_left <= dist_right:
                    left_group.append(cid)
                else:
                    right_group.append(cid)

            if len(left_group) < self.min_cluster_size or len(right_group) < self.min_cluster_size:
                continue
            if len(self.cluster_weights) >= self.max_clusters:
                break

            self._apply_split(cluster_id, left_group, right_group)

    def _cluster_to_clients(self):
        mapping = {idx: [] for idx in range(len(self.cluster_weights))}
        for cid, cluster_id in enumerate(self.cluster_assignments):
            mapping.setdefault(cluster_id, []).append(cid)
        return mapping

    def _cluster_sizes(self):
        mapping = self._cluster_to_clients()
        return [len(mapping.get(idx, [])) for idx in range(len(self.cluster_weights))]

    def _apply_split(self, cluster_id, left_clients, right_clients):
        new_weights = [param.detach().clone() for param in self.cluster_weights[cluster_id]]
        self.cluster_weights.append(new_weights)
        new_cluster_id = len(self.cluster_weights) - 1

        for cid in range(self.args.num_clients):
            m = self.client_membership[cid]
            if m.numel() < len(self.cluster_weights):
                zero = torch.zeros(1, dtype=m.dtype)
                self.client_membership[cid] = torch.cat([m, zero], dim=0)
            

        for cid in right_clients:
            m = torch.zeros(len(self.cluster_weights), dtype=torch.float32)
            m[new_cluster_id] = 1.0
            self.client_membership[cid] = m
            self.cluster_assignments[cid] = new_cluster_id
        for cid in left_clients:
            m = torch.zeros(len(self.cluster_weights), dtype=torch.float32)
            m[cluster_id] = 1.0
            self.client_membership[cid] = m
            self.cluster_assignments[cid] = cluster_id

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

        representation = torch.cat([flat_proto, label_feat], dim=0)
        norm = representation.norm(p=2)
        if norm.item() < 1e-12:
            return None
        return representation / norm

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

    @staticmethod
    def _distance(rep_a, rep_b):
        return 1 - torch.dot(rep_a, rep_b).clamp(-1.0, 1.0)

    @staticmethod
    def _clone_param_list(params):
        return [param.detach().clone() for param in params]