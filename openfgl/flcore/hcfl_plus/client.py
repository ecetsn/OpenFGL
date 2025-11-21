import torch
import torch.nn.functional as F
from openfgl.flcore.base import BaseClient


class HCFLLUSClient(BaseClient):
    """
    Client for the HCFL-PLUS algorithm.

    Each client receives the parameters of the cluster it belongs to, performs
    local training, and reports both the updated weights and simple class-wise
    prototypes so that the server can maintain similarity-aware clusters.
    """

    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super().__init__(args, client_id, data, data_dir, message_pool, device, personalized=True)
        self.current_cluster = 0

    def execute(self):
        """
        Pull all cluster-specific models from the server, compute soft
        responsibilities, and train a personalized model accordingly.
        """
        server_payload = self.message_pool["server"]
        encoder_weights = server_payload.get("encoder_weights")
        cluster_weights = server_payload["cluster_weights"]
        self._load_encoder(encoder_weights)

        labels, mask = self._get_train_labels_and_mask()
        labels = labels.to(self.task.model.parameters().__next__().device)
        mask = mask.to(labels.device).bool()
        if mask.sum() == 0:
            # Nothing to train on; keep current weights
            self.cluster_responsibilities = torch.ones(len(cluster_weights)) / max(len(cluster_weights), 1)
            self.current_cluster = 0
            self.cluster_updates = cluster_weights
            self.sample_resps = [None for _ in cluster_weights]
            return

        cluster_losses = []
        cluster_updates = []
        sample_resps = []
        per_sample_losses = []

        for head_weight in cluster_weights:
            self._load_head(head_weight)
            self.task.model.eval()
            with torch.no_grad():
                _, logits = self.task.model.forward(self.task.splitted_data["data"])
                train_logits = logits[mask]
                train_labels = labels[mask]
                sample_loss = F.cross_entropy(train_logits, train_labels, reduction="none")
                cluster_losses.append(float(sample_loss.mean().detach().cpu()))
                per_sample_losses.append(sample_loss.detach())

        # Compute per-sample responsibilities across clusters with optional annealing/top-k filtering
        loss_stack = torch.stack(per_sample_losses, dim=0)  # [K, N_train]
        base_temp = getattr(self.args, "hcfl_soft_temp", 1.0)
        decay = getattr(self.args, "hcfl_soft_temp_decay", 1.0)
        current_round = self.message_pool.get("round", 0) if isinstance(self.message_pool, dict) else 0
        temp = base_temp * (decay ** max(current_round, 0))
        sample_resp = torch.softmax(-loss_stack / max(temp, 1e-6), dim=0)
        client_resp = sample_resp.mean(dim=1)  # [K]

        topk = getattr(self.args, "hcfl_resp_topk", 0)
        if topk > 0 and topk < sample_resp.size(0):
            # Zero out non-topk clusters per client, then renormalize
            topk_idx = torch.topk(client_resp, topk).indices
            mask = torch.zeros_like(client_resp, dtype=torch.bool)
            mask[topk_idx] = True
            sample_resp = sample_resp * mask.unsqueeze(1)
            sample_resp = sample_resp / sample_resp.sum(dim=0, keepdim=True).clamp_min(1e-6)
            client_resp = client_resp * mask
            client_resp = client_resp / client_resp.sum().clamp_min(1e-6)

        # Train each cluster with weighted loss and KL regularizer
        for k, head_weight in enumerate(cluster_weights):
            self._load_head(head_weight)
            local_resp = self._train_with_soft_weights(sample_resp[k].to(labels.device), client_resp[k].to(labels.device), labels, mask)
            sample_resps.append(local_resp.detach().cpu() if local_resp is not None else None)
            _, head_params = self._extract_encoder_head()
            cluster_updates.append(head_params)

        losses_tensor = torch.tensor(cluster_losses, dtype=torch.float32)
        self.cluster_responsibilities = torch.softmax(-losses_tensor / max(temp, 1e-6), dim=0)
        self.current_cluster = int(torch.argmax(self.cluster_responsibilities).item())
        self._load_head(cluster_updates[self.current_cluster])
        self.cluster_updates = cluster_updates
        self.sample_resps = sample_resps
        self.cluster_losses = losses_tensor.detach().cpu()

    def send_message(self):
        """
        Report the trained weights (for every cluster) together with simple label
        prototypes that summarize the observed data distribution for the chosen
        local model.
        """
        prototypes, counts = self._compute_prototypes()
        encoder_params, _ = self._extract_encoder_head()
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "cluster_updates": self.cluster_updates,
            "encoder_update": encoder_params,
            "cluster_responsibilities": self.cluster_responsibilities.detach().cpu(),
            "sample_responsibilities": self.sample_resps,
            "cluster_losses": self.cluster_losses,
            "cluster_id": self.current_cluster,
            "prototypes": prototypes,
            "label_counts": counts,
        }

    def _compute_prototypes(self):
        """
        Derive class-wise prototypes from encoder features (paper uses features g(x;phi)).
        Falls back to logits if features are unavailable. Returns CPU tensors.
        """
        labels, mask = self._get_train_labels_and_mask()
        if mask.sum() == 0:
            return None, None

        labels = labels.long()
        # Run a single forward pass to get features
        self.task.model.eval()
        with torch.no_grad():
            forward_out = self.task.model.forward(self.task.splitted_data["data"])
        if isinstance(forward_out, tuple) and len(forward_out) >= 1:
            features = forward_out[0]
        else:
            features = forward_out

        if features is None:
            return None, None

        features = features.detach()
        if features.dim() == 1:
            features = features.unsqueeze(-1)

        labels = labels.to(features.device)
        mask = mask.to(features.device).bool()

        selected_feat = features[mask]
        selected_labels = labels[mask]
        num_classes = self.task.num_global_classes

        proto = torch.zeros(num_classes, selected_feat.size(1), device=features.device, dtype=features.dtype)
        counts = torch.zeros(num_classes, device=features.device, dtype=features.dtype)

        if selected_feat.numel() > 0:
            ones = torch.ones(selected_labels.shape[0], device=features.device, dtype=features.dtype)
            proto.index_add_(0, selected_labels, selected_feat)
            counts.index_add_(0, selected_labels, ones)
            proto = proto / counts.clamp_min(1.0).unsqueeze(1)

        return proto.cpu(), counts.cpu()

    def _get_train_labels_and_mask(self):
        """
        Extract the labels and mask that correspond to locally observed training
        data across all supported task types.
        """
        splitted = self.task.splitted_data
        if "merged_edge_label" in splitted:
            labels = splitted["merged_edge_label"]
            mask = splitted["merged_edge_train_mask"]
        else:
            labels = splitted["data"].y
            mask = splitted["train_mask"]
        return labels.clone().detach(), mask.clone().detach()

    def _load_encoder(self, encoder_params):
        if not encoder_params:
            return
        if not hasattr(self.task.model, "layers") or len(self.task.model.layers) < 1:
            return
        enc_iter = iter(encoder_params)
        with torch.no_grad():
            for idx, layer in enumerate(self.task.model.layers):
                if idx >= len(self.task.model.layers) - 1:
                    break
                for p in layer.parameters():
                    try:
                        src = next(enc_iter)
                    except StopIteration:
                        return
                    p.data.copy_(src.data)

    def _load_head(self, head_params):
        if not head_params:
            return
        if not hasattr(self.task.model, "layers") or len(self.task.model.layers) < 1:
            return
        head_iter = iter(head_params)
        with torch.no_grad():
            for p in self.task.model.layers[-1].parameters():
                try:
                    src = next(head_iter)
                except StopIteration:
                    return
                p.data.copy_(src.data)

    def _train_with_soft_weights(self, sample_resp, client_resp, labels, mask):
        """
        Run local epochs with soft sample responsibilities vs cluster and KL term.
        """
        mu = getattr(self.args, "hcfl_mu", 0.0)
        if sample_resp is None or sample_resp.numel() == 0:
            self.task.train()
            for _ in range(self.args.num_epochs):
                self.task.optim.zero_grad()
                embedding, logits = self.task.model.forward(self.task.splitted_data["data"])
                loss = self.task.default_loss_fn(logits[mask], labels[mask])
                loss.backward()
                self.task.optim.step()
            return None

        resp = sample_resp
        if resp.sum() <= 0:
            resp = torch.ones_like(resp) / resp.numel()

        self.task.model.train()
        for _ in range(self.args.num_epochs):
            self.task.optim.zero_grad()
            _, logits = self.task.model.forward(self.task.splitted_data["data"])
            per_sample_loss = F.cross_entropy(logits[mask], labels[mask], reduction="none")
            weighted_loss = (per_sample_loss * resp).sum() / resp.sum()
            kl_term = mu * (client_resp * torch.log(client_resp.clamp_min(1e-8) / resp.clamp_min(1e-8))).mean()
            total_loss = weighted_loss + kl_term
            total_loss.backward()
            self.task.optim.step()
        return resp

    def _extract_encoder_head(self):
        if hasattr(self.task.model, "layers") and len(self.task.model.layers) >= 1:
            encoder = []
            head = []
            for idx, layer in enumerate(self.task.model.layers):
                params = list(layer.parameters())
                if idx < len(self.task.model.layers) - 1:
                    encoder.extend([p.detach().clone() for p in params])
                else:
                    head.extend([p.detach().clone() for p in params])
            return encoder, head
        params = [p.detach().clone() for p in self.task.model.parameters()]
        return [], params
