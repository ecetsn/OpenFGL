import torch
import torch.nn.functional as F
from openfgl.flcore.hcfl_plus.likelihoods import _gaussian_log_density
from openfgl.utils.privacy_utils import clip_gradients, add_noise


class HCFLGraphTaskAdapter:
    """
    Graph-classification HCFL+ adapter with EM-style local training on graph batches.
    """

    def __init__(self, original_task, phi_indices, theta_indices, args):
        self._task = original_task
        self.args = args
        self.device = original_task.device
        self.phi_indices = phi_indices
        self.theta_indices = theta_indices
        self.mu = getattr(args, "hcfl_mu", 0.1)
        self.K = getattr(args, "num_clusters", 1)
        self.sample_weights_omega = None
        self.cluster_gaussians = []
        self._cached_num_samples = None  # Cache to avoid repeated dataloader iteration
        self._initialize_hcfl_optimizer()

    def __getattr__(self, name):
        return getattr(self._task, name)

    def _initialize_hcfl_optimizer(self):
        all_params = list(self._task.model.parameters())
        phi_params = [all_params[i] for i in self.phi_indices]
        theta_params = [all_params[i] for i in self.theta_indices]
        self.optimizer = torch.optim.SGD(
            [
                {"params": phi_params, "lr": getattr(self.args, "hcfl_lr_phi", self.args.lr)},
                {"params": theta_params, "lr": self.args.lr},
            ],
            lr=self.args.lr,
        )
        self._task.optim = self.optimizer

    def _get_current_model_params(self):
        return list(self.model.parameters())

    def _get_theta_params(self, params):
        return [params[i] for i in self.theta_indices]

    def _clone_param_list(self, params):
        return [param.detach().clone() for param in params]

    def _set_theta_params(self, theta_list):
        all_params = self._get_current_model_params()
        with torch.no_grad():
            for local_idx, theta_param in enumerate(theta_list):
                param_idx = self.theta_indices[local_idx]
                all_params[param_idx].data.copy_(theta_param.data)

    def _forward_with_features(self, batch):
        model_output = self.model(batch)
        if isinstance(model_output, tuple):
            if len(model_output) >= 2:
                features, logits = model_output[0], model_output[1]
            else:
                features = model_output[0]
                logits = model_output[0]
        else:
            logits = model_output
            features = model_output
        if features.dim() == 1:
            features = features.unsqueeze(-1)
        return features.detach(), logits

    def _ensure_gaussian_slots(self, num_clusters, feature_dim, device):
        current = len(self.cluster_gaussians)
        if current < num_clusters:
            for _ in range(num_clusters - current):
                self.cluster_gaussians.append(
                    {
                        "mu": torch.zeros(feature_dim, device=device),
                        "sigma": torch.ones(feature_dim, device=device),
                    }
                )
        elif current > num_clusters:
            self.cluster_gaussians = self.cluster_gaussians[:num_clusters]

    def _update_gaussian_params(self, features, gamma):
        if features is None or gamma is None:
            return
        if features.dim() == 1:
            features = features.unsqueeze(-1)
        N, dim = features.shape
        K = gamma.shape[1]
        self._ensure_gaussian_slots(K, dim, features.device)
        eps = 1e-6
        for k in range(K):
            weights = gamma[:, k].unsqueeze(1)
            mass = weights.sum().clamp_min(eps)
            weighted_feats = weights * features
            mu = weighted_feats.sum(dim=0) / mass
            diff = features - mu
            var = (weights * diff * diff).sum(dim=0) / mass
            self.cluster_gaussians[k]["mu"] = mu.detach().clone()
            self.cluster_gaussians[k]["sigma"] = var.detach().clone().clamp_min(eps)

    def _calculate_all_cluster_likelihoods(self, train_loader, server_theta_weights, cluster_label_dists=None):
        formulation = getattr(self.args, "hcfl_formulation", "conditional").lower()
        gaussian_params = self.cluster_gaussians
        global_prior = None
        if cluster_label_dists is not None:
            if not isinstance(cluster_label_dists, torch.Tensor):
                cluster_label_dists = torch.tensor(cluster_label_dists, dtype=torch.float32, device=self.device)
            if cluster_label_dists.dim() == 1:
                prior = cluster_label_dists.clone()
            else:
                prior = cluster_label_dists.sum(dim=0)
            total = prior.sum().clamp_min(1e-6)
            global_prior = (prior / total).to(self.device)

        num_samples = self._infer_num_samples()
        K = len(server_theta_weights)
        L_k_t = torch.zeros(num_samples, K, device=self.device)

        for k in range(K):
            self._set_theta_params(server_theta_weights[k])
            offset = 0
            for batch in train_loader:
                batch = batch.to(self.device)
                features, logits = self._forward_with_features(batch)
                labels = batch.y.long()
                idx = torch.arange(labels.shape[0], device=self.device)
                log_p = F.log_softmax(logits, dim=1)
                log_L_k = log_p[idx, labels]

                if formulation in ("joint", "correlation"):
                    if gaussian_params is not None and k < len(gaussian_params):
                        params = gaussian_params[k]
                        if params is not None:
                            log_L_k = log_L_k + _gaussian_log_density(features, params)
                if formulation == "correlation" and global_prior is not None:
                    log_L_k = log_L_k - torch.log(global_prior[labels].clamp_min(1e-6))

                L_batch = torch.exp(log_L_k)
                L_k_t[offset : offset + labels.shape[0], k] = L_batch
                offset += labels.shape[0]

        return L_k_t

    def train(self, client_instance, server_theta_weights, cluster_label_dists=None, *args, **kwargs):
        self.model.train()
        omega_tilde_t = client_instance.cluster_weights_i.to(self.device)
        train_loader = self.splitted_data["train_dataloader"]
        N_i = self._infer_num_samples()
        if N_i == 0:
            return

        self.K = len(server_theta_weights)
        if (
            self.sample_weights_omega is None
            or self.sample_weights_omega.shape[0] != N_i
            or self.sample_weights_omega.shape[1] != self.K
        ):
            self.sample_weights_omega = omega_tilde_t.unsqueeze(0).repeat(N_i, 1)

        mu_N_factor = self.mu * N_i
        omega_tilde_t_plus_1 = omega_tilde_t.clone()
        local_theta_params = [self._clone_param_list(theta_k) for theta_k in server_theta_weights]
        last_gamma = None

        for _ in range(self.args.num_epochs):
            self.optimizer.zero_grad()
            L_k_t = self._calculate_all_cluster_likelihoods(
                train_loader, local_theta_params, cluster_label_dists=cluster_label_dists
            )
            total_weighted = (self.sample_weights_omega * L_k_t).sum(dim=1, keepdim=True)
            gamma = (self.sample_weights_omega * L_k_t) / total_weighted.clamp_min(1e-12)
            last_gamma = gamma.detach()

            omega_tilde_expanded = omega_tilde_t.unsqueeze(0).repeat(N_i, 1)
            total_weighted_tilde = (omega_tilde_expanded * L_k_t).sum(dim=1, keepdim=True)
            gamma_tilde = (omega_tilde_expanded * L_k_t) / total_weighted_tilde.clamp_min(1e-12)

            omega_tilde_t_plus_1 = gamma_tilde.mean(dim=0)
            entropy_reg = getattr(self.args, "hcfl_entropy_reg", 0.0)
            if entropy_reg > 0:
                uniform = torch.full_like(omega_tilde_t_plus_1, 1.0 / float(self.K))
                omega_tilde_t_plus_1 = (1.0 - entropy_reg) * omega_tilde_t_plus_1 + entropy_reg * uniform

            tilde_mu = 1.0 / (1.0 + mu_N_factor)
            self.sample_weights_omega = tilde_mu * gamma + (1.0 - tilde_mu) * omega_tilde_t_plus_1
            if getattr(self.args, "hcfl_dp_sample_weights", False):
                self.sample_weights_omega = self._apply_dp_to_sample_weights(self.sample_weights_omega)
                if getattr(self.args, "hcfl_dp_accounting", False):
                    client_instance._dp_accounting_counts["sample"] += 1

            # For DP mode, accumulate clipped phi gradients across clusters and batches
            dp_enabled = getattr(self.args, "dp_mech", "no_dp") != "no_dp"
            if dp_enabled:
                phi_grad_accum = {
                    i: torch.zeros_like(list(self.model.parameters())[i])
                    for i in self.phi_indices
                }

            for k in range(self.K):
                # Detach gamma_k to prevent backprop through likelihood computation
                gamma_k = gamma[:, k].detach()
                if gamma_k.sum().item() <= 0:
                    continue

                self._set_theta_params(local_theta_params[k])
                offset = 0
                for batch in train_loader:
                    batch = batch.to(self.device)
                    _, logits = self.model(batch)
                    labels = batch.y.long()
                    sample_losses = F.cross_entropy(logits, labels, reduction="none")
                    weights = gamma_k[offset : offset + labels.shape[0]]
                    weighted_losses = weights * sample_losses  # Per-sample weighted losses
                    batch_size = labels.shape[0]

                    # Model-level DP: clip per-sample gradients if DP enabled
                    if dp_enabled:
                        clip_gradients(
                            self.model,
                            weighted_losses,
                            batch_size,
                            self.args.dp_mech,
                            getattr(self.args, "grad_clip", 1.0),
                        )
                        # Accumulate clipped phi gradients
                        all_params = list(self.model.parameters())
                        for i in self.phi_indices:
                            if all_params[i].grad is not None:
                                phi_grad_accum[i].add_(all_params[i].grad.data)
                    else:
                        loss_k = weighted_losses.sum()
                        loss_k.backward()
                    offset += batch_size

                self._update_theta_cluster(local_theta_params[k])

            # Restore accumulated phi gradients for optimizer.step()
            if dp_enabled:
                all_params = list(self.model.parameters())
                for i in self.phi_indices:
                    all_params[i].grad = phi_grad_accum[i]

            self._apply_optimizer_callback()
            self.optimizer.step()

            # Model-level DP: add noise to model parameters after update
            if dp_enabled:
                add_noise(self.args, self.model, N_i)

        final_theta = [self._clone_param_list(theta) for theta in local_theta_params]
        weighted_theta = self._compute_weighted_theta(final_theta, omega_tilde_t_plus_1)
        self._set_theta_params(weighted_theta)
        if last_gamma is not None:
            features = self._collect_features(train_loader)
            self._update_gaussian_params(features, last_gamma.to(features.device))
        client_instance.delta_theta_per_cluster = self._compute_theta_deltas(final_theta, server_theta_weights)
        client_instance.cluster_weights_i.data.copy_(omega_tilde_t_plus_1.cpu().data)

    def _collect_features(self, train_loader):
        features_all = []
        for batch in train_loader:
            batch = batch.to(self.device)
            features, _ = self._forward_with_features(batch)
            features_all.append(features)
        if not features_all:
            return None
        return torch.cat(features_all, dim=0)

    def _apply_dp_to_sample_weights(self, sample_weights):
        clip = getattr(self.args, "hcfl_sample_clip", 0.0)
        noise = getattr(self.args, "hcfl_sample_noise", 0.0)

        weights = sample_weights.detach().clone()
        if clip > 0:
            norms = weights.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            scale = torch.clamp(clip / norms, max=1.0)
            weights = weights * scale

        if noise > 0:
            weights = weights + torch.randn_like(weights) * noise

        weights = weights.clamp_min(0.0)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return weights

    def _update_theta_cluster(self, local_theta_params_k):
        theta_params = self._get_theta_params(self._get_current_model_params())
        theta_lr = getattr(self.args, "hcfl_lr_theta", self.args.lr)

        with torch.no_grad():
            for idx, param in enumerate(theta_params):
                grad = param.grad
                if grad is None:
                    continue
                local_theta_params_k[idx].data.add_(-theta_lr * grad.data)
                param.grad.zero_()

    def _compute_weighted_theta(self, local_theta_params, membership):
        weighted = [torch.zeros_like(param) for param in local_theta_params[0]]
        with torch.no_grad():
            for k, theta_k in enumerate(local_theta_params):
                if k >= membership.numel():
                    weight_value = 0.0
                else:
                    weight_value = float(membership[k].item())
                if weight_value <= 0:
                    continue
                for pid, param in enumerate(weighted):
                    param.data.add_(weight_value * theta_k[pid].data)
        return weighted

    def _compute_theta_deltas(self, final_theta, initial_theta):
        deltas = []
        for idx in range(len(final_theta)):
            final_params = final_theta[idx]
            init_params = initial_theta[idx] if idx < len(initial_theta) else final_theta[idx]
            cluster_delta = []
            for f_param, i_param in zip(final_params, init_params):
                delta = f_param.detach().clone() - i_param.detach().clone().to(f_param.device)
                cluster_delta.append(delta.cpu())
            deltas.append(cluster_delta)
        return deltas

    def _infer_num_samples(self):
        """Infer number of training samples for graph classification.

        For graph-FL, this counts training graphs from the dataloader.
        Falls back to train_mask if dataloader is unavailable.
        Results are cached to avoid repeated iteration.
        """
        if self._cached_num_samples is not None:
            return self._cached_num_samples

        train_loader = self.splitted_data.get("train_dataloader")
        if train_loader is not None:
            # Try to get length from dataset first (efficient)
            if hasattr(train_loader, "dataset") and hasattr(train_loader.dataset, "__len__"):
                self._cached_num_samples = len(train_loader.dataset)
                return self._cached_num_samples

            # Otherwise iterate to count
            total = 0
            for batch in train_loader:
                if hasattr(batch, "num_graphs"):
                    total += batch.num_graphs
                elif hasattr(batch, "y"):
                    total += batch.y.shape[0]
                else:
                    total += 1
            self._cached_num_samples = total
            return total

        # Fallback to train_mask (for subgraph-FL compatibility)
        train_mask = self.splitted_data.get("train_mask")
        if train_mask is None:
            return 0
        if isinstance(train_mask, torch.Tensor):
            self._cached_num_samples = int(train_mask.sum().item())
        else:
            self._cached_num_samples = int(sum(train_mask))
        return self._cached_num_samples

    def _apply_optimizer_callback(self):
        if not getattr(self.args, "hcfl_freeze_phi", False):
            return

        with torch.no_grad():
            all_params = list(self.model.parameters())
            for idx in self.phi_indices:
                param = all_params[idx]
                if param.grad is not None:
                    param.grad.zero_()
