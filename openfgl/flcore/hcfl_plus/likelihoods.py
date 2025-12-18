import math
from copy import deepcopy
import torch
import torch.nn.functional as F


class ClusterLikelihood:
    """
    Base interface for Tier-1 cluster likelihood formulations.
    """

    def compute(self, adapter, data, labels, server_theta_weights, **kwargs):
        raise NotImplementedError


class ConditionalLikelihood(ClusterLikelihood):
    """
    Standard conditional likelihood: L_k = P_theta_k(y|x).
    """

    def compute(self, adapter, data, labels, server_theta_weights, **kwargs):
        return _compute_likelihood(adapter, data, labels, server_theta_weights)


class JointLikelihood(ClusterLikelihood):
    """
    FedGMM-style joint likelihood: L_k = P(x, y) = P(y|x) * P(x).
    """

    def compute(self, adapter, data, labels, server_theta_weights, **kwargs):
        gaussian_params = kwargs.get("gaussian_params")
        return _compute_likelihood(
            adapter,
            data,
            labels,
            server_theta_weights,
            gaussian_params=gaussian_params,
        )


class CorrelationLikelihood(ClusterLikelihood):
    """
    FedRC-style correlation likelihood: L_k = P(x, y) / (P(x) P(y)).
    """

    def compute(self, adapter, data, labels, server_theta_weights, **kwargs):
        gaussian_params = kwargs.get("gaussian_params")
        global_label_prior = kwargs.get("global_label_prior")
        return _compute_likelihood(
            adapter,
            data,
            labels,
            server_theta_weights,
            gaussian_params=gaussian_params,
            global_label_prior=global_label_prior,
        )


def _compute_likelihood(
    adapter,
    data,
    labels,
    server_theta_weights,
    gaussian_params=None,
    global_label_prior=None,
):
    device = labels.device
    N_i = labels.shape[0]
    K = len(server_theta_weights)
    likelihoods = torch.zeros(N_i, K, device=device)

    original_theta = deepcopy(adapter._get_theta_params(adapter._get_current_model_params()))
    adapter.model.eval()

    with torch.no_grad():
        for k in range(K):
            adapter._set_theta_params(server_theta_weights[k])
            features, logits = adapter._forward_with_features(data)
            logits = logits.to(device)
            log_p = F.log_softmax(logits, dim=1)
            idx = torch.arange(N_i, device=device)
            log_L_k = log_p[idx, labels]

            if gaussian_params is not None and k < len(gaussian_params):
                params = gaussian_params[k]
                if params is not None:
                    log_px = _gaussian_log_density(features, params)
                    log_L_k = log_L_k + log_px

            if global_label_prior is not None:
                log_py = torch.log(global_label_prior[labels].clamp_min(1e-6))
                log_L_k = log_L_k - log_py

            likelihoods[:, k] = torch.exp(log_L_k)

    adapter._set_theta_params(original_theta)
    adapter.model.train()
    return likelihoods.detach()


def _gaussian_log_density(features, params):
    if params is None:
        return torch.zeros(features.shape[0], device=features.device)
    mu = params.get("mu")
    sigma = params.get("sigma")
    if mu is None or sigma is None:
        return torch.zeros(features.shape[0], device=features.device)
    mu = mu.to(features.device)
    sigma = sigma.to(features.device)

    eps = 1e-6
    sigma = sigma.clamp_min(eps)
    diff = features - mu
    exponent = -0.5 * ((diff ** 2) / sigma).sum(dim=1)
    log_det = torch.log(sigma).sum()
    dim = mu.shape[0]
    norm_const = -0.5 * (dim * math.log(2 * math.pi) + log_det)
    return exponent + norm_const


def get_likelihood_strategy(args):
    formulation = getattr(args, "hcfl_formulation", "conditional").lower()
    if formulation == "joint":
        return JointLikelihood()
    if formulation == "correlation":
        return CorrelationLikelihood()
    return ConditionalLikelihood()
