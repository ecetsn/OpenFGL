import torch.nn as nn


def infer_phi_theta_indices(model, args):
    """
    Determine which parameter indices belong to the shared feature extractor (phi)
    and which belong to the predictor head (theta).
    """
    all_params = list(model.parameters())
    theta_param_refs = None

    layers = getattr(model, "layers", None)
    if isinstance(layers, nn.ModuleList) and len(layers) > 0:
        theta_param_refs = list(layers[-1].parameters())
    elif hasattr(model, "lin2"):
        theta_param_refs = list(model.lin2.parameters())
    elif hasattr(model, "classifier"):
        theta_param_refs = list(model.classifier.parameters())
    else:
        num_head_layers = getattr(args, "num_head_layers", 2)
        theta_param_refs = all_params[-num_head_layers:]

    theta_indices = []
    for idx, param in enumerate(all_params):
        if any(param is ref for ref in theta_param_refs):
            theta_indices.append(idx)

    theta_indices = sorted(set(theta_indices))
    phi_indices = [idx for idx in range(len(all_params)) if idx not in theta_indices]
    return phi_indices, theta_indices
