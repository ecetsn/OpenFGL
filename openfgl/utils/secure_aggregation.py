import torch


def mask_tensor(tensor, mask_scale=1.0):
    """
    Placeholder masking for secure aggregation.
    Returns a masked tensor and the mask used.
    """
    if tensor is None:
        return None, None
    mask = torch.randn_like(tensor) * mask_scale
    return tensor + mask, mask


def unmask_tensor(masked_tensor, mask):
    """
    Placeholder unmasking for secure aggregation.
    """
    if masked_tensor is None or mask is None:
        return masked_tensor
    return masked_tensor - mask
