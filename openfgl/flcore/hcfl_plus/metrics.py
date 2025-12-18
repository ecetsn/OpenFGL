import torch
import torch.nn.functional as F


class DistanceMetric:
    """
    Base interface for Tier-4 fine-grained distance metrics.
    """

    def compute(self, rep_a, rep_b, cluster_id):
        raise NotImplementedError


class GradientCosineMetric(DistanceMetric):
    """
    Distance defined by cosine dissimilarity between gradient vectors.
    """

    def compute(self, rep_a, rep_b, cluster_id):
        grad_a = rep_a.get("grads")
        grad_b = rep_b.get("grads")
        if grad_a is None or grad_b is None:
            return 1.0

        flat_a = [g.flatten() for g in grad_a if g is not None]
        flat_b = [g.flatten() for g in grad_b if g is not None]
        if not flat_a or not flat_b:
            return 1.0

        vec_a = torch.cat(flat_a, dim=0)
        vec_b = torch.cat(flat_b, dim=0)
        if vec_a.norm().item() < 1e-12 or vec_b.norm().item() < 1e-12:
            return 1.0

        sim = F.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0), dim=1)
        return 1.0 - torch.clamp(sim, -1.0, 1.0).item()


class PrototypeCosineMetric(DistanceMetric):
    """
    Prototype-based distance. Supports 'ascp' (max of proto/feature) and 'cscp' (proto only).
    """

    def __init__(self, mode="ascp"):
        self.mode = mode

    def compute(self, rep_a, rep_b, cluster_id):
        P_c_a = rep_a.get("P_c")
        P_lf_a = rep_a.get("P_lf")
        P_c_b = rep_b.get("P_c")
        P_lf_b = rep_b.get("P_lf")
        if P_c_a is None or P_c_b is None or P_lf_a is None or P_lf_b is None:
            return 1.0

        m_a = rep_a.get("membership")
        m_b = rep_b.get("membership")
        omega_mult = 1.0
        if m_a is not None and m_b is not None:
            if cluster_id < m_a.numel() and cluster_id < m_b.numel():
                omega_mult = m_a[cluster_id] * m_b[cluster_id]

        P_c_a_norm = P_c_a / P_c_a.norm(dim=1, keepdim=True).clamp_min(1e-12)
        P_c_b_norm = P_c_b / P_c_b.norm(dim=1, keepdim=True).clamp_min(1e-12)
        d_c_y = 1 - torch.clamp(F.cosine_similarity(P_c_a_norm, P_c_b_norm, dim=1), -1.0, 1.0)
        d_c = d_c_y.max()

        d_lf = 1 - torch.clamp(
            F.cosine_similarity(P_lf_a.unsqueeze(0), P_lf_b.unsqueeze(0), dim=1).squeeze(0), -1.0, 1.0
        )

        if self.mode == "cscp":
            dist_components = d_c
        else:
            dist_components = torch.max(d_c, d_lf)
        D_k = dist_components * omega_mult
        return D_k.item()


class FeatureNormMetric(DistanceMetric):
    """
    Distance based solely on normalized feature prototypes.
    """

    def compute(self, rep_a, rep_b, cluster_id):
        lf_a = rep_a.get("P_lf")
        lf_b = rep_b.get("P_lf")
        if lf_a is None or lf_b is None:
            return 1.0

        lf_a = lf_a / lf_a.norm(p=2).clamp_min(1e-12)
        lf_b = lf_b / lf_b.norm(p=2).clamp_min(1e-12)
        dist = 1 - torch.clamp(F.cosine_similarity(lf_a.unsqueeze(0), lf_b.unsqueeze(0)), -1.0, 1.0).item()

        m_a = rep_a.get("membership")
        m_b = rep_b.get("membership")
        if m_a is not None and m_b is not None:
            if cluster_id < m_a.numel() and cluster_id < m_b.numel():
                weight = (m_a[cluster_id] * m_b[cluster_id]).item()
                dist *= weight
        return dist


def get_distance_metric(args):
    metric_type = getattr(args, "hcfl_metric_type", "ascp").lower()
    if metric_type == "gradient":
        return GradientCosineMetric()
    if metric_type in ("feature", "feature_norm"):
        return FeatureNormMetric()
    return PrototypeCosineMetric(mode=metric_type)
