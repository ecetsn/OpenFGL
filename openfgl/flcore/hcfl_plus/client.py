import torch
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
        Pull the cluster-specific model from the server and run local training.
        """
        server_payload = self.message_pool["server"]
        self.current_cluster = server_payload["cluster_assignments"][self.client_id]
        cluster_weight = server_payload["cluster_weights"][self.current_cluster]

        with torch.no_grad():
            for local_param, cluster_param in zip(self.task.model.parameters(), cluster_weight):
                local_param.data.copy_(cluster_param)

        self.task.train()

    def send_message(self):
        """
        Report the trained weights together with simple label prototypes that
        summarize the observed data distribution.
        """
        prototypes, counts = self._compute_prototypes()
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters()),
            "cluster_id": self.current_cluster,
            "prototypes": prototypes,
            "label_counts": counts,
        }

    def _compute_prototypes(self):
        """
        Derive class-wise prototype logits from the current model.
        Returns tensors on CPU to reduce communication overhead.
        """
        eval_output = self.task.evaluate(mute=True)
        logits = eval_output.get("logits")
        if logits is None:
            return None, None

        logits = logits.detach()
        if logits.dim() == 1:
            logits = logits.unsqueeze(-1)

        labels, mask = self._get_train_labels_and_mask()
        labels = labels.to(logits.device).long()
        mask = mask.to(logits.device).bool()

        selected_logits = logits[mask]
        selected_labels = labels[mask]
        num_classes = self.task.num_global_classes

        proto = torch.zeros(num_classes, logits.size(1), device=logits.device, dtype=logits.dtype)
        counts = torch.zeros(num_classes, device=logits.device, dtype=logits.dtype)

        if selected_logits.numel() > 0:
            ones = torch.ones(selected_labels.shape[0], device=logits.device, dtype=logits.dtype)
            proto.index_add_(0, selected_labels, selected_logits)
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
