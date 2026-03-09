"""
VG-SPV DPO Trainer: DPOTrainer subclass with a single override point for VG-fDPO loss.

For now, compute_loss() delegates to the standard DPO loss. Later it can be
replaced with VG-fDPO (Visually-Grounded Fine-Grained DPO) loss (e.g. combining DPO with
vision-guided rewards from models/reward_dino.py).
"""

from trl import DPOTrainer


class VGSPVTrainer(DPOTrainer):
    """
    DPO trainer for VG-SPV. Override compute_loss() to plug in VG-fDPO
    (Visually-Grounded Fine-Grained DPO) loss while keeping the rest of the training loop unchanged.
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss for this batch. Currently uses standard DPO loss.
        Override this method to implement VG-fDPO (Visually-Grounded Fine-Grained DPO) loss.
        """
        return super().compute_loss(model, inputs, return_outputs=return_outputs)
