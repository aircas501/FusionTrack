
import torch
import torch.nn as nn


class UncertaintyWeightedLoss(nn.Module):
    """
    Uncertainty-weighted loss module.

    Uses learnable parameters omega1 and omega2 to dynamically balance Tracking and ReID losses.
    """
    def __init__(self, 
                 init_omega1: float = -1.85,
                 init_omega2: float = -1.05):
        """
        Args:
            init_omega1: Initial value of omega1 (for Tracking loss)
            init_omega2: Initial value of omega2 (for ReID loss)
        """
        super(UncertaintyWeightedLoss, self).__init__()
        
        # Learnable parameters
        self.omega1 = nn.Parameter(torch.tensor(init_omega1, dtype=torch.float32))
        self.omega2 = nn.Parameter(torch.tensor(init_omega2, dtype=torch.float32))
    
    def forward(self, tracking_loss: torch.Tensor, reid_loss: torch.Tensor):
        """
        Compute uncertainty-weighted loss.

        Args:
            tracking_loss: Tracking loss L_T (scalar tensor)
            reid_loss: ReID loss L_R (scalar tensor)

        Returns:
            total_loss: Weighted total loss (scalar tensor)
            loss_dict: Dict containing individual loss terms
        """
        # Compute weighted loss
        # L_total = 0.5 * (e^(-omega1) * L_T + e^(-omega2) * L_R + omega1 + omega2)
        weighted_tracking = torch.exp(-self.omega1) * tracking_loss
        weighted_reid = torch.exp(-self.omega2) * reid_loss
        regularization = self.omega1 + self.omega2
        
        total_loss = 0.5 * (weighted_tracking + weighted_reid + regularization)
        
        loss_dict = {
            'total_loss': total_loss,
            'weighted_tracking': weighted_tracking,
            'weighted_reid': weighted_reid,
            'regularization': regularization,
            'omega1': self.omega1,
            'omega2': self.omega2,
            'raw_tracking': tracking_loss,
            'raw_reid': reid_loss
        }
        
        return total_loss, loss_dict
    
    def get_weights(self):
        """
        Get current weight values.

        Returns:
            dict: Contains omega1, omega2, w1=exp(-omega1), w2=exp(-omega2)
        """
        return {
            'omega1': self.omega1.item(),
            'omega2': self.omega2.item(),
            'weight1': torch.exp(-self.omega1).item(),  # e^(-omega1)
            'weight2': torch.exp(-self.omega2).item(),   # e^(-omega2)
        }


def build_uncertainty_loss(config: dict):
    """
    Build uncertainty-weighted loss module.

    Args:
        config: Config dict containing:
            - UNCERTAINTY_OMEGA1_INIT: initial omega1 (optional, default -1.85)
            - UNCERTAINTY_OMEGA2_INIT: initial omega2 (optional, default -1.05)

    Returns:
        UncertaintyWeightedLoss instance
    """
    init_omega1 = config.get("UNCERTAINTY_OMEGA1_INIT", -1.85)
    init_omega2 = config.get("UNCERTAINTY_OMEGA2_INIT", -1.05)
    
    return UncertaintyWeightedLoss(init_omega1=init_omega1, init_omega2=init_omega2)
