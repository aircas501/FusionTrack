
import torch
import torch.nn as nn


class UncertaintyWeightedLoss(nn.Module):
    """
    不确定性加权损失模块
    
    使用可学习参数ω1和ω2来动态平衡Tracking和ReID任务的损失
    """
    def __init__(self, 
                 init_omega1: float = -1.85,
                 init_omega2: float = -1.05):
        """
        Args:
            init_omega1: ω1的初始值（用于Tracking损失）
            init_omega2: ω2的初始值（用于ReID损失）
        """
        super(UncertaintyWeightedLoss, self).__init__()
        
        # 可学习参数
        self.omega1 = nn.Parameter(torch.tensor(init_omega1, dtype=torch.float32))
        self.omega2 = nn.Parameter(torch.tensor(init_omega2, dtype=torch.float32))
    
    def forward(self, tracking_loss: torch.Tensor, reid_loss: torch.Tensor):
        """
        计算不确定性加权损失
        
        Args:
            tracking_loss: Tracking损失 L_T (scalar tensor)
            reid_loss: ReID损失 L_R (scalar tensor)
        
        Returns:
            total_loss: 加权后的总损失 (scalar tensor)
            loss_dict: 包含各项损失的字典
        """
        # 计算加权损失
        # L_total = 0.5 * (e^(-ω1) * L_T + e^(-ω2) * L_R + ω1 + ω2)
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
        获取当前的权重值
        
        Returns:
            dict: 包含ω1, ω2, w1=exp(-ω1), w2=exp(-ω2)的字典
        """
        return {
            'omega1': self.omega1.item(),
            'omega2': self.omega2.item(),
            'weight1': torch.exp(-self.omega1).item(),  # e^(-ω1)
            'weight2': torch.exp(-self.omega2).item(),   # e^(-ω2)
        }


def build_uncertainty_loss(config: dict):
    """
    构建不确定性加权损失模块
    
    Args:
        config: 配置字典，需要包含：
            - UNCERTAINTY_OMEGA1_INIT: ω1初始值（可选，默认-1.85）
            - UNCERTAINTY_OMEGA2_INIT: ω2初始值（可选，默认-1.05）
    
    Returns:
        UncertaintyWeightedLoss实例
    """
    init_omega1 = config.get("UNCERTAINTY_OMEGA1_INIT", -1.85)
    init_omega2 = config.get("UNCERTAINTY_OMEGA2_INIT", -1.05)
    
    return UncertaintyWeightedLoss(init_omega1=init_omega1, init_omega2=init_omega2)

