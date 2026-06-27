
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReversibilityWeightLearner(nn.Module):
    """
    可逆性损失的可学习权重模块
    
    使用不确定性加权方法，让模型自动学习两个可逆性损失的相对重要性：
    - Level 1: Query ↔ Embedding 可逆性
    - Level 2: ReID特征稳定性
    """
    
    def __init__(self, init_weight1=0.1, init_weight2=0.1):
        """
        Args:
            init_weight1: Level 1 可逆性损失的初始权重
            init_weight2: Level 2 可逆性损失的初始权重
        """
        super(ReversibilityWeightLearner, self).__init__()
        
        # 可学习的log方差参数
        # log_var 初始化：使得 exp(-log_var) ≈ init_weight
        # 即：log_var = -log(init_weight)
        self.log_var1 = nn.Parameter(torch.tensor(-torch.log(torch.tensor(init_weight1))))
        self.log_var2 = nn.Parameter(torch.tensor(-torch.log(torch.tensor(init_weight2))))
        
        print("="*80)
        print("可逆性损失可学习权重模块:")
        print(f"  初始权重1 (Query↔Embedding): {init_weight1:.4f}")
        print(f"  初始权重2 (ReID稳定性): {init_weight2:.4f}")
        print(f"  log_var1 初始化: {self.log_var1.item():.4f}")
        print(f"  log_var2 初始化: {self.log_var2.item():.4f}")
        print("="*80)
    
    def forward(self, rev_loss1, rev_loss2):
        """
        前向传播：应用可学习权重
        
        Args:
            rev_loss1: Level 1 可逆性损失（Query↔Embedding）
            rev_loss2: Level 2 可逆性损失（ReID特征稳定性）
        
        Returns:
            weighted_loss: 加权后的总损失
            loss_dict: 详细的损失信息
        """
        # ⭐ 防止权重爆炸：裁剪log_var到合理范围
        # log_var ∈ [-10, 10] => weight ∈ [exp(-10), exp(10)] ≈ [0.00005, 22026]
        log_var1_clipped = torch.clamp(self.log_var1, min=-10.0, max=10.0)
        log_var2_clipped = torch.clamp(self.log_var2, min=-10.0, max=10.0)
        
        # 计算加权损失
        # weighted_loss = exp(-log_var) * loss + log_var
        # 这样模型会自动平衡两个损失的重要性
        
        # ⭐ 添加数值稳定性检查：如果损失接近0，使用固定权重
        eps = 1e-6
        if rev_loss1.item() < eps and rev_loss2.item() < eps:
            # 两个损失都接近0（例如线性映射），使用固定小权重
            weighted_loss1 = 0.01 * rev_loss1
            weighted_loss2 = 0.01 * rev_loss2
        else:
            weighted_loss1 = torch.exp(-log_var1_clipped) * rev_loss1 + log_var1_clipped
            weighted_loss2 = torch.exp(-log_var2_clipped) * rev_loss2 + log_var2_clipped
        
        total_weighted_loss = weighted_loss1 + weighted_loss2
        
        # 返回详细信息（用于日志）
        loss_dict = {
            'weighted_rev_loss1': weighted_loss1.item(),
            'weighted_rev_loss2': weighted_loss2.item(),
            'total_weighted_rev_loss': total_weighted_loss.item(),
            'weight1': torch.exp(-log_var1_clipped).item(),  # 实际权重
            'weight2': torch.exp(-log_var2_clipped).item(),  # 实际权重
            'log_var1': self.log_var1.item(),
            'log_var2': self.log_var2.item(),
        }
        
        return total_weighted_loss, loss_dict
    
    def get_weights(self):
        """
        获取当前的权重值（用于监控）
        
        Returns:
            dict: 权重信息
        """
        return {
            'weight1': torch.exp(-self.log_var1).item(),
            'weight2': torch.exp(-self.log_var2).item(),
            'log_var1': self.log_var1.item(),
            'log_var2': self.log_var2.item(),
        }


def build_reversibility_weight_learner(config):
    """
    构建可逆性损失权重学习器
    
    Args:
        config: 配置字典
    
    Returns:
        ReversibilityWeightLearner 实例
    """
    init_weight1 = config.get("REVERSIBILITY_WEIGHT1", 0.1)
    init_weight2 = config.get("REVERSIBILITY_WEIGHT2", 0.1)
    
    learner = ReversibilityWeightLearner(
        init_weight1=init_weight1,
        init_weight2=init_weight2
    )
    
    return learner


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 配置
    config = {
        "REVERSIBILITY_WEIGHT1": 0.1,
        "REVERSIBILITY_WEIGHT2": 0.1,
    }
    
    # 创建权重学习器
    learner = build_reversibility_weight_learner(config)
    learner.train()
    
    # 模拟损失
    rev_loss1 = torch.tensor(0.05, requires_grad=True)
    rev_loss2 = torch.tensor(0.03, requires_grad=True)
    
    # 前向传播
    weighted_loss, loss_dict = learner(rev_loss1, rev_loss2)
    
    print(f"\n初始状态:")
    print(f"  原始 rev_loss1: {rev_loss1.item():.6f}")
    print(f"  原始 rev_loss2: {rev_loss2.item():.6f}")
    print(f"  权重1: {loss_dict['weight1']:.4f}")
    print(f"  权重2: {loss_dict['weight2']:.4f}")
    print(f"  加权后总损失: {weighted_loss.item():.6f}")
    
    # 模拟训练
    optimizer = torch.optim.Adam(learner.parameters(), lr=0.01)
    
    print(f"\n模拟训练10步:")
    for step in range(10):
        optimizer.zero_grad()
        
        # 随机损失（模拟）
        rev_loss1 = torch.tensor(0.05 + torch.randn(1).item() * 0.01, requires_grad=True)
        rev_loss2 = torch.tensor(0.03 + torch.randn(1).item() * 0.01, requires_grad=True)
        
        weighted_loss, loss_dict = learner(rev_loss1, rev_loss2)
        weighted_loss.backward()
        optimizer.step()
        
        if step % 3 == 0:
            weights = learner.get_weights()
            print(f"  Step {step}: w1={weights['weight1']:.4f}, w2={weights['weight2']:.4f}, loss={weighted_loss.item():.6f}")
    
    print(f"\n最终权重:")
    final_weights = learner.get_weights()
    print(f"  权重1: {final_weights['weight1']:.4f}")
    print(f"  权重2: {final_weights['weight2']:.4f}")
