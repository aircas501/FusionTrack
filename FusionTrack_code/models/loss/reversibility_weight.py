
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReversibilityWeightLearner(nn.Module):
    """
    Learnable weight module for reversibility losses.

    Uses uncertainty weighting so the model learns the relative importance of:
    - Level 1: Query <-> Embedding reversibility
    - Level 2: ReID feature stability
    """
    
    def __init__(self, init_weight1=0.1, init_weight2=0.1):
        """
        Args:
            init_weight1: Initial weight for Level 1 reversibility loss
            init_weight2: Initial weight for Level 2 reversibility loss
        """
        super(ReversibilityWeightLearner, self).__init__()
        
        # Learnable log-variance parameters
        # log_var init: exp(-log_var) ≈ init_weight, i.e. log_var = -log(init_weight)
        self.log_var1 = nn.Parameter(torch.tensor(-torch.log(torch.tensor(init_weight1))))
        self.log_var2 = nn.Parameter(torch.tensor(-torch.log(torch.tensor(init_weight2))))
        
        print("="*80)
        print("Learnable reversibility loss weight module:")
        print(f"  Initial weight1 (Query<->Embedding): {init_weight1:.4f}")
        print(f"  Initial weight2 (ReID stability): {init_weight2:.4f}")
        print(f"  log_var1 init: {self.log_var1.item():.4f}")
        print(f"  log_var2 init: {self.log_var2.item():.4f}")
        print("="*80)
    
    def forward(self, rev_loss1, rev_loss2):
        """
        Forward pass: apply learnable weights.

        Args:
            rev_loss1: Level 1 reversibility loss (Query <-> Embedding)
            rev_loss2: Level 2 reversibility loss (ReID feature stability)

        Returns:
            weighted_loss: Weighted total loss
            loss_dict: Detailed loss information
        """
        # Prevent weight explosion: clip log_var to a reasonable range
        # log_var in [-10, 10] => weight in [exp(-10), exp(10)] ≈ [0.00005, 22026]
        log_var1_clipped = torch.clamp(self.log_var1, min=-10.0, max=10.0)
        log_var2_clipped = torch.clamp(self.log_var2, min=-10.0, max=10.0)
        
        # Compute weighted loss
        # weighted_loss = exp(-log_var) * loss + log_var
        # so the model automatically balances the importance of the two losses
        
        # Numerical stability: if both losses are near zero, use fixed weights
        eps = 1e-6
        if rev_loss1.item() < eps and rev_loss2.item() < eps:
            # Both losses near zero (e.g. linear mapping); use small fixed weights
            weighted_loss1 = 0.01 * rev_loss1
            weighted_loss2 = 0.01 * rev_loss2
        else:
            weighted_loss1 = torch.exp(-log_var1_clipped) * rev_loss1 + log_var1_clipped
            weighted_loss2 = torch.exp(-log_var2_clipped) * rev_loss2 + log_var2_clipped
        
        total_weighted_loss = weighted_loss1 + weighted_loss2
        
        # Return details for logging
        loss_dict = {
            'weighted_rev_loss1': weighted_loss1.item(),
            'weighted_rev_loss2': weighted_loss2.item(),
            'total_weighted_rev_loss': total_weighted_loss.item(),
            'weight1': torch.exp(-log_var1_clipped).item(),  # Effective weight
            'weight2': torch.exp(-log_var2_clipped).item(),  # Effective weight
            'log_var1': self.log_var1.item(),
            'log_var2': self.log_var2.item(),
        }
        
        return total_weighted_loss, loss_dict
    
    def get_weights(self):
        """
        Get current weight values (for monitoring).

        Returns:
            dict: Weight information
        """
        return {
            'weight1': torch.exp(-self.log_var1).item(),
            'weight2': torch.exp(-self.log_var2).item(),
            'log_var1': self.log_var1.item(),
            'log_var2': self.log_var2.item(),
        }


def build_reversibility_weight_learner(config):
    """
    Build reversibility loss weight learner.

    Args:
        config: Config dict

    Returns:
        ReversibilityWeightLearner instance
    """
    init_weight1 = config.get("REVERSIBILITY_WEIGHT1", 0.1)
    init_weight2 = config.get("REVERSIBILITY_WEIGHT2", 0.1)
    
    learner = ReversibilityWeightLearner(
        init_weight1=init_weight1,
        init_weight2=init_weight2
    )
    
    return learner


# ==================== Usage example ====================
if __name__ == "__main__":
    # Config
    config = {
        "REVERSIBILITY_WEIGHT1": 0.1,
        "REVERSIBILITY_WEIGHT2": 0.1,
    }
    
    # Create weight learner
    learner = build_reversibility_weight_learner(config)
    learner.train()
    
    # Simulated losses
    rev_loss1 = torch.tensor(0.05, requires_grad=True)
    rev_loss2 = torch.tensor(0.03, requires_grad=True)
    
    # Forward pass
    weighted_loss, loss_dict = learner(rev_loss1, rev_loss2)
    
    print(f"\nInitial state:")
    print(f"  Raw rev_loss1: {rev_loss1.item():.6f}")
    print(f"  Raw rev_loss2: {rev_loss2.item():.6f}")
    print(f"  Weight1: {loss_dict['weight1']:.4f}")
    print(f"  Weight2: {loss_dict['weight2']:.4f}")
    print(f"  Weighted total loss: {weighted_loss.item():.6f}")
    
    # Simulated training
    optimizer = torch.optim.Adam(learner.parameters(), lr=0.01)
    
    print(f"\nSimulating 10 training steps:")
    for step in range(10):
        optimizer.zero_grad()
        
        # Random losses (simulation)
        rev_loss1 = torch.tensor(0.05 + torch.randn(1).item() * 0.01, requires_grad=True)
        rev_loss2 = torch.tensor(0.03 + torch.randn(1).item() * 0.01, requires_grad=True)
        
        weighted_loss, loss_dict = learner(rev_loss1, rev_loss2)
        weighted_loss.backward()
        optimizer.step()
        
        if step % 3 == 0:
            weights = learner.get_weights()
            print(f"  Step {step}: w1={weights['weight1']:.4f}, w2={weights['weight2']:.4f}, loss={weighted_loss.item():.6f}")
    
    print(f"\nFinal weights:")
    final_weights = learner.get_weights()
    print(f"  Weight1: {final_weights['weight1']:.4f}")
    print(f"  Weight2: {final_weights['weight2']:.4f}")
