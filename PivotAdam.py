import torch
from torch.optim.optimizer import Optimizer

class CompressedAdam(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), 
                 eps=1e-4, # Increased from 1e-8 for float16 stability
                 proj_dim=128, rotate_every=200):
        """
        Layer-wise Compressed Adam.
        Each parameter tensor gets its own tiny subspace.
        """
        defaults = dict(lr=lr, betas=betas, eps=eps, 
                        proj_dim=proj_dim, rotate_every=rotate_every)
        super().__init__(params, defaults)

    @torch.no_grad()
    def _make_low_rank_basis(self, grad_reshaped, proj_dim):
        """grad_reshaped is [Out, In_features_flattened]"""
        in_features = grad_reshaped.shape[1]
        d = min(proj_dim, in_features)
        
        # 1. Create and QR in float32 for numerical stability
        P = torch.randn(in_features, d, device=grad_reshaped.device, dtype=torch.float32)
        P, _ = torch.linalg.qr(P)
        
        # 2. Cast to match the actual gradient (e.g., float16)
        return P.to(grad_reshaped.dtype)

   
    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                grad = p.grad
                original_shape = grad.shape
                # Flatten the 'input' dimensions: [Out, In, K, K] -> [Out, In*K*K]
                grad_reshaped = grad.view(original_shape[0], -1)

                # 1. Initialize state for this specific layer (ONLY ONCE)
                if len(state) == 0:
                    state['step'] = 0
                    state['P'] = self._make_low_rank_basis(grad_reshaped, group['proj_dim'])
                    d = state['P'].shape[1]
                    
                    # Correctly set device and dtype to match the model (float16/Half)
                    state['exp_avg'] = torch.zeros(
                        grad_reshaped.shape[0], d, 
                        device=p.device, 
                        dtype=p.dtype
                    )
                    state['exp_avg_sq'] = torch.zeros(
                        grad_reshaped.shape[0], d, 
                        device=p.device, 
                        dtype=p.dtype
                    )

                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                P = state['P']
                beta1, beta2 = group['betas']

                # 2. Rotation logic
                if group['rotate_every'] > 0 and state['step'] % group['rotate_every'] == 0:
                    P_new = self._make_low_rank_basis(grad_reshaped, group['proj_dim'])
                    T = P.T @ P_new 
                    state['exp_avg'] = exp_avg @ T
                    # Math is performed in Half precision
                    state['exp_avg_sq'] = (exp_avg_sq.sqrt() @ T).pow(2)
                    state['P'] = P_new
                    P = P_new

                # 3. Project Gradient
                g_hat = grad_reshaped @ P

                # 4. Adam Updates in the low-rank space
                exp_avg.mul_(beta1).add_(g_hat, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).add_(g_hat.pow(2), alpha=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1
                u = -step_size * (exp_avg / denom)

                # 5. Unproject and Reshape back
                delta = u @ P.T
                p.data.add_(delta.view(original_shape))

        return loss