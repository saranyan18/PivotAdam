import torch
from torch.optim.optimizer import Optimizer

class PivotAdam(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), 
                 eps=1e-8, weight_decay=1e-2, 
                 proj_dim=512, rotate_every=200, 
                 ablation_fake_signal=False):
       
    
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        proj_dim=proj_dim, rotate_every=rotate_every,
                        ablation_fake_signal=ablation_fake_signal)
        super().__init__(params, defaults)

    @torch.no_grad()
    def _make_low_rank_basis(self, grad_reshaped, proj_dim):
        in_features = grad_reshaped.shape[1]
        d = min(proj_dim, in_features)
        
        # QR on a random matrix yields a random orthonormal basis 
        P = torch.randn(in_features, d, device=grad_reshaped.device, dtype=torch.float32)
        P, _ = torch.linalg.qr(P)
        return P.to(grad_reshaped.dtype)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        
        for group in self.param_groups:
            weight_decay = group.get('weight_decay', 0.0)
            
            for p in group['params']:
                if p.grad is None:
                    continue

                #Apply Weight decay BEFORE momentum updates
                if weight_decay > 0.0:
                    p.data.mul_(1.0 - group['lr'] * weight_decay)

                state = self.state[p]
                
                
                #1D PARAMETER BYPASS (Biases, LayerNorms)
                
                if len(p.shape) == 1:
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p.data)
                        state['exp_avg_sq'] = torch.zeros_like(p.data)
                        
                    state['step'] += 1
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    beta1, beta2 = group['betas']
                    
                    exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).add_(p.grad.pow(2), alpha=1 - beta2)
                    
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                    step_size = group['lr'] / bias_correction1
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)
                    continue

                
                #2D PARAMETERS: PIVOT ADAM SUBSPACE LOGIC
                
                grad = p.grad
                in_features = grad.shape[-1]
                grad_reshaped = grad.view(-1, in_features)
                
                #Initialization
                if len(state) == 0:
                    state['step'] = 0
                    P = self._make_low_rank_basis(grad_reshaped, group['proj_dim'])
                    state['P'] = P
                    d = P.shape[1]
                    state['exp_avg'] = torch.zeros((grad_reshaped.shape[0], d), dtype=p.dtype, device=p.device)
                    state['exp_avg_sq'] = torch.zeros((grad_reshaped.shape[0], d), dtype=p.dtype, device=p.device)
                
                state['step'] += 1
                beta1, beta2 = group['betas']
                P = state['P']
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                #Subspace Rotation & State Transport
                if state['step'] % group['rotate_every'] == 0:
                    P_new = self._make_low_rank_basis(grad_reshaped, group['proj_dim'])
                    T = P.T @ P_new  # Transition Matrix
                    
                    T32 = T.float()
                    # First moment transport 
                    state['exp_avg'] = (exp_avg.float() @ T32).to(p.dtype)
                    # Second moment transport 
                    state['exp_avg_sq'] = (exp_avg_sq.float() @ (T32 ** 2)).to(p.dtype)
                    
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    state['P'] = P_new
                    P = P_new

                
                real_g_hat = grad_reshaped @ P

              
                #ABLATION :FAKE SIGNAL TEST
    
                if group['ablation_fake_signal']:
                    g_scale = real_g_hat.std().item() + 1e-8 
                    g_hat = torch.randn_like(real_g_hat) * g_scale
                else:
                    g_hat = real_g_hat

                # 4. Adam Updates in the latent subspace
                exp_avg.mul_(beta1).add_(g_hat, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).add_(g_hat.pow(2), alpha=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1
                u = -step_size * (exp_avg / denom)

                # 5. Unproject and Reshape back to high dimension
                delta = u @ P.T
                delta = delta.view_as(p.data)
                
                # 6. Apply update
                p.data.add_(delta)

        return loss