PivotAdam: Memory-Efficient Full-Parameter OptimizationPivotAdam is a low-rank optimizer that enables full-parameter fine-tuning of billion-parameter models on consumer-grade hardware. By projecting gradients into dynamic, orthonormal subspaces, PivotAdam bypasses the massive memory "tax" of standard Adam optimizer states.🚀 The Breakthrough: Breaking the 8GB BarrierStandard AdamW requires $2\times$ the model’s parameters in high-precision (float32) memory for optimizer states ($m$ and $v$).The Problem: A 1.5B parameter model requires ~12GB of VRAM just for the optimizer states, making training impossible on 8GB GPUs.The Solution: PivotAdam reduces this overhead by >90%, allowing full-parameter updates of a 1.5B model using only ~7.3 GB of VRAM.💡 Core Innovations1. Randomized QR-Based ProjectionUnlike GaLore or other methods that rely on computationally expensive Singular Value Decomposition (SVD), PivotAdam utilizes Randomized QR Decomposition. This provides a perfectly orthonormal basis ($P^\top P = \mathbf{I}$) with $O(Nd)$ complexity, significantly reducing per-step overhead.2. Momentum Pivoting ($T = P^\top P_{new}$)When the subspace basis refreshes, historical momentum is preserved through a geometric transition matrix $T$. This "pivots" the existing $m$ and $v$ buffers into the new coordinate system, ensuring trajectory continuity in the latent manifold.3. Numerical Stability FixPivotAdam performs subspace transitions in float32 before casting back to the model's native precision (e.g., float16 or bfloat16). This prevents silent underflow of the variance ($v$) states during coordinate shifts—a common failure point in low-rank optimization.📊 BenchmarksTested on NVIDIA RTX 8GB (Dedicated VRAM)ModelParametersRank (d)OptimizerPeak VRAMStatusQwen-2.5-1.5B1.5B128AdamW~19.5 GB❌ OOMQwen-2.5-1.5B1.5B128PivotAdam~7.33 GB✅ SuccessEmpirical Validation (Ablation Studies)Fake Signal Test: Replacing the projected gradient with isotropic noise led to immediate loss stagnation, proving the projection successfully captures the true descent signal.Chaos Test: Forcing a basis change every step (rotate_every=1) resulted in 7x slower convergence, validating the necessity of subspace stability and the $T$-matrix rotation logic.🛠️ Quick StartPythonfrom pivot_adam import PivotAdam

# proj_dim: the rank of the gradient subspace
# rotate_every: how often to refresh the basis (steps)
optimizer = PivotAdam(
    model.parameters(), 
    lr=1e-5, 
    proj_dim=128, 
    rotate_every=100
)

# Standard training loop
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
🧬 Mathematical OverviewPivotAdam updates weights $\theta$ by projecting gradients $g_t$ into a low-rank subspace $\mathbf{P} \in \mathbb{R}^{n \times d}$:$$\hat{g}_t = g_t \mathbf{P}$$$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\hat{g}_t$$$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \mathbf{P}^\top$$When $\mathbf{P}$ is updated, we apply:$$\hat{m}_{rotated} = \hat{m}_{t-1} (P_{old}^\top P_{new})$$📝 CitationCode snippet@software{PivotAdam2026,
  author = {Saranyan M},
  title = {PivotAdam: Memory-Efficient Full-Parameter Optimization via Randomized Subspace Pivoting},
  year = {2026},
  url = {https://github.com/your-username/PivotAdam}
}
