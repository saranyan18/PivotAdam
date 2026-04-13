# PivotAdam
Pivot Adam is a high-efficiency, low-rank optimizer designed to bypass the O(N) memory "tax" of standard Adam. By projecting gradients into a dynamic, orthonormal subspace and utilizing a novel momentum-rotation mechanism, Pivot Adam enables full-parameter fine-tuning of 1.1B+ models on consumer hardware with as little as 8GB of VRAM
