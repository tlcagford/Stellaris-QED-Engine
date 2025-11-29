import numpy as np
from numba import jit, prange
import time

@jit(nopython=True, parallel=True)
def accelerate_field_evolution(E, B, sources, dt):
    """
    Numba-accelerated core field update
    Critical for magnetar-scale simulations
    """
    E_new = np.empty_like(E)
    B_new = np.empty_like(B)
    
    for i in prange(E.shape[1]):
        for j in prange(E.shape[2]):
            # Core field update logic here
            E_new[:, i, j] = E[:, i, j] + dt * sources[:, i, j]
            B_new[:, i, j] = B[:, i, j] - dt * (np.roll(E[:, i, j], -1) - E[:, i, j])
    
    return E_new, B_new

def benchmark_schwinger_fields():
    """Benchmark performance at Schwinger-limit field scales"""
    sizes = [128, 256, 512]
    for size in sizes:
        E = np.random.randn(3, size, size).astype(np.float64)
        B = np.random.randn(3, size, size).astype(np.float64)
        sources = np.zeros_like(E)
        
        start = time.time()
        E_new, B_new = accelerate_field_evolution(E, B, sources, 1e-12)
        elapsed = time.time() - start
        
        print(f"Grid {size}x{size}: {elapsed:.3f}s")
