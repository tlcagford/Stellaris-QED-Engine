#!/usr/bin/env python3
"""
Quick test of QED effects in magnetar-strength fields
"""

import numpy as np
import matplotlib.pyplot as plt
from physics.euler_heisenberg import EulerHeisenbergVacuum
from environments.magnetar import MagnetarEnvironment

def test_schwinger_limit():
    """Test QED effects near Schwinger limit"""
    print("=== Testing QED Effects at Schwinger Limit ===")
    
    qed = EulerHeisenbergVacuum()
    magnetar = MagnetarEnvironment(B_surface=1e15)
    
    # Test field strengths from lab to magnetar scales
    test_fields = [1e6, 1e9, 1e12, 1e15]  # Gauss
    
    for B_gauss in test_fields:
        B = np.array([0.0, 0.0, B_gauss * 1e-4])  # Convert to Tesla
        E = np.array([0.0, 0.0, 0.0])
        
        D, H = qed.nonlinear_polarization(E, B)
        
        # Calculate correction strength
        correction = np.linalg.norm(D - E) / np.linalg.norm(E) if np.linalg.norm(E) > 0 else 0
        schwinger_ratio = B_gauss / 4.4e13  # Schwinger limit in Gauss
        
        print(f"B = {B_gauss:.1e} G | Schwinger ratio: {schwinger_ratio:.2e} | "
              f"QED correction: {correction:.2e}")

if __name__ == "__main__":
    test_schwinger_limit()
    
    # Quick visualization
    qed = EulerHeisenbergVacuum()
    B_fields = np.logspace(10, 15, 50)  # 10^10 to 10^15 Gauss
    conversions = []
    
    for B in B_fields:
        B_vec = np.array([0.0, B * 1e-4, 0.0])  # Tesla
        E_vec = np.array([0.0, 0.0, 0.0])
        prob = qed.dark_photon_mixing_probability(E_vec, B_vec, 1e-12)
        conversions.append(prob)
    
    plt.loglog(B_fields, conversions)
    plt.xlabel('Magnetic Field (Gauss)')
    plt.ylabel('Dark Photon Conversion Probability')
    plt.title('QED-Enhanced Dark Photon Conversion')
    plt.grid(True)
    plt.savefig('qed_conversion_vs_field.png')
    print("Plot saved as 'qed_conversion_vs_field.png'")
