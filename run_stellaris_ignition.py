#!/usr/bin/env python3
"""
STELLARIS IGNITION SEQUENCE - Main execution script
Week 1: Euler-Heisenberg + Dark Photon coupling in magnetar fields
"""

import numpy as np
import matplotlib.pyplot as plt
from physics.euler_heisenberg import EulerHeisenbergVacuum
from solvers.qed_field_solver import QEDFieldSolver
from environments.magnetar import MagnetarEnvironment
from visualization.field_visualizer import FieldVisualizer

def main():
    print("🚀 STELLARIS QED ENGINE - IGNITION SEQUENCE STARTED")
    print("=" * 60)
    
    # Initialize extreme environment
    magnetar = MagnetarEnvironment(B_surface=1e15)  # 10^15 Gauss - magnetar strength
    print(f"✅ Magnetar initialized: B_surface = {magnetar.B_surface:.1e} G")
    print(f"✅ Schwinger ratio: {magnetar.schwinger_ratio:.2e}")
    
    # Create computational grid
    grid_size = 256
    x = np.linspace(-100, 100, grid_size)  # 200 km box
    y = np.linspace(-100, 100, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Initialize dipole magnetic field (magnetar configuration)
    B_field = magnetar.create_dipole_field((X, Y), center=(0, 0))
    E_field = np.zeros_like(B_field)  # Start with pure magnetic field
    
    print("✅ Computational grid initialized")
    print(f"✅ Field shape: {B_field.shape}, Grid: {grid_size}x{grid_size}")
    
    # Initialize QED solver
    dt = 1e-12  # Small timestep for stability
    solver = QEDFieldSolver(B_field.shape[1:], dt, dark_photon_mass=1e-12)
    
    # Test Euler-Heisenberg effects
    qed_vacuum = EulerHeisenbergVacuum()
    
    print("\n🧪 TESTING QED VACUUM POLARIZATION:")
    test_B = 1e15  # Gauss
    test_B_tesla = test_B * 1e-4  # Convert to Tesla
    B_test_vec = np.array([0.0, test_B_tesla, 0.0])
    E_test_vec = np.array([0.0, 0.0, 0.0])
    
    D, H = qed_vacuum.nonlinear_polarization(E_test_vec, B_test_vec)
    correction_strength = np.linalg.norm(H - B_test_vec) / np.linalg.norm(B_test_vec)
    
    print(f"   Field strength: {test_B:.1e} G")
    print(f"   QED correction: {correction_strength:.2e}")
    print(f"   Schwinger critical: 4.4e13 G")
    print(f"   Ratio to critical: {test_B/4.4e13:.1f}")
    
    # Calculate conversion probability
    conversion_prob = qed_vacuum.dark_photon_mixing_probability(
        E_test_vec, B_test_vec, dark_photon_mass=1e-12)
    print(f"   Dark photon conversion probability: {conversion_prob:.2e}")
    
    # Run short simulation
    print("\n⏳ Running QED field evolution (10 steps)...")
    energy_history = []
    conversion_history = []
    
    for step in range(10):
        E_field, B_field = solver.evolve_fields(E_field, B_field)
        energy_density = 0.5 * (np.sum(E_field**2) + np.sum(B_field**2))
        energy_history.append(energy_density)
        conversion_history.append(solver.energy_converted)
        
        if step % 2 == 0:
            print(f"   Step {step}: Energy = {energy_density:.2e}, "
                  f"Converted = {solver.energy_converted:.2e}")
    
    # Visualization
    print("\n📊 Generating diagnostics...")
    visualizer = FieldVisualizer()
    
    # Plot field configuration
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Magnetic field strength
    B_magnitude = np.sqrt(B_field[0]**2 + B_field[1]**2)
    im1 = axes[0,0].imshow(B_magnitude, cmap='plasma', extent=(-100, 100, -100, 100))
    axes[0,0].set_title('Magnetic Field Strength (Gauss)')
    plt.colorbar(im1, ax=axes[0,0])
    
    # Conversion hotspots
    conversion_map = magnetar.get_conversion_hotspots((X, Y), (E_field, B_field))
    im2 = axes[0,1].imshow(conversion_map, cmap='hot', extent=(-100, 100, -100, 100))
    axes[0,1].set_title('Dark Photon Conversion Probability')
    plt.colorbar(im2, ax=axes[0,1])
    
    # Energy history
    axes[1,0].plot(energy_history)
    axes[1,0].set_title('Total Field Energy')
    axes[1,0].set_xlabel('Time step')
    axes[1,0].set_ylabel('Energy Density')
    
    # Conversion history
    axes[1,1].plot(conversion_history)
    axes[1,1].set_title('Cumulative Dark Energy Conversion')
    axes[1,1].set_xlabel('Time step')
    axes[1,1].set_ylabel('Energy Converted')
    
    plt.tight_layout()
    plt.savefig('stellaris_ignition_diagnostics.png', dpi=150, bbox_inches='tight')
    
    print("✅ Diagnostics saved as 'stellaris_ignition_diagnostics.png'")
    
    # Conservation check
    conservation_violation = solver.get_conservation_violation()
    print(f"\n🔍 CONSERVATION LAW ANALYSIS:")
    print(f"   Energy conservation violation: {conservation_violation:.2e}")
    
    if conservation_violation < 1e-8:
        print("   ✅ Conservation laws satisfied within tolerance")
    else:
        print("   ⚠️  Significant conservation violation detected!")
        print("   This indicates either:")
        print("   - Numerical instability")
        print("   - Physics implementation error") 
        print("   - GENUINE NEW PHYSICS (unlikely but exciting!)")
    
    print(f"\n🎯 CONVERSION EVENTS DETECTED: {len(solver.conversion_events)}")
    for i, event in enumerate(solver.conversion_events[-3:]):  # Last 3 events
        print(f"   Event {i}: t={event['time']:.2e}, E={event['energy']:.2e}")
    
    print("\n🚀 STELLARIS IGNITION COMPLETE")
    print("Next: Scale to full GR + plasma coupling (Month 1 target)")

if __name__ == "__main__":
    main()
