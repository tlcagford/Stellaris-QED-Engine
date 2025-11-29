#!/usr/bin/env python3
"""
CLOSED-LOOP SIMULATION - STELLARIS QED ENGINE
Copyright (c) 2024 Tony Eugene Ford

Dual-Licensed:
- ACADEMIC_LICENSE: Free for non-commercial use
- COMMERCIAL_LICENSE: Required for commercial use

Comprehensive simulation testing the full QED + Dark Photon system
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import os
import time

# Add project path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from physics.euler_heisenberg import EulerHeisenbergVacuum
    from solvers.qed_field_solver import QEDFieldSolver
    from environments.magnetar import MagnetarEnvironment
    from solvers.base_solver import FieldSolver
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Creating minimal implementations for simulation...")
    
    # Minimal implementations if imports fail
    class EulerHeisenbergVacuum:
        def nonlinear_polarization(self, E, B):
            return E, B
        def dark_photon_mixing_probability(self, E, B, mass):
            return 1e-10 * np.linalg.norm(B)**2

class ClosedLoopSimulation:
    """
    Complete closed-loop simulation of QED vacuum + dark photon conversion
    """
    
    def __init__(self, grid_size=128, duration=100, field_strength=1e14):
        self.grid_size = grid_size
        self.duration = duration
        self.field_strength = field_strength
        self.setup_simulation()
        
    def setup_simulation(self):
        """Initialize all simulation components"""
        print("🚀 INITIALIZING CLOSED-LOOP SIMULATION")
        print("=" * 50)
        
        # Physical parameters
        self.dt = 1e-12  # Timestep (s)
        self.total_steps = self.duration
        self.current_step = 0
        
        # Initialize components
        self.qed_vacuum = EulerHeisenbergVacuum()
        self.magnetar = MagnetarEnvironment(B_surface=self.field_strength)
        
        # Create computational grid
        self.x = np.linspace(-50, 50, self.grid_size)  # 100 km box
        self.y = np.linspace(-50, 50, self.grid_size)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Initialize fields
        self.initialize_fields()
        
        # Simulation data storage
        self.energy_history = []
        self.conversion_history = []
        self.field_history = []
        self.conservation_violation = []
        
        print(f"✅ Grid: {self.grid_size}x{self.grid_size}")
        print(f"✅ Duration: {self.duration} steps")
        print(f"✅ Field strength: {self.field_strength:.1e} G")
        
    def initialize_fields(self):
        """Initialize electromagnetic fields with magnetar configuration"""
        # Create dipole magnetic field
        self.B_field = self.magnetar.create_dipole_field((self.X, self.Y), center=(0, 0))
        
        # Add some initial electric field perturbations
        self.E_field = np.zeros_like(self.B_field)
        
        # Add noise to break symmetry and trigger dynamics
        noise_amplitude = 1e3  # Small perturbations
        self.E_field += noise_amplitude * np.random.randn(*self.E_field.shape)
        
    def calculate_energy_density(self):
        """Calculate total field energy density"""
        E_energy = 0.5 * np.sum(self.E_field**2)
        B_energy = 0.5 * np.sum(self.B_field**2)
        return E_energy + B_energy
    
    def apply_qed_corrections(self):
        """Apply Euler-Heisenberg QED corrections"""
        E_corrected = np.zeros_like(self.E_field)
        B_corrected = np.zeros_like(self.B_field)
        
        # Apply QED corrections pointwise
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                E_vec = self.E_field[:, i, j]
                B_vec = self.B_field[:, i, j]
                D, H = self.qed_vacuum.nonlinear_polarization(E_vec, B_vec)
                E_corrected[:, i, j] = D
                B_corrected[:, i, j] = H
        
        return E_corrected, B_corrected
    
    def calculate_dark_photon_conversion(self):
        """Calculate dark photon conversion across the grid"""
        conversion_map = np.zeros((self.grid_size, self.grid_size))
        total_conversion = 0.0
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                E_vec = self.E_field[:, i, j]
                B_vec = self.B_field[:, i, j]
                
                # Calculate conversion probability
                prob = self.qed_vacuum.dark_photon_mixing_probability(
                    E_vec, B_vec, dark_photon_mass=1e-12)
                
                # Convert some field energy (simplified model)
                local_energy = 0.5 * (np.sum(E_vec**2) + np.sum(B_vec**2))
                converted_energy = local_energy * prob * self.dt
                
                conversion_map[i, j] = converted_energy
                total_conversion += converted_energy
                
                # Remove converted energy from fields (energy conservation)
                if converted_energy > 0:
                    scale_factor = np.sqrt(1 - converted_energy / (local_energy + 1e-30))
                    self.E_field[:, i, j] *= scale_factor
                    self.B_field[:, i, j] *= scale_factor
        
        return conversion_map, total_conversion
    
    def evolve_fields(self):
        """Evolve electromagnetic fields one timestep"""
        # Simple wave equation evolution (replace with full Maxwell later)
        E_new = np.zeros_like(self.E_field)
        B_new = np.zeros_like(self.B_field)
        
        # Finite difference field evolution
        for comp in range(3):  # For each vector component
            # Simple wave propagation (placeholder for full Maxwell)
            E_new[comp] = self.E_field[comp] + self.dt * (
                np.roll(self.B_field[(comp+1)%3], -1, axis=0) - 
                np.roll(self.B_field[(comp+1)%3], 1, axis=0) +
                np.roll(self.B_field[(comp+2)%3], -1, axis=1) - 
                np.roll(self.B_field[(comp+2)%3], 1, axis=1)
            ) / (4.0)  # Crude finite difference
            
            B_new[comp] = self.B_field[comp] - self.dt * (
                np.roll(self.E_field[(comp+1)%3], -1, axis=0) - 
                np.roll(self.E_field[(comp+1)%3], 1, axis=0) +
                np.roll(self.E_field[(comp+2)%3], -1, axis=1) - 
                np.roll(self.E_field[(comp+2)%3], 1, axis=1)
            ) / (4.0)
        
        return E_new, B_new
    
    def check_conservation(self, initial_energy, current_energy, converted_energy):
        """Check energy conservation"""
        expected_energy = initial_energy - converted_energy
        actual_energy = current_energy
        violation = abs(actual_energy - expected_energy) / (initial_energy + 1e-30)
        
        self.conservation_violation.append(violation)
        return violation
    
    def run_simulation(self):
        """Run the complete closed-loop simulation"""
        print("\n⏳ RUNNING CLOSED-LOOP SIMULATION")
        print("=" * 50)
        
        start_time = time.time()
        total_converted_energy = 0.0
        
        for step in range(self.total_steps):
            self.current_step = step
            
            # Store initial state
            initial_energy = self.calculate_energy_density()
            
            # 1. Apply QED corrections
            self.E_field, self.B_field = self.apply_qed_corrections()
            
            # 2. Calculate dark photon conversion
            conversion_map, step_conversion = self.calculate_dark_photon_conversion()
            total_converted_energy += step_conversion
            
            # 3. Evolve fields
            self.E_field, self.B_field = self.evolve_fields()
            
            # 4. Calculate current energy
            current_energy = self.calculate_energy_density()
            
            # 5. Store data
            self.energy_history.append(current_energy)
            self.conversion_history.append(total_converted_energy)
            self.field_history.append((self.E_field.copy(), self.B_field.copy()))
            
            # 6. Check conservation
            violation = self.check_conservation(initial_energy, current_energy, step_conversion)
            
            if step % 20 == 0:
                progress = (step + 1) / self.total_steps * 100
                print(f"   Step {step:3d}/{self.total_steps} ({progress:5.1f}%) | "
                      f"Energy: {current_energy:.2e} | "
                      f"Converted: {total_converted_energy:.2e} | "
                      f"Violation: {violation:.2e}")
        
        simulation_time = time.time() - start_time
        print(f"\n✅ Simulation completed in {simulation_time:.2f} seconds")
        
        return self.analyze_results()
    
    def analyze_results(self):
        """Analyze and visualize simulation results"""
        print("\n📊 ANALYZING SIMULATION RESULTS")
        print("=" * 50)
        
        # Calculate final statistics
        initial_energy = self.energy_history[0] if self.energy_history else 0
        final_energy = self.energy_history[-1] if self.energy_history else 0
        total_converted = self.conversion_history[-1] if self.conversion_history else 0
        
        energy_conserved = abs(final_energy + total_converted - initial_energy) / initial_energy
        
        print(f"📈 Initial Energy:    {initial_energy:.2e}")
        print(f"📈 Final Energy:      {final_energy:.2e}")
        print(f"🔄 Energy Converted:  {total_converted:.2e}")
        print(f"⚖️  Energy Conservation: {energy_conserved:.2e}")
        
        if energy_conserved < 1e-8:
            print("✅ Excellent energy conservation!")
        elif energy_conserved < 1e-5:
            print("⚠️  Good energy conservation")
        else:
            print("❌ Significant energy conservation issues")
        
        self.create_diagnostic_plots()
        return self.generate_report()
    
    def create_diagnostic_plots(self):
        """Create comprehensive diagnostic plots"""
        print("\n🎨 GENERATING DIAGNOSTIC PLOTS")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('STELLARIS QED ENGINE - Closed Loop Simulation Results', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Magnetic field at final step
        B_magnitude = np.sqrt(self.B_field[0]**2 + self.B_field[1]**2 + self.B_field[2]**2)
        im1 = axes[0,0].imshow(B_magnitude, cmap='plasma', 
                              extent=(-50, 50, -50, 50))
        axes[0,0].set_title('Final Magnetic Field (Gauss)')
        axes[0,0].set_xlabel('X (km)')
        axes[0,0].set_ylabel('Y (km)')
        plt.colorbar(im1, ax=axes[0,0])
        
        # Plot 2: Energy evolution
        axes[0,1].plot(self.energy_history)
        axes[0,1].set_title('Total Field Energy Evolution')
        axes[0,1].set_xlabel('Time Step')
        axes[0,1].set_ylabel('Energy Density')
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Conversion history
        axes[0,2].plot(self.conversion_history)
        axes[0,2].set_title('Cumulative Dark Energy Conversion')
        axes[0,2].set_xlabel('Time Step')
        axes[0,2].set_ylabel('Total Energy Converted')
        axes[0,2].grid(True, alpha=0.3)
        
        # Plot 4: Conservation violation
        axes[1,0].semilogy(self.conservation_violation)
        axes[1,0].set_title('Energy Conservation Violation')
        axes[1,0].set_xlabel('Time Step')
        axes[1,0].set_ylabel('Relative Violation')
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 5: Field components over time
        time_steps = range(len(self.energy_history))
        axes[1,1].plot(time_steps, self.energy_history, label='Total Energy')
        axes[1,1].plot(time_steps, self.conversion_history, label='Converted Energy')
        axes[1,1].set_title('Energy Balance')
        axes[1,1].set_xlabel('Time Step')
        axes[1,1].set_ylabel('Energy')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # Plot 6: Conversion efficiency
        if len(self.energy_history) > 1:
            efficiency = [conv/(energy+conv) if (energy+conv) > 0 else 0 
                         for energy, conv in zip(self.energy_history, self.conversion_history)]
            axes[1,2].plot(efficiency)
            axes[1,2].set_title('Dark Photon Conversion Efficiency')
            axes[1,2].set_xlabel('Time Step')
            axes[1,2].set_ylabel('Efficiency')
            axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('closed_loop_simulation_results.png', dpi=150, bbox_inches='tight')
        print("✅ Results saved as 'closed_loop_simulation_results.png'")
        
        # Create animation of field evolution
        self.create_field_animation()
    
    def create_field_animation(self):
        """Create animation of field evolution"""
        print("🎬 Creating field evolution animation...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        def animate(frame):
            ax1.clear()
            ax2.clear()
            
            E_frame, B_frame = self.field_history[frame]
            
            # Plot magnetic field
            B_mag = np.sqrt(B_frame[0]**2 + B_frame[1]**2)
            im1 = ax1.imshow(B_mag, cmap='plasma', animated=True,
                           extent=(-50, 50, -50, 50))
            ax1.set_title(f'Magnetic Field - Step {frame}')
            ax1.set_xlabel('X (km)')
            ax1.set_ylabel('Y (km)')
            
            # Plot energy conversion
            conversion_prob = np.zeros((self.grid_size, self.grid_size))
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    E_vec = E_frame[:, i, j]
                    B_vec = B_frame[:, i, j]
                    prob = self.qed_vacuum.dark_photon_mixing_probability(
                        E_vec, B_vec, dark_photon_mass=1e-12)
                    conversion_prob[i, j] = prob
            
            im2 = ax2.imshow(conversion_prob, cmap='hot', animated=True,
                           extent=(-50, 50, -50, 50))
            ax2.set_title(f'Conversion Probability - Step {frame}')
            ax2.set_xlabel('X (km)')
            ax2.set_ylabel('Y (km)')
            
            return im1, im2
        
        # Create animation (every 5th frame for speed)
        frames = range(0, min(len(self.field_history), 100), 5)
        anim = FuncAnimation(fig, animate, frames=frames, interval=100, blit=False)
        
        anim.save('field_evolution.gif', writer='pillow', fps=10)
        print("✅ Animation saved as 'field_evolution.gif'")
        
        plt.close()
    
    def generate_report(self):
        """Generate comprehensive simulation report"""
        report = f"""
        CLOSED-LOOP SIMULATION REPORT - STELLARIS QED ENGINE
        ===================================================
        Simulation Parameters:
        - Grid Size: {self.grid_size}x{self.grid_size}
        - Duration: {self.duration} steps
        - Field Strength: {self.field_strength:.1e} G
        - Timestep: {self.dt:.1e} s
        
        Results:
        - Initial Energy: {self.energy_history[0]:.2e}
        - Final Energy: {self.energy_history[-1]:.2e}
        - Total Converted: {self.conversion_history[-1]:.2e}
        - Energy Conservation: {self.conservation_violation[-1]:.2e}
        - Peak Conversion Rate: {max(np.diff(self.conversion_history)):.2e}
        
        Status: {'✅ SUCCESS' if self.conservation_violation[-1] < 1e-5 else '⚠️ WARNING'}
        """
        return report

def main():
    """Run the closed-loop simulation"""
    print("🔬 STELLARIS QED ENGINE - CLOSED LOOP SIMULATION")
    print("=" * 60)
    
    # Run simulation with different parameters
    simulations = [
        # (grid_size, duration, field_strength)
        (64, 50, 1e13),   # Quick test
        (128, 100, 1e14), # Standard run
    ]
    
    for grid_size, duration, field_strength in simulations:
        print(f"\n🧪 Running simulation: {grid_size} grid, {duration} steps, {field_strength:.1e} G")
        
        sim = ClosedLoopSimulation(
            grid_size=grid_size,
            duration=duration, 
            field_strength=field_strength
        )
        
        report = sim.run_simulation()
        print(report)
        
        # Brief pause between simulations
        time.sleep(1)

if __name__ == "__main__":
    main()
