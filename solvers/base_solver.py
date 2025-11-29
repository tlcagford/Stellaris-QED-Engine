import numpy as np
from abc import ABC, abstractmethod

class BaseSolver(ABC):
    """
    Abstract base class for all physical solvers
    Ensures consistent interface for coupled simulations
    """
    
    def __init__(self, grid_shape, dt):
        self.grid_shape = grid_shape
        self.dt = dt
        self.current_time = 0.0
        self.conservation_data = []
        
    @abstractmethod
    def evolve(self, fields, sources=None):
        """Evolve fields one timestep"""
        pass
    
    def check_conservation(self, fields, energy_density, momentum_density):
        """Track conservation laws - CRITICAL for our physics"""
        total_energy = np.sum(energy_density)
        total_momentum = np.sum(momentum_density)
        
        self.conservation_data.append({
            'time': self.current_time,
            'energy': total_energy,
            'momentum': total_momentum
        })
        
        return len(self.conservation_data) == 1 or abs(
            self.conservation_data[-1]['energy'] - self.conservation_data[-2]['energy']
        ) < 1e-10
    
    def get_conservation_violation(self):
        """Calculate total conservation violation"""
        if len(self.conservation_data) < 2:
            return 0.0
        energy_change = abs(self.conservation_data[-1]['energy'] - self.conservation_data[0]['energy'])
        return energy_change / self.conservation_data[0]['energy']
