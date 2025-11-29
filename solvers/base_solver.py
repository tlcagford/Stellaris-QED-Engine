"""
Base Solver Class for STELLARIS QED ENGINE
Copyright (c) 2024 Tony Eugene Ford (tlcagford@gmail.com)
Dual Licensed under Apache 2.0 AND MIT
"""

import numpy as np
from abc import ABC, abstractmethod
import logging

class BaseSolver(ABC):
    """
    Abstract base class for all physical solvers in the STELLARIS project
    Ensures consistent interface for coupled simulations and conservation tracking
    """
    
    def __init__(self, grid_shape, dt, solver_name="BaseSolver"):
        self.grid_shape = grid_shape
        self.dt = dt
        self.current_time = 0.0
        self.solver_name = solver_name
        self.conservation_data = []
        self.setup_logging()
        
    def setup_logging(self):
        """Initialize logging for conservation monitoring"""
        self.logger = logging.getLogger(f"{self.solver_name}")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    @abstractmethod
    def evolve(self, fields, sources=None):
        """
        Evolve fields one timestep - must be implemented by subclasses
        
        Parameters:
        fields: Current field state (tuple of arrays)
        sources: Source terms (optional)
        
        Returns:
        Updated fields after one timestep
        """
        pass
    
    def check_conservation(self, fields, energy_density, momentum_density):
        """
        Track conservation laws - CRITICAL for physics validation
        
        Parameters:
        fields: Current field arrays
        energy_density: Energy density field
        momentum_density: Momentum density field
        
        Returns:
        Boolean indicating if conservation is satisfied
        """
        total_energy = np.sum(energy_density)
        total_momentum = np.sum(momentum_density)
        
        conservation_entry = {
            'time': self.current_time,
            'energy': total_energy,
            'momentum_x': total_momentum[0] if hasattr(total_momentum, '__len__') else total_momentum,
            'momentum_y': total_momentum[1] if hasattr(total_momentum, '__len__') and len(total_momentum) > 1 else 0.0,
            'fields_shape': [f.shape for f in fields]
        }
        
        self.conservation_data.append(conservation_entry)
        
        # Check energy conservation
        if len(self.conservation_data) > 1:
            energy_change = abs(self.conservation_data[-1]['energy'] - self.conservation_data[-2]['energy'])
            energy_violation = energy_change / (self.conservation_data[-2]['energy'] + 1e-30)
            
            if energy_violation > 1e-8:
                self.logger.warning(f"Energy conservation violation: {energy_violation:.2e}")
                return False
                
        return True
    
    def get_conservation_violation(self):
        """Calculate total conservation violation over simulation"""
        if len(self.conservation_data) < 2:
            return {'energy': 0.0, 'momentum': 0.0}
        
        initial_energy = self.conservation_data[0]['energy']
        final_energy = self.conservation_data[-1]['energy']
        energy_violation = abs(final_energy - initial_energy) / (initial_energy + 1e-30)
        
        return {
            'energy': energy_violation,
            'timesteps': len(self.conservation_data),
            'duration': self.current_time
        }
    
    def get_conservation_report(self):
        """Generate detailed conservation report"""
        violation = self.get_conservation_violation()
        
        report = f"""
        CONSERVATION LAW REPORT - {self.solver_name}
        =========================================
        Total timesteps: {violation['timesteps']}
        Simulation time: {violation['duration']:.2e} s
        Energy violation: {violation['energy']:.2e}
        
        STATUS: {'✅ PASS' if violation['energy'] < 1e-10 else '❌ FAIL'}
        """
        return report
    
    def reset(self):
        """Reset solver to initial state"""
        self.current_time = 0.0
        self.conservation_data = []
        self.logger.info(f"{self.solver_name} reset")

class FieldSolver(BaseSolver):
    """
    Specialized base class for field evolution solvers
    """
    
    def __init__(self, grid_shape, dt, field_names=None):
        super().__init__(grid_shape, dt, "FieldSolver")
        self.field_names = field_names or ["Field_X", "Field_Y", "Field_Z"]
        
    def calculate_energy_density(self, fields):
        """Calculate energy density from fields - to be overridden"""
        # Default: sum of squares
        return sum(np.sum(f**2) for f in fields) / 2
    
    def calculate_momentum_density(self, fields):
        """Calculate momentum density - to be overridden"""
        # Default: zero momentum (override for specific physics)
        return np.zeros(2)
