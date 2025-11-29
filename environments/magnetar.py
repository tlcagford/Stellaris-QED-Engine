import numpy as np
from physics.euler_heisenberg import EulerHeisenbergVacuum

class MagnetarEnvironment:
    """
    Models the extreme electromagnetic environment around a magnetar
    """
    
    def __init__(self, B_surface=1e15, radius=10e3, period=10.0):
        # Magnetar parameters (realistic values)
        self.B_surface = B_surface  # Surface field in Gauss
        self.radius = radius        # Neutron star radius in cm
        self.period = period        # Rotation period in seconds
        
        # Derived quantities
        self.schwinger_ratio = self._calculate_schwinger_ratio()
        
    def _calculate_schwinger_ratio(self):
        """Calculate how close we are to Schwinger limit"""
        E_schwinger = 1.3e18  # V/m, Schwinger critical field
        E_equivalent = self.B_surface * 1e-4 * 3e8  # Convert Gauss to V/m equivalent
        return E_equivalent / E_schwinger
    
    def create_dipole_field(self, grid, center):
        """
        Create dipole magnetic field configuration
        """
        x, y = grid
        x_centered = x - center[0]
        y_centered = y - center[1]
        r = np.sqrt(x_centered**2 + y_centered**2)
        
        # Avoid division by zero
        r = np.maximum(r, self.radius)
        
        # Dipole field components
        Bx = 3 * x_centered * y_centered / r**5
        By = (3 * y_centered**2 - r**2) / r**5
        
        # Normalize to surface field strength
        Bx *= self.B_surface * (self.radius / r)**3
        By *= self.B_surface * (self.radius / r)**3
        
        return np.stack([Bx, By])
    
    def get_conversion_hotspots(self, grid, field_configuration):
        """
        Identify regions of maximum dark photon conversion probability
        """
        E, B = field_configuration
        qed = EulerHeisenbergVacuum()
        
        conversion_map = np.zeros_like(E[0])
        for i in range(E.shape[1]):
            for j in range(E.shape[2]):
                E_vec = E[:, i, j]
                B_vec = B[:, i, j]
                prob = qed.dark_photon_mixing_probability(
                    E_vec, B_vec, dark_photon_mass=1e-12)
                conversion_map[i, j] = prob
                
        return conversion_map
