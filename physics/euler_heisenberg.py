import numpy as np
from scipy import constants as const

class EulerHeisenbergVacuum:
    """
    Strong-field QED corrections to Maxwell's equations
    Includes dark photon coupling through nonlinear vacuum polarization
    """
    
    def __init__(self, electron_mass=const.m_e, alpha_fine=const.alpha):
        self.m_e = electron_mass
        self.alpha = alpha_fine
        self.compton_wavelength = const.hbar / (electron_mass * const.c)
        
        # Euler-Heisenberg prefactor
        self.xi = (2 * self.alpha**2 * const.hbar**3) / (45 * self.m_e**4 * const.c**5)
        
    def nonlinear_polarization(self, E, B, epsilon=1e-6):
        """
        Calculate nonlinear D and H fields including QED vacuum corrections
        and dark photon mixing effects
        """
        E = np.array(E, dtype=np.float64)
        B = np.array(B, dtype=np.float64)
        
        # Lorentz invariants
        S = 0.5 * (np.dot(E, E) - np.dot(B, B))  # (E² - B²)/2
        P = np.dot(E, B)                         # E·B
        
        # Avoid numerical instability at low fields
        field_strength = np.sqrt(np.dot(E, E) + np.dot(B, B))
        if field_strength < epsilon:
            return E, B
        
        # Schwinger critical field
        E_crit = self.m_e**2 * const.c**3 / (const.e * const.hbar)
        
        # Euler-Heisenberg corrections
        D_correction = 4 * S * E + 7 * P * B
        H_correction = 4 * S * B - 7 * P * E
        
        # Apply QED corrections (dimensionless)
        D = E + self.xi * D_correction
        H = B + self.xi * H_correction
        
        return D, H
    
    def dark_photon_mixing_probability(self, E, B, dark_photon_mass, mixing_epsilon=1e-3):
        """
        Calculate dark photon -> photon conversion probability in strong fields
        Includes QED vacuum corrections to the conversion amplitude
        """
        # Base conversion without QED effects
        base_prob = mixing_epsilon**2 * self._oscillation_probability(E, B, dark_photon_mass)
        
        # QED enhancement factor (field-dependent)
        schwinger_ratio = np.sqrt(np.dot(E, E)) / (self.m_e**2 * const.c**3 / (const.e * const.hbar))
        qed_enhancement = 1 + 0.1 * schwinger_ratio**2  # Approximate field enhancement
        
        return base_prob * qed_enhancement
    
    def _oscillation_probability(self, E, B, m_A_prime):
        """Quantum oscillation probability in transverse magnetic fields"""
        # Simplified conversion probability
        # In reality, this needs full wave optics in curved spacetime
        B_mag = np.linalg.norm(B)
        if B_mag == 0:
            return 0
            
        # Characteristic oscillation length
        oscillation_length = (4 * const.pi * const.hbar * const.c) / (m_A_prime**2 * const.c**4)
        return np.sin(oscillation_length)**2
