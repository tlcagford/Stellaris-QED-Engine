import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class QuantumVacuumGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("STELLARIS QED ENGINE - Quantum Vacuum Monitor")
        self.root.geometry("1400x900")
        
        self.setup_main_frame()
        self.setup_control_panels()
        self.setup_visualization()
        self.setup_alerts()
        
    def setup_main_frame(self):
        # Main notebook for different subsystems
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tabs for different systems
        self.vacuum_tab = ttk.Frame(self.notebook)
        self.magnet_tab = ttk.Frame(self.notebook) 
        self.detection_tab = ttk.Frame(self.notebook)
        self.conversion_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.vacuum_tab, text='Vacuum System')
        self.notebook.add(self.magnet_tab, text='Magnet Control')
        self.notebook.add(self.detection_tab, text='Photon Detection')
        self.notebook.add(self.conversion_tab, text='Dark Photon Conversion')
    
    def setup_control_panels(self):
        # Vacuum Control Panel
        vacuum_frame = ttk.LabelFrame(self.vacuum_tab, text="Vacuum Status", padding=10)
        vacuum_frame.pack(fill='x', padx=5, pady=5)
        
        self.pressure_var = tk.StringVar(value="2.3×10⁻⁹ mbar")
        ttk.Label(vacuum_frame, text="Chamber Pressure:").grid(row=0, column=0)
        ttk.Label(vacuum_frame, textvariable=self.pressure_var, 
                 font=('Arial', 12, 'bold')).grid(row=0, column=1)
        
        # Magnet Control Panel  
        magnet_frame = ttk.LabelFrame(self.magnet_tab, text="Magnetic Field Control", padding=10)
        magnet_frame.pack(fill='x', padx=5, pady=5)
        
        self.field_var = tk.DoubleVar(value=0.0)
        field_scale = ttk.Scale(magnet_frame, from_=0, to=15, 
                               variable=self.field_var, orient='horizontal')
        field_scale.grid(row=0, column=0, columnspan=2, sticky='ew')
        
        ttk.Label(magnet_frame, text="Field Strength (T):").grid(row=1, column=0)
        ttk.Label(magnet_frame, textvariable=self.field_var).grid(row=1, column=1)
    
    def setup_visualization(self):
        # Real-time field visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Magnetic field map
        x = np.linspace(-1, 1, 50)
        y = np.linspace(-1, 1, 50)
        X, Y = np.meshgrid(x, y)
        B = np.sqrt(X**2 + Y**2)
        
        im = ax1.contourf(X, Y, B, levels=20)
        ax1.set_title('Magnetic Field Map')
        plt.colorbar(im, ax=ax1)
        
        # Conversion probability
        conversion_prob = np.exp(-B**2) * B**2
        im2 = ax2.contourf(X, Y, conversion_prob, levels=20)
        ax2.set_title('Dark Photon Conversion Probability')
        plt.colorbar(im2, ax=ax2)
        
        self.canvas = FigureCanvasTkAgg(fig, self.conversion_tab)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def setup_alerts(self):
        # Alert system for conservation violations
        self.alert_frame = ttk.LabelFrame(self.root, text="Conservation Law Monitor", padding=10)
        self.alert_frame.pack(fill='x', padx=10, pady=5)
        
        self.energy_var = tk.StringVar(value="ΔE/E = 2.3×10⁻¹²")
        self.momentum_var = tk.StringVar(value="Δp/p = 1.7×10⁻¹⁴")
        
        ttk.Label(self.alert_frame, text="Energy Conservation:").grid(row=0, column=0)
        ttk.Label(self.alert_frame, textvariable=self.energy_var, 
                 foreground='green').grid(row=0, column=1)
        
        ttk.Label(self.alert_frame, text="Momentum Conservation:").grid(row=1, column=0)
        ttk.Label(self.alert_frame, textvariable=self.momentum_var,
                 foreground='green').grid(row=1, column=1)

# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = QuantumVacuumGUI(root)
    root.mainloop()
