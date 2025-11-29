#!/usr/bin/env python3
"""
CREATE FIELD EVOLUTION ANIMATION - STELLARIS QED ENGINE
Generates field_evolution.gif
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os

print("🎬 Creating Field Evolution Animation...")

# Create the figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('STELLARIS QED ENGINE - Real-time Field Evolution\nDark Photon Conversion Dynamics', 
             fontsize=14, fontweight='bold')

# Create coordinate grid
x = np.linspace(-50, 50, 150)
y = np.linspace(-50, 50, 150)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
theta = np.arctan2(Y, X)

# Precompute field evolution data
frames = 30
field_data = []
conversion_data = []

print("📊 Generating frame data...")
for i in range(frames):
    # Time-dependent field pulsation
    time_factor = 0.7 + 0.3 * np.sin(i * 0.3)
    
    # Magnetic field evolution (pulsing dipole)
    B_r = 1e14 * 2 * np.cos(theta) / (R**3 + 1e-6) * time_factor
    B_theta = 1e14 * np.sin(theta) / (R**3 + 1e-6) * time_factor
    B_x = B_r * np.cos(theta) - B_theta * np.sin(theta)
    B_y = B_r * np.sin(theta) + B_theta * np.cos(theta)
    B_magnitude = np.sqrt(B_x**2 + B_y**2)
    
    field_data.append(B_magnitude)
    
    # Conversion probability evolution (follows field strength with phase lag)
    conversion_prob = (B_magnitude / 1e14)**4 * 1e-8 * (0.8 + 0.2 * np.sin(i * 0.3 + 1))
    conversion_prob[R > 45] = 0
    conversion_data.append(conversion_prob)

# Animation function
def animate(frame):
    ax1.clear()
    ax2.clear()
    
    # Left panel: Magnetic field
    im1 = ax1.imshow(field_data[frame], cmap='plasma', 
                    extent=(-50, 50, -50, 50), 
                    vmin=1e10, vmax=1e14)
    ax1.set_title(f'Magnetic Field Evolution\nFrame {frame+1}/{frames}', fontweight='bold')
    ax1.set_xlabel('X Position (km)')
    ax1.set_ylabel('Y Position (km)')
    
    # Right panel: Conversion probability
    im2 = ax2.imshow(conversion_data[frame], cmap='hot', 
                    extent=(-50, 50, -50, 50), 
                    vmin=0, vmax=1e-8)
    ax2.set_title(f'Dark Photon Conversion\nProbability Distribution', fontweight='bold')
    ax2.set_xlabel('X Position (km)')
    ax2.set_ylabel('Y Position (km)')
    
    # Add progress indicator
    fig.text(0.5, 0.02, f'Time: {frame * 0.1:.1f} ns | Conversion Active: {"✅" if frame > 5 else "⏳"}', 
             ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    return [im1, im2]

print("🎞️ Rendering animation...")
# Create animation
anim = FuncAnimation(fig, animate, frames=frames, interval=200, blit=False)

# Save as GIF
output_path = 'field_evolution.gif'
try:
    writer = PillowWriter(fps=5)
    anim.save(output_path, writer=writer, dpi=150)
    print(f"✅ SAVED: {output_path}")
except Exception as e:
    print(f"❌ GIF creation failed: {e}")
    # Fallback: save last frame
    plt.savefig('field_evolution_static.png', dpi=150, bbox_inches='tight')
    print("✅ Saved static frame as backup")

plt.close()

# Verify
if os.path.exists(output_path):
    file_size = os.path.getsize(output_path) / 1024
    print(f"📏 Animation size: {file_size:.1f} KB")
    print("🎉 Animation ready! Drag and drop into your project.")
else:
    print("❌ Animation file not created")
