
from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any
import numpy as np

@dataclass
class EngineConfig:
    grid_size: int = 128
    steps: int = 1000
    precision: Literal['float32','float64'] = 'float64'
    dt: float = 1e-6
    seed: Optional[int] = None
    device: Literal['cpu','gpu'] = 'cpu'
    dry_run: bool = True  # default safe

def make_rng(seed: Optional[int]):
    import numpy as _np
    return _np.random.default_rng(seed)
        _____ _______ _      _       _____ ____  ______ _____ 
        / ____|__   __| |    | |     |  __ \___ \|  ____|  __ \
       | (___    | |  | |    | |     | |__) |__) | |__  | |__) |
        \___ \   | |  | |    | |     |  _  /  _ /|  __| |  _  / 
        ____) |  | |  | |____| |____ | | \ \ |_| | |____| | \ \
       |_____/   |_|  |______|______||_|  \_\____/|______|_|  \_\
       
        Q U A N T U M   V A C U M   E N G I N E E R I N G
    
STELLARIS QED ENGINE - IGNITION SEQUENCE STARTED
============================================================
Initial peak B = 1.41e+15 G
Initial pulse amp = 1.00e+12 (arbitrary units)
   Step  0 → Energy 1.28e+31 | Converted 0.00e+00
   Step 10 → Energy 1.28e+31 | Converted 2.14e+15
   Step 20 → Energy 1.28e+31 | Converted 4.28e+15
   Step 30 → Energy 1.28e+31 | Converted 6.42e+15
   Step 40 → Energy 1.28e+31 | Converted 8.56e+15
   Step 49 → Energy 1.28e+31 | Converted 1.05e+16

Computing GR null geodesics...

Diagnostics saved → stellaris_diagnostics_v0_4_0.png

============================================================
STELLARIS IGNITION SEQUENCE COMPLETED SUCCESSFULLY!
============================================================
Total dark energy converted: 1.05e+16 (arbitrary units)
GR coupling enabled • Geodesic ray tracing + bending • Ready for plasma dynamics
