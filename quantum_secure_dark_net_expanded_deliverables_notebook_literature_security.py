# Expanded deliverables for Quantum-Secure-Dark-Net
# This document contains the three expanded artifacts you asked for:
# 1) simulations/decoherence_calc.ipynb (notebook JSON) — save as .ipynb in simulations/
# 2) literature/papers curated list + README (with DOIs/arXiv ids) — save into literature/
# 3) security/model.md (adversary model + security proof sketch) — save into security/

# ------------------------- (1) JUPYTER NOTEBOOK JSON -------------------------
# Save the JSON below to simulations/decoherence_calc.ipynb
notebook_json = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Decoherence simulator — interactive notebook\n",
    "\n",
    "Interactive notebook that exposes the toy Hamiltonian and a phenomenological decoherence function.\n",
    "Use the widgets to sweep parameters (g, m, Q, eta, background noise) and export plots."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from scipy.linalg import expm\n",
    "import matplotlib.pyplot as plt\n",
    "from ipywidgets import interact, FloatLogSlider, FloatSlider, IntSlider\n",
    "\n",
    "hbar = 1.0\n",
    "def decoherence_rate_from_params(g, m_ev, Q, eta, background_noise):\n",
    "    eV_to_J = 1.602176634e-19\n",
    "    hbar_si = 1.054571817e-34\n",
    "    omega_d = (m_ev * eV_to_J) / hbar_si\n",
    "    gamma0 = 1e3\n",
    "    gamma = gamma0 * (omega_d / (1 + Q)) * (1.0 / (1 + g)) * (1.0 / max(1e-6, eta)) * (1 + background_noise)\n",
    "    return gamma\n",
    "\n",
    "def plot_decoherence(g, m_ev, Q, eta, background_noise):\n",
    "    gamma = decoherence_rate_from_params(g, m_ev, Q, eta, background_noise)\n",
    "    distances = np.logspace(2, 7, 200)\n",
    "    c = 3e8\n",
    "    t_prop = distances / c\n",
    "    P = 1.0 - np.exp(-gamma * t_prop)\n",
    "    plt.figure(figsize=(8,4))\n",
    "    plt.loglog(distances, P)\n",
    "    plt.xlabel('distance (m)')\n",
    "    plt.ylabel('decoherence probability P')\n",
    "    plt.title(f'gamma={gamma:.3e}  (g={g}, m={m_ev} eV, Q={Q}, eta={eta})')\n",
    "    plt.grid(True, which='both', ls=':')\n",
    "    plt.show()\n",
    "\n",
    "interact(plot_decoherence,\n",
    "         g=FloatLogSlider(value=1e-2, base=10, min=-6, max=2, step=0.1, description='g'),\n",
    "         m_ev=FloatLogSlider(value=1e-22, base=10, min=-24, max=-10, step=0.1, description='m (eV)'),\n",
    "         Q=IntSlider(value=10000, min=100, max=10000000, step=100, description='Q'),\n",
    "         eta=FloatSlider(value=0.2, min=1e-6, max=1.0, step=0.01, description='eta'),\n",
    "         background_noise=FloatSlider(value=1.0, min=0.0, max=100.0, step=0.1, description='noise'))\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.x"}
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

# Save the JSON to file instructions (paste into terminal on your machine):
notebook_save_instructions = (
"mkdir -p simulations && python - <<'PY'\n"
"import json\n"
"nb = " + repr(notebook_json) + "\n"
"with open('simulations/decoherence_calc.ipynb','w') as f:\n"
"    json.dump(nb, f)\n"
"PY\n"
)

# ------------------------- (2) LITERATURE EXPANDED ---------------------------
literature_expanded = '''
# literature/README.md (expanded)

This folder contains curated canonical papers you should include in `literature/papers/`.

Papers (suggested file names):

1) yin2017_micius_entanglement.pdf
   - Yin, J. et al., "Satellite-based entanglement distribution over 1200 kilometers," Science 2017.
     DOI: 10.1126/science.aan3211
   - Notes: experimental entanglement distribution, link budgets, atmospheric channel models.

2) liao2017_micius_qkd.pdf
   - Liao, S.-K. et al., "Satellite-to-ground quantum key distribution," Nature 2017.
     DOI: 10.1038/nature23655
   - Notes: practical QKD payloads, rates, and authenticated key exchange.

3) admx_overview_latest.pdf
   - ADMX Collaboration (multiple): review recent ADMX/haloscope instrumentation papers (search arXiv for latest results).
   - Notes: cavity designs, Q factors, quantum-limited amplification chains.

4) jaeckel2010_low_energy_frontier.pdf
   - Jaeckel & Ringwald, "The Low-Energy Frontier of Particle Physics", Ann. Rev. Nucl. Part. Sci. 2010.
     arXiv:1002.0329
   - Notes: review of hidden-photon models and kinetic mixing constraints.

5) hui2017_ultralight_scalars.pdf
   - Hui et al., "Ultralight scalars as cosmological dark matter", Phys. Rev. D 2017. arXiv:1610.08297
   - Notes: fuzzy DM properties and astrophysical constraints.

6) sikivie1983_axion_tests.pdf
   - Sikivie, P., "Experimental Tests of the 'Invisible' Axion", Phys. Rev. Lett. 1983. DOI:10.1103/PhysRevLett.51.1415
   - Notes: Primakoff conversion theory in cavities.

7) recent_dark_photon_limits.pdf
   - Collect HPS, APEX, NA64 papers (search arXiv/INSPIRE) and include their arXiv IDs/DOIs.

Usage:
- Place PDFs in `literature/papers/` and create a short note file per paper in `literature/notes/` explaining the key parameters
  it constrains (e.g., link loss, Q, coupling g limits).
'''

# ------------------------- (3) SECURITY MODEL FILE --------------------------
security_model_md = '''
# security/model.md

(Contents identical to the security_model provided earlier)

# Security model & proof sketch — Quantum-Secure Dark Net

This file sketches a defensible adversary model and the steps required to convert physical assumptions
into an information-theoretic security statement. It is intentionally conservative and lists the physical
assumptions that must be validated experimentally.

## 1) Adversary model (Eve)
Assume Eve has the following capabilities unless proven otherwise by experiment:
- Can place arbitrary measurement devices on classical light channels (free-space, fiber) and perform joint
  quantum operations on any captured photons with access to arbitrary quantum computers (Shor/Grover-capable).
- Can attempt to intercept or probe any physical apparatus in the public domain (ground stations, satellites)
  but cannot magically bypass physically secured hardware if tamper-evident and authenticated.
- **Unknown dark-sector access**: Eve may have access to devices tuned to detect dark-sector excitations if
  those excitations couple to standard model fields above some coupling threshold g_Eve. Define g_Eve as the
  maximum coupling strength an adversary can exploit given their best practical sensors.

## 2) Physical assumptions (to be validated experimentally)
- A1: There exists a converter that maps a visible/IR photon mode to a dark-sector excitation with coupling g and
  efficiency η_conv, preserving quantum coherence to fidelity F_conv.
- A2: The dark-sector excitation propagates with negligible coupling to standard detectors along the link (coupling
  << g_Eve), or if coupling exists it is bounded and can be included in the security proof.
- A3: Receivers can reconvert dark excitations back to photonic modes with efficiency η_rec and fidelity F_rec.
- A4: Classical authentication channels exist and are secure (standard public-key or pre-shared symmetric methods)
  to prevent MitM at the classical control level.

## 3) Security proof sketch
If A1–A4 hold with experimentally demonstrated parameters, follow this path to an information-theoretic proof:

1. **Model the physical channel** as a quantum channel \(\mathcal{E}_{g,\eta}\) parameterized by coupling g and efficiencies.
   Derive a bound on the mutual information \(I(E;K)\) between Eve's best measurement outcomes and the generated key K
   as a function of g and detector noise limits.

2. **Parameter estimation**: During key generation runs, perform tomography/estimation on a subset of exchanged
   entangled pairs or test states to estimate channel parameters (error rates, losses). This is analogous to BB84 parameter
   estimation but with the additional physical parameters (g, η_conv, η_rec).

3. **Key rate bound**: Using Devetak–Winter formula (or equivalent entropic bounds), compute secure key rate R_secure
   = S(A|E) - leak_EC, where S(A|E) is the conditional entropy given Eve's accessible information, which must be upper-bounded
   using the channel model and parameter estimates.

4. **Composable security**: Use a composable framework (e.g., universal composability) to ensure security when keys are used in larger protocols.

## 4) Required experimental deliverables for a rigorous security claim
- Measured bounds on g, η_conv, η_rec, dark-sector background coupling, and noise spectral density.
- Repeatable parameter estimation protocol and observed error rates across many trials.
- Proof that any residual information accessible to Eve given bounded g_Eve is below epsilon (choose epsilon small enough for application).

## 5) Authentication & practical matters
- Classical authentication is mandatory: the system must use authenticated classical channels to avoid MiTM attacks on control messages.
- Physical tamper-evidence, supply-chain security for converters, and hardware attestation are necessary engineering controls.
'''

# ------------------------- WRITE-INSTRUCTIONS ---------------------------------
write_instructions = '''
I created this expanded deliverables document in the canvas. To pick these up into your repo:

1) Copy the JSON printed under `notebook_json` and save it as `simulations/decoherence_calc.ipynb`.
   Or, run the small Python snippet in `notebook_save_instructions` to write it automatically on a machine with Python.

2) Create `literature/README.md` and `literature/papers/` and paste the `literature_expanded` contents into the README;
   download the referenced PDFs (DOIs/arXiv) into the papers folder.

3) Create `security/model.md` and paste the `security_model_md` contents.

If you want, I can now (choose one or more):
- produce the actual `.ipynb` file here and attach a download link (I can generate a runnable notebook and provide it),
- or commit these files as a git patch you can apply.

Tell me which and I'll generate it now.
'''

# End of expanded deliverables document
print('Expanded deliverables prepared — save the notebook JSON and the README/security files into your repo as instructed.')
