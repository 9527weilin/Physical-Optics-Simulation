# Physical Optics Simulation Suite (PyOpticalSim)

**PyOpticalSim** is a comprehensive Python-based GUI suite designed to simulate and visualize fundamental physical optics phenomena. Built with **PyQt5** for the interface and **Matplotlib** for high-quality rendering, this project provides interactive tools for education and research in optics.

## 🚀 Features & Modules

This repository contains four independent simulation applications:

### 1. Diffraction Simulator (`diffractionaApp.py`)
Compare Analytical solutions against Numerical (FFT) solutions for various diffraction apertures.
* **Apertures:** Circular, Rectangular, and Grating.
* **Analysis:** Real-time calculation of **Relative Error**, **MSE**, and **RMS** between analytical and FFT models.
* **Visualization:** Switch between Electric Field ($E$) and Intensity ($I$); 1D cross-sections and 2D heatmaps.

### 2. Fresnel Equations & Polarization (`fresnel_app.py`)
Visualize light behavior at optical interfaces.
* **Coefficients:** Plot Reflection/Transmission amplitudes ($r_s, r_p, t_s, t_p$) and Power ($R, T$) vs. Incident Angle.
* **Phase:** Analyze phase shifts ($\Delta \phi$) relative to the incident angle.
* **Polarization:** Real-time visualization of polarization states (Linear, Circular, Elliptical) using **Stokes Parameters**.
* **Key Angles:** Automatic calculation of Brewster's Angle and Critical Angle.

### 3. White Light Interferometry (`intefereApp.py`)
Simulation of a Mirau-type White Light Interferometer (WLI) for 3D surface topography.
* **Surface Generation:** Create virtual samples (Step, Sphere, Tilt, Random Roughness).
* **Process:** Simulate vertical scanning (VSI), coherence envelopes, and phase recovery.
* **Reconstruction:** Compare Measured Topography ($Z_{meas}$) vs. True Topography ($Z_{true}$) with error mapping.

### 4. Lens Imaging System (`imgApp.py`)
Simulate image formation through optical lens systems using Wave Optics (Fresnel Diffraction).
* **Setup:** Support for No-Lens, Single-Lens, and Double-Lens configurations.
* **Propagation:** FFT-based Fresnel propagation method for accurate wave evolution.
* **Metrics:** Image quality assessment using **PSNR**, **MSE**, **RMSE**, and Correlation Coefficient.
* **Interactive:** Optical path diagrams and Z-axis propagation scanning.

---

## 🛠️ Installation

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install numpy matplotlib PyQt5 scipy