# -*- coding: utf-8 -*-
from scipy.signal import hilbert, savgol_filter
from dataclasses import dataclass
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import pyqtSlot
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import math
import sys
from Ui_WhiteInterfere import Ui_WhiteInterfere 

# =========================================================
#                     参数结构体
# =========================================================
@dataclass
class InterfereData:
    # 光源参数
    Wavelength_start: float = 380.0
    Wavelength_end: float = 650.0
    Central_wavelength: float = (Wavelength_start + Wavelength_end) / 2
    Bandwidth: float = 80.0
    Wavelength_samples: int = 200
    # 扫描参数
    Scan_steps: int = 320
    # 干涉光强参数
    I1: float = 1.0
    I2: float = 1.0
    Phase_offset: float = 0.0
    # CCD 参数
    CCD_Pixels_X: int = 256
    CCD_Pixels_Y: int = 128
    Pixel_size: float = 6.5e-6
    # 臂长设置
    ReflectMirror_distance: float = 100e-3
    SampleMirror_distance: float = 100e-3
    CCD_Distance: float = 200e-3
    # 信号处理参数
    Baseline_mode: str = 'local'
    Local_baseline_halfwidth: int = 4
    Envelope_smooth_sigma: float = 1.0
    # 其他选项
    Seed: int = 42
    Plot_results: bool = False  

# =========================================================
#                  主窗口
# =========================================================
class InterfereMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_WhiteInterfere()
        self.ui.setupUi(self)
        self.ui.SureButton.clicked.connect(self.run_simulation)
        self.params = InterfereData()

        # Matplotlib Canvas
        self.fig = Figure(figsize=(5,4))
        self.canvas = FigureCanvas(self.fig)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.ui.ObjectPlot.setLayout(layout)

        # 默认参数填充
        self.fill_default_params()

    def fill_default_params(self):
        cfg = self.params
        self.ui.Wavelength_start.setText(str(cfg.Wavelength_start))
        self.ui.Wavelength_end.setText(str(cfg.Wavelength_end))
        self.ui.Wavelength_samples.setText(str(cfg.Wavelength_samples))
        self.ui.Phase_offset.setText(str(cfg.Phase_offset))
        self.ui.Scan_step_size.setText(str(cfg.Scan_steps))
        self.ui.Screen_Hight.setText(str(cfg.CCD_Pixels_Y))
        self.ui.Screen_Weight.setText(str(cfg.CCD_Pixels_X))
        self.ui.Coherence_length.setText(f"{coherence_length(cfg.Central_wavelength*1e-9, cfg.Bandwidth)*1e6:.3f}")

    @pyqtSlot()
    def run_simulation(self):
        try:
            self.params.Wavelength_start = float(self.ui.Wavelength_start.text())
            self.params.Wavelength_end = float(self.ui.Wavelength_end.text())
            self.params.Central_wavelength = (self.params.Wavelength_start + self.params.Wavelength_end)/2
            self.params.Wavelength_samples = int(self.ui.Wavelength_samples.text())
            self.params.Phase_offset = float(self.ui.Phase_offset.text())
            self.params.Scan_steps = int(self.ui.Scan_step_size.text())
            self.params.CCD_Pixels_X = int(self.ui.Screen_Weight.text())
            self.params.CCD_Pixels_Y = int(self.ui.Screen_Hight.text())
        except ValueError:
            print("输入参数有误，使用默认值")

        # 仿真
        result = white_light_sim(self.params)

        # 绘制测量高度
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        im = ax.imshow(result["Z_measured"]*1e6, cmap='jet')
        ax.set_title("测量高度 (µm)")
        self.fig.colorbar(im, ax=ax)
        self.canvas.draw()

# =========================================================
#                核心函数
# =========================================================
def gaussian_spectrum_k(lambda0_nm, bandwidth_nm, N):
    lambda0_m = lambda0_nm * 1e-9
    delta_lambda = bandwidth_nm * 1e-9
    k0 = 2*np.pi/lambda0_m
    delta_k = 2*np.pi*delta_lambda/(lambda0_m**2)
    if N<3: N=3
    k = np.linspace(k0-3*delta_k, k0+3*delta_k, N)
    sigma_k = delta_k/2.355
    S = np.exp(-0.5*((k-k0)/sigma_k)**2)
    S /= np.sum(S)
    return k,S

def coherence_length(lambda0_m, bandwidth_m):
    return 0.44*lambda0_m**2/bandwidth_m if bandwidth_m>0 else np.nan

def sample_surface(X,Y):
    # 高斯凸台 + X>0 台阶
    Z = 0.4e-6*np.exp(-( (X/50e-6)**2 + (Y/40e-6)**2 ))
    Z[X>0] += 0.6e-6
    return Z

def compute_interference_stream(OPD, k, S, I1, I2, phase0, dtype=np.float32):
    OPD_local = np.array(OPD, dtype=dtype, copy=False)
    out = np.zeros(OPD_local.shape, dtype=dtype)
    coef = 2.0 * math.sqrt(I1*I2)
    for ki, sk in zip(k,S):
        out += sk*(I1+I2 + coef*np.cos(OPD_local*ki + phase0))
    return out

def baseline_remove_sg(Icube):
    N = Icube.shape[2]
    window = min(31, N-(1-N%2))
    if window<5:
        return np.zeros_like(Icube)
    return savgol_filter(Icube, window_length=window, polyorder=3, axis=2)

def get_envelope(I_ac):
    analytic = hilbert(I_ac, axis=2)
    env = np.abs(analytic)
    N = env.shape[2]
    win = min(11, N-(1-N%2))
    if win>=5:
        env = savgol_filter(env, win, 2, axis=2)
    return env

def white_light_sim(data: InterfereData):
    if data.Seed is not None:
        np.random.seed(int(data.Seed))

    lambda0 = data.Central_wavelength*1e-9
    bandwidth = data.Bandwidth*1e-9

    k,S = gaussian_spectrum_k(data.Central_wavelength, data.Bandwidth, data.Wavelength_samples)
    Lc = coherence_length(lambda0, bandwidth)

    px = (np.arange(data.CCD_Pixels_X)-data.CCD_Pixels_X/2)*data.Pixel_size
    py = (np.arange(data.CCD_Pixels_Y)-data.CCD_Pixels_Y/2)*data.Pixel_size
    X,Y = np.meshgrid(px, py)
    Z = sample_surface(X,Y)

    d_ref = np.linspace(Z.min()-1.5*Lc, Z.max()+1.5*Lc, data.Scan_steps)
    OPD = 2*(Z[...,None]-d_ref)

    Icube = compute_interference_stream(OPD, k, S, data.I1, data.I2, data.Phase_offset)
    baseline = baseline_remove_sg(Icube)
    I_ac = Icube - baseline
    env = get_envelope(I_ac)

    idx = np.argmax(env, axis=2)
    Z_meas = d_ref[idx]
    error = Z_meas - Z

    return {
        "Z_true": Z,
        "Z_measured": Z_meas,
        "error": error,
        "Icube": Icube,
        "Envelope": env,
        "d_ref": d_ref,
        "Lc": Lc
    }

# =========================================================
#                     程序入口
# =========================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = InterfereMainWindow()
    mainWin.show()
    sys.exit(app.exec_())
