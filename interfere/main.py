# -*- coding: utf-8 -*-
"""
极简版白光干涉 VSI 仿真（模块化增强版 + 精度增强函数）
添加内容：
- 去基线 baseline_remove()
- Hilbert 包络 envelope_hilbert()
- 自动 ±Lc 扫描范围 auto_scan_range()
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
from scipy.signal import hilbert, savgol_filter

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# =========================================================
#                      参数结构体
# =========================================================
@dataclass
class InterfereData:
    # 光源参数：用于模拟宽带或准单色光源
    Wavelength_start: float = 450.0 # 光谱起始波长（nm）
    Wavelength_end: float = 650.0 # 光谱结束波长（nm）
    Central_wavelength: float = 550.0 # 光源中心波长（nm）
    Bandwidth: float = 150.0 # 光源带宽（nm）
    Wavelength_samples: int = 160 # 光谱采样点数

    # 扫描参数：模拟台阶扫描或相移步数
    Scan_steps: int = 240 # 扫描步数

    # 干涉光强参数：两束光的强度
    I1: float = 1.0 # 参考光强
    I2: float = 1.0 # 测量光强
    Phase_offset: float = 0.0 # 初始相位偏置
    
    # CCD 参数：定义二维探测面的大小
    CCD_Pixels_X: int = 128 # CCD 水平方向像素数
    CCD_Pixels_Y: int = 64 # CCD 垂直方向像素数
    Pixel_size: float = 10e-6 # 单个像素大小（um）
    # 臂长设置：参考臂和样品臂的距离
    ReflectMirror_distance: float = 100e-3
    SampleMirror_distance: float = 100e-3
    CCD_Distance: float = 200e-3
    # 信号处理参数
    Baseline_mode: str = 'local' # 去基线方法：'local' 用局部背景
    Local_baseline_halfwidth: int = 5 # 局部背景窗口半宽，用于减噪
    Envelope_smooth_sigma: float = 1.0 # 包络平滑系数
    # 随机种子，保证可复现
    Seed = 42
    np.random.seed(Seed)
    Plot_results: bool = True # 是否绘图


# =========================================================
#             样品形貌模型（可替换）
# =========================================================
def sample_surface_m(X, Y):
    return 1e-6 * np.exp(-((X/30e-6)**2 + (Y/30e-6)**2))



# =========================================================
#             相干长度（Gaussian 近似）
# =========================================================
def coherence_length(lambda0_m: float, bandwidth_m: float) -> float:
    return lambda0_m**2 / bandwidth_m if bandwidth_m > 0 else np.nan



# =========================================================
#          NEW①：自动生成扫描范围 ±Lc
# =========================================================
def auto_scan_range(Z_sample, Lc_m, steps):
    zmin = Z_sample.min() - 1.2 * Lc_m
    zmax = Z_sample.max() + 1.2 * Lc_m
    return np.linspace(zmin, zmax, steps)



# =========================================================
#       白光干涉积分（与原版相同）
# =========================================================
def white_light_interference(OPD, wavelengths, spectrum, I1, I2, phase0):
    cos_term = np.cos(2*np.pi*OPD[..., None] / wavelengths + phase0)
    return (spectrum * (I1 + I2 + 2*np.sqrt(I1*I2) * cos_term)).sum(axis=2)



# =========================================================
#         去基线（局部滑动均值）
# =========================================================
def baseline_remove(Icube, halfwidth=5):
    """
    对干涉立方体 Icube(y,x,n) 沿 scan 方向做局部均值减法
    """
    N = Icube.shape[2]
    out = np.zeros_like(Icube)

    for k in range(N):
        i0 = max(0, k - halfwidth)
        i1 = min(N, k + halfwidth + 1)
        baseline = Icube[..., i0:i1].mean(axis=2)
        out[..., k] = Icube[..., k] - baseline

    return out



# =========================================================
#        Hilbert 包络
# =========================================================
def envelope_hilbert(Icube_ac):
    """
    输入：去基线后的 Icube_ac(y,x,n)
    输出：包络 envelope(y,x,n)
    """
    analytic = hilbert(Icube_ac, axis=2)
    envelope = np.abs(analytic)

    envelope = savgol_filter(envelope, 9, 2, axis=2)

    return envelope



# =========================================================
#                 VSI
# =========================================================
def white_light_simulation(data: InterfereData) -> Dict[str, Any]:
    if data.Seed is not None:
        np.random.seed(data.Seed)

    # ===== 波长 =====
    wavelengths = np.linspace(
        data.Wavelength_start, data.Wavelength_end, data.Wavelength_samples
    ) * 1e-9
    spectrum = np.ones_like(wavelengths)
    spectrum /= spectrum.sum()

    # ===== 相干长度 =====
    lambda0_m = data.Central_wavelength * 1e-9
    bandwidth_m = data.Bandwidth * 1e-9
    Lc = coherence_length(lambda0_m, bandwidth_m)
    print(f"相干长度 Lc = {Lc*1e6:.3f} µm")

    # ===== 空间网格 =====
    px = (np.arange(data.CCD_Pixels_X) - data.CCD_Pixels_X/2) * (data.Pixel_size)
    py = (np.arange(data.CCD_Pixels_Y) - data.CCD_Pixels_Y/2) * (data.Pixel_size)
    X, Y = np.meshgrid(px, py)

    # ===== 样品形貌 =====
    Z_sample = sample_surface_m(X, Y)
    Z_sample_um = Z_sample * 1e6

    # ===== NEW：自动 ±Lc 扫描 =====
    d_ref_array = auto_scan_range(Z_sample, Lc, data.Scan_steps)

    # ===== 干涉立方体 =====
    Icube = np.zeros((data.CCD_Pixels_Y, data.CCD_Pixels_X, data.Scan_steps))
    for i, d_ref in enumerate(d_ref_array):
        OPD = 2*(Z_sample - d_ref)
        Icube[..., i] = white_light_interference(
            OPD, wavelengths, spectrum, data.I1, data.I2, data.Phase_offset
        )

    # ===== 去基线 =====
    I_ac = baseline_remove(Icube, halfwidth=data.Local_baseline_halfwidth)

    # ===== Hilbert 包络 =====
    envelope = envelope_hilbert(I_ac)

    # ===== 峰值位置（包络的最大点） =====
    idx = np.argmax(envelope, axis=2)
    Z_measured = d_ref_array[idx]
    Z_measured_um = Z_measured * 1e6

    # ===== 误差 =====
    error_um = Z_measured_um - Z_sample_um

    # ===== 绘图 =====
    if data.Plot_results:
        plt.figure(figsize=(10, 5))

        plt.subplot(221)
        plt.imshow(Z_sample_um, cmap='jet')
        plt.title("真实高度 (µm)")
        plt.colorbar()

        plt.subplot(222)
        plt.imshow(Z_measured_um, cmap='jet')
        plt.title("测量高度 (µm)")
        plt.colorbar()

        plt.subplot(223)
        plt.imshow(error_um, cmap='jet')
        plt.title("误差 (µm)")
        plt.colorbar()

        plt.subplot(224)
        centre = Z_sample_um.shape[0] // 2
        plt.plot(px*1e6, Z_sample_um[centre], 'k-', label="真实")
        plt.plot(px*1e6, Z_measured_um[centre], 'r--', label="测量")
        plt.title("中心行剖面")
        plt.legend()
        plt.show()

    return {
        "Z_sample_um": Z_sample_um,
        "Z_measured_um": Z_measured_um,
        "error_um": error_um,
        "I_cube": Icube,
        "Envelope": envelope,
        "d_ref_array": d_ref_array,
        "Lc_um": Lc * 1e6,
        "X_um": px * 1e6
    }


# =========================================================
#                     MAIN
# =========================================================
if __name__ == "__main__":
    data = InterfereData()
    result = white_light_simulation(data)
    print("误差均值(um):", result["error_um"].mean())
