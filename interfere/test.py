# -*- coding: utf-8 -*-


import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.signal import hilbert, savgol_filter
import warnings
import math

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# =========================================================
#                     参数结构体
# =========================================================
@dataclass
class InterfereData:
    
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
#              高斯光谱（k 空间表达）
# =========================================================
def gaussian_spectrum_k(lambda0_nm, bandwidth_nm, N):
    """
    生成高斯波数谱 S(k)，用于模拟宽带光源。

    波数 k = 2π/λ
    高斯谱的宽度由光源带宽决定，越宽则相干长度越短。
    """
    lambda0_m = lambda0_nm * 1e-9
    delta_lambda = bandwidth_nm * 1e-9

    k0 = 2 * np.pi / lambda0_m
    delta_k = 2 * np.pi * delta_lambda / (lambda0_m**2)

    if N < 3:
        N = 3

    # 构造 k 轴（±3σ）
    k = np.linspace(k0 - 3*delta_k, k0 + 3*delta_k, N, dtype=np.float64)

    # 高斯分布
    sigma_k = delta_k / 2.355
    S = np.exp(-0.5 * ((k - k0) / sigma_k)**2)

    # 归一化光谱强度
    S /= np.sum(S)
    return k, S


# =========================================================
#                高斯光源相干长度
# =========================================================
def coherence_length(lambda0_m, bandwidth_m):
    """Lc = 0.44 * λ0^2 / Δλ（高斯光源）"""
    if bandwidth_m <= 0:
        return np.nan
    return 0.44 * lambda0_m**2 / bandwidth_m


# =========================================================
#                样品高度模型：高斯凸包 + 台阶
# =========================================================
def sample_surface(X, Y):
    """
    简单双重结构：
    1. 中心高斯凸台（模拟光学元件）
    2. X>0 区域添加 600 nm 台阶
    """
    Z = 0.4e-6 * np.exp(-( (X/50e-6)**2 + (Y/40e-6)**2 ))
    Z[X > 0] += 0.6e-6
    return 1e-6 * np.exp(-((X/30e-6)**2 + (Y/30e-6)**2))


# =========================================================
#               逐谱点干涉累加（VSI 的核心）
# =========================================================
def compute_interference_stream(OPD, k, S, I1, I2, phase0, dtype=np.float32):
    """
    对每个波数 k_i 单独计算干涉条纹并累加：
        I = I1 + I2 + 2√(I1 I2) * S(k_i) * cos(k_i * OPD + phase0)

    OPD shape: (Y, X, Scan)
    返回 Icube: (Y, X, Scan)
    """
    OPD_local = np.array(OPD, dtype=dtype, copy=False)
    out = np.zeros(OPD_local.shape, dtype=dtype)

    coef = 2.0 * math.sqrt(I1 * I2)  # 干涉项系数

    for ki, sk in zip(k, S):
        phase = OPD_local * np.float32(ki) + np.float32(phase0)
        cos_term = np.cos(phase)
        out += np.float32(sk) * (I1 + I2 + coef * cos_term)

    return out


# =========================================================
#                Savitzky-Golay 基线去除
# =========================================================
def baseline_remove_sg(Icube):
    """对扫描轴执行多项式平滑，去除 DC 趋势成分。"""
    N = Icube.shape[2]
    window = min(31, N - (1 - N%2))
    if window < 5:
        return np.zeros_like(Icube)
    return savgol_filter(Icube, window_length=window, polyorder=3, axis=2)


# =========================================================
#             Hilbert 包络提取（用于峰值定位）
# =========================================================
def get_envelope(I_ac):
    """
    Hilbert 变换 → 解析信号 → 包络（场干涉的能量包络）

    注意：VSI 中，包络峰值即为干涉中心，对应真实高度。
    """
    analytic = hilbert(I_ac, axis=2)
    env = np.abs(analytic)

    # 可选平滑
    N = env.shape[2]
    win = min(11, N - (1 - N%2))
    if win >= 5:
        env = savgol_filter(env, win, 2, axis=2)
    return env


# =========================================================
#                VSI 主流程
# =========================================================
def white_light_sim(data: InterfereData):

    if data.Seed is not None:
        np.random.seed(int(data.Seed))

    lambda0 = data.Central_wavelength * 1e-9
    bandwidth = data.Bandwidth * 1e-9

    # ---------- 生成光谱 ----------
    k, S = gaussian_spectrum_k(
        data.Central_wavelength, data.Bandwidth, data.Wavelength_samples
    )

    # ---------- 相干长度 ----------
    Lc = coherence_length(lambda0, bandwidth)
    print(f"相干长度 Lc = {Lc*1e6:.3f} µm")

    # ---------- 空间网格 ----------
    px = (np.arange(data.CCD_Pixels_X) - data.CCD_Pixels_X/2) * data.Pixel_size
    py = (np.arange(data.CCD_Pixels_Y) - data.CCD_Pixels_Y/2) * data.Pixel_size
    X, Y = np.meshgrid(px, py)

    # ---------- 样品高度 ----------
    Z = sample_surface(X, Y)

    # ---------- 扫描路径 ----------
    d_ref = np.linspace(Z.min() - 1.5*Lc, Z.max() + 1.5*Lc, data.Scan_steps)

    # ---------- OPD ----------
    OPD = 2.0 * (Z[..., None] - d_ref)

    # ---------- 干涉条纹 ----------
    Icube = compute_interference_stream(OPD, k, S, data.I1, data.I2, data.Phase_offset)
    Icube = Icube.astype(np.float64)

    # ---------- 去基线 ----------
    baseline = baseline_remove_sg(Icube)
    I_ac = Icube - baseline

    # ---------- 包络 ----------
    env = get_envelope(I_ac)

    # ---------- 峰值定位（核心结果） ----------
    idx = np.argmax(env, axis=2)
    Z_meas = d_ref[idx]

    # ---------- 误差 ----------
    error = Z_meas - Z

    # ---------- 可视化 ----------
    if data.Plot_results:
        plt.figure(figsize=(11,6))

        plt.subplot(221)
        plt.imshow(Z*1e6, cmap='jet')
        plt.title("真实高度 (µm)")
        plt.colorbar()

        plt.subplot(222)
        plt.imshow(Z_meas*1e6, cmap='jet')
        plt.title("测量高度 (µm)")
        plt.colorbar()

        plt.subplot(223)
        plt.imshow(error*1e6, cmap='bwr')
        plt.title("误差 (µm)")
        plt.colorbar()

        center = data.CCD_Pixels_Y//2
        plt.subplot(224)
        plt.plot(px*1e6, Z[center]*1e6, label='真实')
        plt.plot(px*1e6, Z_meas[center]*1e6, '--', label='测量')
        plt.title("中心剖面 (µm)")
        plt.legend()

        plt.show()

    return {
        "Z_true": Z,
        "Z_measured": Z_meas,
        "error": error,
        "Icube": Icube,
        "Envelope": env,
        "d_ref": d_ref,
        "Lc": Lc
    }


if __name__ == "__main__":
    cfg = InterfereData()
    result = white_light_sim(cfg)
    print("误差均值 (nm):", np.mean(result["error"]) * 1e9)

