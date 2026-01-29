# -*- coding: utf-8 -*-

import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.signal import hilbert, savgol_filter
from scipy.interpolate import RegularGridInterpolator
from sample_surface import sample_surface_selector
from interfereLight import LightSource, generate_spectrum, coherence_length
import math

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# =========================================================
#                     参数结构体
# =========================================================
@dataclass
class InterfereData:
    # 光谱参数 (单位: nm = 纳米 = 10e-9米)
    Wavelength_start: float = 380.0  # nm - 起始波长
    Wavelength_end: float = 650.0    # nm - 终止波长
    Central_wavelength: float = (Wavelength_start + Wavelength_end) / 2  # nm - 中心波长
    Bandwidth: float = Wavelength_end - Wavelength_start     # nm - 光谱带宽(FWHM)
    Wavelength_samples: int = 64     # 无量纲 - 波长采样点数
    
    # 相干参数 (单位: um = 微米 = 10e-6米)
    Lc: float = 0.432              # um - 相干长度
    
    # 扫描参数
    Scan_steps: int = 320           # 无量纲 - 扫描步数
    multipleLc: float = 6.0          # 无量纲 - 扫描范围为多少个相干长度
    
    # 干涉光强参数
    I1: float = 1.0                # 无量纲 - 参考光强度
    I2: float = 1.0                # 无量纲 - 样品光强度
    Phase_offset: float = 0.0      # rad - 相位偏移量
    
    # CCD 参数
    CCD_Pixels_X: int = 256        # 无量纲 - X方向像素数
    CCD_Pixels_Y: int = 128        # 无量纲 - Y方向像素数
    CCD_Distance: float = 500e-3   # m - CCD到样品的距离
    Pixel_size: float = 6.5e-6     # m - 像素尺寸(6.5um)
    Magnification: float = 10.0   # 光学放大倍率
    
    # 臂长设置 (单位: m = 米)
    Fixed_mirror_distance: float = 100e-3      # m - 固定反射镜的距离
    Sample_postion_distance: float = 100e-3    # m - 样品反射镜的初始距离
    
    # 信号处理参数
    Baseline_mode: str = 'local'               # 无量纲 - 基线去除模式
    Local_baseline_halfwidth: int = 4          # 无量纲 - 局部基线半宽
    Envelope_smooth_sigma: float = 1.0         # 无量纲 - 包络平滑参数
    
    # 其他选项
    Seed = 24
    Plot_results: bool = True       # 无量纲 - 是否绘制结果
    Sample_width: float = 3e-3      # m - 样品宽度（X方向）
    Sample_length: float = 3e-3     # m - 样品长度（Y方向）
    Sample_mode: str = "gaussian_step"  # 无量纲 - 样品表面模式
    Nx_true: int = 1024             # 无量纲 - X方向真实分辨率
    Ny_true: int = 1024             # 无量纲 - Y方向真实分辨率
    
# =========================================================
#               米劳式干涉
# =========================================================
def compute_interference_stream_mirau(OPD, k, S, I1, I2, phase0, dtype=np.float32):
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
    N = Icube.shape[2]
    window = min(31, N - (1 - N%2))
    if window < 5:
        return np.zeros_like(Icube)
    return savgol_filter(Icube, window_length=window, polyorder=3, axis=2)


# =========================================================
#             Hilbert 包络提取（用于峰值定位） 
# =========================================================
def get_envelope(I_ac):
    analytic = hilbert(I_ac, axis=2)
    env = np.abs(analytic)

    # 可选平滑
    N = env.shape[2]
    win = min(11, N - (1 - N%2))
    if win >= 5:
        env = savgol_filter(env, win, 2, axis=2)
    return env

# =========================================================
#                米劳式白光干涉仪仿真主流程
# =========================================================
def mirau_white_light_sim(data: InterfereData):
    np.random.seed(data.Seed)

    # 光谱
    light = LightSource("gaussian", data.Central_wavelength, data.Bandwidth, data.Wavelength_samples)
    k, S = generate_spectrum(light)
    Lc = coherence_length(data.Central_wavelength*1e-9, data.Bandwidth*1e-9)

    # 高分辨率真实样品
    px_true = np.linspace(-data.Sample_width/2, data.Sample_width/2, data.Nx_true)
    py_true = np.linspace(-data.Sample_length/2, data.Sample_length/2, data.Ny_true)
    X_true, Y_true = np.meshgrid(px_true, py_true)
    Z_true = sample_surface_selector(X_true, Y_true, data.Sample_mode, data.Sample_width, data.Sample_length)

    #CCD 采样
    px = np.linspace(-data.Sample_width/2, data.Sample_width/2, data.CCD_Pixels_X)
    py = np.linspace(-data.Sample_length/2, data.Sample_length/2, data.CCD_Pixels_Y)
    interp = RegularGridInterpolator((py_true, px_true), Z_true, bounds_error=False, fill_value=np.nan)
    Xc, Yc = np.meshgrid(px, py)
    pts = np.stack([Yc.ravel(), Xc.ravel()], axis=-1)
    Z_meas_grid = interp(pts).reshape(data.CCD_Pixels_Y, data.CCD_Pixels_X)


    # # 对 CCD 像素做面积平均 (物空间)
    # pixel_obj_size = data.Pixel_size / data.Magnification
    # Nx_pixel = data.CCD_Pixels_X
    # Ny_pixel = data.CCD_Pixels_Y
    # Z_meas_grid = np.zeros((Ny_pixel, Nx_pixel), dtype=np.float64)
    # Z_meas = np.zeros((Ny_pixel, Nx_pixel), dtype=np.float64)
    # px = np.linspace(-data.Sample_width/2, data.Sample_width/2, Nx_pixel)
    # py = np.linspace(-data.Sample_height/2, data.Sample_height/2, Ny_pixel)
    # interp = RegularGridInterpolator((py_true, px_true), Z_true, bounds_error=False, fill_value=np.nan)
    # Xc, Yc = np.meshgrid(px, py)

    # for i in range(Ny_pixel):
    #     for j in range(Nx_pixel):
    #         x0 = px[j] - pixel_obj_size/2
    #         x1 = px[j] + pixel_obj_size/2
    #         y0 = py[i] - pixel_obj_size/2
    #         y1 = py[i] + pixel_obj_size/2
    #         x_sample = np.linspace(x0, x1, 4)
    #         y_sample = np.linspace(y0, y1, 4)
    #         Xs, Ys = np.meshgrid(x_sample, y_sample)
    #         pts = np.stack([Ys.ravel(), Xs.ravel()], axis=-1)
    #         Z_meas_grid[i,j] = np.nanmean(interp(pts))
    # 扫描路径
    d_ref = np.linspace(-data.multipleLc*Lc, data.multipleLc*Lc, data.Scan_steps)

    # OPD
    OPD = 2.0*(Z_meas_grid[..., None] - d_ref - (data.Sample_postion_distance - data.Fixed_mirror_distance))

    # 干涉条纹
    Icube = compute_interference_stream_mirau(OPD, k, S, data.I1, data.I2, data.Phase_offset).astype(np.float64)

    # 信号处理
    baseline = baseline_remove_sg(Icube)
    I_ac = Icube - baseline
    env = get_envelope(I_ac)
    idx = np.argmax(env, axis=2)
    Z_meas = d_ref[idx]
    error = (Z_meas - Z_meas_grid)/Z_meas_grid

    return {
        "Z_true": Z_true,
        "Z_measured": Z_meas,
        "error": error,
        "px": px,
        "py": py,
        "px_true": px_true,
        "py_true": py_true,
        "X_true": X_true,
        "Y_true": Y_true,
        "Xc": Xc,
        "Yc": Yc,
        "Icube": Icube,
        "Envelope": env,
        "d_ref": d_ref,
        "Lc": Lc,
        "Sample_mode": data.Sample_mode
    }




# =========================================================
# 主程序入口
# =========================================================
if __name__ == "__main__":
    # 配置参数
    cfg = InterfereData()

    # 调用模拟函数并获取结果
    result = mirau_white_light_sim(cfg)

    # 从返回的结果字典中提取所需的参数
    Z_true, Z_meas, error, px, py, px_true, py_true, X_true, Y_true, Xc, Yc, Icube, env, d_ref, Lc, Sample_mode = result.values()

    # 打印误差均值
    print("误差均值 (nm):", np.mean(error) * 1e9)

    # 打印相干长度
    print("相干长度 Lc = {:.3f} um".format(Lc * 1e6))
    # 打印扫描路径和 Icube 的维度
    print("d_ref shape:", d_ref.shape)  # 应该是 (Scan_steps,)
    print("Icube shape:", Icube.shape)  # 应该是 (CCD_Pixels_Y, CCD_Pixels_X, Scan_steps)

    # 确保扫描路径覆盖了足够的范围
    print("Scan range:", np.min(d_ref), "to", np.max(d_ref))

    # 选择中心点的强度，并绘制光强随扫描路径变化的曲线
    center_x = cfg.CCD_Pixels_X // 2
    center_y = cfg.CCD_Pixels_Y // 2
    intensity_at_center = Icube[center_y, center_x, :]

    # 绘图
    plt.plot(d_ref, intensity_at_center, label='Intensity at center')
    plt.xlabel("Scan Path (d_ref)")
    plt.ylabel("Intensity")
    plt.title("Interference Intensity at center point vs. Scan Path")
    plt.legend()
    plt.show()
