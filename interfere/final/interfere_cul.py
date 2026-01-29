# -*- coding: utf-8 -*-
"""
干涉仪的主要计算部分
InterfereData: 参数结构体
compute_interference_stream_mirau： 米劳式干涉计算主函数
mirau_white_light_sim： 米劳式白光干涉仪仿真主流程

"""
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
    # 光谱参数 
    Wavelength_start: float = 450.0  # 起始波长
    Wavelength_end: float = 650.0    # 终止波长
    Central_wavelength: float = (Wavelength_start + Wavelength_end) / 2  # 中心波长
    Bandwidth: float = Wavelength_end - Wavelength_start     #  光谱带宽(FWHM)
    Wavelength_samples: int = 128   #  波长采样点数
    
    # 相干参数 (单位: um )
    Lc: float = 1.512               # 相干长度
    
    # 扫描参数
    Scan_steps: int = 320           # 扫描步数
    multipleLc: float = 2    
    
    # 干涉光强参数
    I1: float = 1.0                # 参考光强度
    I2: float = 1.0                # 样品光强度
    Phase_offset: float = 0.0      # 相位偏移量
    
    # CCD 参数
    CCD_Pixels_X: int = 128        #  X方向像素数
    CCD_Pixels_Y: int = 128        #  Y方向像素数
    CCD_Distance: float = 500e-3   #  CCD到样品的距离
    Pixel_size: float = 6.5e-6     #  像素尺寸(6.5um)
    Magnification: float = 10.0   # 光学放大倍率
    
    # 臂长设置 (单位: m = 米)
    Fixed_mirror_distance: float = 100e-3      # m - 固定反射镜的距离
    Sample_postion_distance: float = 100e-3    # m - 样品反射镜的初始距离
    
    # 信号处理参数
    Baseline_mode: str = 'local'               # 基线去除模式
    Local_baseline_halfwidth: int = 4          # 局部基线半宽
    Envelope_smooth_sigma: float = 1.0         # 包络平滑参数
    
    # 其他选项
    Seed: int = 24
    Plot_results: bool = True       #  是否绘制结果
    Sample_width: float = 3e-3      # 样品宽度（X方向）
    Sample_length: float = 3e-3     #  样品长度（Y方向）
    Sample_height: float = 10e-6   #  样品高度范围（Z方向）
    Sample_mode: str = "sphere"  # 样品表面模式 gaussian_step、sphere、tilt、random、multi_step
    Nx_true: int = 1024             # X方向真实分辨率
    Ny_true: int = 1024             #  Y方向真实分辨率
    
# =========================================================
#               米劳式干涉
# =========================================================
def compute_interference_stream_mirau(OPD, k, S, I1, I2, phase0, dtype=np.float32):
    OPD_local = np.array(OPD, dtype=dtype, copy=False)
    out = np.zeros(OPD_local.shape, dtype=dtype)

    coef = 2.0 * math.sqrt(I1 * I2)  # 干涉项系数
    #=======计算光程差带来的相位变化=======
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
#             Hilbert 包络提取
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
    Z_true = sample_surface_selector(X_true, Y_true, data.Sample_mode, data.Sample_width, data.Sample_length, data.Sample_height)

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
    d_ref = np.linspace(0 - data.multipleLc*Lc, data.Sample_height + data.multipleLc*Lc, data.Scan_steps)
    # d_ref = np.linspace( - data.multipleLc*Lc,  data.multipleLc*Lc, data.Scan_steps)

    # OPD
    OPD = 2.0*(Z_meas_grid[..., None] - d_ref - (data.Sample_postion_distance - data.Fixed_mirror_distance))

    # 干涉条纹
    Icube = compute_interference_stream_mirau(OPD, k, S, data.I1, data.I2, data.Phase_offset).astype(np.float64)

    # 信号处理
    baseline = baseline_remove_sg(Icube)
    I_ac = Icube - 2
    env = get_envelope(I_ac)
    idx = np.argmax(env, axis=2)
    Z_meas = d_ref[idx]

    # 误差计算
    epsilon = 1e-12  # 很小的阈值
    abs_error = Z_meas - Z_meas_grid
    rel_error = np.zeros_like(Z_meas_grid)
    non_zero_mask = np.abs(Z_meas_grid) > epsilon
    rel_error[non_zero_mask] = abs_error[non_zero_mask] / Z_meas_grid[non_zero_mask]
    rel_error[~non_zero_mask] = 0
    error = np.abs(rel_error)

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
        "Sample_mode": data.Sample_mode,
        "I_ac": I_ac,
        "baseline": baseline
    }


def estimate_peak_spacing(Central_wavelength_nm, Bandwidth_nm):
    """
    理论估计白光干涉仪干涉峰重复出现的距离
    
    参数:
        Central_wavelength_nm : float
            光源中心波长，单位 nm
        Bandwidth_nm : float
            光源带宽（FWHM），单位 nm

    返回:
        Lc : float
            相干长度 (m)
        peak_repeat_distance : float
            干涉峰理论重复间距 (m)
    """
    # 转换为米
    lambda0 = Central_wavelength_nm * 1e-9
    delta_lambda = Bandwidth_nm * 1e-9

    # 高斯光源相干长度公式：Lc = 2 ln(2)/pi * lambda0^2 / delta_lambda
    # 简化常用近似：Lc ≈ lambda0^2 / delta_lambda
    Lc = lambda0**2 / delta_lambda  # 米

    # 对于白光干涉仪，峰值的重复距离近似为 2 * Lc
    peak_repeat_distance = 2 * Lc  # 米

    print(f"中心波长: {Central_wavelength_nm} nm, 带宽: {Bandwidth_nm} nm")
    print(f"理论相干长度 Lc ≈ {Lc*1e6:.3f} μm")
    print(f"理论干涉峰重复距离 ≈ {peak_repeat_distance*1e6:.3f} μm")

    return 0.44 *Lc, peak_repeat_distance


def calculate_visibility_map(Icube):
    """
    计算干涉图像的可见度图

    参数:
        Icube : ndarray
            三维数组，形状为 (Ny, Nx, Nscan)，表示干涉强度数据

    返回:
        visibility_map : ndarray
            二维数组，形状为 (Ny, Nx)，表示每个像素的可见度
    """
    I_max = np.max(Icube, axis=2)
    I_min = np.min(Icube, axis=2)

    # 避免除以零
    epsilon = 1e-12
    visibility_map = (I_max - I_min) / (I_max + I_min + epsilon)

    return visibility_map
# =========================================================
# 主程序入口
# =========================================================
if __name__ == "__main__":
    # 配置参数
    cfg = InterfereData()

    # 调用模拟函数并获取结果
    result = mirau_white_light_sim(cfg)

    # 从返回的结果字典中提取所需的参数
    Z_true, Z_meas, error, px, py, px_true, py_true, X_true, Y_true, Xc, Yc, Icube, env, d_ref, Lc, Sample_mode, I_ac, baseline = result.values()

    # 打印相干长度
    Lc, peak_repeat_distance = estimate_peak_spacing(cfg.Central_wavelength, cfg.Bandwidth)

    # 打印扫描路径和 Icube 的维度
    print("d_ref shape:", d_ref.shape)  
    print("Icube shape:", Icube.shape)  

    print("Scan range:", np.min(d_ref), "to", np.max(d_ref))

    # 选择中心点的强度，并绘制光强随扫描路径变化的曲线
    center_x = cfg.CCD_Pixels_X // 2
    center_y = cfg.CCD_Pixels_Y // 2

    # 获取原始信号（Icube）、基线矫正后的信号（I_ac）以及包络（env）
    intensity_at_center = Icube[center_y, center_x, :]
    intensity_baseline_corrected = I_ac[center_y, center_x, :]
    envelope_at_center = env[center_y, center_x, :]

    # 绘制图形
    plt.figure(figsize=(10, 6))

    # 绘制基线矫正前的信号
    plt.plot(d_ref, intensity_at_center, label='基线矫正前的强度', color='gray', alpha=0.7)

    # 绘制基线矫正后的信号
    plt.plot(d_ref, intensity_baseline_corrected, label='基线矫正后的强度', color='blue', linewidth=2)

    # 绘制包络
    plt.plot(d_ref, envelope_at_center, label='包络', color='red', linestyle='--', linewidth=2)

    #绘制基线
    plt.plot(d_ref, baseline[center_y, center_x, :], label='基线', color='green', linestyle=':', linewidth=2)

    # 设置标签和标题
    plt.xlabel("扫描路径 (d_ref)")
    plt.ylabel("强度")
    plt.title("中心点强度与包络，基线矫正前后对比")
    plt.legend()

    # 显示图形
    plt.grid(True)
    plt.show()

    visibility_map = calculate_visibility_map(Icube)
    plt.figure(figsize=(6, 5))
    plt.imshow(visibility_map, extent=(px[0]*1e3, px[-1]*1e3, py[0]*1e3, py[-1]*1e3), cmap='jet')
    plt.colorbar(label='可见度')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.title('干涉图像可见度图')
    plt.show()
