# -*- coding: utf-8 -*-
"""
菲涅尔衍射 + 单/双透镜数值模拟 + 滑动条调节传播距离 z
长度单位：米 (输入时用 e-3 表示 mm)
支持几何光学焦点计算、孔径函数，并计算成像放大率
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import cv2
from testplot import plot_optical_system
from img_data import OpticalParams
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# -----------------------------
# 读取图像并更新 Nx, Ny
# -----------------------------
def load_image(path, dx, dy):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    Ny, Nx = img.shape
    img = img.astype(np.float64)/img.max()
    return img, Nx, Ny

# -----------------------------
# 圆形孔径函数
# -----------------------------
def aperture_mask(p: OpticalParams):
    radius = p.radius_mm
    Lx = p.Nx * p.dx
    Ly = p.Ny * p.dy
    x = np.linspace(-Lx/2, Lx/2, p.Nx)
    y = np.linspace(-Ly/2, Ly/2, p.Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    R = np.sqrt(X**2 + Y**2)
    mask = np.zeros_like(R)
    mask[R <= radius] = 1.0
    return mask

# -----------------------------
# 透镜相位叠加
# -----------------------------
def lens_phase(U0, p: OpticalParams, f: float):
    k = 2*np.pi / p.wavelength
    Lx = p.Nx * p.dx
    Ly = p.Ny * p.dy
    x = np.linspace(-Lx/2, Lx/2, p.Nx)
    y = np.linspace(-Ly/2, Ly/2, p.Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    H_lens = np.exp(-1j * k * (X**2 + Y**2) / (2*f))
    mask = aperture_mask(p)
    H_lens *= mask
    return U0 * H_lens

# -----------------------------
# 菲涅尔传播 (空间域频域)
# -----------------------------
# def fresnel_propagation_fft(U0, p: OpticalParams, z):
#     """
#     使用空域卷积模拟菲涅尔衍射
#     U0: 输入图像（波前）
#     p: 光学参数
#     z: 传播距离
#     """
#     lambda_ = p.wavelength  # 波长
#     k = 2 * np.pi / lambda_  # 波数
    
#     # 创建坐标网格
#     x = np.linspace(-p.Nx * p.dx / 2, p.Nx * p.dx / 2, p.Nx)
#     y = np.linspace(-p.Ny * p.dy / 2, p.Ny * p.dy / 2, p.Ny)
#     X, Y = np.meshgrid(x, y, indexing='ij')

#     # 计算菲涅尔核

#     h = np.exp(1j*k*z)/(1j*lambda_*z)*np.exp(1j * k * (X**2 + Y**2) / (2 * z))  # 菲涅尔衍射的传播核
#     H = np.fft.fft2(np.fft.fftshift(h))

#     U0_fft = np.fft.fft2(np.fft.fftshift(U0))
#     U_fft = U0_fft * H
#     U = np.fft.ifftshift(np.fft.ifft2(U_fft))


#     return U

# -----------------------------
# 菲涅尔传播 - 传递函数法
# -----------------------------
def fresnel_propagation_fft(U0, p: OpticalParams, z, pad_factor=2):
    """
    改进的菲涅尔传播，添加零填充防止混叠
    pad_factor: 零填充倍数（2表示将尺寸扩大2倍）
    """
    lambda_ = p.wavelength
    k = 2 * np.pi / lambda_
    
    # 零填充
    Ny, Nx = U0.shape
    Ny_pad = int(Ny * pad_factor)
    Nx_pad = int(Nx * pad_factor)
    
    # 在中心放置原图，周围补零
    U0_pad = np.zeros((Nx_pad, Ny_pad), dtype=complex)
    start_x = (Nx_pad - Nx) // 2
    start_y = (Ny_pad - Ny) // 2
    U0_pad[start_x:start_x+Nx, start_y:start_y+Ny] = U0
    
    # 创建频率坐标
    fx = np.fft.fftfreq(Nx_pad, p.dx)
    fy = np.fft.fftfreq(Ny_pad, p.dy)
    FX, FY = np.meshgrid(fx, fy, indexing='ij')
    
    # 传递函数
    H = np.exp(1j * k * z) * np.exp(-1j * np.pi * lambda_ * z * (FX**2 + FY**2))
    
    # 传播计算
    U0_fft = np.fft.fft2(np.fft.ifftshift(U0_pad))
    Uz_pad = np.fft.fftshift(np.fft.ifft2(U0_fft * H))
    
    # 裁剪回原始尺寸（中心部分）
    result = Uz_pad[start_x:start_x+Nx, start_y:start_y+Ny]
    return result
# -----------------------------
# 几何光学焦点计算及放大率
# -----------------------------
def geometric_focus_and_mag(p: OpticalParams):
    if p.mode == 1:
        s = p.imageTolens
        f = p.f1
        s_prime = 1 / (1/f - 1/s)
        M = - s_prime / s
        print(f"单透镜焦点 s' = {s_prime*1e3:.1f} mm, 放大率 M = {M:.3f}")
        return s_prime, M
    elif p.mode == 2:
        # 修正双透镜几何光学计算
        # 第一个透镜的成像
        s1 = p.imageTolens
        f1 = p.f1
        s1_prime = 1 / (1/f1 - 1/s1)
        M1 = - s1_prime / s1
        
        # 第二个透镜的成像
        s2 = p.f1Tof2 - s1_prime  # 注意：这是从透镜1的像到透镜2的距离
        f2 = p.f2
        s2_prime = 1 / (1/f2 - 1/s2)
        M2 = - s2_prime / s2
        
        # 总放大率和总像距
        M = M1 * M2
        total_s_prime = p.f1Tof2 + s2_prime  # 从透镜1到最终像面的距离
        
        print(f"双透镜：")
        print(f"  透镜1像距: {s1_prime*1e3:.1f} mm, 放大率 M1 = {M1:.3f}")
        print(f"  透镜2物距: {s2*1e3:.1f} mm, 像距: {s2_prime*1e3:.1f} mm, 放大率 M2 = {M2:.3f}")
        print(f"  总像距: {total_s_prime*1e3:.1f} mm, 总放大率 M = {M:.3f}")
        
        # 返回从透镜2到像面的距离
        return s2_prime, M
    else:
        return p.z, 1.0
# -----------------------------
# 主程序
# -----------------------------
if __name__ == "__main__":
    img_path = "photo/cat.png"
    U0, Nx, Ny = load_image(img_path, dx=4e-6, dy=4e-6)
    params = OpticalParams(
        wavelength=671e-9,
        dx=6.5e-6, 
        dy=6.5e-6,
        Nx=Nx, 
        Ny=Ny,
        imageTolens=150e-3,
        f1=100e-3, 
        f2=200e-3, 
        f1Tof2=100e-3,
        z=20e-3,
        mode=1,
        radius_mm=25e-3
    )

    # ------------------ 透镜处理 ------------------
    if params.mode == 0:
        U_lens = U0
    elif params.mode == 1:
        U0_img = fresnel_propagation_fft(U0, params, params.imageTolens)
        U_lens = lens_phase(U0_img, params, params.f1)
    elif params.mode == 2:
        U1 = fresnel_propagation_fft(U0, params, params.imageTolens)
        U1 = lens_phase(U1, params, params.f1)
        U2 = fresnel_propagation_fft(U1, params, params.f1Tof2)
        U_lens = lens_phase(U2, params, params.f2)

    # ------------------ 几何光学焦点及放大率 ------------------
    z_init, mag = geometric_focus_and_mag(params)

    # ------------------ 初始传播 ------------------
    Uz = fresnel_propagation_fft(U_lens, params, z_init)
    I = np.abs(Uz)**2

    # ------------------ 显示 ------------------
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))  

    # 显示原图在左边
    ax_left.imshow(U0, cmap='gray')
    ax_left.set_title("原图")
    ax_left.axis('off')  # 关闭坐标轴显示

    # 显示传播后的图像在右边
    im = ax_right.imshow(I, cmap='gray')
    ax_right.set_title(f"像距 z = {z_init*1e3:.0f} mm, 放大率 M = {mag:.3f}")
    ax_right.axis('off')  # 关闭坐标轴显示
    plt.colorbar(im, ax=ax_right)  # 添加右侧图像的颜色条

    # ------------------ 滑条 ------------------
    z_best_mm = z_init*1e3
    z_min = max(0, z_best_mm - 100)
    z_max = z_best_mm + 100
    ax_z = plt.axes([0.25, 0.1, 0.5, 0.03])
    slider_z = Slider(ax_z, 'z (mm)', z_min, z_max, valinit=z_best_mm, valfmt='%1.0f')

    def update(val):
        z_mm = slider_z.val
        z = z_mm*1e-3
        Uz = fresnel_propagation_fft(U_lens, params, z)  # 更新传播后的图像
        I = np.abs(Uz)**2
        im.set_data(I)  # 更新右边图像
        ax_right.set_title(f"传播后强度 z = {z_mm:.0f} mm, 放大率 M = {mag:.3f}")  # 更新标题
        fig.canvas.draw_idle()  # 使画布立即更新
        
    slider_z.on_changed(update)

    plt.show()
