import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# =====================================================
# 生成不同类型的透射屏
# =====================================================
def generate_aperture(aperture_type, Nx=128, Ny=128):
    x = np.linspace(-1, 1, Nx)
    y = np.linspace(-1, 1, Ny)
    X, Y = np.meshgrid(x, y)
    aperture = np.zeros((Ny, Nx))

    if aperture_type == "circle":
        R = 0.5
        aperture = ((X**2 + Y**2) <= R**2).astype(float)

    elif aperture_type == "rect":
        aperture = np.zeros_like(X)
        aperture[np.abs(X) < 0.4] = 1.0
        aperture[np.abs(Y) < 0.2] = aperture[np.abs(Y) < 0.2]

    elif aperture_type == "grating":
        period_pix = 10
        line = (np.mod(np.arange(Nx), period_pix) < period_pix/2).astype(float)
        aperture = np.tile(line, (Ny, 1))

    else:
        raise ValueError("aperture_type must be circle/rect/grating")

    return aperture


# =====================================================
# Fraunhofer 直接积分（双重求和）
# =====================================================
def fraunhofer_direct(aperture, dx, wavelength, z):
    """
    直接积分实现（严格公式）
    U(fx,fy) = exp(i k z) / (i λ z) * ∬ A(x,y) exp(-i 2π (fx x + fy y)) dx dy
    其中 fx,fy = frequency = cycles/m (和 np.fft.fftfreq 对齐)
    """
    Ny, Nx = aperture.shape

    # 物理坐标（源面）
    x_real = (np.arange(Nx) - Nx/2) * dx
    y_real = (np.arange(Ny) - Ny/2) * dx
    Xr, Yr = np.meshgrid(x_real, y_real)

    # 频率坐标 fx, fy (cycles/m) 与采样 dx 对应
    fx = np.linspace(-1/(2*dx), 1/(2*dx), Nx)
    fy = np.linspace(-1/(2*dx), 1/(2*dx), Ny)
    F_x, F_y = np.meshgrid(fx, fy)

    k = 2*np.pi / wavelength
    prefactor = np.exp(1j * k * z) / (1j * wavelength * z)  # e^{ikz}/(i λ z)

    U = np.zeros((Ny, Nx), dtype=np.complex128)
    for m in range(Ny):
        for n in range(Nx):
            # kernel uses fx,fy times x,y: exp(-i 2π (fx*x + fy*y))
            phase = np.exp(-1j * 2 * np.pi * (F_x[m, n] * Xr + F_y[m, n] * Yr))
            U[m, n] = prefactor * np.sum(aperture * phase) * (dx * dx)

    I = np.abs(U) ** 2
    # 防止全零（极端情况）
    if I.max() != 0:
        I /= I.max()
    return I, fx, fy


# =====================================================
# Fraunhofer FFT 实现（前置因子与 direct 保持一致）
# =====================================================
def fraunhofer_fft(aperture, dx, wavelength, z):
    """
    用 FFT 计算 Fraunhofer 衍射：
    continuous FT ≈ fft2(aperture) * dx*dx
    加上相同的 prefactor e^{ikz}/(i λ z)
    """
    k = 2 * np.pi / wavelength
    prefactor = np.exp(1j * k * z) / (1j * wavelength * z)  # same as direct

    # 注意使用 fftshift/ifftshift 以保证物理坐标对齐
    U = fftshift(fft2(aperture)) * (dx * dx)
    U = prefactor * U

    I = np.abs(U) ** 2
    if I.max() != 0:
        I /= I.max()
    return I


# =====================================================
# 可视化
# =====================================================
def plot_results(aperture, I_direct, I_fft, dx, wavelength, z):
    Ny, Nx = aperture.shape

    # 观察平面物理坐标（米 -> mm）
    fx = np.linspace(-1/(2*dx), 1/(2*dx), Nx)
    fy = np.linspace(-1/(2*dx), 1/(2*dx), Ny)
    x_obs = wavelength * z * fx * 1e3  # mm
    y_obs = wavelength * z * fy * 1e3  # mm
    extent_obs = [x_obs.min(), x_obs.max(), y_obs.min(), y_obs.max()]

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(aperture, cmap='gray', origin='lower')
    plt.title("Aperture")

    plt.subplot(1, 3, 2)
    plt.imshow(I_direct, cmap='hot', extent=extent_obs, origin='lower')
    plt.title(f"Fraunhofer Direct (z={z:.3f} m)")
    plt.xlabel("x_obs (mm)"); plt.ylabel("y_obs (mm)")

    plt.subplot(1, 3, 3)
    plt.imshow(I_fft, cmap='hot', extent=extent_obs, origin='lower')
    plt.title(f"Fraunhofer FFT (z={z:.3f} m)")
    plt.xlabel("x_obs (mm)")

    plt.tight_layout()
    plt.show()


# =====================================================
# 主程序示例（保持你原来风格）
# =====================================================
def run_demo():
    aperture_type = "grating"

    Nx = Ny = 128
    wavelength = 500e-9
    dx = 10e-6
    z = 0.5

    aperture = generate_aperture(aperture_type, Nx, Ny)
    I_direct, fx, fy = fraunhofer_direct(aperture, dx, wavelength, z)
    I_fft = fraunhofer_fft(aperture, dx, wavelength, z)

    plot_results(aperture, I_direct, I_fft, dx, wavelength, z)


if __name__ == "__main__":
    run_demo()
