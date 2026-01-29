import numpy as np
from diffraction_data import Light

def grating_aperture(x: np.ndarray, y: np.ndarray, params: Light) -> tuple:
    """
    光栅夫琅和费衍射
    """
    # -------- 参数 --------
    a = params.Grating_d1          # 单缝宽度
    d = params.Grating_d2          # 光栅周期
    num_slits = params.Grating_N   # 缝数
    lam = params.wavelength        # 波长
    z = params.Focal_length        # 衍射距离
    k = 2 * np.pi / lam
    I0 = params.EPower
    N = params.N

    # ========== 解析法 ==========
    x_flat = x[0, :]
    eps = 1e-15
    
    # 单缝衍射因子
    alpha = np.pi * a * x_flat / (lam * z)
    with np.errstate(divide='ignore', invalid='ignore'):
        single = a * np.sin(alpha) / (alpha + eps)
        single[np.abs(alpha) < eps] = a
    
    # 多缝干涉因子
    beta = np.pi * d * x_flat / (lam * z)
    with np.errstate(divide='ignore', invalid='ignore'):
        multi = np.sin(num_slits * beta) / (np.sin(beta) + eps)
        multi[np.abs(np.sin(beta)) < eps] = num_slits

    #=========常数项、二次相位==========
    prefactor = np.exp(1j * k * z) / (1j * lam * z)
    E_analytical_1d = prefactor * single * multi
    I_analytical_1d = np.abs(E_analytical_1d)**2* np.sqrt(I0) 
    
    # 扩展为二维
    E_analytical = np.tile(E_analytical_1d, (x.shape[0], 1))
    I_analytical = np.tile(I_analytical_1d, (x.shape[0], 1))

    # ========== 一维FFT方法 ==========
    X_aperture, Y_aperture, fx, fy, X_fft, Y_fft, dx_aperture, dy_aperture = params.generate_aperture_plane()
    
    # 提取x方向的一维坐标
    x_aperture = X_aperture[0, :]  # 第一行的x坐标
    
    # 计算狭缝中心位置
    slit_centers = np.linspace(-(num_slits-1)/2*d, (num_slits-1)/2*d, num_slits)
    
    # 构造一维光栅
    aperture_1d = np.zeros(N, dtype=float)
    
    # 构建透射函数
    for center in slit_centers:
        mask = np.abs(x_aperture - center) <= a/2
        aperture_1d[mask] = 1.0
    
    # 将一维孔径函数扩展为二维
    aperture_func = np.tile(aperture_1d, (N, 1))
    
    # 一维FFT计算衍射
    F_1d = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(aperture_1d)))
    
    # 提取x方向的一维观察屏坐标
    x_fft_1d = X_fft[0, :]  # 第一行的x坐标
    
    
    # 计算电场和强度
    quadratic_phase_1d = np.exp(1j * k * (x_flat**2) / (2 * z))
    scale_factor = dx_aperture 
    
    E_fft_1d = prefactor * F_1d * quadratic_phase_1d * scale_factor
    I_fft_1d = np.abs(E_fft_1d)**2 * I0
    
    
    # 扩展为二维
    E_fft = np.tile(E_fft_1d, (x.shape[0], 1))
    I_fft = np.tile(I_fft_1d, (x.shape[0], 1))
    
    # ========== 孔径函数信息 ==========
    # 光栅在x方向的范围
    x_range = [-(num_slits-1)/2*d - a*1.2, (num_slits-1)/2*d + a*1.2]
    
    
    y_range = [-a*2, a*2]
    
    aperture_info = {
        'func': aperture_func,
        'X': X_aperture,
        'Y': Y_aperture,
        'x_range': x_range,
        'y_range': y_range
    }
    
    return E_analytical, I_analytical, E_fft, I_fft, aperture_info