import numpy as np
from scipy.interpolate import RectBivariateSpline
from diffraction_data import Light


def rectangular_aperture(x: np.ndarray, y: np.ndarray, params: Light) -> tuple:
    """
    矩形孔夫琅和费衍射
    """
    # 参数
    a = params.Rect_Width
    b = params.Rect_Height
    lam = params.wavelength
    z = params.Focal_length
    k = 2 * np.pi / lam
    I0 = params.EPower
    
    # 观察屏坐标
    X_obs = x
    Y_obs = y
    
    # ========== 解析解方法 ==========
    # 夫琅和费衍射公式
    alpha = np.pi * a * X_obs / (lam * z)
    beta = np.pi * b * Y_obs / (lam * z)
    
    # sinc函数计算
    sinc_x = np.where(alpha == 0, 1.0, np.sin(alpha) / alpha)
    sinc_y = np.where(beta == 0, 1.0, np.sin(beta) / beta)
    
    #=========面积、常数项、二次相位==========
    prefactor = np.exp(1j * k * z) / (1j * lam * z)
    quadratic_phase = np.exp(1j * k * (X_obs**2 + Y_obs**2) / (2 * z))
    E_analytic = prefactor * a * b * sinc_x * sinc_y * quadratic_phase
    I_analytic = np.abs(E_analytic)**2 * I0
    
    # ========== FFT方法==========
    X_aperture, Y_aperture, fx, fy, X_fft, Y_fft, dx_aperture, dy_aperture = params.generate_aperture_plane()
    
    # 矩形孔径
    aperture_func = ((np.abs(X_aperture) <= a/2) & (np.abs(Y_aperture) <= b/2)).astype(float)
    
    # FFT
    F = np.fft.fft2(aperture_func)
    F_shift = np.fft.fftshift(F)
    
    # FFT场
    scale_factor = dx_aperture * dy_aperture
    E_fft = prefactor * F_shift * scale_factor
    
    # 添加二次相位因子
    E_fft = E_fft * np.exp(1j * k * (X_obs**2 + Y_obs**2) / (2 * z))

    I_fft_raw = np.abs(E_fft)**2

    I_fft = I_fft_raw * I0
    
    # 孔径函数信息
    aperture_info = {
        'func': aperture_func,
        'X': X_aperture,
        'Y': Y_aperture,
        'x_range': [-a*0.6, a*0.6],
        'y_range': [-b*0.6, b*0.6]
    }
    
    return E_analytic, I_analytic, E_fft, I_fft, aperture_info