import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 


# ---------- 参数 ----------
n1 = 1.5
n2 = 1.0
theta_i = np.linspace(0, 90, 10000) * math.pi / 180   # 0-90°
sin_theta_t = np.sin(theta_i) * (n1 / n2)


# ---------- cos_theta_t ----------
cos_theta_t = np.sqrt(1.0 - sin_theta_t**2)
cos_theta_t = np.where(sin_theta_t > 1,
                       1j * np.sqrt(sin_theta_t**2 - 1.0),
                       cos_theta_t)



# ---------- 计算相位变化 ----------
def fresnel_with_phase(n1, n2, theta_i):
    thetas = np.zeros_like(theta_i)
    thetap = np.zeros_like(theta_i)

    br   = np.arctan(n2 / n1)              # Brewster 角（弧度，数组友好）
    crit = np.arcsin(np.minimum(n2/n1, 1.0))

    for idx, th in enumerate(theta_i):
        if th < br:                                        # 1. Brewster 前
            thetas[idx] = 0.0
            thetap[idx] = np.pi
        elif th < crit:                                    # 2. Brewster → 临界
            thetas[idx] = 0.0
            thetap[idx] = 0.0
        else:                                              # 3. TIR
            sin2 = np.sin(th)**2
            sin_tc2 = (n2/n1)**2
            # 先 clip 再 sqrt，杜绝负数
            gamma = np.sqrt(np.maximum(0.0, sin2 - sin_tc2))
            thetas[idx] = np.abs(2.0 * np.arctan(-gamma / np.cos(th)))
            thetap[idx] = np.abs(2.0 * np.arctan(-gamma / np.cos(th) * (n1/n2)**2))
    return thetas, thetap


# ---------- 偏振光 Stokes 参数 ----------
def stokes_elliptical(a, b, delta_deg, intensity=1.0):
    delta = np.deg2rad(delta_deg)
    I  = a**2 + b**2
    S1 = a**2 - b**2
    S2 = 2*a*b*np.cos(delta)
    S3 = 2*a*b*np.sin(delta)
    S  = np.array([I, S1, S2, S3])
    return (intensity / I) * S


# ---------- Fresnel ----------
rs = (n1 * np.cos(theta_i) - n2 * cos_theta_t) / (n1 * np.cos(theta_i) + n2 * cos_theta_t)
rp = (n2 * np.cos(theta_i) - n1 * cos_theta_t) / (n2 * np.cos(theta_i) + n1 * cos_theta_t)
ts = (2 * n1 * np.cos(theta_i)) / (n1 * np.cos(theta_i) + n2 * cos_theta_t)
tp = (2 * n1 * np.cos(theta_i)) / (n2 * np.cos(theta_i) + n1 * cos_theta_t)

Rous = np.abs(rs)**2
Roup = np.abs(rp)**2
Tous = (n2 * np.real(cos_theta_t) / (n1 * np.cos(theta_i))) * np.abs(ts)**2
Toup = (n2 * np.real(cos_theta_t) / (n1 * np.cos(theta_i))) * np.abs(tp)**2

# ==================== 绘图 ====================
plt.figure(figsize=(7, 4))
plt.plot(np.degrees(theta_i), np.real(rs), label='rs (Re)')
plt.plot(np.degrees(theta_i), np.real(rp), label='rp (Re)')
plt.plot(np.degrees(theta_i), np.real(ts), label='ts (Re)')
plt.plot(np.degrees(theta_i), np.real(tp), label='tp (Re)')

plt.xlim(0, 90)
plt.xlabel('Incident Angle (degrees)')
plt.ylabel('Fresnel Coefficients (real part)')
plt.title('Fresnel Coefficients – Real Part')
plt.legend(ncol=2, fontsize=9)
plt.grid(True, ls=':', alpha=0.4)
plt.tight_layout()
plt.show()


thetas,thetap = fresnel_with_phase(n1, n2, theta_i)
plt.figure(figsize=(7, 4))

plt.plot(np.degrees(theta_i), thetas, label='thetas')
plt.plot(np.degrees(theta_i), thetap, label='thetap')
plt.xlim(0, 90)
plt.xlabel('Incident Angle (degrees)')
plt.ylabel('Position Phase Change (radians)')
plt.title('Fresnel Coefficients – Phase Change')
plt.legend(ncol=2, fontsize=9)
plt.grid(True, ls=':', alpha=0.4)
plt.tight_layout()
plt.show()


# ==================== 能量系数（50° 单点） ====================
idx      = np.argmin(np.abs(theta_i - np.deg2rad(50)))
cosθ_i   = np.cos(theta_i[idx])
cosθ_t   = np.real(cos_theta_t[idx])   # TIR 时 =0，已安全

# 能量系数（单点）
Rs = np.abs(rs[idx])**2
Rp = np.abs(rp[idx])**2
Ts = (n2 * cosθ_t) / (n1 * cosθ_i) * np.abs(ts[idx])**2
Tp = (n2 * cosθ_t) / (n1 * cosθ_i) * np.abs(tp[idx])**2

# ==================== 三光束椭圆（能量缩放） ====================
a_in, b_in, δ_in = 1.0, 0.6, 60        # 入射电场振幅

# 1. 入射（基准强度 = 1）
S_in = stokes_elliptical(a_in, b_in, δ_in, intensity=1.0)

# 2. 反射（形状由复系数定，强度乘 R）
Ex_re = a_in * rs[idx]          # 复振幅
Ey_re = b_in * rp[idx] * np.exp(1j * np.deg2rad(δ_in))
S_re  = stokes_elliptical(np.abs(Ex_re), np.abs(Ey_re),
                          np.degrees(np.angle(Ey_re) - np.angle(Ex_re)),
                          intensity=(Rs + Rp)/2)   # 自然光平均反射强度

# 3. 透射（形状由复系数定，强度乘 T）
Ex_tr = a_in * ts[idx]
Ey_tr = b_in * tp[idx] * np.exp(1j * np.deg2rad(δ_in))
S_tr  = stokes_elliptical(np.abs(Ex_tr), np.abs(Ey_tr),
                          np.degrees(np.angle(Ey_tr) - np.angle(Ex_tr)),
                          intensity=(Ts + Tp)/2)   # 自然光平均透射强度

# ==================== 一图三椭圆（能量版） ====================
fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

def draw_ellipse(ax, a, b, delta_deg, color='b', title=''):
    delta = np.deg2rad(delta_deg)
    t = np.linspace(0, 2*np.pi, 200)
    Ex = a * np.cos(t)
    Ey = b * np.cos(t + delta)
    ax.plot(Ex, Ey, color=color, lw=2)
    ax.plot(Ex[0], Ey[0], 'o', color=color)
    ax.set_aspect('equal'); ax.grid(True, ls=':', alpha=0.4)
    ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel('Ex'); ax.set_ylabel('Ey')
    ax.set_title(title)

# 三子图（能量缩放）
draw_ellipse(axes[0], a_in, b_in, δ_in,
             color='b', title=f'Incident (50°)\\n$a={a_in}$, $b={b_in}$, $\\delta={δ_in}°$\\n$I=1.00$')

draw_ellipse(axes[1], np.abs(Ex_re), np.abs(Ey_re),
             np.degrees(np.angle(Ey_re) - np.angle(Ex_re)),
             color='r', title=f'Reflected\\n$a={np.abs(Ex_re):.2f}$, $b={np.abs(Ey_re):.2f}$\\n$\\delta={np.degrees(np.angle(Ey_re)-np.angle(Ex_re)):.1f}°$\\n$I_{{avg}}={((Rs+Rp)/2):.3f}$')

draw_ellipse(axes[2], np.abs(Ex_tr), np.abs(Ey_tr),
             np.degrees(np.angle(Ey_tr) - np.angle(Ex_tr)),
             color='g', title=f'Transmitted\\n$a={np.abs(Ex_tr):.2f}$, $b={np.abs(Ey_tr):.2f}$\\n$\\delta={np.degrees(np.angle(Ey_tr)-np.angle(Ex_tr)):.1f}°$\\n$I_{{avg}}={((Ts+Tp)/2):.3f}$')

fig.suptitle('Polarization Ellipse: Energy-Scaled (50°)', fontsize=13)
plt.show()