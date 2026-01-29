"""
菲涅尔方程的计算部分
stokes_elliptical: 根据振幅和相位差生成 Stokes 矢量
get_reflected_stokes: 计算反射光的 Stokes 矢量
get_transmitted_stokes: 计算透射光的 Stokes 矢量
fresnel_coeffs_single: 计算单个入射角的 Fresnel 系数
phase_calc: 计算相位变化
compute_all_arrays: 计算所有入射角的 Fresnel 系数和 Stokes
"""

import math
import numpy as np
from dataclasses import dataclass

# ---------------------- 参数类 ----------------------
@dataclass
class FresnelParams:
    n1: float = 1.5
    n2: float = 1.0
    angle_deg: float = 30.0
    wave_mode: str = "S波"  # "S波"、"P波"、"圆偏振光"、"任意偏振光"、"椭圆偏振光"
    amp_s: float = 1.0
    amp_p: float = 0.0
    delta_deg: float = 0.0  # S/P 相位差

# ---------------------- Stokes 函数 ----------------------
def stokes_elliptical(a, b, delta_deg, intensity=1.0):
    """根据振幅 a,b 和相位差 delta_deg 生成 Stokes 矢量"""
    delta = np.deg2rad(delta_deg)
    I  = a**2 + b**2
    S1 = a**2 - b**2
    S2 = 2*a*b*np.cos(delta)
    S3 = 2*a*b*np.sin(delta)
    S  = np.array([I, S1, S2, S3])
    
    # 归一化：确保总强度为1（如果不为零）
    if I > 1e-12:
        return (intensity / I) * S
    else:
        return np.array([0.0, 0.0, 0.0, 0.0])


# ---------------------- Fresnel 计算类 ----------------------
class FresnelCalculator:
    def __init__(self, params: FresnelParams):
        self.params = params

    def get_incident_stokes(self, As, Ap, wave_mode):
        """获取入射光的归一化 Stokes 参数"""
        # 先归一化入射振幅，使总强度为1
        total_intensity = As**2 + Ap**2
        if total_intensity > 1e-12:
            As_norm = As / np.sqrt(total_intensity)
            Ap_norm = Ap / np.sqrt(total_intensity)
        else:
            As_norm = 0.0
            Ap_norm = 0.0
        
        delta_deg = self.params.delta_deg if wave_mode == "椭圆偏振光" or wave_mode == '任意偏振光'or wave_mode == "圆偏振光" else 0.0
        return stokes_elliptical(As_norm, Ap_norm, delta_deg, intensity=1.0)

    def get_reflected_stokes(self, angle_deg, As, Ap):
        """
        计算反射光的归一化 Stokes 参数。
        """
        idx = np.argmin(np.abs(self.deg - angle_deg))
        rs = self.rs_arr[idx]  # 复数反射系数
        rp = self.rp_arr[idx]  # 复数反射系数
        
        # 归一化入射振幅，使入射总强度为1
        incident_intensity = As**2 + Ap**2
        if incident_intensity > 1e-12:
            As_norm = As / np.sqrt(incident_intensity)
            Ap_norm = Ap / np.sqrt(incident_intensity)
        else:
            As_norm = 0.0
            Ap_norm = 0.0
        
        # 获取入射光的相对相位差
        if self.params.wave_mode in ["任意偏振光", "椭圆偏振光", "圆偏振光"]:
            delta = np.deg2rad(self.params.delta_deg)
        else:
            delta = 0.0
        
        # 构造归一化入射复振幅
        Es_in = As_norm * 1.0
        Ep_in = Ap_norm * np.exp(1j * delta)
        
        # 反射后复振幅
        Es_out = rs * Es_in
        Ep_out = rp * Ep_in
        
        # 计算反射Stokes参数（已经归一化，总强度在0-1之间）
        I = np.abs(Es_out)**2 + np.abs(Ep_out)**2
        S1 = np.abs(Es_out)**2 - np.abs(Ep_out)**2
        S2 = 2.0 * np.real(Es_out * np.conj(Ep_out))
        S3 = 2.0 * np.imag(Es_out * np.conj(Ep_out))
        
        # 反射光强应该等于反射率
        R_s = np.abs(rs)**2
        R_p = np.abs(rp)**2
        expected_intensity = (As_norm**2) * R_s + (Ap_norm**2) * R_p
        
        # 计算Stokes参数应该具有的总强度
        if I > 1e-12 and expected_intensity > 1e-12:
            scale = expected_intensity / I
            return np.array([
                expected_intensity,  # 反射光强（0-1之间）
                S1 * scale,
                S2 * scale,
                S3 * scale
            ])
        else:
            return np.array([0.0, 0.0, 0.0, 0.0])

    def get_transmitted_stokes(self, angle_deg, As, Ap):
        """
        计算透射光的归一化 Stokes 参数。
        返回的光强是相对入射光强的透射率（0-1之间）。
        """
        idx = np.argmin(np.abs(self.deg - angle_deg))
        ts = self.ts_arr[idx]  # 复数振幅透射系数
        tp = self.tp_arr[idx]  # 复数振幅透射系数
        
        # 归一化入射振幅，使入射总强度为1
        incident_intensity = As**2 + Ap**2
        if incident_intensity > 1e-12:
            As_norm = As / np.sqrt(incident_intensity)
            Ap_norm = Ap / np.sqrt(incident_intensity)
        else:
            As_norm = 0.0
            Ap_norm = 0.0
        
        # 获取入射光的相对相位差
        if self.params.wave_mode in ["任意偏振光", "椭圆偏振光", "圆偏振光"]:
            delta = np.deg2rad(self.params.delta_deg)
        else:
            delta = 0.0
        
        # 构造归一化入射复振幅
        Es_in = As_norm * 1.0
        Ep_in = Ap_norm * np.exp(1j * delta)
        
        # 检查是否全反射
        n1, n2 = self.params.n1, self.params.n2
        theta1 = np.deg2rad(angle_deg)
        sin_theta1 = np.sin(theta1)
        cos_theta1 = np.cos(theta1)
        sin_theta2 = (n1 / n2) * sin_theta1
        
        if np.abs(sin_theta2) >= 1.0:
            # 全反射：无透射能量
            return np.array([0.0, 0.0, 0.0, 0.0])
        
        # 非全反射情况
        cos_theta2 = np.sqrt(1.0 - sin_theta2**2)
        
        # 计算能量透射率
        T_s_energy = (n2 * cos_theta2) / (n1 * cos_theta1) * np.abs(ts)**2
        T_p_energy = (n2 * cos_theta2) / (n1 * cos_theta1) * np.abs(tp)**2
        
        # 计算透射Stokes参数（电场振幅形式）
        Es_out = ts * Es_in
        Ep_out = tp * Ep_in
        
        I_field = np.abs(Es_out)**2 + np.abs(Ep_out)**2
        S1_field = np.abs(Es_out)**2 - np.abs(Ep_out)**2
        S2_field = 2.0 * np.real(Es_out * np.conj(Ep_out))
        S3_field = 2.0 * np.imag(Es_out * np.conj(Ep_out))
        
        # 计算透射光强
        transmitted_intensity = (As_norm**2) * T_s_energy + (Ap_norm**2) * T_p_energy
        
        if I_field > 1e-12 and transmitted_intensity > 1e-12:
            scale = transmitted_intensity / I_field
            return np.array([
                transmitted_intensity,  
                S1_field * scale,
                S2_field * scale,
                S3_field * scale
            ])
        else:
            return np.array([0.0, 0.0, 0.0, 0.0])
    
    def fresnel_coeffs_single(self, n1, n2, theta_deg):
        """计算单角度 Fresnel 系数，包括 TIR"""
        theta = math.radians(theta_deg)
        sin1, cos1 = math.sin(theta), math.cos(theta)
        sin2 = (n1 / n2) * sin1
        if sin2 > 1.0:  # TIR
            sin2 = 1.0
            cos2 = complex(0.0, math.sqrt(sin2**2 - 1.0))
        else:
            cos2 = complex(math.sqrt(1 - sin2**2), 0.0)
        cos1c = complex(cos1, 0.0)
        rs = (n1 * cos1c - n2 * cos2) / (n1 * cos1c + n2 * cos2)
        rp = (n2 * cos1c - n1 * cos2) / (n2 * cos1c + n1 * cos2)
        ts = 2 * n1 * cos1c / (n1 * cos1c + n2 * cos2)
        tp = 2 * n1 * cos1c / (n2 * cos1c + n1 * cos2)
        return rs, rp, ts, tp

    @staticmethod
    def phase_calc(deg, n1, n2, brewster_deg, crit_deg):
        """读取入射角数组，计算相位变化"""
        theta_rad = np.radians(deg)

        sin_th = np.sin(theta_rad)
        cos_th = np.cos(theta_rad)

        sin_tc2 = (n2 / n1)**2

        ph_s = np.zeros_like(deg)
        ph_p = np.zeros_like(deg)
        #========半波损失，导致相位差距180==========
        ph_p[deg < brewster_deg] = 180.0
        #布儒斯特角，P波无反射，相位无定义
        mask = deg >= crit_deg
        gamma = np.sqrt(np.maximum(0.0, sin_th**2 - sin_tc2))
        #代入公式计算
        phi_s = -2.0 * np.arctan2(gamma, cos_th)
        phi_p = -2.0 * np.arctan2(gamma * (n1/n2)**2, cos_th)

        ph_s[mask] = np.degrees(np.abs(phi_s[mask]))
        ph_p[mask] = np.degrees(np.abs(phi_p[mask]))

        return ph_s, ph_p

    # ---------------------- 核心计算 ----------------------
    def compute_all_arrays(self, step_deg=0.01):
        deg = np.arange(0.0, 90.0 + step_deg, step_deg)
        rs_arr = np.zeros_like(deg, dtype=complex)
        rp_arr = np.zeros_like(deg, dtype=complex)
        ts_arr = np.zeros_like(deg, dtype=complex)
        tp_arr = np.zeros_like(deg, dtype=complex)
        Rous = np.zeros_like(deg)
        Roup = np.zeros_like(deg)
        Taos = np.zeros_like(deg)
        Taop = np.zeros_like(deg)

        n1, n2 = self.params.n1, self.params.n2

        for i, th in enumerate(deg):
            rs, rp, ts, tp = self.fresnel_coeffs_single(n1, n2, th)
            rs_arr[i], rp_arr[i], ts_arr[i], tp_arr[i] = rs, rp, ts, tp

            # 强度计算（反射率和透射率）
            theta = math.radians(th)
            sin1, cos1 = math.sin(theta), math.cos(theta)
            sin2 = (n1 / n2) * sin1
            if abs(sin2) > 1.0:
                # 全反射
                Rous[i], Roup[i] = abs(rs)**2, abs(rp)**2
                Taos[i], Taop[i] = 1 - Rous[i], 1 - Roup[i]
                # Taos[i], Taop[i] = 0.0, 0.0
            else:
                cos2 = math.sqrt(max(0.0, 1 - sin2**2))
                Rous[i], Roup[i] = abs(rs)**2, abs(rp)**2
                # 透射率计算
                Taos[i] = (n2 * cos2) / (n1 * cos1) * abs(ts)**2
                Taop[i] = (n2 * cos2) / (n1 * cos1) * abs(tp)**2

        brewster_deg = math.degrees(math.atan2(n2, n1))
        crit_deg = 90.0 if n1 <= n2 else math.degrees(math.asin(min(1.0, n2 / n1)))
        phase_s_arr, phase_p_arr = self.phase_calc(deg, n1, n2, brewster_deg, crit_deg)

        # ---------- 偏振光 Stokes 参数 ----------
        mode = self.params.wave_mode
        As, Ap, delta = self.params.amp_s, self.params.amp_p, self.params.delta_deg
        
        # 归一化入射振幅
        incident_intensity = As**2 + Ap**2
        if incident_intensity > 1e-12:
            As_norm = As / np.sqrt(incident_intensity)
            Ap_norm = Ap / np.sqrt(incident_intensity)
        else:
            As_norm = 0.0
            Ap_norm = 0.0
            
        S_in = stokes_elliptical(a=As_norm, b=Ap_norm, delta_deg=delta)

        # 反射 / 透射 Stokes I 分量
        R_total = np.zeros_like(deg)
        T_total = np.zeros_like(deg)
        for i in range(len(deg)):
            # Fresnel 系数
            rs, rp = rs_arr[i], rp_arr[i]
            ts, tp = ts_arr[i], tp_arr[i]
            
            # 反射光强：反射率
            R_total[i] = (As_norm**2) * np.abs(rs)**2 + (Ap_norm**2) * np.abs(rp)**2
            
            # 透射光强：透射率
            theta = math.radians(deg[i])
            sin1, cos1 = math.sin(theta), math.cos(theta)
            sin2 = (n1 / n2) * sin1
            if abs(sin2) < 1.0:
                cos2 = math.sqrt(1 - sin2**2)
                T_s = (n2 * cos2) / (n1 * cos1) * np.abs(ts)**2
                T_p = (n2 * cos2) / (n1 * cos1) * np.abs(tp)**2
                T_total[i] = (As_norm**2) * T_s + (Ap_norm**2) * T_p
            else:
                T_total[i] = 0.0

        # 保存结果
        self.deg = deg
        self.rs_arr, self.rp_arr = rs_arr, rp_arr
        self.ts_arr, self.tp_arr = ts_arr, tp_arr
        self.Rous_arr, self.Roup_arr = Rous, Roup
        self.Taos_arr, self.Taop_arr = Taos, Taop
        self.R_total, self.T_total = R_total, T_total
        self.phase_s, self.phase_p = phase_s_arr, phase_p_arr
        self.brewster_deg, self.crit_deg = brewster_deg, crit_deg