"""
Fresnel 可视化主程序 
"""
from dataclasses import dataclass
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QAction
from PyQt5.QtCore import pyqtSlot
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import numpy as np
import sys

from Ui_Fresnel_Window import Ui_Fresnel
from fresnel_calc import FresnelParams, FresnelCalculator

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------- 主程序 ----------------------
class FresnelApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Fresnel()
        self.ui.setupUi(self)

        # ------------------- 参数 -------------------
        self.params = FresnelParams()
        self.calc = FresnelCalculator(self.params)
        self.data_ready = False
        self.show_mode = "r_t"  # 显示模式：rs/rp/ts/tp 或 R/T

        # ------------------- 替换 QLabel 为 FigureCanvas -------------------
        self.canvas_main = self._replace_label_with_canvas(self.ui.DataShowLabel)
        self.canvas_phase = self._replace_label_with_canvas(self.ui.PositionShowLabel)
        self.canvas_polar = self._replace_label_with_canvas(self.ui.PolarisationShow)
        self.polar_mode = 0  # 0: 入射光, 1: 反射光, 2: 折射光

        # ------------------- 菜单 -------------------
        self._setup_wave_menu()
        self._setup_show_menu()

        # ------------------- 信号 -------------------
        self.ui.SureButton.clicked.connect(self.on_sure_clicked)
        self.ui.AngelRead.valueChanged.connect(self.on_slider_changed)
        self.ui.AngelRead.setRange(0, 90)
        self.ui.change_button.clicked.connect(self.on_change_polar)  # 切换偏振显示

        # ------------------- 初始化 -------------------
        self.__on_build__()

    # ---------------------- 构建与初始化 ----------------------
    def __on_build__(self):
        """初始化界面显示值与空坐标系"""
        self.ui.IncidentNData.setText(str(self.params.n1))
        self.ui.RefractingNData.setText(str(self.params.n2))
        self.ui.AngelRead.setValue(int(self.params.angle_deg))
        self.ui.ChoseWave.setText(self.params.wave_mode)
        self.ui.SintensityData.setText(str(self.params.amp_s))
        self.ui.PintensityData.setText(str(self.params.amp_p))
        self.ui.phi_sub.setText(str(self.params.delta_deg))
        self.draw_initial_axes()
        self.draw_polarisation()  # 初始化偏振显示

    def draw_initial_axes(self):
        """绘制空坐标系"""
        # 主图
        fig = self.canvas_main.figure
        fig.clear()
        ax = fig.subplots()
        ax.set_facecolor("white")
        ax.set_xlim(0, 90)
        ax.set_ylim(0, 1)
        ax.set_xlabel("入射角 θ (°)")
        ax.set_ylabel("反射/透射数值")
        ax.grid(True, linestyle=":", alpha=0.5)
        fig.tight_layout()
        self.canvas_main.draw()

        # 相位图
        fig3 = self.canvas_phase.figure
        fig3.clear()
        ax3 = fig3.subplots()
        ax3.set_facecolor("white")
        ax3.set_xlim(0, 90)
        ax3.set_ylim(0, 360)
        ax3.set_xlabel("入射角 θ (°)")
        ax3.set_ylabel("相位差 |Δφ| (°)")
        ax3.grid(True, linestyle=":", alpha=0.5)
        fig3.tight_layout()
        self.canvas_phase.draw()

    # ---------------------- 替换 QLabel ----------------------
    def _replace_label_with_canvas(self, label):
        canvas = FigureCanvas(Figure(facecolor="white"))
        layout = label.parent().layout()
        layout.replaceWidget(label, canvas)
        label.deleteLater()
        return canvas

    # ---------------------- 菜单设置 ----------------------
    def _setup_show_menu(self):
        menu = QMenu(self.ui.ShowRTrt)
        a1 = QAction("rs, rp, ts, tp", self)
        a2 = QAction("R_s, R_p, T_s, T_p", self)
        menu.addActions([a1, a2])
        self.ui.ShowRTrt.setMenu(menu)
        a1.triggered.connect(lambda: self._set_show_mode("r_t"))
        a2.triggered.connect(lambda: self._set_show_mode("R_T"))
        self.ui.ShowRTrt.setText("rs, rp, ts, tp")

    def _setup_wave_menu(self):
        menu = QMenu(self.ui.ChoseWave)
        modes = ["S波", "P波", "圆偏振光", "任意偏振光", "椭圆偏振光"]
        for mode in modes:
            action = QAction(mode, self)
            action.triggered.connect(lambda checked, m=mode: self._select_wave_mode(m))
            menu.addAction(action)
        self.ui.ChoseWave.setMenu(menu)
        self.ui.ChoseWave.setText("光线选择")

    def _set_show_mode(self, mode):
        self.show_mode = mode
        self.ui.ShowRTrt.setText("rs, rp, ts, tp" if mode == "r_t" else "R_s, R_p, T_s, T_p")
        if self.data_ready:
            self.draw_main_plot()

    def _select_wave_mode(self, mode: str):
        self.params.wave_mode = mode
        self.ui.ChoseWave.setText(mode)
        if mode == "S波":
            self.ui.SintensityData.setText("1.0")
            self.ui.PintensityData.setText("0.0")
            self.ui.phi_sub.setText("0")
        elif mode == "P波":
            self.ui.SintensityData.setText("0.0")
            self.ui.PintensityData.setText("1.0")
            self.ui.phi_sub.setText("0")
        elif mode == "圆偏振光":
            self.ui.SintensityData.setText("1.0")
            self.ui.PintensityData.setText("1.0")
            self.ui.phi_sub.setText("90")
        elif mode == "任意偏振光":
            self.ui.SintensityData.setText("0.7")
            self.ui.PintensityData.setText("0.5")
            self.ui.phi_sub.setText("30")
        elif mode == "椭圆偏振光":
            self.ui.SintensityData.setText("1.0")
            self.ui.PintensityData.setText("0.5")
            self.ui.phi_sub.setText("60")

    # ---------------------- 绘图 ----------------------
    def draw_main_plot(self):
        fig = self.canvas_main.figure
        fig.clear()
        ax = fig.subplots()
        ax.set_facecolor("white")

        if not self.data_ready:
            ax.set_xlim(0, 90)
            ax.set_ylim(0, 1)
            fig.tight_layout()
            self.canvas_main.draw()
            return

        if self.show_mode == "r_t":
            # 判断是否有角度超过全反射角
            rs_arr = self.calc.rs_arr
            rp_arr = self.calc.rp_arr
            ts_arr = self.calc.ts_arr
            tp_arr = self.calc.tp_arr
            
            # 使用 mask 来分隔全反射角前后的部分
            mask_below_critical = self.calc.deg < self.calc.crit_deg
            mask_above_critical = ~mask_below_critical  # 反转条件，找到大于等于临界角的部分

            # 在全反射角之前显示实部，之后显示幅值
            ax.plot(self.calc.deg[mask_below_critical], rs_arr.real[mask_below_critical], label="rs", color="red")
            ax.plot(self.calc.deg[mask_below_critical], rp_arr.real[mask_below_critical], label="rp", color="orange")
            ax.plot(self.calc.deg[mask_below_critical], ts_arr.real[mask_below_critical], "--", label="ts", color="blue")
            ax.plot(self.calc.deg[mask_below_critical], tp_arr.real[mask_below_critical], "--", label="tp", color="cyan")

            ax.plot(self.calc.deg[mask_above_critical], np.abs(rs_arr[mask_above_critical]),  color="red")
            ax.plot(self.calc.deg[mask_above_critical], np.abs(rp_arr[mask_above_critical]),  color="orange")
            ax.plot(self.calc.deg[mask_above_critical], np.abs(ts_arr[mask_above_critical]), "--",  color="blue")
            ax.plot(self.calc.deg[mask_above_critical], np.abs(tp_arr[mask_above_critical]), "--", color="cyan")

        else:
            ax.plot(self.calc.deg, self.calc.Rous_arr, label="R_s", color="red")
            ax.plot(self.calc.deg, self.calc.Roup_arr, label="R_p", color="orange")
            ax.plot(self.calc.deg, self.calc.Taos_arr, "--", label="T_s", color="blue")
            ax.plot(self.calc.deg, self.calc.Taop_arr, "--", label="T_p", color="cyan")

        ax.axvline(self.params.angle_deg, color="gray", linestyle=":")
        ymax = ax.get_ylim()[1]
        ax.text(min(90.0, self.params.angle_deg + 1.0), ymax * 0.92,
                f"θ={self.params.angle_deg:.1f}°", color="gray")

        ax.set_xlabel("入射角 θ (°)")
        ax.set_ylabel("反射 / 透射系数")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize="small")
        fig.tight_layout()
        self.canvas_main.draw()


    def draw_phase_plot(self):
        if not self.data_ready:
            return
        fig = self.canvas_phase.figure
        fig.clear()
        ax = fig.subplots()
        deg = self.calc.deg
        ph_s = self.calc.phase_s
        ph_p = self.calc.phase_p
        bs = self.calc.brewster_deg
        cs = self.calc.crit_deg

        ax.plot(deg, ph_s, "r", lw=1.5, label="|Δφs|")
        ax.plot(deg, ph_p, "b--", lw=1.5, label="|Δφp|")
        ax.axvline(bs, color="green", ls=":", lw=1, label=f"Brewster {bs:.1f}°")
        ax.axvline(cs, color="purple", ls=":", lw=1, label=f"Critical {cs:.1f}°")
        ax.axvspan(cs, 90, color="gray", alpha=0.15, label="TIR 区域")

        ax.set_xlabel("入射角 θ (°)")
        ax.set_ylabel("相位差 |Δφ| (°)")
        ax.set_xlim(0, 90)
        ax.set_ylim(0, 200)
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(fontsize="small")
        self.canvas_phase.draw()

    def update_value_labels_at_current_angle(self):
        if not self.data_ready:
            return
        idx = int(round((self.params.angle_deg - self.calc.deg[0]) / (self.calc.deg[1] - self.calc.deg[0])))
        idx = max(0, min(len(self.calc.deg)-1, idx))
        rs, rp, ts, tp = self.calc.rs_arr[idx], self.calc.rp_arr[idx], self.calc.ts_arr[idx], self.calc.tp_arr[idx]
        Rous, Roup, Taos, Taop = self.calc.Rous_arr[idx], self.calc.Roup_arr[idx], self.calc.Taos_arr[idx], self.calc.Taop_arr[idx]

        self.ui.rsData.setText(f"{rs.real:.5f} {'+' if rs.imag>=0 else '-'} {abs(rs.imag):.5f}j")
        self.ui.rpData.setText(f"{rp.real:.5f} {'+' if rp.imag>=0 else '-'} {abs(rp.imag):.5f}j")
        self.ui.tsData.setText(f"{ts.real:.5f} {'+' if ts.imag>=0 else '-'} {abs(ts.imag):.5f}j")
        self.ui.tpData.setText(f"{tp.real:.5f} {'+' if tp.imag>=0 else '-'} {abs(tp.imag):.5f}j")
        self.ui.RousData.setText(f"{Rous:.5f}")
        self.ui.RoupData.setText(f"{Roup:.5f}")
        self.ui.TaosData.setText(f"{Taos:.5f}")
        self.ui.TaopData.setText(f"{Taop:.5f}")
        self.ui.BRSTAngel.setText(f"{self.calc.brewster_deg:.2f}°")

        self.draw_polarisation()  # 每次更新角度也更新偏振显示

    # ---------------------- 偏振显示 ----------------------
    def draw_polarisation(self):
        fig = self.canvas_polar.figure
        fig.clear()
        ax = fig.subplots()
        ax.set_facecolor("white")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.axis("equal")

        if not self.data_ready:
            fig.tight_layout()
            self.canvas_polar.draw()
            return

        # 根据当前偏振模式选择显示的光类型和颜色
        if self.polar_mode == 0:
            label = "入射光"
            S = self.calc.get_incident_stokes(self.params.amp_s, self.params.amp_p, self.params.wave_mode)
            color = "black"
        elif self.polar_mode == 1:
            label = "反射光"
            S = self.calc.get_reflected_stokes(self.params.angle_deg, self.params.amp_s, self.params.amp_p)
            color = "red"
        else:
            label = "折射光"
            S = self.calc.get_transmitted_stokes(self.params.angle_deg, self.params.amp_s, self.params.amp_p)
            color = "blue"

        I, S1, S2, S3 = S
        
        # 计算偏振度（对于圆偏振和椭圆偏振，DoP应该接近1）
        DoP = np.sqrt(S1**2 + S2**2 + S3**2) / I if I > 1e-12 else 0
        
        # 计算归一化的斯托克斯参数
        if I > 1e-12:
            s1, s2, s3 = S1/I, S2/I, S3/I
        else:
            s1 = s2 = s3 = 0
        
        # 计算S和P分量的振幅
        As = np.sqrt((I + S1) / 2)  # S分量振幅
        Ap = np.sqrt((I - S1) / 2)  # P分量振幅
        
        # 计算相位差δ（S分量相对于P分量的相位差）
        if As * Ap > 1e-12:
            cos_delta = S2 / (2 * As * Ap)
            sin_delta = S3 / (2 * As * Ap)
            delta = np.arctan2(sin_delta, cos_delta)  # 弧度
        else:
            delta = 0
        
        # 判断偏振类型
        if DoP < 0.01:
            # 未偏振或非常弱的偏振
            polarization_type = "未偏振"
            ax.text(0, 0, '未偏振光', ha='center', va='center', color=color, fontsize=12)
            
        elif np.abs(s3) < 0.01 and np.abs(delta) < 0.01:
            # 线偏振（S3接近0，相位差接近0）
            polarization_type = "线偏振"
            
            # 计算线偏振方向
            if np.abs(s1) > 1e-6 or np.abs(s2) > 1e-6:
                psi = 0.5 * np.arctan2(s2, s1)  # 弧度
                if psi < 0:
                    psi += np.pi
                psi_deg = psi * 180 / np.pi
            else:
                psi = 0
                psi_deg = 0
            
            # 绘制一条线表示线偏振
            length = 1.0  # 归一化长度
            x_line = np.array([-length * np.cos(psi), length * np.cos(psi)])
            y_line = np.array([-length * np.sin(psi), length * np.sin(psi)])
            
            ax.plot(x_line, y_line, color=color, lw=3)
            
        else:
            # 椭圆偏振或圆偏振
            # 判断是否为圆偏振：S和P振幅相等且相位差为±90°
            is_circular = (np.abs(As - Ap) < 0.01 * (As + Ap) and 
                        np.abs(np.abs(delta) - np.pi/2) < 0.01)
            
            if is_circular:
                polarization_type = "圆偏振"
            else:
                polarization_type = "椭圆偏振"
            
            # 判断旋转方向
            if delta > 0:
                rotation = "右旋（逆时针）"
            elif delta < 0:
                rotation = "左旋（顺时针）"
            else:
                rotation = "无旋转"
            
            # 使用电场方程绘制椭圆/圆
            t = np.linspace(0, 2 * np.pi, 400)
            
            # P分量（x方向）：Ap * cos(ωt)
            # S分量（y方向）：As * cos(ωt + δ)
            x = Ap * np.cos(t)
            y = As * np.cos(t + delta)
            
            # 归一化显示，使椭圆在[-1, 1]范围内
            max_amplitude = max(np.max(np.abs(x)), np.max(np.abs(y)))
            if max_amplitude > 1e-6:
                x = x / max_amplitude
                y = y / max_amplitude
            
            # 绘制椭圆/圆
            ax.plot(x, y, color=color, lw=2)
        
        # 设置标题
        if polarization_type == "线偏振":
            if 'psi_deg' in locals():
                title = f"{label} - {polarization_type} ({psi_deg:.1f}°)"
            else:
                title = f"{label} - {polarization_type}"
        elif polarization_type in ["圆偏振", "椭圆偏振"]:
            title = f"{label} - {polarization_type} ({rotation})"
        else:
            title = f"{label} - {polarization_type}"
        
        ax.set_title(title, fontsize=12, pad=10)
        
        # 显示详细信息
        info_text = f"光强: {I:.3f}"
        if DoP > 0.01 and polarization_type != "未偏振":
            info_text += f"\n偏振度: {DoP:.3f}"
            if polarization_type in ["椭圆偏振", "圆偏振"]:
                info_text += f"\nδ: {delta*180/np.pi:.1f}°"
        
        ax.text(0.95, 0.95, info_text, transform=ax.transAxes,
                horizontalalignment='right', verticalalignment='top',
                fontsize=9, color=color,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # 设置坐标轴标签
        ax.set_xlabel("P分量（平行于入射面）", fontsize=10)
        ax.set_ylabel("S分量（垂直于入射面）", fontsize=10)
        
        # 设置图形范围
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        
        # 添加坐标轴参考线
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
        
        fig.tight_layout()
        self.canvas_polar.draw()

    #---------------------- 偏振切换 ----------------------
    @pyqtSlot()
    def on_change_polar(self):
        self.polar_mode = (self.polar_mode + 1) % 3
        self.draw_polarisation()

    # ---------------------- 事件 ----------------------
    @pyqtSlot()
    def on_sure_clicked(self):
        try:
            self.params.n1 = float(self.ui.IncidentNData.text())
            self.params.n2 = float(self.ui.RefractingNData.text())
            self.params.angle_deg = float(self.ui.AngelRead.value())
            self.params.wave_mode = self.ui.ChoseWave.text().strip()
            self.params.amp_s = float(self.ui.SintensityData.text())
            self.params.amp_p = float(self.ui.PintensityData.text())
            self.params.delta_deg = float(self.ui.phi_sub.text())
        except Exception:
            pass

        self.calc = FresnelCalculator(self.params)
        self.calc.compute_all_arrays(step_deg=0.01)
        self.data_ready = True
        self.draw_main_plot()
        self.draw_phase_plot()
        self.update_value_labels_at_current_angle()

    @pyqtSlot(int)
    def on_slider_changed(self, val):
        self.params.angle_deg = val
        if self.data_ready:
            self.draw_main_plot()
            self.update_value_labels_at_current_angle()

# ---------------------- 启动 ----------------------
def main():
    app = QApplication(sys.argv)
    win = FresnelApp()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
