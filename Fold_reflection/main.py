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
import math
import sys
from Ui_Fresnel_Window import Ui_Fresnel

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 


# ---------------------- 数据类 ----------------------
@dataclass
class FresnelParams:
    n1: float = 1.5
    n2: float = 1.0
    angle_deg: float = 30.0
    wave_mode: str = "S波"  # “S波”、“P波”、“自然光”、“任意偏振光”、“特殊椭圆偏振光”
    amp_s: float = 1.0
    amp_p: float = 0.0


# ---------------------- 主程序 ----------------------
class FresnelApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Fresnel()
        self.ui.setupUi(self)

        self.params = None
        self.data_ready = False
        self.show_mode = "r_t"

        # 用 matplotlib 替换 QLabel
        self.canvas_main = self._replace_label_with_canvas(self.ui.DataShowLabel)
        self.canvas_phase = self._replace_label_with_canvas(self.ui.PositionShowLabel)

        # 菜单与信号
        self._setup_wave_menu()       # 光线选择菜单
        self._setup_show_menu()       # 显示模式菜单
        self.ui.SureButton.clicked.connect(self.on_sure_clicked)
        self.ui.AngelRead.valueChanged.connect(self.on_slider_changed)
        self.ui.AngelRead.setRange(0, 90)  # 限制角度范围

        # 初始化参数并显示
        self.__on_build__()

    # ---------------------- 构建函数 ----------------------
    def __on_build__(self):
        """构建 Fresnel 参数并显示在界面上"""
        self.params = FresnelParams()

        # 设置默认 UI 值
        self.ui.IncidentNData.setText(str(self.params.n1))
        self.ui.RefractingNData.setText(str(self.params.n2))
        self.ui.AngelRead.setValue(int(self.params.angle_deg))
        self.ui.ChoseWave.setText(self.params.wave_mode)
        self.ui.SintensityData.setText(str(self.params.amp_s))
        self.ui.PintensityData.setText(str(self.params.amp_p))

        # 初始化空坐标轴
        self.draw_initial_axes()

    def draw_initial_axes(self):
        """在未点击确定前显示空坐标系"""
        # 主图用角度 X 轴（0..90），其它两个用 -1..1 的物理示意坐标
        # main: 0..90 x, 0..1 y
        fig = self.canvas_main.figure
        fig.clear()
        ax = fig.subplots()
        ax.set_facecolor("white")
        ax.set_xlim(0, 90)
        ax.set_ylim(0, 1)
        ax.set_xlabel("入射角 θ (°)")
        ax.set_ylabel("数值")
        ax.grid(True, linestyle=":", alpha=0.5)
        fig.tight_layout()
        self.canvas_main.draw()

       
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

    # ---------------------- 界面构造 ----------------------
    def _replace_label_with_canvas(self, label):
        canvas = FigureCanvas(Figure(facecolor="white"))
        layout = label.parent().layout()
        layout.replaceWidget(label, canvas)
        label.deleteLater()
        return canvas

    def _setup_show_menu(self):
        """左上角显示选项菜单"""
        menu = QMenu(self.ui.ShowRTrt)
        a1 = QAction("rs, rp, ts, tp", self)
        a2 = QAction("Rous, Roup, Taos, Taop", self)
        menu.addActions([a1, a2])
        self.ui.ShowRTrt.setMenu(menu)
        a1.triggered.connect(lambda: self._set_show_mode("r_t"))
        a2.triggered.connect(lambda: self._set_show_mode("R_T"))
        self.ui.ShowRTrt.setText("rs, rp, ts, tp")

    def _setup_wave_menu(self):
        """为“光线选择”按钮添加下拉菜单（五种模式）"""
        menu = QMenu(self.ui.ChoseWave)
        modes = ["S波", "P波", "自然光", "任意偏振光", "特殊椭圆偏振光"]
        for mode in modes:
            action = QAction(mode, self)
            # 必须用默认参数把 mode 绑定到 lambda，否则循环变量会覆盖
            action.triggered.connect(lambda checked, m=mode: self._select_wave_mode(m))
            menu.addAction(action)
        self.ui.ChoseWave.setMenu(menu)
        # 设置初始显示为当前 mode
        self.ui.ChoseWave.setText("光线选择")

    def _set_show_mode(self, mode):
        self.show_mode = mode
        self.ui.ShowRTrt.setText("rs, rp, ts, tp" if mode == "r_t" else "Rous, Roup, Taos, Taop")
        if self.data_ready:
            self.draw_main_plot()

    def _select_wave_mode(self, mode: str):
        """选中某种偏振光模式后更新 UI 与参数"""
        # 更新 dataclass 与按钮文字
        if self.params is None:
            self.params = FresnelParams()
        self.params.wave_mode = mode
        self.ui.ChoseWave.setText(mode)
        # 根据模式自动调整 S / P 强度输入框
        if mode == "S波":
            self.ui.SintensityData.setText("1.0")
            self.ui.PintensityData.setText("0.0")
        elif mode == "P波":
            self.ui.SintensityData.setText("0.0")
            self.ui.PintensityData.setText("1.0")
        elif mode == "自然光":
            self.ui.SintensityData.setText("1.0")
            self.ui.PintensityData.setText("1.0")
        elif mode == "任意偏振光":
            self.ui.SintensityData.setText("0.7")
            self.ui.PintensityData.setText("0.5")
        elif mode == "特殊椭圆偏振光":
            self.ui.SintensityData.setText("1.0")
            self.ui.PintensityData.setText("0.5")

    # ---------------------- Fresnel 计算 ----------------------
    def fresnel_coeffs_single(self, n1, n2, theta_deg):
        """计算单一角度的 rs, rp, ts, tp（严格数学版 Fresnel，包括 TIR）"""
        theta = math.radians(theta_deg)
        sin1 = math.sin(theta)
        cos1 = math.cos(theta)

        sin2 = (n1 / n2) * sin1

        # ----------- 真实 Fresnel TIR 处理 -----------
        if sin2 > 1.0:
            # TIR：cos2 = i·sqrt(sin2^2 - 1)
            sin2 = 1.0
            cos2 = complex(0.0, math.sqrt(sin2 * sin2 - 1.0))
        else:
            # 正常折射
            cos2 = complex(math.sqrt(1.0 - sin2 * sin2), 0.0)

        cos1c = complex(cos1, 0.0)

        # Fresnel 系数（复数）
        rs = (n1 * cos1c - n2 * cos2) / (n1 * cos1c + n2 * cos2)
        rp = (n2 * cos1c - n1 * cos2) / (n2 * cos1c + n1 * cos2)

        ts = 2 * n1 * cos1c / (n1 * cos1c + n2 * cos2)
        tp = 2 * n1 * cos1c / (n2 * cos1c + n1 * cos2)

        return rs, rp, ts, tp


    def compute_all_arrays(self, n1, n2, step_deg=0.001):
        """计算 Fresnel 反射/透射系数及相位差，包含偏振叠加"""
        deg = np.arange(0.0, 90.0 + step_deg, step_deg)
        rs_arr = np.zeros_like(deg, dtype=complex)
        rp_arr = np.zeros_like(deg, dtype=complex)
        ts_arr = np.zeros_like(deg, dtype=complex)
        tp_arr = np.zeros_like(deg, dtype=complex)
        Rous = np.zeros_like(deg)
        Roup = np.zeros_like(deg)
        Taos = np.zeros_like(deg)
        Taop = np.zeros_like(deg)
        phase_s = np.zeros_like(deg)
        phase_p = np.zeros_like(deg)

        for i, th in enumerate(deg):
            rs, rp, ts, tp = self.fresnel_coeffs_single(n1, n2, th)
            rs_arr[i], rp_arr[i], ts_arr[i], tp_arr[i] = rs, rp, ts, tp

            # 根据实际角度计算反射 / 透射强度
            theta = math.radians(th)
            sin1, cos1 = math.sin(theta), math.cos(theta)
            sin2 = (n1 / n2) * sin1

            if abs(sin2) > 1.0:
                # 全反射：透射能量为 0
                Rous[i], Roup[i], Taos[i], Taop[i] = abs(rs) ** 2, abs(rp) ** 2, 0.0, 0.0
            else:
                cos2 = math.sqrt(max(0.0, 1.0 - sin2 ** 2))
                Rous[i], Roup[i] = abs(rs) ** 2, abs(rp) ** 2
                Taos[i] = (n2 * cos2) / (n1 * cos1) * abs(ts) ** 2
                Taop[i] = (n2 * cos2) / (n1 * cos1) * abs(tp) ** 2

            # 相位（度）
            phase_s[i] = np.angle(rs, deg=True)
            phase_p[i] = np.angle(rp, deg=True)

        # Brewster / 临界角
        brewster_deg = math.degrees(math.atan2(n2, n1))
        crit_deg = 90.0 if n1 <= n2 else math.degrees(math.asin(min(1.0, n2 / n1)))

        # ---------------------- 偏振组合 ----------------------
        # 使用 self.params 的 amp_s / amp_p / wave_mode
        if self.params is None:
            self.params = FresnelParams()
        mode = self.params.wave_mode
        As, Ap = self.params.amp_s, self.params.amp_p

        if mode == "S波":
            weight_s, weight_p = 1.0, 0.0
        elif mode == "P波":
            weight_s, weight_p = 0.0, 1.0
        elif mode == "自然光":
            weight_s, weight_p = 0.5, 0.5
        elif mode == "任意偏振光":
            total = max(As + Ap, 1e-9)
            weight_s, weight_p = As / total, Ap / total
        elif mode == "特殊椭圆偏振光":
            weight_s, weight_p = 2.0 / 3.0, 1.0 / 3.0
        else:
            weight_s, weight_p = 0.5, 0.5

        R_total = weight_s * Rous + weight_p * Roup
        T_total = weight_s * Taos + weight_p * Taop

        # 保存结果到对象属性
        self.deg = deg
        self.rs_arr, self.rp_arr = rs_arr, rp_arr
        self.ts_arr, self.tp_arr = ts_arr, tp_arr
        self.Rous_arr, self.Roup_arr = Rous, Roup
        self.Taos_arr, self.Taop_arr = Taos, Taop
        self.R_total, self.T_total = R_total, T_total
        self.phase_s, self.phase_p = phase_s, phase_p
        self.brewster_deg, self.crit_deg = brewster_deg, crit_deg

    # ---------------------- 绘图 ----------------------
    def draw_main_plot(self):
        fig = self.canvas_main.figure
        fig.clear()
        ax = fig.subplots()
        ax.set_facecolor("white")
        if not hasattr(self, "deg"):
            # 绘空图（axs 已在初始化绘制）
            ax.set_xlim(0, 90)
            ax.set_ylim(0, 1)
            ax.set_xlabel("入射角 θ (°)")
            ax.set_ylabel("数值")
            ax.grid(True, linestyle=":", alpha=0.5)
            fig.tight_layout()
            self.canvas_main.draw()
            return

        if self.show_mode == "r_t":
            ax.plot(self.deg, self.rs_arr.real, label="rs (Re)", color="red")
            ax.plot(self.deg, self.rp_arr.real, label="rp (Re)", color="orange")
            ax.plot(self.deg, self.ts_arr.real, "--", label="ts (Re)", color="blue")
            ax.plot(self.deg, self.tp_arr.real, "--", label="tp (Re)", color="cyan")
        else:
            ax.plot(self.deg, self.Rous_arr, label="R_s (Rous)", color="red")
            ax.plot(self.deg, self.Roup_arr, label="R_p (Roup)", color="orange")
            ax.plot(self.deg, self.Taos_arr, "--", label="T_s (Taos)", color="blue")
            ax.plot(self.deg, self.Taop_arr, "--", label="T_p (Taop)", color="cyan")

        ax.set_xlabel("入射角 θ (°)")
        ax.set_ylabel("反射 / 透射系数")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize="small")

        # 当前角度竖线与标签
        ax.axvline(self.params.angle_deg, color="gray", linestyle=":")
        ymax = ax.get_ylim()[1]
        ax.text(min(90.0, self.params.angle_deg + 1.0), ymax * 0.92, f"θ={self.params.angle_deg:.1f}°", color="gray")

        fig.tight_layout()
        self.canvas_main.draw()


    def draw_phase_plot(self):
        """右下角：Fresnel 真实相位（Brewster 跳变 + TIR）"""
        if not self.data_ready:
            return

        fig = self.canvas_phase.figure
        fig.clear()
        ax = fig.subplots()

        deg  = self.deg
        n1, n2 = self.params.n1, self.params.n2
        bs, cs = self.brewster_deg, self.crit_deg

        theta_rad = np.radians(deg)               # 0-90° → 0-π/2
        sin_th    = np.sin(theta_rad)
        cos_th    = np.cos(theta_rad)
        sin_tc2   = (n2 / n1)**2                  # sin²(θc)

        # ---------- 三段相位（矢量化） ----------
        # 1. Brewster 前
        ph_s = np.where(deg < bs, 0.0, 0.0)
        ph_p = np.where(deg < bs, 180.0, 0.0)

        # 2. Brewster → 临界
        mask2 = (deg >= bs) & (deg < cs)
        ph_s  = np.where(mask2, 0.0, ph_s)
        ph_p  = np.where(mask2, 0.0, ph_p)

        # 3. TIR 区
        mask3 = deg >= cs
        gamma = np.sqrt(np.maximum(0.0, sin_th**2 - sin_tc2))
        phi_s = -2.0 * np.arctan2(gamma, cos_th)            
        phi_p = -2.0 * np.arctan2(gamma * (n1/n2)**2, cos_th)
        ph_s  = np.where(mask3, np.degrees(np.abs(phi_s)), ph_s)
        ph_p  = np.where(mask3, np.degrees(np.abs(phi_p)), ph_p)

        # ---------- 绘图 ----------
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
        # 找到最接近的索引
        idx = int(round((self.params.angle_deg - self.deg[0]) / (self.deg[1] - self.deg[0])))
        idx = max(0, min(len(self.deg) - 1, idx))
        rs, rp, ts, tp = self.rs_arr[idx], self.rp_arr[idx], self.ts_arr[idx], self.tp_arr[idx]
        Rous, Roup, Taos, Taop = self.Rous_arr[idx], self.Roup_arr[idx], self.Taos_arr[idx], self.Taop_arr[idx]
        # 显示复数实/虚部
        self.ui.rsData.setText(f"{rs.real:.5f} {'+' if rs.imag >= 0 else '-'} {abs(rs.imag):.5f}j")
        self.ui.rpData.setText(f"{rp.real:.5f} {'+' if rp.imag >= 0 else '-'} {abs(rp.imag):.5f}j")
        self.ui.tsData.setText(f"{ts.real:.5f} {'+' if ts.imag >= 0 else '-'} {abs(ts.imag):.5f}j")
        self.ui.tpData.setText(f"{tp.real:.5f} {'+' if tp.imag >= 0 else '-'} {abs(tp.imag):.5f}j")
        # 显示反射率 / 透射率
        self.ui.RousData.setText(f"{Rous:.5f}")
        self.ui.RoupData.setText(f"{Roup:.5f}")
        self.ui.TaosData.setText(f"{Taos:.5f}")
        self.ui.TaopData.setText(f"{Taop:.5f}")
        # Brewster 角显示
        self.ui.BRSTAngel.setText(f"{self.brewster_deg:.2f}°")

    # ---------------------- 事件 ----------------------
    @pyqtSlot()
    def on_sure_clicked(self):
        try:
            # 读取输入到 params
            if self.params is None:
                self.params = FresnelParams()
            self.params.n1 = float(self.ui.IncidentNData.text())
            self.params.n2 = float(self.ui.RefractingNData.text())
            self.params.angle_deg = float(self.ui.AngelRead.value())
            self.params.wave_mode = self.ui.ChoseWave.text().strip()
            # 如果输入框为空或非法，使用默认值
            try:
                self.params.amp_s = float(self.ui.SintensityData.text())
            except:
                self.params.amp_s = self.params.amp_s
            try:
                self.params.amp_p = float(self.ui.PintensityData.text())
            except:
                self.params.amp_p = self.params.amp_p
        except Exception:
            # 忽略解析错误，保留旧值
            pass

        # 计算并绘图
        self.compute_all_arrays(self.params.n1, self.params.n2)
        self.data_ready = True
        self.draw_main_plot()
        self.draw_phase_plot()
        self.update_value_labels_at_current_angle()

    @pyqtSlot(int)
    def on_slider_changed(self, val):
        if self.params is None:
            self.params = FresnelParams()
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
