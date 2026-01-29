import sys
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QHBoxLayout, QMessageBox, QMenu

from diffraction_data import Light
from diffraction_circular import circular_aperture
from diffraction_rectangular import rectangular_aperture
from diffraction_grating import grating_aperture
from Ui_AnalyticalvsFFT import Ui_MainWindow

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class MplCanvas(FigureCanvas):
    """自定义Matplotlib画布"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.axes = self.fig.add_subplot(111)
        
    def clear(self):
        """完全清空图形"""
        self.fig.clf()
        self.axes = self.fig.add_subplot(111)
        self.draw()
        
    def plot_2d(self, data, x_range, y_range, title, cmap='viridis', log_scale=False, colorbar_label='强度'):
        """绘制2D图像"""
        # 完全清除之前的图形
        self.fig.clf()
        self.axes = self.fig.add_subplot(111)
        
        if log_scale:
            # 避免log(0)的问题
            data_plot = np.log(data + 1e-10)
        else:
            data_plot = data
            
        im = self.axes.imshow(data_plot, 
                             extent=[x_range[0]*1e3, x_range[1]*1e3, 
                                    y_range[0]*1e3, y_range[1]*1e3],
                             cmap=cmap, origin='lower', aspect='auto')
        
        self.axes.set_xlabel('x (mm)')
        self.axes.set_ylabel('y (mm)')
        self.axes.set_title(title)
        
        # 添加颜色条
        cbar = self.fig.colorbar(im, ax=self.axes)
        cbar.set_label(colorbar_label)
        
        self.fig.tight_layout()
        self.draw()
        
    def plot_1d(self, x, y, title, xlabel='x (mm)', ylabel='强度'):
        """绘制1D剖面图"""
        # 完全清除之前的图形
        self.fig.clf()
        self.axes = self.fig.add_subplot(111)
            
        self.axes.plot(x, y, 'b-', linewidth=2)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_title(title)
        self.axes.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.draw()


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        # 初始化参数
        self.params = Light() 
        
        # 存储计算结果
        self.calculation_results = None
        self.current_display_mode = 'E'  # 'E' 或 'I'
        self.current_dimension = '1D'    # '2D' 或 '1D'
        self.show_error_aperture = False  # 控制是否显示误差和孔径图
        
        # 设置UI初始值
        self.init_ui_with_defaults()
        
        # 设置下拉菜单
        self.init_menus()
        
        # 设置Matplotlib画布
        self.setup_matplotlib()
        
        # 连接信号和槽
        self.connect_signals()
        
    def init_ui_with_defaults(self):
        """初始化UI"""
        # 波长：550 nm
        self.Wavelength.setText(f"{self.params.wavelength * 1e9:.0f}")
        
        # 焦距：120 mm
        self.Focal_length.setText(f"{self.params.Focal_length * 1e3:.1f}")
        
        # 观察屏尺寸：20 mm
        self.objectx.setText(f"{self.params.obs_range_x * 1e3:.1f}")
        self.objecty.setText(f"{self.params.obs_range_y * 1e3:.1f}")
        
        # 分辨率：1024
        self.Nx.setText(f"{self.params.N}")
        self.Ny.setText(f"{self.params.N}")
        
        # 圆孔半径：0.025 mm
        self.Rdata.setText(f"{self.params.R * 1e3:.3f}")
        
        # 矩形孔尺寸：0.05 mm × 0.05 mm
        self.Rectx.setText(f"{self.params.Rect_Width * 1e3:.3f}")
        self.Recty.setText(f"{self.params.Rect_Height * 1e3:.3f}")
        
        # 光栅参数
        self.gratingd1.setText(f"{self.params.Grating_d1 * 1e3:.3f}")
        self.gratingd2.setText(f"{self.params.Grating_d2 * 1e3:.3f}")
        self.gratingN.setText(f"{self.params.Grating_N}")
        
        # 误差显示框设为只读
        self.relative_error.setReadOnly(True)
        self.mse.setReadOnly(True)
        self.max_error.setReadOnly(True)
        self.rms_error.setReadOnly(True)
        
        # 初始孔径类型
        self.Screenchoice.setText(self.params.PingMode)
        
    def init_menus(self):
        """初始化下拉菜单"""
        # 衍射屏类型菜单
        screen_menu = QMenu()
        screen_types = ["circular", "rectangular", "grating"]
        for screen_type in screen_types:
            action = screen_menu.addAction(screen_type)
            action.triggered.connect(lambda checked, st=screen_type: self.set_screen_type(st))
        self.Screenchoice.setMenu(screen_menu)
        self.Screenchoice.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        
        # 显示类型菜单（电场/光强）
        display_menu = QMenu()
        display_types = [("电场", "E"), ("光强", "I")]
        for display_name, display_code in display_types:
            action = display_menu.addAction(display_name)
            action.triggered.connect(lambda checked, dc=display_code: self.set_display_type(dc))
        self.EIchoice.setMenu(display_menu)
        self.EIchoice.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.EIchoice.setText("电场")
        
        # 维度显示菜单
        dim_menu = QMenu()
        dim_types = ["2D", "1D"]
        for dim_type in dim_types:
            action = dim_menu.addAction(dim_type)
            action.triggered.connect(lambda checked, dt=dim_type: self.set_dimension_type(dt))
        self.DChoice.setMenu(dim_menu)
        self.DChoice.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.DChoice.setText("1D")
        
    def set_screen_type(self, screen_type):
        """设置衍射屏类型"""
        self.Screenchoice.setText(screen_type)
        self.show_error_aperture = False  # 切换孔径类型时，返回衍射图模式
        self.errorAndap.setText("切换孔径/误差图")
            
    def set_display_type(self, display_code):
        """设置显示类型（E或I）"""
        self.current_display_mode = display_code
        display_name = "电场" if display_code == 'E' else "光强"
        self.EIchoice.setText(display_name)
            
    def set_dimension_type(self, dimension_type):
        """设置维度显示类型"""
        self.current_dimension = dimension_type
        self.DChoice.setText(dimension_type)
            
    def setup_matplotlib(self):
        """设置Matplotlib画布和工具栏"""
        # 移除原来的QLabel
        self.AnalyticalPlot.setParent(None)
        self.FFTPlot.setParent(None)
        
        # 创建新的容器和布局
        analytical_widget = QtWidgets.QWidget()
        analytical_layout = QVBoxLayout(analytical_widget)
        self.analytical_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.analytical_toolbar = NavigationToolbar(self.analytical_canvas, self)
        analytical_layout.addWidget(self.analytical_toolbar)
        analytical_layout.addWidget(self.analytical_canvas)
        
        fft_widget = QtWidgets.QWidget()
        fft_layout = QVBoxLayout(fft_widget)
        self.fft_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.fft_toolbar = NavigationToolbar(self.fft_canvas, self)
        fft_layout.addWidget(self.fft_toolbar)
        fft_layout.addWidget(self.fft_canvas)
        
        # 添加到水平布局
        self.horizontalLayout.addWidget(analytical_widget)
        self.horizontalLayout.addWidget(fft_widget)
        
    def connect_signals(self):
        """连接信号和槽"""
        self.SureButton.clicked.connect(self.compute_and_plot)
        self.errorAndap.clicked.connect(self.toggle_error_aperture_view)
        
    def toggle_error_aperture_view(self):
        """切换显示误差图/孔径图"""
        if not self.calculation_results:
            QMessageBox.warning(self, "提示", "请先点击'绘制'按钮进行计算")
            return
            
        # 切换视图模式
        self.show_error_aperture = not self.show_error_aperture
        
        if self.show_error_aperture:
            self.errorAndap.setText("返回衍射图")
        else:
            self.errorAndap.setText("切换孔径/误差图")

        # 更新显示
        self.update_display()
        
    def get_parameters_from_ui(self):
        """从UI获取参数并更新Light对象"""
        try:
            # 基本参数
            self.params.wavelength = float(self.Wavelength.text()) * 1e-9
            self.params.Focal_length = float(self.Focal_length.text()) * 1e-3
            self.params.obs_range_x = float(self.objectx.text()) * 1e-3
            self.params.obs_range_y = float(self.objecty.text()) * 1e-3
            self.params.N = int(self.Nx.text())
            self.params.PingMode = self.Screenchoice.text()
            
            # 根据衍射屏类型设置相应参数
            screen_type = self.params.PingMode
            if screen_type == 'circular':
                self.params.R = float(self.Rdata.text()) * 1e-3
            elif screen_type == 'rectangular':
                self.params.Rect_Width = float(self.Rectx.text()) * 1e-3
                self.params.Rect_Height = float(self.Recty.text()) * 1e-3
            elif screen_type == 'grating':
                self.params.Grating_d1 = float(self.gratingd1.text()) * 1e-3
                self.params.Grating_d2 = float(self.gratingd2.text()) * 1e-3
                self.params.Grating_N = int(self.gratingN.text())
                
        except ValueError as e:
            QMessageBox.warning(self, "参数错误", f"请输入有效的数值参数: {str(e)}")
            return False
            
        return True
        
    def compute_errors(self, I_analytic, I_fft, epsilon=1e-10):
        """计算误差"""
        # 归一化
        I_analytic_norm = I_analytic / np.max(I_analytic)
        I_fft_norm = I_fft / np.max(I_fft)
        
        # 相对误差
        relative_error = np.mean(np.abs(I_fft_norm - I_analytic_norm) / (I_analytic_norm + epsilon))
        
        # 均方误差
        mse = np.mean((I_fft_norm - I_analytic_norm) ** 2)
        
        # 最大误差
        max_error = np.max(np.abs(I_fft_norm - I_analytic_norm))
        
        # 均方根误差
        rms_error = np.sqrt(np.mean((I_fft_norm - I_analytic_norm) ** 2))
        
        return relative_error, mse, max_error, rms_error
        
    def compute_diffraction(self):
        """计算衍射结果"""
        # 获取参数
        if not self.get_parameters_from_ui():
            return False
            
        # 创建观察屏坐标
        x = np.linspace(-self.params.obs_range_x/2, self.params.obs_range_x/2, self.params.N)
        y = np.linspace(-self.params.obs_range_y/2, self.params.obs_range_y/2, self.params.N)
        X, Y = np.meshgrid(x, y)
        
        # 根据衍射屏类型选择计算函数
        screen_type = self.params.PingMode
        
        try:
            if screen_type == 'circular':
                E_analytic, I_analytic, E_fft, I_fft, aperture_info = circular_aperture(X, Y, self.params)
            elif screen_type == 'rectangular':
                E_analytic, I_analytic, E_fft, I_fft, aperture_info = rectangular_aperture(X, Y, self.params)
            elif screen_type == 'grating':
                E_analytic, I_analytic, E_fft, I_fft, aperture_info = grating_aperture(X, Y, self.params)
            else:
                QMessageBox.warning(self, "错误", f"不支持的衍射屏类型: {screen_type}")
                return False
                
        except Exception as e:
            QMessageBox.critical(self, "计算错误", f"计算过程中发生错误: {str(e)}")
            return False
        
        # 保存计算结果
        self.calculation_results = {
            'E_analytic': E_analytic,
            'I_analytic': I_analytic,
            'E_fft': E_fft,
            'I_fft': I_fft,
            'X': X,
            'Y': Y,
            'x': x,
            'y': y,
            'aperture_info': aperture_info,
            'screen_type': screen_type
        }
        
        return True
        
    def update_display(self):
        """根据当前设置更新显示"""
        if not self.calculation_results:
            return
            
        if self.show_error_aperture:
            self.show_error_aperture_plots()
        else:
            self.show_diffraction_plots()
            
    def show_diffraction_plots(self):
        """显示衍射图"""
        # 选择要显示的数据
        if self.current_display_mode == 'E':
            analytical_data = np.abs(self.calculation_results['E_analytic'])
            fft_data = np.abs(self.calculation_results['E_fft'])
            data_label = "电场强度"
        else:
            analytical_data = self.calculation_results['I_analytic']
            fft_data = self.calculation_results['I_fft']
            data_label = "光强"
        
        # 根据维度显示
        if self.current_dimension == '2D':
            x_range = (-self.params.obs_range_x/2, self.params.obs_range_x/2)
            y_range = (-self.params.obs_range_y/2, self.params.obs_range_y/2)
            
            self.analytical_canvas.plot_2d(
                analytical_data, x_range, y_range, 
                f"解析解 - {self.params.PingMode}孔径", 
                log_scale=True,
                colorbar_label=data_label
            )
            
            self.fft_canvas.plot_2d(
                fft_data, x_range, y_range,
                f"FFT解 - {self.params.PingMode}孔径", 
                log_scale=True,
                colorbar_label=data_label
            )
            
        else:  # 1D显示
            # 取中心剖面 (y=0)
            center_idx = self.params.N // 2
            x_profile = self.calculation_results['x'] * 1e3 
            
            analytical_profile = analytical_data[center_idx, :]
            fft_profile = fft_data[center_idx, :]
            
            self.analytical_canvas.plot_1d(
                x_profile, analytical_profile,
                f"解析解 - {self.params.PingMode}孔径 (y=0剖面)",
                xlabel='x (mm)', ylabel=data_label
            )
            
            self.fft_canvas.plot_1d(
                x_profile, fft_profile,
                f"FFT解 - {self.params.PingMode}孔径 (y=0剖面)",
                xlabel='x (mm)', ylabel=data_label
            )
            
    def show_error_aperture_plots(self):
        """显示误差图和孔径图"""
        # 计算误差
        if self.current_display_mode == 'I':
            analytic_data = self.calculation_results['I_analytic']
            fft_data = self.calculation_results['I_fft']
        else:
            analytic_data = np.abs(self.calculation_results['E_analytic'])
            fft_data = np.abs(self.calculation_results['E_fft'])
            
        # 绝对误差
        abs_error = np.abs(fft_data - analytic_data)
        
        # 相对误差
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_error = np.abs(fft_data - analytic_data) / (analytic_data + 1e-10)
            rel_error[analytic_data < 1e-10] = 0
        
        # 获取孔径信息
        aperture_info = self.calculation_results['aperture_info']
        aperture_func = aperture_info['func']
        
        # 绝对误差
        x_range = (-self.params.obs_range_x/2, self.params.obs_range_x/2)
        y_range = (-self.params.obs_range_y/2, self.params.obs_range_y/2)
        
        self.analytical_canvas.plot_2d(
            abs_error, x_range, y_range, 
            f"绝对误差 - {self.params.PingMode}孔径",
            cmap='viridis',
            colorbar_label='绝对误差'
        )
        
        # 右图：孔径函数
        # 计算孔径函数的显示范围
        screen_type = self.params.PingMode
        if screen_type == 'circular':
            R = self.params.R
            aperture_x_range = [-R*1.2, R*1.2]
            aperture_y_range = [-R*1.2, R*1.2]
        elif screen_type == 'rectangular':
            a = self.params.Rect_Width
            b = self.params.Rect_Height
            aperture_x_range = [-a*0.6, a*0.6]
            aperture_y_range = [-b*0.6, b*0.6]
        elif screen_type == 'grating':
            d = self.params.Grating_d2
            num_slits = self.params.Grating_N
            aperture_x_range = [-d*num_slits*0.6, d*num_slits*0.6]
            aperture_y_range = [-d*0.6, d*0.6]
        else:
            aperture_x_range = [-1e-3, 1e-3]
            aperture_y_range = [-1e-3, 1e-3]
        
        self.fft_canvas.plot_2d(
            aperture_func, aperture_x_range, aperture_y_range,
            f"孔径函数 - {self.params.PingMode}孔径",
            cmap='gray',
            colorbar_label='透射率'
        )
        
    def compute_and_plot(self):
        """计算衍射并绘图"""
        # 清空之前的计算
        self.calculation_results = None
        
        # 重置视图模式为衍射图
        self.show_error_aperture = False
        self.errorAndap.setText("切换孔径/误差图")
        # 清空画布
        self.analytical_canvas.clear()
        self.fft_canvas.clear()
        
        # 计算衍射
        if not self.compute_diffraction():
            return
            
        # 更新显示
        self.update_display()
        
        # 计算并显示误差
        if self.current_display_mode == 'I':
            I_analytic = self.calculation_results['I_analytic']
            I_fft = self.calculation_results['I_fft']
            relative_error, mse, max_error, rms_error = self.compute_errors(I_analytic, I_fft)
            
        if self.current_display_mode == 'E':
            E_analytic = np.abs(self.calculation_results['E_analytic'])
            E_fft = np.abs(self.calculation_results['E_fft'])
            relative_error, mse, max_error, rms_error = self.compute_errors(E_analytic, E_fft)
        
        self.relative_error.setText(f"{relative_error:.6e}")
        self.mse.setText(f"{mse:.6e}")
        self.max_error.setText(f"{max_error:.6e}")
        self.rms_error.setText(f"{rms_error:.6e}")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle('光学衍射模拟 - 解析解 vs FFT解')
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()