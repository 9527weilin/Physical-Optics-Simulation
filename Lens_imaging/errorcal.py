# errorcal.py
import numpy as np
import cv2


def calculate_image_metrics(original, processed):
    """
    计算图像质量指标
    
    参数:
        original: 原始图像 (numpy数组)
        processed: 处理后的图像 (numpy数组)
        
    返回:
        包含各项指标的字典
    """
    try:
        # 确保图像大小相同
        if original.shape != processed.shape:
            # 调整大小
            processed_resized = cv2.resize(processed, (original.shape[1], original.shape[0]))
        else:
            processed_resized = processed
        
        # 归一化
        original_norm = original / np.max(original) if np.max(original) > 0 else original
        processed_norm = processed_resized / np.max(processed_resized) if np.max(processed_resized) > 0 else processed_resized
        
        # 计算均方误差 (MSE)
        mse = np.mean((original_norm - processed_norm) ** 2)
        
        # 计算均方根误差 (RMSE)
        rmse = np.sqrt(mse)
        
        # 计算峰值信噪比 (PSNR)
        if mse == 0:
            psnr = 100
        else:
            psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        
        # 计算相关系数
        if len(original_norm.flatten()) > 1:
            corr_coef = np.corrcoef(original_norm.flatten(), processed_norm.flatten())[0, 1]
        else:
            corr_coef = 0
        
        # 返回指标字典
        return {
            'mse': mse,
            'rmse': rmse,
            'psnr': psnr,
            'corr_coef': corr_coef
        }
        
    except Exception as e:
        print(f"计算图像质量指标错误: {e}")
        return {
            'mse': float('nan'),
            'rmse': float('nan'),
            'psnr': float('nan'),
            'corr_coef': float('nan')
        }