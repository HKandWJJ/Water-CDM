import numpy as np
import cv2
from cv2.ximgproc import guidedFilter

def singleScaleRetinex(img, sigma):

    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))

    return retinex

def multiScaleRetinex(img, sigma_list):

    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        retinex += singleScaleRetinex(img, sigma)

    retinex = retinex / len(sigma_list)

    return retinex

def simplestColorBalance(img, low_clip, high_clip):    

    total = img.shape[0] * img.shape[1]
    for i in range(img.shape[2]):
        unique, counts = np.unique(img[:, :, i], return_counts=True)
        current = 0
        for u, c in zip(unique, counts):            
            if float(current) / total < low_clip:
                low_val = u
            if float(current) / total < high_clip:
                high_val = u
            current += c
                
        img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)

    return img    

def gamma_correction(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def adaptive_gamma_correction(image, guide_image):
    # 计算引导图像的灰度平均值
    gray_guide = cv2.cvtColor(guide_image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray_guide)

    # 根据平均亮度自适应调整Gamma值
    gamma = np.clip(1.5 - mean_brightness / 128, 0.5, 2.0)
    
    # 应用Gamma校正到目标图像
    corrected_image = gamma_correction(image, gamma)
    
    return corrected_image

# 这个函数求的是 色彩恢复因子Ci
def colorRestoration(img, alpha, beta):  

    img_sum = np.sum(img, axis=2, keepdims=True)   # 按通道求和

    color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))

    return color_restoration

def adjust_brightness_contrast(img1, img2, guide_img):
    img1_hsl = cv2.cvtColor(img1, cv2.COLOR_BGR2HLS)
    img2_hsl = cv2.cvtColor(img2, cv2.COLOR_BGR2HLS)
    
    img1_l = img1_hsl[:, :, 1]
    img2_l = img2_hsl[:, :, 1]

    # 计算局部对比度
    contrast1 = compute_local_contrast(img1_l, guide_img)
    contrast2 = compute_local_contrast(img2_l, guide_img)

    # 计算自适应权重
    weight1 = contrast1 / (contrast1 + contrast2 + 1e-5)
    weight2 = contrast2 / (contrast1 + contrast2 + 1e-5)
    
    # 融合亮度通道
    blended_l_channel = (weight1 * img1_l + weight2 * img2_l).astype(np.uint8)
    img1_hsl[:, :, 1] = blended_l_channel
    blended_image = cv2.cvtColor(img1_hsl, cv2.COLOR_HLS2BGR)
    
    return blended_image

def automatedMSRCR(img, guide_img, sigma_list=[100, 200, 300]):
    img1 = np.float64(img) + 1.0

    img_retinex = multiScaleRetinex(img1, sigma_list)   # 求 MSR
    for i in range(img_retinex.shape[2]):  # 维度循环
        # unique()：返回参数数组中所有不同的值，并按照从小到大排序
        # unique 去重后重新排序的数组，count去重后 不同数据的个数
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)  # 这里*了100 数据放大100倍
        
        zero_count = 0
        for u, c in zip(unique, count):  
            if u == 0:
                zero_count = c  # 数组中0的个数
                break
        #  下面数据/100 都是为了还原数据 因为上面*了100  
        low_val = unique[0] / 100.0    #  MSR 结果 img_retinex中的最小值
        high_val = unique[-1] / 100.0  #  MSR 结果 img_retinex中的最大值
        
        # 把最大值和最小值给收缩了，防止两极化现象
        for u, c in zip(unique, count):  
            if u < 0 and c < zero_count * 0.05:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.05:
                high_val = u / 100.0
                break
        
        # 限定范围  把高于high_val 和低于low_val 用 high_val、low_val代替 。
        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)
        # 直接线性量化  0-255
        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255

    img_retinex = np.uint8(img_retinex)
    img_retinex = adjust_brightness_contrast(img_retinex, img, guide_img)
    # img_retinex = adaptive_gamma_correction(img_retinex, guide_img)

        
    return img_retinex

def MSRCP(img, guide_img, sigma_list=[100, 200, 300], low_clip=0.01, high_clip=0.99):

    img1 = np.float64(img) + 1.0

    intensity = np.sum(img1, axis=2) / img1.shape[2]  

    retinex = multiScaleRetinex(intensity, sigma_list)

    intensity = np.expand_dims(intensity, 2)
    retinex = np.expand_dims(retinex, 2)

    intensity1 = simplestColorBalance(retinex, low_clip, high_clip)

    intensity1 = (intensity1 - np.min(intensity1)) / \
                 (np.max(intensity1) - np.min(intensity1)) * \
                 255.0 + 1.0

    img_msrcp = np.zeros_like(img1)
    
    for y in range(img_msrcp.shape[0]):
        for x in range(img_msrcp.shape[1]):
            B = np.max(img1[y, x])
            A = np.minimum(256.0 / B, intensity1[y, x, 0] / intensity[y, x, 0])
            img_msrcp[y, x, 0] = A * img1[y, x, 0]
            img_msrcp[y, x, 1] = A * img1[y, x, 1]
            img_msrcp[y, x, 2] = A * img1[y, x, 2]

    img_msrcp = np.uint8(img_msrcp - 1.0)
    img_msrcp = adjust_brightness_contrast(img_msrcp, img, guide_img)

    return img_msrcp

def HSL_L(img, guide_img):

    image = automatedMSRCR(img, guide_img)
    hsl_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    l_channel = hsl_image[:, :, 1]  # 1表示亮度通道

    return l_channel

def median_fusion(img1, img2):

    l_channel1 = HSL_L(img1)
    hsl_image = cv2.cvtColor(img2, cv2.COLOR_BGR2HLS)
    l_channel2 = hsl_image[:, :, 1]
    l_channels = np.stack([l_channel1, l_channel2], axis=0)
    # 对合并后的张量进行中值融合
    fused_l_channel = np.median(l_channels, axis=0)
    hsl_image[:, :, 1] = fused_l_channel.astype(np.uint8)
    fused_image = cv2.cvtColor(hsl_image, cv2.COLOR_HLS2BGR)

    return fused_image


def compute_local_contrast(image, guide_img, kernel_size=15, method='guided'):
    if method == 'mean':
        # 均值模糊
        mean = cv2.blur(image, (kernel_size, kernel_size))
        squared_mean = cv2.blur(image**2, (kernel_size, kernel_size))
    elif method == 'gaussian':
        # 高斯模糊
        mean = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        squared_mean = cv2.GaussianBlur(image**2, (kernel_size, kernel_size), 0)
    elif method == 'bilateral':
        # 双边滤波
        mean = cv2.bilateralFilter(image, kernel_size, 75, 75)
        squared_mean = cv2.bilateralFilter(image**2, kernel_size, 75, 75)
    elif method == 'median':
        # 中值滤波
        mean = cv2.medianBlur(image, kernel_size)
        squared_mean = cv2.medianBlur(image**2, kernel_size)
    elif method == 'guided':
        # 引导滤波
        radius = kernel_size // 2
        eps = 0.01
        mean = guidedFilter(guide_img, image, radius, eps)
        squared_mean = guidedFilter(guide_img**2, image, radius, eps)
    elif method == 'nlm':
        # 非局部均值滤波
        h = 10
        mean = cv2.fastNlMeansDenoising(image, h=h)
        squared_mean = cv2.fastNlMeansDenoising(image**2, h=h)
    else:
        raise ValueError("Unknown method: choose from 'mean', 'gaussian', 'bilateral', 'median', 'guided', 'adaptive_mean', 'nlm'")
    
    variance = squared_mean - mean**2
    contrast = np.sqrt(variance)
    return contrast

def adaptive_weighted_fusion(img1, img2, guide_img):
    l_channel1 = HSL_L(img1, guide_img)
    hsl_image = cv2.cvtColor(img2, cv2.COLOR_BGR2HLS)
    l_channel2 = hsl_image[:, :, 1]

    # 计算局部对比度
    contrast1 = compute_local_contrast(l_channel1, guide_img)
    contrast2 = compute_local_contrast(l_channel2, guide_img)

    # 计算自适应权重
    weight1 = contrast1 / (contrast1 + contrast2 + 1e-5)
    weight2 = contrast2 / (contrast1 + contrast2 + 1e-5)

    # 融合亮度通道
    blended_l_channel = (weight1 * l_channel1 + weight2 * l_channel2).astype(np.uint8)
    hsl_image[:, :, 1] = blended_l_channel
    blended_image = cv2.cvtColor(hsl_image, cv2.COLOR_HLS2BGR)

    # 添加滤波算法
    # smoothed_image = cv2.bilateralFilter(blended_image, 5, 40, 40)
    smoothed_image = guidedFilter(guide_img, blended_image, 7, 0.01)

    return smoothed_image