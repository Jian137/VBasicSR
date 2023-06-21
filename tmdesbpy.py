import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
# 读取图像
image = np.load("/home/vipsl416-4-zhanglize/Datasets/IXI/IXI_LR/bicubic_2x/PD/test_slice/500_000.npy")


import numpy as np
import matplotlib.pyplot as plt

def plot_frequency_spectrum(image,save_name):
    # 读取图像
    #image = np.load(image_path)
    #image = plt.imread(image_path)
    #image=np.expand_dims(image,axis=2)

    # 将图像转换为灰度图
    grayscale_image = np.mean(image, axis=2)

    # 进行傅里叶变换
    frequency_spectrum = np.fft.fft2(grayscale_image)

    # 移动零频率分量到频谱的中心
    frequency_spectrum_shifted = np.fft.fftshift(frequency_spectrum)

    # 计算频率谱的幅度谱
    amplitude_spectrum = np.abs(frequency_spectrum_shifted)
    #amplitude_spectrum=amplitude_spectrum/ np.max(amplitude_spectrum)

    # 统计不同频率的大小
    normalized_amplitude_spectrum = amplitude_spectrum / np.max(amplitude_spectrum)

    frequency_counts = np.sum(normalized_amplitude_spectrum, axis=0)
    frequency_counts = frequency_counts / sum(frequency_counts) *50

    #if max(frequency_counts)>10000:
    #    frequency_counts=frequency_counts//16
    frequency_axis = np.fft.fftshift(np.fft.fftfreq(grayscale_image.shape[0]))*image.shape[0]
    # 绘制频率统计图
    plt.figure()
    plt.plot(frequency_axis,frequency_counts)
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.title('Frequency Spectrum')

    plt.show()
    plt.savefig(save_name+".png")
    return frequency_axis,frequency_counts



# 传入图像路径调用函数
# image_path = "/home/vipsl416-4-zhanglize/Datasets/IXI/IXI_LR/bicubic_2x/PD/test_slice/500_000.npy"
image_path = "results/test_SwinIRM_SRx4_IXI_PD/visualization/IXI_PD_X4_test/500_000_test_SwinIRM_SRx4_IXI_PD.png"


# 绘制统计图

plt.title('Frequency Magnitude')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')

plt.show()

#lr_path="/home/vipsl416-4-zhanglize/Datasets/IXI/IXI_LR/bicubic_4x/PD/test_slice/500_000.npy"
lr_path="/home/vipsl416-4-zhanglize/Datasets/IXI/IXI_HR/PD/test_slice/500_000.npy"
lr_image=np.load(lr_path)
lr_image=np.expand_dims(lr_image,axis=2)

h,w,_ = lr_image.shape
#scale=4
#lr_image=transform.resize(lr_image, (h*scale, w*scale),order=3)



lr_axis,lr_count=plot_frequency_spectrum(lr_image,"LR_Fre")

sr_path = "results/test_SwinIRM_SRx4_IXI_PD/visualization/IXI_PD_X4_test/500_000_test_SwinIRM_SRx4_IXI_PD.png"



sr_image=plt.imread(sr_path)
sr_image=np.expand_dims(sr_image,axis=2)
sr_image=transform.resize(sr_image, (h, w),order=3)
sr_axis,sr_count=plot_frequency_spectrum(sr_image,"SR_Fre")


plt.figure()
plt.plot(lr_axis,lr_count-sr_count)
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum')

plt.show()
plt.savefig("cy"+".png")