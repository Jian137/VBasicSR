from math import floor, ceil
import torch
import torch.nn as nn


#  原始的数据形式：slice * c * w * h
#  要变成的数据形式: slice * c * (w * crop_size), (h * crop_size)

def lr_padding(volume: torch.tensor, crop_size_scale):  # w表示倒数第二个维度， h表示倒数第一个维度
    w, h = volume.shape[-2], volume.shape[-1]
    # assert (crop_size_w_scale >= 1 and crop_size_w_scale >= 1)

    crop_size_w = ceil(w / crop_size_scale) * crop_size_scale
    crop_size_h = ceil(h / crop_size_scale) * crop_size_scale

    rest_w = crop_size_w - w
    rest_h = crop_size_h - h
    # 如果rest是奇数，则 右侧和下侧 比 左侧和上侧 多补一个
    m = nn.ReflectionPad2d(
        (floor(rest_h / 2), rest_h - floor(rest_h / 2), floor(rest_w / 2), rest_w - floor(rest_w / 2)))

    # new_tensor = torch.zeros([volume.shape[0], volume.shape[1], crop_size_w, crop_size_h],
    #                          dtype=volume.dtype)  # 保证数据类型一致

    # for i in range(volume.shape[0]):  # 遍历每一个2d图像
    #     cur_image = volume[i]
    #     new_tensor[i] = m(cur_image).unsqueeze(0)

    # return new_tensor
    volume = m(volume)
    return volume


def hr_de_padding(volume: torch.tensor, original_tensor_shape: tuple):
    w, h = original_tensor_shape[0], original_tensor_shape[1]
    assert (w < volume.shape[-2] and h < volume.shape[-1])
    rest_w = volume.shape[-2] - w
    rest_h = volume.shape[-1] - h
    return volume[:, :, floor(rest_w / 2) : floor(rest_w / 2) + w, floor(rest_h / 2) : floor(rest_h / 2) + h]


if __name__ == "__main__":
    """test_tensor = torch.randn(1, 1, 2, 3)
    new_tensor = lr_padding(test_tensor, 4, 6)
    print(test_tensor)
    print(new_tensor)
    de_padding = hr_de_padding(new_tensor, (2, 3))
    print(de_padding)"""
    lr = torch.rand(3,1,240,240)
    lr_padding = lr_padding(lr, 32)
    hr_re = hr_de_padding(lr_padding,(240,240))
    print(lr == hr_re)
