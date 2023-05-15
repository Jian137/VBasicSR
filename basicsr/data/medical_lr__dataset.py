import os
import numpy as np
from tqdm import tqdm
import shutil

def get_numpy_slice(numpy_file_dir, target):
    slice_dir = target
    if not os.path.exists(slice_dir):
        os.mkdir(slice_dir)

    for file in os.listdir(numpy_file_dir):
        if os.path.isdir(file):  # 去除是文件夹的情况，此时不读文件夹
            continue
        file_name = os.path.join(numpy_file_dir, file)
        numpy_file = np.load(file_name)

        for i in range(numpy_file.shape[2]):
            new_slice_name = os.path.join(slice_dir, file[:-4] + "_" + str(i) + file[-4:])
            # print(new_slice_name)
            cur_slice = numpy_file[:, :, i]
            cur_slice = np.expand_dims(cur_slice, axis=2)
            print(cur_slice.shape)
            # np.save(new_slice_name, cur_slice)


def get_numpy_slice_to_directory(numpy_file_dir, target_directory, modal="PD_"):  # target_directory 代表哪个全是文件夹的文件夹
    slice_dir = target_directory
    if not os.path.exists(slice_dir):
        os.mkdir(slice_dir)

    for file in tqdm(os.listdir(numpy_file_dir)):
        if os.path.isdir(file):  # 去除是文件夹的情况，此时不读文件夹
            continue
        file_num = file.split('_')[-1].split('.')[0]
        # print(file_num)
        dir_name = os.path.join(target_directory, file_num.zfill(3))
        file_name = os.path.join(numpy_file_dir, file)

        if not os.path.exists(dir_name):  # 创建的文件夹是用来放当前file对应的slice
            os.mkdir(dir_name)
            # print(dir_name)


        numpy_file = np.load(file_name)

        for i in range(numpy_file.shape[2]):
            new_file_name = str(i).zfill(8) + ".npy"
            cur_slice = numpy_file[:, :, i]
            cur_slice = np.expand_dims(cur_slice, axis=2)
            # print(cur_slice.shape)
            # print(os.path.join(dir_name, new_file_name))
            np.save(os.path.join(dir_name, new_file_name), cur_slice)

def remove_folder(folder_name):
    for dirname in folder_name:
        if len(dirname) != 6:
            shutil.rmtree(os.path.join(folder_name, dirname))


if __name__ == "__main__":
    base_numpy_dir = "/home/vipsl416-4-zhanglize/Datasets/IXI"
    base_numpy_dir_lr = os.path.join(base_numpy_dir, "IXI_LR")
    base_numpy_dir_lr_list = []
    base_numpy_dir_hr = os.path.join(base_numpy_dir, "IXI_HR")
    base_numpy_dir_hr_list = []


    # this is for slicing the
    # for dir in os.listdir(base_numpy_dir_hr):
    #     if os.path.isdir(os.path.join(base_numpy_dir_hr, dir)):
    #         base_numpy_dir_hr_list.append(os.path.join(base_numpy_dir_hr, dir))

    # print(base_numpy_dir_hr_list)

    # for model_dir in base_numpy_dir_hr_list:
    # model_dir = '/home/vipsl416-4-zhanglize/Datasets/IXI/IXI_HR/PD'
    # model_dir_train = os.path.join(model_dir, "train")
    # model_dir_val = os.path.join(model_dir, "valid")
    # model_dir_test = os.path.join(model_dir, "test")
    # print(model_dir_train)
    # print(model_dir_val)
    # print(model_dir_test)
    # get_numpy_slice_to_directory(model_dir_train, model_dir_train + "_slice_video", modal="PD_")
    # get_numpy_slice_to_directory(model_dir_val, model_dir_val + "_slice_video", modal="PD_")
    # get_numpy_slice_to_directory(model_dir_test, model_dir_test + "_slice_video", modal="PD_")


    # for dir in os.listdir(base_numpy_dir_lr):
    #     if os.path.isdir(os.path.join(base_numpy_dir_lr, dir)):
    #         base_numpy_dir_lr_list.append(os.path.join(base_numpy_dir_lr, dir))

    # # print(base_numpy_dir_lr_list)

    # for lr_dir in base_numpy_dir_lr_list:
    #     for model_dir in os.listdir(lr_dir):
    #         model_dir_path = os.path.join(lr_dir, model_dir)
    #         model_dir_train = os.path.join(model_dir_path, "train")
    #         model_dir_val = os.path.join(model_dir_path, "valid")
    #         model_dir_test = os.path.join(model_dir_path, "test")
    #         get_numpy_slice(model_dir_train, model_dir_train + "_slice")
    #         get_numpy_slice(model_dir_val, model_dir_val + "_slice")
    #         get_numpy_slice(model_dir_test, model_dir_test + "_slice")


    ##################################################################################################

    # base_lr_dir = "/home/vipsl416-4-zhanglize/Datasets/IXI/IXI_LR/bicubic_2x/PD/valid"
    # base_hr_dir = "/home/vipsl416-4-zhanglize/Datasets/IXI/IXI_HR/PD/train"

    # # # base_lr_list = sorted(os.listdir(base_lr_dir))
    # # # base_hr_list = sorted(os.listdir(base_hr_dir))
    # # # base_hr_list_split=[i.split("_2X_")[-1] for i in base_hr_list]
    # # # os.rename()

    # for file in os.listdir(base_lr_dir):
    #     file_path = os.path.join(base_lr_dir, file)
    #     print(file_path)
    #     new_name = os.path.join(base_lr_dir, file[11:])
    #     print(new_name)
        # os.rename(file_path, new_name)

    #


    model_dir_path = "/home/vipsl416-4-zhanglize/Datasets/IXI/IXI_HR/T2"
    model_dir_train = os.path.join(model_dir_path, "train")
    model_dir_val = os.path.join(model_dir_path, "valid")
    model_dir_test = os.path.join(model_dir_path, "test")
    get_numpy_slice_to_directory(model_dir_train, model_dir_train + "_slice_video")
    get_numpy_slice_to_directory(model_dir_val, model_dir_val + "_slice_video")
    get_numpy_slice_to_directory(model_dir_test, model_dir_test + "_slice_video")