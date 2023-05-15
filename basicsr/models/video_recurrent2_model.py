from itertools import count
import torch
from collections import Counter
from os import path as osp
from torch import distributed as dist
from tqdm import tqdm
import numpy as np
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img,tensor2npy
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.registry import MODEL_REGISTRY
from .video_base_model import VideoBaseModel
import os
#TODO 医学超分专用
@MODEL_REGISTRY.register()
class VideoRecurrentModel2(VideoBaseModel):

    def __init__(self, opt):
        super(VideoRecurrentModel2, self).__init__(opt)
        if self.is_train:
            self.fix_flow_iter = opt['train'].get('fix_flow')

    def setup_optimizers(self):
        train_opt = self.opt['train']
        flow_lr_mul = train_opt.get('flow_lr_mul', 1)
        logger = get_root_logger()
        logger.info(f'Multiple the learning rate for flow network with {flow_lr_mul}.')
        if flow_lr_mul == 1:
            optim_params = self.net_g.parameters()
        else:  # separate flow params and normal params for different lr
            normal_params = []
            flow_params = []
            for name, param in self.net_g.named_parameters():
                if 'spynet' in name:
                    flow_params.append(param)
                else:
                    normal_params.append(param)
            optim_params = [
                {  # add normal params first
                    'params': normal_params,
                    'lr': train_opt['optim_g']['lr']
                },
                {
                    'params': flow_params,
                    'lr': train_opt['optim_g']['lr'] * flow_lr_mul
                },
            ]

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def optimize_parameters(self, current_iter):
        if self.fix_flow_iter:
            logger = get_root_logger()
            if current_iter == 1:
                logger.info(f'Fix flow network and feature extractor for {self.fix_flow_iter} iters.')
                for name, param in self.net_g.named_parameters():
                    if 'spynet' in name or 'edvr' in name:
                        param.requires_grad_(False)
            elif current_iter == self.fix_flow_iter:
                logger.warning('Train all the parameters.')
                self.net_g.requires_grad_(True)

        super(VideoRecurrentModel2, self).optimize_parameters(current_iter)
    def counter(self,info_list):
        item={}
        for key in info_list:
            folder,_ = key.split("/")
            if folder in item.keys():
                item[folder]+=1
            else:
                item[folder]=0
        return item
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # TODO padding: reflection
        dataset = dataloader.dataset
        dataset_name = dataset.opt['name']
        if not osp.exists(osp.join(self.opt['path']['visualization'], dataset_name)):
            os.mkdir(osp.join(self.opt['path']['visualization'], dataset_name))
        with_metrics = self.opt['val']['metrics'] is not None
        # initialize self.metric_results
        # It is a dict: {
        #    'folder1': tensor (num_frame x len(metrics)),
        #    'folder2': tensor (num_frame x len(metrics))
        # }
        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {}
                num_frame_each_folder = Counter(dataset.data_info['folder'])
                #num_frame_each_folder = self.counter(dataset.keys)
                for folder, num_frame in num_frame_each_folder.items():
                    self.metric_results[folder] = torch.zeros(
                        num_frame, len(self.opt['val']['metrics']), dtype=torch.float32, device='cuda')
            # initialize the best metric results
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        rank, world_size = get_dist_info()
        if with_metrics:
            for _, tensor in self.metric_results.items():
                tensor.zero_()

        metric_data = dict()
        metric_data['max']=dataset.opt['max']
        num_folders = len(dataset)
        num_pad = (world_size - (num_folders % world_size)) % world_size
        if rank == 0:
            pbar = tqdm(total=len(dataset), unit='folder')
        # Will evaluate (num_folders + num_pad) times, but only the first num_folders results will be recorded.
        # (To avoid wait-dead)
        for i in range(rank, num_folders + num_pad, world_size):
            idx = min(i, num_folders - 1)
            val_data = dataset[idx]
            folder = val_data['folder']

            # TODO 预处理 改val_data['lq]
            try:
                val_data['lq'] = lr_padding(val_data['lq'],self.opt['val']['crop'])
            except:
                pass
            # compute outputs
            val_data['lq'].unsqueeze_(0)
            val_data['gt'].unsqueeze_(0)
            self.feed_data(val_data)
            val_data['lq'].squeeze_(0)
            val_data['gt'].squeeze_(0)

            _,_,h,w = val_data['gt'].shape

            self.test()
            visuals = self.get_current_visuals()
            # TODO 改成是否存在检测
            try:
                visuals['result'] = hr_de_padding(visuals['result'][0],(h,w)).unsqueeze(0)
            except:
                pass
            # tentative for out of GPU memory
            del self.lq
            del self.output
            if 'gt' in visuals:
                del self.gt
            torch.cuda.empty_cache()

            if self.center_frame_only:
                visuals['result'] = visuals['result'].unsqueeze(1)
                if 'gt' in visuals:
                    visuals['gt'] = visuals['gt'].unsqueeze(1)

            # evaluate
            if i < num_folders:
                imgs=[]
                for idx in range(visuals['result'].size(1)):
                    result = visuals['result'][0, idx, :, :, :]
                    # result_img = tensor2img([result])  # uint8, bgr
                    result_img = result
                    metric_data['img'] = result_img
                    if 'gt' in visuals:
                        gt = visuals['gt'][0, idx, :, :, :]
                        # gt_img = tensor2img([gt])  # uint8, bgr
                        gt_img = gt
                        metric_data['img2'] = gt_img

                    if save_img:
                        if self.opt['is_train']:
                            raise NotImplementedError('saving image is not supported during training.')
                        else:
                            if self.center_frame_only:  # vimeo-90k
                                clip_ = val_data['lq_path'].split('/')[-3]
                                seq_ = val_data['lq_path'].split('/')[-2]
                                name_ = f'{clip_}_{seq_}'
                                img_path = osp.join(self.opt['path']['visualization'], dataset_name, folder,
                                                    f"{name_}_{self.opt['name']}.png")
                            else:  # others
                                img_path = osp.join(self.opt['path']['visualization'], dataset_name, folder,
                                                    f"{idx:08d}_{self.opt['name']}.npy")
                        # image name only for REDS dataset


                    # TODO 保存npy文件
                    # calculate metrics
                    if with_metrics:
                        for metric_idx, opt_ in enumerate(self.opt['val']['metrics'].values()):
                            result = calculate_metric(metric_data, opt_)
                            self.metric_results[folder][idx, metric_idx] += result
                    # imgs.append(np.around(result_img*dataset.opt['max']))
                    imgs.append(tensor2npy(result_img,max=dataset.opt['max']))
                npyname = osp.join(self.opt['path']['visualization'], dataset_name, folder+f"_{self.opt['name']}.npy")
                imgs= np.stack(imgs,axis=2)
                np.save(npyname,imgs)
                # progress bar
                if rank == 0:
                    for _ in range(world_size):
                        pbar.update(1)
                        pbar.set_description(f'Folder: {folder}')

        if rank == 0:
            pbar.close()

        if with_metrics:
            if self.opt['dist']:
                # collect data among GPUs
                for _, tensor in self.metric_results.items():
                    dist.reduce(tensor, 0)
                dist.barrier()

            if rank == 0:
                self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def test(self):
        n = self.lq.size(1)
        self.net_g.eval()

        flip_seq = self.opt['val'].get('flip_seq', False)
        self.center_frame_only = self.opt['val'].get('center_frame_only', False)

        if flip_seq:
            self.lq = torch.cat([self.lq, self.lq.flip(1)], dim=1)

        with torch.no_grad():
            self.output = self.net_g(self.lq)
            self.output = torch.clamp(self.output,0,1)
        if flip_seq:
            output_1 = self.output[:, :n, :, :, :]
            output_2 = self.output[:, n:, :, :, :].flip(1)
            self.output = 0.5 * (output_1 + output_2)

        if self.center_frame_only:
            self.output = self.output[:, n // 2, :, :, :]

        self.net_g.train()
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
