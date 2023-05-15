import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

@MODEL_REGISTRY.register()
class DAN4MSRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(DAN4MSRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()
        # TODO
        # self.boundaries_loss = BoundariesLoss(k_size=opt["kernel_size"]).to(self.device)
        self.boundaries_loss = BoundariesLoss(k_size=10).to(self.device)
        self.centralized_loss = CentralizedLoss(k_size=opt["kernel_size"], scale_factor=opt["scale"]).to(self.device)
    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
    def calc_kernel_loss(self, curr_k, g_output, g_input):
        # Calculate K which is equivalent to G
        # self.calc_curr_k()
        # Calculate constraints
        # loss_bicubic = self.bicubic_loss(g_input=g_input, g_output=g_output)
        loss_boundaries = self.boundaries_loss(kernel=curr_k)
        # loss_sum2one = self.sum2one_loss(kernel=curr_k) #useless
        loss_centralized = self.centralized_loss.forward(kernel=curr_k) #chuxina nan
        # loss_sparse = self.sparse_loss(kernel=curr_k) # chuxian nan
        # Apply constraints co-efficients
        lambda_sum2one = 0.01
        lambda_bicubic = 5
        lambda_boundaries = 1e-0
        lambda_centralized = 1e-2
        lambda_sparse = 5
        # return loss_sum2one * lambda_sum2one + \
        return loss_boundaries * lambda_boundaries + \
               loss_centralized * lambda_centralized #+ \
               # loss_sparse * lambda_sparse #+ \
        # return loss_bicubic * lambda_bicubic
    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.outputs,self.kermaps = self.net_g(self.lq)

        self.output = self.outputs[-1]
        self.fake_ker = self.kermaps[-1]

        l_total = 0
        loss_dict = OrderedDict()


        for ind in range(len(self.kermaps)):
            # d_kr = 1e-5*self.calc_kernel_loss(self.kermaps[ind], self.lq, self.gt)
            if self.cri_pix:
                l_pix = self.cri_pix(self.outputs[ind], self.gt)
                l_total += l_pix
                loss_dict['l_pix%d'%ind] = l_pix
            # l_total += d_kr
            ## loss_dict['l_ker%d'%ind]=d_kr.item()
            # pixel loss




        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.outputs,self.kermaps = self.net_g_ema(self.lq)
                self.output=self.outputs[-1]
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.outputs,self.kermaps = self.net_g(self.lq)
                self.output=self.outputs[-1]
            self.net_g.train()

    def test_selfensemble(self):
        # TODO: to be tested
        # 8 augmentations
        # modified from https://github.com/thstkdgus35/EDSR-PyTorch

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        # prepare augmented data
        lq_list = [self.lq]
        for tf in 'v', 'h', 't':
            lq_list.extend([_transform(t, tf) for t in lq_list])

        # inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
            self.net_g.train()

        # merge results
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')
        output = torch.cat(out_list, dim=0)

        self.output = output.mean(dim=0, keepdim=True)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()


            # sr_img = tensor2img([visuals['result']])
            metric_data['img'] = torch.clamp(visuals['result'],0,1)
            metric_data['img2'] = visuals['gt']

            # tentative for out of GPU memory


            if save_img: #TODO save npy文件
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t{metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)


class BoundariesLoss(nn.Module):
    """ Encourages sparsity of the boundaries by penalizing non-zeros far from the center """

    def __init__(self, k_size):
        super(BoundariesLoss, self).__init__()
        self.mask = map2tensor(create_penalty_mask(k_size, 30))
        self.zero_label = Variable(torch.zeros(k_size).cuda(), requires_grad=False)
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        #print("mask:",self.mask.shape)
        inputs = kernel * self.mask
        labels = torch.zeros_like(inputs)
        #print("BoundariesLoss inputs:", inputs.shape, "labels:", labels.shape)
        return self.loss(labels, inputs)
def map2tensor(gray_map):
    """Move gray maps to GPU, no normalization is done"""
    return torch.FloatTensor(gray_map).unsqueeze(0).unsqueeze(0).cuda()
def create_penalty_mask(k_size, penalty_scale):
    """Generate a mask of weights penalizing values close to the boundaries"""
    center_size = k_size // 2 + k_size % 2
    mask = create_gaussian(size=k_size, sigma1=k_size, is_tensor=False)
    mask = 1 - mask / np.max(mask)
    margin = (k_size - center_size) // 2 - 1
    mask[margin:-margin, margin:-margin] = 0
    return penalty_scale * mask
def create_gaussian(size, sigma1, sigma2=-1, is_tensor=False):
    """Return a Gaussian"""
    func1 = [np.exp(-z ** 2 / (2 * sigma1 ** 2)) / np.sqrt(2 * np.pi * sigma1 ** 2) for z in range(-size // 2 + 1, size // 2 + 1)]
    func2 = func1 if sigma2 == -1 else [np.exp(-z ** 2 / (2 * sigma2 ** 2)) / np.sqrt(2 * np.pi * sigma2 ** 2) for z in range(-size // 2 + 1, size // 2 + 1)]
    return torch.FloatTensor(np.outer(func1, func2)).cuda() if is_tensor else np.outer(func1, func2)
class CentralizedLoss(nn.Module):
    """ Penalizes distance of center of mass from K's center"""

    def __init__(self, k_size, scale_factor=.5):
        super(CentralizedLoss, self).__init__()
        self.indices = Variable(torch.arange(0., float(k_size)).cuda(), requires_grad=False)
        wanted_center_of_mass = k_size // 2 + 0.5 * (int(1 / scale_factor) - k_size % 2)
        # print("wanted:",wanted_center_of_mass)
        self.center = Variable(torch.FloatTensor([wanted_center_of_mass, wanted_center_of_mass]).cuda(), requires_grad=False)
        self.loss = nn.MSELoss()

    def forward(self, kernel):
        """Return the loss over the distance of center of mass from kernel center """
        B, k, k = kernel.shape
        r_sum, c_sum = torch.sum(kernel, 2).reshape(B, -1, k), torch.sum(kernel, 1).reshape(B, -1, k)
        # print(self.center.shape)
        # print(c_sum.shape)
        # print(self.indices.unsqueeze(0).unsqueeze(0).expand(B,-1,-1).shape)
        inputs = torch.stack((torch.matmul(r_sum, self.indices.unsqueeze(1).unsqueeze(0).expand(B,-1,-1)) / (torch.sum(kernel)),
                                      torch.matmul(c_sum, self.indices.unsqueeze(1).unsqueeze(0).expand(B,-1,-1)) / (torch.sum(kernel)))).squeeze(3).permute(1,2,0)
        labels = self.center.unsqueeze(0).expand(B,-1,-1)
        #print("CentralizedLoss inputs:", inputs.shape, "labels:", labels.shape)
        return self.loss(labels, inputs)
