# train for MSR IMDN
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 master_port=4321 basicsr/train.py -opt options/train/MSR/train_EDSR_Lx2.yml --launcher pytorch