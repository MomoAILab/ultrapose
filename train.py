"""
created by songkey@pku.edu.cn
date @ 2021-10-15
"""
import os
import os.path as osp
import cv2
import numpy as np
import random
import torch
import utils.misc as misc

from options.train_options import TrainOptions
from dataset import create_dataset
from networks import create_model
from utils.visualizer import Visualizer

cur_dir = osp.dirname(__file__)

def train(args, train_dataset, train_dataloader, model, visualizer, it_start=0):
    model.train()
    for iteration, batch_data in enumerate(train_dataloader):

        fwd_dict = model(batch_data)
        loss_dict = fwd_dict['loss_dict']

        losses = sum(loss_dict[k] for k in loss_dict.keys())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = misc.reduce_dict(loss_dict)
        loss_dict_reduced = sum(loss_dict_reduced.values())
        loss_value = loss_dict_reduced.item()

        model.optimizer_g.zero_grad()
        losses.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.net_g.parameters(), args.grad_clip_thresh)
        model.optimizer_g.step()

        if args.rank == 0 and iteration % 2 == 0:
            log_info = "ep:{} it:{} ct:{} total:{:.3f} norm:{:.3f}".format(epoch, iteration, it_start, losses, grad_norm)
            for key, val in fwd_dict['loss_dict'].items():
                log_info += " {}:{:.3f}".format(key, val)

            print(log_info)
            visualizer.display_train_hist(log_info)
            loss_keys =list(fwd_dict['loss_dict'].keys()) + ['total']
            loss_vec = []
            for value in fwd_dict['loss_dict'].values():
                loss_vec.append(value.detach().cpu())
            loss_vec = np.array(loss_vec + [losses.cpu().detach()], dtype=np.float32)
            # loss_vec = np.array(list(fwd_dict['loss_dict'].values()) + [losses.cpu()], dtype=np.float32)
            visualizer.plot_train_loss(epoch, iteration, it_start, loss_vec, loss_keys)

            rand_idx = random.randint(0, args.batch_size-1)
            show = train_dataset.show_training_results(batch_data, fwd_dict, rand_idx)
            visualizer.show_image(show, 'train_images', 'train_images')

            if not args.use_visdom:
                cv2.imshow('show', show)
                cv2.waitKey(1)
        it_start += 1
    return it_start

if __name__ == '__main__':

    train_options = TrainOptions()
    args = train_options.parse()

    # model
    visualizer = Visualizer(args)

    misc.init_distributed_mode(args)
    if args.rank == 0:
        train_options.print_options(args)
    if args.distributed_run:
        seed = args.seed + misc.get_rank()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # torch.backends.cudnn.benchmark = True

    if args.distributed:
        device = torch.device("cuda:%d" % args.gpu)
    else:
        device = torch.device(args.device)

    model = create_model(args, device)

    learning_rate = args.lr
    begin_epoch, it_start = 0, 0
    if osp.exists(args.finetune_model_path):
         begin_epoch, learning_rate, it_start = model.load_checkpoint(visualizer, args)

    net = model.net_g
    if args.distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])
        model_without_ddp = net.module

    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    if misc.get_rank() == 0: print('number of params:', n_parameters)

    for epoch in range(begin_epoch, args.total_epoch_num):
        args.data_pool = epoch
        train_dataloader, train_dataset = create_dataset(args, is_train=True)
        it_start = train(args, train_dataset, train_dataloader, model, visualizer, it_start)
        if args.rank == 0 and epoch % args.check_per_epoch == 0 and epoch != 0:
            if not osp.exists(osp.join(args.checkpoints_dir)): os.makedirs(args.checkpoints_dir)
            save_path = osp.join(args.checkpoints_dir, "dp_ep{}_ct{}_bs{}_ws{}_{}_data_{}_model.pth".format(
                                  epoch,
                                  it_start,
                                  args.batch_size,
                                  args.world_size,
                                  args.dataset_name,
                                  args.model_name))
            model.checkpoint(epoch, it_start, visualizer, learning_rate, save_path)
        del train_dataloader
        del train_dataset


