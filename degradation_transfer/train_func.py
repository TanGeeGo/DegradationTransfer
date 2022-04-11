import time
import torch
import numpy as np
from torch.utils.data import dataset
from kernelnet import KernelNet
from dataloader import DataGenerator

def train_function(conf, args, input, label, ch_indicator):
    """
        train function
            input args:
                conf:
                args:
                input:
                label:
                ch_indcator:
    """
    # initialize the network model
    gan = KernelNet(conf, args, input, ch_indicator)
    # data generation
    dataset = DataGenerator(conf, gan, input, label)
    # batch size
    batch_size = conf.batch_size
    for iteration in range(conf.max_iters):
        # learning rate adjust
        if iteration % conf.update_l_rate_freq == 0:
            for params in gan.optimizer_G.param_groups:
                params['lr'] /= conf.update_l_rate_rate

        # generate a batch of input and label
        for batch_index in range(iteration*batch_size, (iteration+1)*batch_size, 1):
            [g_in, d_in] = dataset.__getitem__(batch_index)
            if batch_index == iteration*batch_size:
                # if the batch index is the first of the batch, copy it
                g_batch = g_in
                d_batch = d_in
            else:
                # generate the batch, concat the rest input
                g_batch = np.concatenate((g_batch, g_in), axis=0)
                d_batch = np.concatenate((d_batch, d_in), axis=0)

        g_batch = torch.from_numpy(g_batch).float()
        d_batch = torch.from_numpy(d_batch).float()
        g_batch = g_batch.cuda()
        d_batch = d_batch.cuda()
        
        gan.train(g_batch, d_batch, iteration)