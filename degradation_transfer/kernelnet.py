import os
import cv2
import torch
import loss
import time
import numpy as np
import scipy.io as sio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.image as mpimg
from utils import create_dir, unprocess, process
from scipy.ndimage import measurements, interpolation

class Generator(nn.Module):
    def __init__(self, conf):
        super(Generator, self).__init__()
        struct = conf.G_structure
        # First layer - Converting RGB image to latent space
        self.first_layer = nn.Conv2d(in_channels=1, out_channels=conf.G_chan, kernel_size=struct[0], bias=False)

        feature_block = []  # Stacking intermediate layer
        for layer in range(1, len(struct) - 1):
            # feature_block += [nn.Conv2d(in_channels=conf.G_chan, out_channels=conf.G_chan, kernel_size=struct[layer], bias=False),
            #                   nn.ReLU(True)]
            feature_block += [nn.Conv2d(in_channels=conf.G_chan, out_channels=conf.G_chan, kernel_size=struct[layer], bias=False)]
        self.feature_block = nn.Sequential(*feature_block)
        # Final layer - Down-sampling and converting back to image
        self.final_layer = nn.Conv2d(in_channels=conf.G_chan, out_channels=1, kernel_size=struct[-1],
                                     stride=int(1 / conf.scale_factor), bias=False)

        # Calculate number of pixels shaved in the forward pass
        self.output_size = self.forward(torch.FloatTensor(torch.ones([1, 1, conf.input_crop_size, conf.input_crop_size])), conf.G_kernel_size).shape[-1]
        self.forward_shave = int(conf.input_crop_size * conf.scale_factor) - self.output_size
        self.G_kernel_size = conf.G_kernel_size


    def forward(self, input_tensor, kernel_size):
        [B, C, H, W] = input_tensor.shape
        downscaled = self.first_layer(input_tensor)
        features = self.feature_block(downscaled)
        output = self.final_layer(features)
        res_tensor = input_tensor[:, :, int((kernel_size-1) / 2) : int((kernel_size-1) / 2) + (H-kernel_size+1),
                                        int((kernel_size-1) / 2) : int((kernel_size-1) / 2) + (W-kernel_size+1)]

        return output + res_tensor
        # return output

def weights_init_G(m):
    """ initialize weights of the generator """
    if m.__class__.__name__.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight, 0.1)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)

class Discriminator(nn.Module):

    def __init__(self, conf):
        super(Discriminator, self).__init__()

        # First layer - Convolution (with no ReLU)
        self.first_layer = nn.utils.spectral_norm(nn.Conv2d(in_channels=1, out_channels=conf.D_chan, kernel_size=conf.D_kernel_size, bias=True))
        feature_block = []  # Stacking layers with 1x1 kernels
        for _ in range(1, conf.D_n_layers - 1):
            feature_block += [nn.utils.spectral_norm(nn.Conv2d(in_channels=conf.D_chan, out_channels=conf.D_chan, kernel_size=1, bias=True)),
                              nn.BatchNorm2d(conf.D_chan),
                              nn.ReLU(True)]
        self.feature_block = nn.Sequential(*feature_block)
        self.final_layer = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(in_channels=conf.D_chan, out_channels=1, kernel_size=1, bias=True)),
                                         nn.Sigmoid())

        # Calculate number of pixels shaved in the forward pass
        self.forward_shave = conf.input_crop_size - self.forward(torch.FloatTensor(torch.ones([1, 1, conf.input_crop_size, conf.input_crop_size]))).shape[-1]

    def forward(self, input_tensor):
        receptive_extraction = self.first_layer(input_tensor)
        features = self.feature_block(receptive_extraction)
        return self.final_layer(features)

def weights_init_D(m):
    """ initialize weights of the discriminator """
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight, 0.1)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif class_name.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class KernelNet:

    def __init__(self, conf, args, input, ch_indicator):
        # acquire the configuration
        self.conf = conf
        self.args = args

        # define the generator
        self.G = Generator(self.conf).cuda()
        
        # batch size of the network
        self.batch_size = self.conf.batch_size

        # Initialize a kernel K for emitating
        self.curr_k = torch.FloatTensor(self.conf.G_kernel_size, self.conf.G_kernel_size).cuda()

        # Losses
        self.sum2one_loss = loss.SumOfWeightsLoss().cuda()
        self.boundaries_loss = loss.BoundariesLoss(k_size=self.conf.G_kernel_size).cuda()
        self.centralized_loss = loss.CentralizedLoss(k_size=self.conf.G_kernel_size, scale_factor=self.conf.scale_factor).cuda()
        self.sparse_loss = loss.SparsityLoss().cuda()

        # Constraint co-efficients
        self.lambda_sum2one = self.conf.lambda_sum2one
        self.lambda_boundaries = self.conf.lambda_boundaries

        self.loss_L1_sum = 0
        self.loss_boundaries_sum = 0
        self.loss_sum2one_sum = 0
        self.total_loss_L1_sum = 0

        # Define loss function
        self.criterionMSE = torch.nn.MSELoss().cuda()
        self.criterionL1 = torch.nn.L1Loss().cuda()

        # Initialize the weights of networks
        self.G.apply(weights_init_G)
        # self.D.apply(weights_init_D)

        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=self.conf.g_lr, betas=(self.conf.beta1, 0.999))
        
        # test image used for output the results after convolution
        self.test_input_ori = input
        self.test_input_val = mpimg.imread(os.path.join(self.conf.v_input_dir, 'validation_img.png'))
        ########################################################################
        if (ch_indicator == 0) or (ch_indicator == 1):
            self.color_idx = 'r'
        elif ch_indicator == 2:
            self.color_idx = 'g'
        elif ch_indicator == 3:
            self.color_idx = 'b'
        self.wb = args.white_balance
        ########################################################################
        self.test_input_ori = torch.FloatTensor(self.test_input_ori)
        self.test_input_val = torch.FloatTensor(self.test_input_val)
        self.test_input_ori = F.pad(self.test_input_ori,
                                    (self.conf.G_kernel_size-1, self.conf.G_kernel_size-1,
                                     self.conf.G_kernel_size-1, self.conf.G_kernel_size-1),
                                    "constant", value=0).unsqueeze(0).unsqueeze(0).cuda()
        self.test_input_val = F.pad(self.test_input_val,
                                    (self.conf.G_kernel_size-1, self.conf.G_kernel_size-1,
                                     self.conf.G_kernel_size-1, self.conf.G_kernel_size-1),
                                    "constant", value=0).unsqueeze(0).unsqueeze(0).cuda()

        self.test_input_ori = unprocess(self.test_input_ori, self.wb, self.color_idx)
        self.test_input_val = unprocess(self.test_input_val, self.wb, self.color_idx)

        self.curr_img_ori = torch.zeros_like(self.test_input_ori).cuda()
        self.curr_img_val = torch.zeros_like(self.test_input_val).cuda()

    def train(self, g_input, d_input, iteration):
        self.g_input = unprocess(g_input, self.wb, self.color_idx)

        self.d_input = d_input
        ########################################################################
        ## train generator
        # zeroize gradients
        self.optimizer_G.zero_grad()
        # Generator forward pass
        g_pred = self.G.forward(self.g_input, self.conf.G_kernel_size)
        # Pass Generators output through Discriminator
        # d_pred_fake = self.D.forward(g_pred)

        # inverse pipeline, after process we compute the loss
        g_pred = process(g_pred, self.wb, self.color_idx)

        # Calculate generator loss, based on discriminator prediction on generator result
        # loss_g = self.criterionGAN(d_last_layer=d_pred_fake, is_d_input_real=True)
        # Calculate the mseloss
        # loss_mse = self.criterionMSE(g_pred, self.d_input)
        # Calculate the L1loss
        loss_L1 = self.criterionL1(g_pred, self.d_input)

        # Sum generator losses
        [loss_boundaries, loss_sum2one] = self.calc_constraints(g_pred, iteration, self.conf.G_kernel_size)
        # total_loss_g = loss_g + loss_sum2one * self.lambda_sum2one + \
        #                loss_boundaries * self.lambda_boundaries + \
        #                loss_centralized * self.lambda_centralized + \
        #                loss_sparse * self.lambda_sparse
        # Sum the fidelity losses
        # total_loss_mse = loss_mse + loss_sum2one*0.001 + loss_boundaries*100000
        # total_loss_L1 = loss_L1
        # total_loss_L1 = loss_L1 + loss_sum2one*0.001
        # total_loss_L1 = loss_L1 + loss_boundaries*100000
        total_loss_L1 = loss_L1 + loss_sum2one*self.lambda_sum2one + loss_boundaries*self.lambda_boundaries

        ########################################################################
        #mse_test = torch.nn.MSELoss().cuda()
        #mse_test_value = mse_test(self.g_input[:, :, 6:58, 6:58], self.d_input)
        #print(mse_test_value)
        ########################################################################

        # Calculate gradients
        total_loss_L1.backward()
        # Update weights
        self.optimizer_G.step()
        ########################################################################
        ## train discriminator
        # Zeroize gradients
        # self.optimizer_D.zero_grad()
        # Discriminator forward pass over real example
        # d_pred_real = self.D.forward(self.d_input)
        # Discriminator forward pass over fake example (generated by generator)
        # Note that generator result is detached so that gradients are not propagating back through generator
        # g_output = self.G.forward(self.g_input, self.conf.G_kernel_size)
        # d_pred_fake = self.D.forward(g_output.detach())
        # Calculate discriminator loss
        # loss_d_fake = self.criterionGAN(d_pred_fake, is_d_input_real=False)
        # loss_d_real = self.criterionGAN(d_pred_real, is_d_input_real=True)
        # loss_d = (loss_d_fake + loss_d_real) * 0.5
        # Calculate gradients, note that gradients are not propagating back through generator
        # loss_d.backward()
        # Update weights, note that only discriminator weights are updated (by definition of the D optimizer)
        # self.optimizer_D.step()

        ########################################################################
        # add loss together
        self.add_loss(loss_L1, loss_boundaries, loss_sum2one, total_loss_L1)

        # every 1000 iteration, print the loss results
        if ((iteration % 1000 == 0) and (iteration != 0)) or (iteration == self.conf.max_iters - 1):
            print("Iteration[{:0>3}]".format(iteration))
            if (iteration == self.conf.max_iters - 1):
                # save the kernel and the image result of each iteration
                self.gen_curr_k(iteration, self.color_idx)
                self.gen_curr_img(iteration, self.conf.G_kernel_size, self.color_idx)
            [loss_L1_avg, loss_boundaries_avg, loss_sum2one_avg, total_loss_L1_avg] = self.avg_loss()
            print("L1 Loss: {:.8f} Sum2one Loss: {:.8f} Boundaries Loss: {:.8f} Total L1 Loss: {:.8f}".format(
                  loss_L1_avg, loss_sum2one_avg, loss_boundaries_avg, total_loss_L1_avg))
            #print("GAN Loss: {:.8f} Boundaries Loss: {:.8f} Sum2one Loss: {:.8f} Centralized Loss: {:.8f} Sparse Loss: {:.8f}".format(
            #    loss_g_avg, loss_boundaries_avg, loss_sum2one_avg, loss_centralized_avg, loss_sparse_avg))
            #print("Total Generator Loss: {:.8f} D_Loss_Fake: {:.8f} D_Loss_Real: {:.8f} Total Discriminator Loss: {:.8f}".format(
            #    total_loss_g_avg, loss_d_fake_avg, loss_d_real_avg, loss_d_avg))

        # at last iteration, save the model
        if (iteration == self.conf.max_iters - 1):
            # save the model
            # torch.save(model.state_dict(), os.path.join(self.conf.ckpt_dir, 'model_%04d_dict.pth' % (epoch+1)))
            torch.save(self.G.state_dict(), os.path.join(self.conf.output_dir,
                                                         self.conf.img_name,
                                                         "ckpt",
                                                         "model_dict_{:s}.pth".format(self.color_idx)))


    ############################################################################
    ## utility function
    def calc_constraints(self, g_pred, iteration, kernel_size):
        # Calculate K which is equivalent to G
        self.calc_curr_k(kernel_size)
        # Calculate constraints
        #self.loss_bicubic = self.bicubic_loss.forward(g_input=self.g_input, g_output=g_pred)
        loss_boundaries = self.boundaries_loss.forward(kernel=self.curr_k)
        loss_sum2one = self.sum2one_loss.forward(kernel=self.curr_k)
        # loss_centralized = self.centralized_loss.forward(kernel=self.curr_k)
        # loss_sparse = self.sparse_loss.forward(kernel=self.curr_k)
        # Apply constraints co-efficients
        return loss_boundaries, loss_sum2one

    def calc_curr_k(self, kernel_size):
        """given a generator network, the function calculates the kernel it is imitating"""
        delta = torch.Tensor([1.]).unsqueeze(0)
        delta_pad = F.pad(delta, (self.conf.G_kernel_size-1, self.conf.G_kernel_size-1,
                                  self.conf.G_kernel_size-1, self.conf.G_kernel_size-1),
                                  "constant", value=0)
        delta_pad = delta_pad.unsqueeze(0).unsqueeze(0).cuda()

        # model method
        curr_k = self.G.forward(delta_pad, kernel_size)

        # full convolution method
        # for ind, w in enumerate(self.G.parameters()):
        #     curr_k = F.conv2d(delta, w, padding=self.conf.G_kernel_size - 1) if ind == 0 else F.conv2d(curr_k, w)

        self.curr_k = curr_k.squeeze()

    def gen_curr_k(self, iteration, color_idx):
        """generate the kernel after each iteration"""
        knl_of_iter = self.curr_k.detach().cpu().float().numpy()
        # calculate the current center of mass for the kernel
        current_center_of_knl = measurements.center_of_mass(knl_of_iter)
        wanted_center_of_knl = np.array(knl_of_iter.shape) // 2 + 0.5 * (np.array(1) - (np.array(knl_of_iter.shape) % 2))
        # calculate the shift of the kernel
        shift_vec = wanted_center_of_knl - current_center_of_knl
        knl_shift = interpolation.shift(knl_of_iter, shift_vec)
        """saves the final kernel and the shifted final kernel to the results folder"""
        #print(os.path.join(self.conf.output_dir, self.conf.img_name, '%d_kernel.mat' % iteration))
        sio.savemat(os.path.join(self.conf.output_dir,
                                 self.conf.img_name,
                                 "kernel_pred",
                                 '{:05d}_kernel_{:s}.mat'.format(iteration, color_idx)),
                    {'Kernel': knl_of_iter})
        sio.savemat(os.path.join(self.conf.output_dir,
                                 self.conf.img_name,
                                 "kernel_pred",
                                 '{:05d}_kernel_{:s}.mat'.format(iteration, color_idx)),
                    {'Kernel': knl_shift})

    def gen_curr_img(self, iteration, kernel_size, color_idx):
        """generate the convolution result after training"""
        # model method
        curr_img_ori = self.G.forward(self.test_input_ori, kernel_size)
        curr_img_val = self.G.forward(self.test_input_val, kernel_size)

        # process to rgb
        curr_img_ori = process(curr_img_ori, self.wb, self.color_idx)
        curr_img_val = process(curr_img_val, self.wb, self.color_idx)

        # full convolution method
        # for ind, w in enumerate(self.G.parameters()):
        #     curr_img = F.conv2d(self.test_input, w, padding=self.conf.G_kernel_size - 1) if ind == 0 else F.conv2d(curr_img, w)

        """transform it to cpu array"""
        curr_img_ori = curr_img_ori.squeeze().detach().cpu().float().numpy()
        curr_img_val = curr_img_val.squeeze().detach().cpu().float().numpy()

        curr_img_ori = curr_img_ori[int((self.conf.G_kernel_size-1)/2) : int(-(self.conf.G_kernel_size-1)/2),
                                    int((self.conf.G_kernel_size-1)/2) : int(-(self.conf.G_kernel_size-1)/2)]
        curr_img_val = curr_img_val[int((self.conf.G_kernel_size-1)/2) : int(-(self.conf.G_kernel_size-1)/2),
                                    int((self.conf.G_kernel_size-1)/2) : int(-(self.conf.G_kernel_size-1)/2)]
        cv2.imwrite(os.path.join(self.conf.output_dir,
                                 self.conf.img_name,
                                 "ori_img_pred",
                                 "{:05d}_image_{:s}.png".format(iteration, color_idx)), curr_img_ori*255.)
        cv2.imwrite(os.path.join(self.conf.output_dir,
                                 self.conf.img_name,
                                 "val_img_pred",
                                 "{:05d}_image_{:s}.png".format(iteration, color_idx)), curr_img_val*255.)

    def add_loss(self, loss_L1, loss_boundaries, loss_sum2one, total_loss_L1):
        self.loss_L1_sum += loss_L1.item()
        self.loss_boundaries_sum += loss_boundaries.item()
        self.loss_sum2one_sum += loss_sum2one.item()
        self.total_loss_L1_sum += total_loss_L1.item()

    def avg_loss(self):
        loss_L1_avg = self.loss_L1_sum / 1000
        loss_boundaries_avg = self.loss_boundaries_sum / 1000
        loss_sum2one_avg = self.loss_sum2one_sum / 1000
        total_loss_L1_avg = self.total_loss_L1_sum / 1000

        self.loss_L1_sum = 0
        self.loss_boundaries_sum = 0
        self.loss_sum2one_sum = 0
        self.total_loss_L1_sum = 0

        return loss_L1_avg, loss_boundaries_avg, loss_sum2one_avg, total_loss_L1_avg

    def kernel_shift(self, kernel, sf):
        # There are two reasons for shifting the kernel :
        # 1. Center of mass is not in the center of the kernel which creates ambiguity. There is no possible way to know
        #    the degradation process included shifting so we always assume center of mass is center of the kernel.
        # 2. We further shift kernel center so that top left result pixel corresponds to the middle of the sfXsf first
        #    pixels. Default is for odd size to be in the middle of the first pixel and for even sized kernel to be at the
        #    top left corner of the first pixel. that is why different shift size needed between odd and even size.
        # Given that these two conditions are fulfilled, we are happy and aligned, the way to test it is as follows:
        # The input image, when interpolated (regular bicubic) is exactly aligned with ground truth.

        # First calculate the current center of mass for the kernel
        current_center_of_mass = measurements.center_of_mass(kernel)

        # The second term ("+ 0.5 * ....") is for applying condition 2 from the comments above
        wanted_center_of_mass = np.array(kernel.shape) // 2 + 0.5 * (np.array(sf) - (np.array(kernel.shape) % 2))
        # Define the shift vector for the kernel shifting (x,y)
        shift_vec = wanted_center_of_mass - current_center_of_mass
        # Before applying the shift, we first pad the kernel so that nothing is lost due to the shift
        # (biggest shift among dims + 1 for safety)
        kernel = np.pad(kernel, np.int(np.ceil(np.max(np.abs(shift_vec)))) + 1, 'constant')

        # Finally shift the kernel and return
        kernel = interpolation.shift(kernel, shift_vec)

        return kernel
    ############################################################################
