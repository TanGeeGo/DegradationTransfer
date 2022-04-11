import argparse
import torch
import os

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='full field kernel esti')
        self.conf = None

        # System parameters
        self.parser.add_argument('--img_name', type=str, default='image1',
                                 help='image name for saving purposes')
        self.parser.add_argument('--output_dir', type=str, default="/Users",
                                 help='results path')
        self.parser.add_argument('--v_input_dir', type=str, default="/Users",
                                 help='valid path')

        # Sizes
        self.parser.add_argument('--input_crop_size', type=int, default=128,
                                 help='Generators crop size')
        self.parser.add_argument('--batch_size', type=int, default=16,
                                 help='batch size of input and the label')
        self.parser.add_argument('--scale_factor', type=float, default=1,
                                 help='The downscaling scale factor')

        # Network architecture
        self.parser.add_argument('--G_structure', type=list, default=[7, 5, 3, 3, 3, 3, 3, 3, 3, 3],
                                 help='# kernel size of the G')
        self.parser.add_argument('--G_chan', type=int, default=64,
                                 help='# of channels in hidden layer in the G')
        self.parser.add_argument('--D_chan', type=int, default=64,
                                 help='# of channels in hidden layer in the D')
        self.parser.add_argument('--G_kernel_size', type=int, default=27,
                                 help='The kernel size G is estimating')
        self.parser.add_argument('--D_n_layers', type=int, default=7,
                                 help='Discriminators depth')
        self.parser.add_argument('--D_kernel_size', type=int, default=7,
                                 help='Discriminators convolution kernels size')

        # Iterations
        self.parser.add_argument('--max_iters', type=int, default=10000,
                                 help='# of iterations')

        # Optimization hyper-parameters
        self.parser.add_argument('--g_lr', type=float, default=1e-4,
                                 help='initial learning rate for generator')
        self.parser.add_argument('--d_lr', type=float, default=1e-4,
                                 help='initial learning rate for discriminator')
        self.parser.add_argument('--beta1', type=float, default=0.9,
                                 help='Adam momentum')
        self.parser.add_argument('--update_l_rate_freq', type=int, default=5000,
                                 help='The frequency to update learning rate')
        self.parser.add_argument('--update_l_rate_rate', type=float, default=1.0,
                                 help='The rate to update learning rate')

        # loss hyper-parameters
        self.parser.add_argument('--lambda_sum2one', type=float, default=0.001,
                                 help='The lambda of sum2one loss')
        self.parser.add_argument('--lambda_boundaries', type=float, default=1,
                                 help='The lambda of boundaries loss')

    def parse(self, args=None):
        """Parse the configuration"""
        self.conf = self.parser.parse_args(args=args)

        # kernel directory
        create_dir(os.path.join(self.conf.output_dir, self.conf.img_name, 'kernel_pred'))
        # original image directory
        create_dir(os.path.join(self.conf.output_dir, self.conf.img_name, 'ori_img_pred'))
        # valid image directory
        create_dir(os.path.join(self.conf.output_dir, self.conf.img_name, 'val_img_pred'))
        # log and ckpt
        create_dir(os.path.join(self.conf.output_dir, self.conf.img_name, "ckpt"))
        create_dir(os.path.join(self.conf.output_dir, self.conf.img_name, "log"))
        return self.conf
