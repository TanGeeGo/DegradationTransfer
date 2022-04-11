from model.fov_kpn import FOVKPN

def make_model(input_channel, output_channel, args):
    if args.NetName == 'FOVKPN':
        return FOVKPN(input_channel, output_channel, args.n_channel, args.offset_channel, args.fov_att, 
                      args.kernel_size, args.color)

