import sys
sys.path.append("/data3/masx/code/Volume-C-CAM")
from misc import pyutils
from options.experiment_options import MyOptions
import os
import cam_gen

import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')


if __name__ == "__main__":
    args = MyOptions().parse()
    pyutils.Logger(args.log_name + '_causal_cam.log')
    args.cam_num_epoches = 15
    print(vars(args))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    it_n = 1
    print(f'round = {it_n}')
    args.currRound = it_n
    args.round_out_dir = args.output_dir + 'round_' + str(it_n) + '/'
    args.round_in_dir = args.output_dir + 'round_' + str(it_n-1) + '/'
    if not os.path.exists(os.path.join(args.round_out_dir, 'checkpoints')):
        os.makedirs(os.path.join(args.round_out_dir, 'checkpoints'))

    if os.path.exists(os.path.join(args.round_out_dir, 'checkpoints', args.cam_weights_name)):
        args.train_cam_pass = True
    else:
        args.train_cam_pass = True

    if not os.path.exists(os.path.join(args.round_out_dir, 'prediction', args.cam_out_dir)):
        os.makedirs(os.path.join(args.round_out_dir, 'prediction', args.cam_out_dir))
        args.make_cam_pass = True
        args.eval_cam_pass = True
    else:
        args.make_cam_pass = True
        args.eval_cam_pass = False

    cam_gen.run(args)