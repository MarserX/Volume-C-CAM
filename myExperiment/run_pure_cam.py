import sys
sys.path.append("/data3/masx/code/Volume-C-CAM")
from misc import pyutils
from options.experiment_options import MyOptions
import os
import cam_gen
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == "__main__":
    args = MyOptions().parse()
    pyutils.Logger(args.log_name + '_run_pure_cam.log')
    print(vars(args))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    it_n = 0
    print(f'round = {it_n}')
    args.currRound = it_n
    args.round_out_dir = args.output_dir + 'round_' + str(it_n) + '/'
    args.round_in_dir = args.output_dir + 'round_' + str(it_n) + '/'
    if not os.path.exists(os.path.join(args.round_out_dir, 'checkpoints')):
        os.makedirs(os.path.join(args.round_out_dir, 'checkpoints'))

    if os.path.exists(os.path.join(args.round_out_dir, 'checkpoints', args.cam_weights_name)):
        args.train_cam_pass = False
    else:
        args.train_cam_pass = True

    if not os.path.exists(os.path.join(args.round_out_dir, 'prediction', args.cam_out_dir)):
        os.makedirs(os.path.join(args.round_out_dir, 'prediction', args.cam_out_dir))
        args.make_cam_pass = True
        args.eval_cam_pass = True
    else:
        args.make_cam_pass = False
        args.eval_cam_pass = False

    if not os.path.exists(os.path.join(args.round_out_dir, 'prediction', args.lpcam_out_dir)):
        os.makedirs(os.path.join(args.round_out_dir, 'prediction', args.lpcam_out_dir))
        args.make_cam_pass = True
        args.eval_cam_pass = True
    else:
        args.make_cam_pass = False
        args.eval_cam_pass = False

    cam_gen.run(args)