import sys
sys.path.append("/data3/masx/code/Volume-C-CAM")
from misc import pyutils
from options.experiment_options import MyOptions
import os
import aff_refine

if __name__ == "__main__":
    args = MyOptions().parse()
    pyutils.Logger(args.log_name + '_causal_refine_volume.log')
    print(vars(args))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    it_n = 1
    print(f'round = {it_n}')
    args.currRound = it_n
    args.round_out_dir = args.output_dir + 'round_' + str(it_n) + '/'
    args.round_in_dir = args.output_dir + 'round_' + str(it_n-1) + '/'
    args.cam_out_dir = 'lpcam'
    args.ir_label_out_dir = 'volume_ir_label'
    args.irn_weights_name = 'volume_res50_irn.pth'
    args.save_rw_dir = 'volume_rw'

    if not os.path.exists(os.path.join(args.round_out_dir, 'prediction', args.ir_label_out_dir)):
        os.makedirs(os.path.join(args.round_out_dir, 'prediction', args.ir_label_out_dir))
        args.cam_to_ir_label_pass = True
    else:
        args.cam_to_ir_label_pass = False

    if os.path.exists(os.path.join(args.round_out_dir, 'checkpoints', args.irn_weights_name)):
        args.train_irn_pass = False
    else:
        args.train_irn_pass = True

    if not os.path.exists(os.path.join(args.round_out_dir, 'prediction/volume_rw')):
        os.makedirs(os.path.join(args.round_out_dir, 'prediction/volume_rw'))
        args.make_sem_seg_pass = True
    else:
        args.make_sem_seg_pass = False

    # if not os.path.exists(os.path.join(args.round_out_dir, 'prediction', args.sem_seg_out_dir)):
    #     os.makedirs(os.path.join(args.round_out_dir, 'prediction', args.sem_seg_out_dir))
    #     args.eval_sem_seg_pass = True
    # else:
    #     args.eval_sem_seg_pass = False

    aff_refine.run(args)