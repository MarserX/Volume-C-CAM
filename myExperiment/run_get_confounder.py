import sys
sys.path.append("/data3/masx/code/Volume-C-CAM")
from misc import pyutils
from options.experiment_options import MyOptions
import os
import get_confounder

if __name__ == "__main__":
    args = MyOptions().parse()
    pyutils.Logger(os.path.join('/data3/masx/code/Volume-C-CAM', args.log_name + '_get_confounder.log'))
    # pyutils.Logger(args.log_name + '_get_confounder.log')
    print(vars(args))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    it_n = 0
    print(f'round = {it_n}')
    args.currRound = it_n
    args.round_out_dir = args.output_dir + 'round_' + str(it_n) + '/'
    args.round_in_dir = args.output_dir + 'round_' + str(it_n) + '/'

    if not os.path.exists(os.path.join(args.round_out_dir, 'volume_confounder.npy')):
        args.make_confounder_pass = True
    else:
        args.make_confounder_pass = False

    get_confounder.run(args)