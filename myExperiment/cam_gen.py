import os


# to train a cams model and generate cams
def camGenerator(args):
    if args.camTrainerName == 'irnet':
        from step import irnet
        args.seg_pred_dir = os.path.join(args.round_in_dir, 'prediction/volume_rw_png')
        # args.seg_pred_dir = os.path.join(args.round_in_dir, 'prediction/seg/train')
        # args.seg_pred_dir_val = os.path.join(args.round_in_dir, 'prediction/seg/val')
        if args.train_cam_pass is True:
            irnet.train_cam(args)
        if args.make_cam_pass is True:
            irnet.make_cam(args)
        # if args.make_cam_pass is True:
        #     irnet.make_lpcam(args)
        # if args.eval_cam_pass is True:
        #     irnet.eval_cam(args)



def run(args):
    camGenerator(args)



