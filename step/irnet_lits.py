from misc import pyutils
import os


def train_cam(args):
    if args.currRound == 0:
        from step.Irnet import train_cam_lits
        args.cam_ckpt = None
        timer = pyutils.Timer('step.irnet.train_cam_lits:')
        train_cam_lits.run(args)
    if args.currRound >= 1:
        from step.Irnet import train_cam_tx
        args.train_list = '../data/all_valid_1_2.txt'
        args.cam_learning_rate = 0.001
        args.cam_network = "net.resnet50_cam_tx"
        args.cam_ckpt = os.path.join(args.round_in_dir, 'checkpoints', args.cam_weights_name)
        timer = pyutils.Timer('step.irnet.train_cam_tx:')
        train_cam_tx.run(args)


def make_cam(args):
    if args.currRound == 0:
        from step.Irnet import make_cam_lits
        timer = pyutils.Timer('step.irnet.make_cam_lits:')
        make_cam_lits.run(args)
    if args.currRound >= 1:
        from step.Irnet import make_cam_tx
        args.train_list = '../data/all_valid_1_2.txt'
        args.cam_network = "net.resnet50_cam_tx"
        timer = pyutils.Timer('step.irnet.make_cam_tx:')
        make_cam_tx.run(args)


def make_lpcam(args):
    if args.currRound == 0:
        from step.Irnet import make_lpcam_lits
        timer = pyutils.Timer('step.irnet.make_cam:')
        make_lpcam_lits.run(args)
    if args.currRound >= 1:
        from step.Irnet import make_cam_tx
        args.train_list = '../data/all_valid_1_2.txt'
        args.cam_network = "net.resnet50_cam_tx"
        timer = pyutils.Timer('step.irnet.make_cam_tx:')
        make_cam_tx.run(args)


def eval_cam(args):
    from step.Irnet import eval_cam
    timer = pyutils.Timer('step.irnet.eval_cam:')
    eval_cam.run(args)


def cam_to_ir_label(args):
    from step.Irnet import cam_to_ir_label
    timer = pyutils.Timer('step.irnet.cam_to_ir_label:')
    cam_to_ir_label.run(args)


def train_irn(args):
    from step.Irnet import train_irn
    timer = pyutils.Timer('step.irnet.train_irn:')
    train_irn.run(args)


def make_sem_seg_labels(args):
    from step.Irnet import make_sem_seg_labels
    timer = pyutils.Timer('step.irnet.make_sem_seg_labels:')
    make_sem_seg_labels.run(args)


def eval_sem_seg(args):
    from step.Irnet import eval_sem_seg
    timer = pyutils.Timer('step.irnet.eval_sem_seg:')
    eval_sem_seg.run(args)
