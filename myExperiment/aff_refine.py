def cam2mask(args):
    if args.camTrainerName == 'irnet':
        import step.irnet
        if args.cam_to_ir_label_pass is True:
            step.irnet.cam_to_ir_label(args)
        if args.train_irn_pass is True:
            step.irnet.train_irn(args)
        if args.make_sem_seg_pass is True:
            step.irnet.make_sem_seg_labels(args)
        # if args.eval_sem_seg_pass is True:
        #     step.irnet.eval_sem_seg(args)


def run(args):
    cam2mask(args)