import argparse
from .option import Option
import os


class IRNetOptions(Option):
    def initialize(self):
        Option.initialize(self)
        # Environment
        self.parser.add_argument("--data_root", default='/data3/masx/data/CZ_ACDC', type=str, help="Path to dataset.")
        self.parser.add_argument("--num_workers", default=os.cpu_count() // 2, type=int)

        # Dataset
        self.parser.add_argument("--train_list", default="data/train_all_acdc.txt", type=str)
        self.parser.add_argument("--val_list", default="data/test_all_acdc.txt", type=str)
        self.parser.add_argument("--infer_list", default="data/train_all_valid_acdc.txt", type=str)
        # self.parser.add_argument("--chainer_eval_set", default="train", type=str)

        # Class Activation Map
        self.parser.add_argument("--cam_network", default="net.resnet50_cam", type=str)
        self.parser.add_argument("--cam_crop_size", default=512, type=int)
        self.parser.add_argument("--cam_batch_size", default=16, type=int)
        self.parser.add_argument("--cam_num_epoches", default=10, type=int)
        self.parser.add_argument("--cam_learning_rate", default=0.01, type=float)
        self.parser.add_argument("--cam_weight_decay", default=1e-4, type=float)
        self.parser.add_argument("--cam_eval_thres", default=0.5, type=float)
        self.parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5, 2.0),
                            help="Multi-scale inferences")

        # Mining Inter-pixel Relations
        self.parser.add_argument("--conf_fg_thres", default=0.8, type=float)
        self.parser.add_argument("--conf_bg_thres", default=0.5, type=float)

        # Inter-pixel Relation Network (IRNet)
        self.parser.add_argument("--irn_network", default="net.resnet50_irn", type=str)
        self.parser.add_argument("--irn_crop_size", default=512, type=int)
        self.parser.add_argument("--irn_batch_size", default=16, type=int)
        self.parser.add_argument("--irn_num_epoches", default=6, type=int)
        self.parser.add_argument("--irn_learning_rate", default=0.1, type=float)
        self.parser.add_argument("--irn_weight_decay", default=1e-4, type=float)

        # Random Walk Params
        self.parser.add_argument("--beta", default=10)
        self.parser.add_argument("--exp_times", default=4,
                            help="Hyper-parameter that controls the number of random walk iterations,"
                                 "The random walk is performed 2^{exp_times}.")
        self.parser.add_argument("--ins_seg_bg_thres", default=0.5)
        self.parser.add_argument("--sem_seg_bg_thres", default=0.75)

        # Output Path
        self.parser.add_argument("--cam_weights_name", default="res50_cam.pth", type=str)
        self.parser.add_argument("--irn_weights_name", default="volume_res50_irn.pth", type=str)
        self.parser.add_argument("--cam_out_dir", default="cam", type=str)
        self.parser.add_argument("--lpcam_out_dir", default="lpcam", type=str)
        self.parser.add_argument("--ir_label_out_dir", default="ir_label", type=str)
        self.parser.add_argument("--sem_seg_out_dir", default="volume_rw_png", type=str)
        # self.parser.add_argument("--ins_seg_out_dir", default="ins_seg", type=str)

        # Step
        self.parser.add_argument("--train_cam_pass", default=True)
        self.parser.add_argument("--make_cam_pass", default=True)
        self.parser.add_argument("--eval_cam_pass", default=True)
        self.parser.add_argument("--cam_to_ir_label_pass", default=True)
        self.parser.add_argument("--train_irn_pass", default=True)
        self.parser.add_argument("--make_ins_seg_pass", default=True)
        self.parser.add_argument("--eval_ins_seg_pass", default=True)
        self.parser.add_argument("--make_sem_seg_pass", default=True)
        self.parser.add_argument("--eval_sem_seg_pass", default=True)



