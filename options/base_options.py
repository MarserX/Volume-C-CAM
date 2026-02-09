import argparse
from .option import Option


class BaseOptions(Option):
    # def __init__(self):
    #     super().__init__()

    def initialize(self):
        # experiment specifics
        self.parser.add_argument("--roundNumber", default=4, type=str, help="round of EM iteration")
        self.parser.add_argument("--output_dir", default='myExperiment/results/EM_rounds_acdc/', type=str,
                            help="Path to save output results.")
        self.parser.add_argument("--camTrainerName", default='irnet', type=str, help="chose in [customer, irnet, ...]")
        self.parser.add_argument("--segModelName", default='deeplabv3', type=str, help="chose in [customer, deeplabv3, ...]")

        self.parser.add_argument("--train_seg_pass", default=True)
        self.parser.add_argument("--eval_seg_pass", default=True)
        self.parser.add_argument("--make_confounder_pass", default=True)

        self.parser.add_argument("--log_name", default="myExperiment/logs/acdc/experiment_acdc", type=str)
        self.parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")

