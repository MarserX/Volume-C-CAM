from .base_options import BaseOptions
from .irnet_options import IRNetOptions


class MyOptions(BaseOptions, IRNetOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        IRNetOptions.initialize(self)

