import argparse


class Option():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        pass

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
            self.initialized = True
        self.opt = self.parser.parse_args()
        return self.opt