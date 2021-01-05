# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Benchmarking utils for timing code snippets
"""


class Meter(object):
    """ Computes and stores the average, variance, and current value """

    def __init__(self, init_dict=None, ptag="Time", stateful=False, csv_format=True):
        """
        :param init_dict: Dictionary to initialize meter values
        :param ptag: Print tag used in __str__() to identify meter
        :param stateful: Whether to store value history and compute MAD
        """
        self.reset()
        self.ptag = ptag
        self.value_history = None
        self.stateful = stateful
        if self.stateful:
            self.value_history = []
        self.csv_format = csv_format
        if init_dict is not None:
            for key in init_dict:
                try:
                    # TODO: add type checking to init_dict values
                    self.__dict__[key] = init_dict[key]
                except Exception:
                    print("(Warning) Invalid key {} in init_dict".format(key))

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.std = 0
        self.sqsum = 0
        self.mad = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.sqsum += (val ** 2) * n
        if self.count > 1:
            self.std = (
                (self.sqsum - (self.sum ** 2) / self.count) / (self.count - 1)
            ) ** 0.5
        if self.stateful:
            self.value_history.append(val)
            mad = 0
            for v in self.value_history:
                mad += abs(v - self.avg)
            self.mad = mad / len(self.value_history)

    def __str__(self):
        if self.csv_format:
            if self.stateful:
                return str("{dm.val:.3f},{dm.avg:.3f},{dm.mad:.3f}".format(dm=self))
            else:
                return str("{dm.val:.3f},{dm.avg:.3f},{dm.std:.3f}".format(dm=self))
        else:
            if self.stateful:
                return str(self.ptag) + str(
                    ": {dm.val:.3f} ({dm.avg:.3f} +- {dm.mad:.3f})".format(dm=self)
                )
            else:
                return str(self.ptag) + str(
                    ": {dm.val:.3f} ({dm.avg:.3f} +- {dm.std:.3f})".format(dm=self)
                )