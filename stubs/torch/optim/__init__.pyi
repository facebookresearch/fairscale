# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from .sgd import SGD as SGD
from .adam import Adam as Adam
from .madgrad import MADGRAD as MADGRAD
from . import lr_scheduler as lr_scheduler
from .optimizer import Optimizer as Optimizer
#MODIFIED BY TORCHGPIPE
from .rmsprop import RMSprop as RMSprop
#END
