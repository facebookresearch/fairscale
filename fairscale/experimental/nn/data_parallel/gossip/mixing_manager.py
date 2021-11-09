# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Mixing Manager Class

:description: Class provides an API for dynamically selecting mixing weights
              for gossip
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

import torch

from .graph_manager import GraphManager


class MixingManager(ABC):
    def __init__(self, graph: GraphManager, device: Optional[torch.device]) -> None:
        self.graph_manager = graph
        self.device = device

    def is_regular(self) -> bool:
        """
        Whether there is bias accumulated in local entry of stationary
        distribution of mixing matrix
        """
        return self.graph_manager.is_regular_graph() and self.is_uniform()

    @abstractmethod
    def is_uniform(self) -> bool:
        """Whether mixing weights are distributed uniformly over peers"""
        raise NotImplementedError

    @abstractmethod
    def get_mixing_weights(self, residual_adjusted: bool = True) -> Dict[Union[str, int], torch.Tensor]:
        """Create mixing weight dictionary using uniform allocation"""
        raise NotImplementedError


class UniformMixing(MixingManager):
    def get_mixing_weights(self, residual_adjusted: bool = True) -> Dict[Union[str, int], torch.Tensor]:
        """Create mixing weight dictionary using uniform allocation"""
        mixing_weights: Dict[Union[str, int], torch.Tensor] = {}
        out_peers, _ = self.graph_manager.get_peers()

        w = torch.tensor([1.0 / (len(out_peers) + 1.0)], device=self.device)
        mixing_weights["lo"] = w.clone()
        w_op = w if not residual_adjusted else w / mixing_weights["lo"]
        mixing_weights["uniform"] = w_op.clone()
        for op in out_peers:
            mixing_weights[op] = w_op.clone()
        return mixing_weights

    def is_uniform(self) -> bool:
        return True
