# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Generic, TypeVar

ConsumerType = TypeVar("ConsumerType")


@dataclass
class DataConsumer(Generic[ConsumerType]):
    """A data class for representating a consumer of an output of a module."""

    consumer: ConsumerType
    consumer_input_idx: int  # indicating which input of the consumer module
    output_idx: int  # indicating which output of the producer module
