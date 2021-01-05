# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Benchmarking utils for timing cuda executions
"""

import collections

import numpy as np
import torch


def create_and_record_event():
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


def create_event_recorder(event_name, dummy=False):
    if not dummy:
        return CudaEventRecorder(event_name)
    return DummyCudaEventRecorder()


class CudaEventRecorder(object):
    """ Allows recording cuda events in an easy manner """

    event_recorders = collections.defaultdict(list)
    all_event_recorders = collections.defaultdict(list)

    def __init__(self, event_name):
        self.event_name = event_name
        self.start_event = create_and_record_event()
        self.end_event = None

        # Adding it to global tracker
        CudaEventRecorder.event_recorders[event_name].append(self)
        CudaEventRecorder.all_event_recorders[event_name].append(self)

    def stop(self):
        self.end_event = create_and_record_event()

    def find_time_elapsed(self):
        if self.end_event is None:
            raise Exception(
                f"stopEvent was not called for event with name {self.event_name}"
            )

        self.end_event.synchronize()
        return self.start_event.elapsed_time(self.end_event)

    @classmethod
    def reset(cls):
        cls.event_recorders = collections.defaultdict(list)

    @classmethod
    def get_common_timings(cls, event_recorders, description):
        all_timings_str = f"{description}:\n"

        # Iterating over different types of events, eg., forward, backward
        for event_name, event_recorder_list in event_recorders.items():
            time_taken_list = []

            # Iterating over different occurences of an event type
            for event_recorder in event_recorder_list:
                time_taken_list.append(event_recorder.find_time_elapsed())

            all_timings_str += (
                "{}: Time taken: avg: {}, std: {}, count: " "{}\n"
            ).format(
                event_name,
                np.mean(time_taken_list),
                np.std(time_taken_list),
                len(time_taken_list),
            )

        return all_timings_str

    @classmethod
    def get_timings(cls):
        return cls.get_common_timings(cls.event_recorders, "Timings since last reset")

    @classmethod
    def get_all_timings(cls):
        return cls.get_common_timings(cls.all_event_recorders, "All timings")


class DummyCudaEventRecorder(object):
    def stop(self):
        pass
