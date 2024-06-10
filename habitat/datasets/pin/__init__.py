#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.dataset import Dataset
from habitat.core.registry import registry


# and into simulators specifically. As a result of that the connection points
# for our tasks and datasets for actions is coming from inside habitat-sim
# which makes it impossible for anyone to use habitat-api without having
# habitat-sim installed. In a future PR we will implement a base simulator
# action class which will be the connection point for tasks and datasets.
# Post that PR we would no longer need try register blocks.


def _try_register_pindatasetv1():
    try:
        from habitat.datasets.pin.pin import (
            PINDatasetV1,
        )

        has_pin = True
    except ImportError as e:
        has_pin = False
        pin_import_error = e

    if has_pin:
        from habitat.datasets.pin.pin import PINDatasetV1
    else:

        @registry.register_dataset(name="PIN-v1")
        class PINDatasetImportError(Dataset):
            def __init__(self, *args, **kwargs):
                raise pin_import_error
