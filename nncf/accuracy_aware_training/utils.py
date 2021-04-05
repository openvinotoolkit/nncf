"""
 Copyright (c) 2021 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import os
import datetime
import os.path as osp

from texttable import Texttable
from torch import distributed as dist
from nncf.common.utils.logger import logger as nncf_logger


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def print_statistics(stats, logger=nncf_logger):
    for key, val in stats.items():
        if isinstance(val, Texttable):
            logger.info("{}:".format(key))
            logger.info(val.draw())
        else:
            logger.info("{}: {}".format(key, val))


def configure_paths(log_dir):
    d = datetime.datetime.now()
    run_id = '{:%Y-%m-%d__%H-%M-%S}'.format(d)
    log_dir = osp.join(log_dir, "accuracy_aware_training/{run_id}".format(run_id=run_id))
    os.makedirs(log_dir)
    return log_dir
