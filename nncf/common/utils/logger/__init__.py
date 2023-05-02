# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from nncf.common.deprecation import warning_deprecated

# pylint:disable=wrong-import-position

warning_deprecated(
    "Importing from nncf.common.utils.logger is deprecated. "
    "Import `from nncf` directly instead, i.e.: \n"
    "`from nncf import set_log_level` instead of `from nncf.common.utils.logger import set_log_level`, and:\n"
    "`from nncf import nncf_logger` instead of `from nncf.common.utils.logger import logger as nncf_logger`"
)

# pylint:disable=unused-import
from nncf.common.logging.logger import disable_logging
from nncf.common.logging.logger import nncf_logger as logger
from nncf.common.logging.logger import set_log_level
