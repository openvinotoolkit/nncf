# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from logging import Logger
from typing import Generic, Iterable, Iterator, Optional, TypeVar

from nncf.common.logging import nncf_logger

TObj = TypeVar("TObj")


class ProgressBar(Generic[TObj]):
    """
    A basic progress bar specifically for the logging.
    It does not print at the same line, instead it logs multiple lines intentionally to avoid multiprocessing issues.
    Works as a decorator for iterable.
    :param iterable: iterable to decorate with a progressbar.
    :param logger: logging to print a progress, nncf_logger by default
    :param desc: prefix for the printed line, empty by default
    :param num_lines:  defines the number of lines to log
    :param total: the expected total number of iterations
    """

    def __init__(
        self,
        iterable: Iterable[TObj],
        logger: Logger = nncf_logger,
        desc: str = "",
        num_lines: int = 10,
        total: Optional[int] = None,
    ):
        self._logger = logger
        self._iterable = iterable
        self._desc = desc
        self._num_lines = num_lines

        self._index: int = 0
        self._width = 16
        self._is_enabled = False
        self._total = None
        if total is not None:
            if not isinstance(total, (int, float)) or total <= 0:
                logger.error(
                    "Progress bar is disabled because the expected total number of iterations is invalid: "
                    "it should be an integer and more than 0"
                )
                return
            self._total = int(total)

        if iterable is not None and self._total is None:
            try:
                self._total = len(iterable)  # type: ignore[arg-type]
            except (TypeError, AttributeError):
                logger.error(
                    "Progress bar is disabled because the given iterable is invalid: "
                    "it does not implement __len__ method"
                )
                return

        if not isinstance(num_lines, int) or num_lines <= 1:
            logger.error(
                "Progress bar is disabled because the given number of lines for logging is invalid: "
                "it should be an integer and more than 1"
            )
            return

        self._step = max(1, self._total // (self._num_lines - 1))
        self._is_enabled = True

    def __iter__(self) -> Iterator[TObj]:
        for obj in self._iterable:
            yield obj
            if self._is_enabled:
                self._print_next()

    def _print_next(self) -> None:
        self._index += 1
        if self._total is None or self._index > self._total:
            return

        if self._index % self._step == 0 or self._index == self._total:
            num_filled = int(self._index * self._width / self._total)
            num_empty = self._width - num_filled
            filled = "█" * num_filled
            empty = " " * num_empty
            self._logger.info(
                "{desc} |{filled}{empty}| {index} / {total}".format(
                    desc=self._desc, filled=filled, empty=empty, index=self._index, total=self._total
                )
            )
