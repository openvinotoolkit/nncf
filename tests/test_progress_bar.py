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
import logging

import pytest

from nncf.progress_bar import ProgressBar

TEST_RANGE = range(3)


def test_can_print_by_default(caplog):
    for _ in ProgressBar(TEST_RANGE):
        pass

    assert caplog.record_tuples == [
        ('nncf', 20, ' █████             | 1 / 3'),
        ('nncf', 20, ' ██████████        | 2 / 3'),
        ('nncf', 20, ' ████████████████  | 3 / 3')
    ]


def test_can_print_by_default__with_enumerate_and_total(caplog):
    for _ in ProgressBar(enumerate(TEST_RANGE), total=3.0):
        pass

    assert caplog.record_tuples == [
        ('nncf', 20, ' █████             | 1 / 3'),
        ('nncf', 20, ' ██████████        | 2 / 3'),
        ('nncf', 20, ' ████████████████  | 3 / 3')
    ]


def test_can_print_with_another_logger(caplog):
    name = "test"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    for _ in ProgressBar(enumerate(TEST_RANGE), logger=logger, total=2):
        pass

    assert caplog.record_tuples == [
        ('test', 20, ' ████████          | 1 / 2'),
        ('test', 20, ' ████████████████  | 2 / 2')
    ]


@pytest.mark.parametrize('num_lines', [0, 1, -1, 's'])
def test_invalid_num_lines_leads_to_disabling_progress_bar(num_lines, caplog):
    for _ in ProgressBar(TEST_RANGE, num_lines=num_lines):
        pass

    assert len(caplog.records) == 1
    record = next(iter(caplog.records))
    assert record.levelno == logging.WARNING


@pytest.mark.parametrize('total', [0, -1, 's'])
def test_invalid_total_leads_to_disabling_progress_bar(total, caplog):
    for _ in ProgressBar(TEST_RANGE, total=total):
        pass

    assert len(caplog.records) == 1
    record = next(iter(caplog.records))
    assert record.levelno == logging.WARNING


def test_can_iterate_over_empty_iterable(caplog):
    for _ in ProgressBar([]):
        pass

    assert caplog.record_tuples == []


def test_type_error_happens_for_iteration_none(caplog):
    with pytest.raises(TypeError):
        for _ in ProgressBar(None):
            pass


def test_can_print_collections_less_than_num_lines(caplog):
    desc = 'desc'
    for _ in ProgressBar(TEST_RANGE, desc=desc, num_lines=4):
        pass

    assert caplog.record_tuples == [
        ('nncf', 20, 'desc █████             | 1 / 3'),
        ('nncf', 20, 'desc ██████████        | 2 / 3'),
        ('nncf', 20, 'desc ████████████████  | 3 / 3')
    ]


def test_can_print_collections_bigger_than_num_lines(caplog):
    for _ in ProgressBar(range(11), num_lines=3):
        pass

    assert caplog.record_tuples == [
        ('nncf', 20, ' ███████           | 5 / 11'),
        ('nncf', 20, ' ██████████████    | 10 / 11'),
        ('nncf', 20, ' ████████████████  | 11 / 11')
    ]


def test_can_iterate_with_warning_for_iterable_without_len(caplog):
    for _ in ProgressBar(enumerate(TEST_RANGE)):
        pass

    assert len(caplog.records) == 1
    record = next(iter(caplog.records))
    assert record.levelno == logging.WARNING


def test_can_limit_number_of_iterations(caplog):
    for _ in ProgressBar(TEST_RANGE, total=2):
        pass

    assert caplog.record_tuples == [
        ('nncf', 20, ' ████████          | 1 / 2'),
        ('nncf', 20, ' ████████████████  | 2 / 2')
    ]
