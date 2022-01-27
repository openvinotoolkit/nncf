"""
 Copyright (c) 2022 Intel Corporation
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
import pytest

from nncf.experimental.post_training_api.compression_builder import PriorityQueue

# TODO: add compression algorithms with priority fields
TEST_LISTS = [[5, 7, 3, 4, 1, 6, 8, 9, 2], [1, 10, 2, 5, 7, 13, 99, 0, 1, 1, 5]]


@pytest.mark.parametrize("test_list", TEST_LISTS)
def test_queue(test_list):
    queue = PriorityQueue(test_list)
    sorted_test_list = sorted(test_list)
    for i in range(len(sorted_test_list)):
        queue_elem = queue.pop()
        ref_elem = sorted_test_list.pop()
        assert ref_elem == queue_elem
