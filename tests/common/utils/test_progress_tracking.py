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
import pytest

from nncf.common.logging.track_progress import WeightedProgress
from nncf.common.logging.track_progress import track

N = 10


def get_sequence(n):
    return [i for i in range(n)]


class track_wrapper:
    def __init__(self, track):
        self.track = track

    def __iter__(self):
        completed = 0
        for i, value in enumerate(self.track):
            if not self.track.progress.live.auto_refresh:
                # There is no easy way to check this when auto refresh is enabled because _TrackThread is used
                assert completed == self.track.progress._tasks[self.track.task].completed
            yield value
            completed += self.track.weights[i] if isinstance(self.track.progress, WeightedProgress) else 1


@pytest.mark.parametrize("n", [N])
@pytest.mark.parametrize("is_weighted", [False, True])
@pytest.mark.parametrize("auto_refresh", [False, True])
def test_track(n, is_weighted, auto_refresh):
    original_sequence = get_sequence(n)
    retrieved_sequence = [None] * n
    for i, it in enumerate(
        track_wrapper(
            track(
                original_sequence,
                description="Progress...",
                weights=original_sequence if is_weighted else None,
                auto_refresh=auto_refresh,
            )
        )
    ):
        retrieved_sequence[i] = it
    assert all(original_sequence[i] == retrieved_sequence[i] for i in range(n))


@pytest.mark.parametrize("n", [N])
@pytest.mark.parametrize("is_weighted", [False, True])
@pytest.mark.parametrize("auto_refresh", [False, True])
def test_track_no_length(n, is_weighted, auto_refresh):
    original_sequence = get_sequence(n)
    original_sequence_iterable = iter(original_sequence)
    retrieved_sequence = [None] * n
    for i, it in enumerate(
        track_wrapper(
            track(
                original_sequence_iterable,
                total=n,
                description="Progress...",
                weights=original_sequence if is_weighted else None,
                auto_refresh=auto_refresh,
            )
        )
    ):
        retrieved_sequence[i] = it
    assert all(original_sequence[i] == retrieved_sequence[i] for i in range(n))


@pytest.mark.parametrize("n", [N])
@pytest.mark.parametrize("is_weighted", [False, True])
def test_track_context_manager(n, is_weighted):
    weights = get_sequence(n)
    with track(total=n, description="Progress...", weights=weights if is_weighted else None) as pbar:
        for i in range(n):
            assert pbar.progress._tasks[pbar.task].completed == (sum(weights[:i]) if is_weighted else i)
            pbar.update(advance=1)
