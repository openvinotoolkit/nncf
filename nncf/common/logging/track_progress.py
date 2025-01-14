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
from __future__ import annotations

from typing import Any, Callable, Dict, Generic, Iterable, Iterator, List, Optional, Sequence, Union

from rich.console import Console
from rich.progress import BarColumn
from rich.progress import Progress
from rich.progress import ProgressColumn
from rich.progress import ProgressType
from rich.progress import Task
from rich.progress import TaskID
from rich.progress import TaskProgressColumn
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.progress import TimeRemainingColumn
from rich.style import StyleType
from rich.table import Column
from rich.text import Text

INTEL_BLUE_COLOR = "#0068b5"


class IterationsColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        if task.total is None:
            return Text("")
        text = f"{int(task.completed)}/{int(task.total)}"
        return Text(text, style=INTEL_BLUE_COLOR)


class SeparatorColumn(ProgressColumn):
    def __init__(self, table_column: Optional[Column] = None, disable_if_no_total: bool = False) -> None:
        super().__init__(table_column)
        self.disable_if_no_total = disable_if_no_total

    def render(self, task: Task) -> Text:
        if self.disable_if_no_total and task.total is None:
            return Text("")
        return Text("•")


class TimeElapsedColumnWithStyle(TimeElapsedColumn):
    def render(self, task: Task) -> Text:
        text = super().render(task)
        return Text(text._text[0], style=INTEL_BLUE_COLOR)


class TimeRemainingColumnWithStyle(TimeRemainingColumn):
    def render(self, task: Task) -> Text:
        text = super().render(task)
        return Text(text._text[0], style=INTEL_BLUE_COLOR)


class WeightedProgress(Progress):
    """
    A class to perform a weighted progress tracking.
    """

    def update(self, task_id: TaskID, **kwargs: Any) -> None:
        task = self._tasks[task_id]

        advance = kwargs.get("advance", None)
        if advance is not None:
            kwargs["advance"] = self.weighted_advance(task, advance)

        completed = kwargs.get("completed", None)
        if completed is not None:
            kwargs["completed"] = self.get_weighted_completed(task, completed)

        super().update(task_id, **kwargs)

    def advance(self, task_id: TaskID, advance: float = 1) -> None:
        if advance is not None:
            task = self._tasks[task_id]
            advance = self.weighted_advance(task, advance)
        super().advance(task_id, advance)

    def reset(self, task_id: TaskID, **kwargs: Any) -> None:
        task = self._tasks[task_id]

        completed = kwargs.get("completed", None)
        if completed is not None:
            kwargs["completed"] = self.get_weighted_completed(task, completed)

        super().reset(task_id, **kwargs)

        if completed == 0:
            task.fields["completed_steps"] = 0

    @staticmethod
    def weighted_advance(task: Task, advance: float) -> float:
        """
        Perform weighted advancement based on an integer step value.
        """
        if advance % 1 != 0:
            raise Exception(f"Unexpected `advance` value: {advance}.")
        advance = int(advance)
        current_step: int = task.fields["completed_steps"]
        weighted_advance: float = sum(task.fields["weights"][current_step : current_step + advance])
        task.fields["completed_steps"] = current_step + advance
        return weighted_advance

    @staticmethod
    def get_weighted_completed(task: Task, completed: float) -> float:
        """
        Get weighted `completed` corresponding to an integer `completed` field.
        """
        if completed % 1 != 0:
            raise Exception(f"Unexpected `completed` value: {completed}.")
        return float(sum(task.fields["weights"][: int(completed)]))


class track(Generic[ProgressType]):
    def __init__(
        self,
        sequence: Union[Sequence[ProgressType], Iterable[ProgressType], None] = None,
        description: str = "Working...",
        total: Optional[float] = None,
        auto_refresh: bool = True,
        console: Optional[Console] = None,
        transient: bool = False,
        get_time: Optional[Callable[[], float]] = None,
        refresh_per_second: float = 10,
        style: StyleType = "bar.back",
        complete_style: StyleType = "bar.complete",
        finished_style: StyleType = "bar.finished",
        pulse_style: StyleType = "bar.pulse",
        update_period: float = 0.1,
        disable: bool = False,
        show_speed: bool = True,
        weights: Optional[List[float]] = None,
    ):
        """
        Track progress by iterating over a sequence.

        This function is very similar to rich.progress.track(), but with some customizations.

        Usage:

        ```
        arr = [1,2]

        for i in track(arr, description="Processing..."):
            print(i)

        with track[None](total=len(arr), description="Processing...") as pbar:
            for i in arr:
                pbar.update(advance=1)
        ```

        :param sequence: An iterable (must support "len") you wish to iterate over.
        :param description: Description of the task to show next to the progress bar. Defaults to "Working".
        :param total: Total number of steps. Default is len(sequence).
        :param auto_refresh: Automatic refresh. Disable to force a refresh after each iteration. Default is True.
        :param transient: Clear the progress on exit. Defaults to False.
        :param get_time: A callable that gets the current time, or None to use Console.get_time. Defaults to None.
        :param console: Console to write to. Default creates an internal Console instance.
        :param refresh_per_second: Number of times per second to refresh the progress information. Defaults to 10.
        :param style: Style for the bar background. Defaults to "bar.back".
        :param complete_style: Style for the completed bar. Defaults to "bar.complete".
        :param finished_style: Style for a finished bar. Defaults to "bar.finished".
        :param pulse_style: Style for pulsing bars. Defaults to "bar.pulse".
        :param update_period: Minimum time (in seconds) between calls to update(). Defaults to 0.1.
        :param disable: Disable display of progress.
        :param show_speed: Show speed if the total isn't known. Defaults to True.
        :param weights: List of progress weights for each sequence element. Weights should be proportional to the time
            it takes to process sequence elements. Useful when processing time is strongly non-uniform.
        :return: An iterable of the values in the sequence.
        """

        self.sequence = sequence
        self.weights = weights
        self.total = sum(self.weights) if self.weights is not None else total
        self.description = description
        self.update_period = update_period
        self.task: Optional[TaskID] = None

        self.columns: List[ProgressColumn] = (
            [TextColumn("[progress.description]{task.description}")] if description else []
        )
        self.columns.extend(
            (
                BarColumn(
                    style=style,
                    complete_style=complete_style,
                    finished_style=finished_style,
                    pulse_style=pulse_style,
                    bar_width=None,
                ),
                TaskProgressColumn(show_speed=show_speed),
            )
        )
        # Do not add iterations column for weighted tracking because steps will be in weighted coordinates
        if self.weights is None:
            self.columns.append(IterationsColumn())
        self.columns.extend(
            (
                SeparatorColumn(),
                TimeElapsedColumnWithStyle(),
                SeparatorColumn(disable_if_no_total=True),  # disable because time remaining will be empty
                TimeRemainingColumnWithStyle(),
            )
        )

        disable = disable or (hasattr(sequence, "__len__") and len(sequence) == 0)  # type: ignore[arg-type]

        progress_cls = Progress if weights is None else WeightedProgress
        self.progress = progress_cls(
            *self.columns,
            auto_refresh=auto_refresh,
            console=console,
            transient=transient,
            get_time=get_time,
            refresh_per_second=refresh_per_second or 10,
            disable=disable,
        )

    def __iter__(self) -> Iterator[ProgressType]:
        if self.sequence is None:
            raise RuntimeError("__iter__ called without set sequence.")
        with self:
            yield from self.progress.track(
                self.sequence,
                total=self.total,
                task_id=self.task,
                description=self.description,
                update_period=self.update_period,
            )

    def __enter__(self) -> track[ProgressType]:
        kwargs: Dict[str, Any] = {}
        if self.weights is not None:
            kwargs["weights"] = self.weights
            kwargs["completed_steps"] = 0
        self.task = self.progress.add_task(self.description, total=self.total, **kwargs)
        self.progress.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        self.progress.__exit__(*args)
        if self.task is not None:
            self.progress.remove_task(self.task)
            self.task = None

    def update(self, advance: float, **kwargs: Any) -> None:
        if self.task is None:
            raise RuntimeError("update is available only inside context manager.")
        self.progress.update(self.task, advance=advance, **kwargs)
