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
from datetime import timedelta
from typing import Callable, Iterable, List, Optional, Sequence, Union

from rich.console import Console
from rich.progress import BarColumn
from rich.progress import Column
from rich.progress import Progress
from rich.progress import ProgressColumn
from rich.progress import ProgressType
from rich.progress import Task
from rich.progress import TaskProgressColumn
from rich.progress import TextColumn
from rich.style import StyleType
from rich.text import Text

INTEL_BLUE_COLOR = (0, 113, 197)


class IterationsColumn(ProgressColumn):
    def __init__(self, style: Union[str, StyleType]):
        super().__init__()
        self.style = style

    def render(self, task: Task) -> Text:
        if task.total is None:
            return Text("")
        text = f"{int(task.completed)}/{int(task.total)}"
        return Text(text, style=self.style)


class SeparatorColumn(ProgressColumn):
    def __init__(self, table_column: Optional[Column] = None, disable_if_no_total: bool = False) -> None:
        super().__init__(table_column)
        self.disable_if_no_total = disable_if_no_total

    def render(self, task: Task) -> Text:
        if self.disable_if_no_total and task.total is None:
            return Text("")
        return Text("•")


class TimeElapsedColumnWithStyle(ProgressColumn):
    """
    Renders time elapsed.

    Similar to TimeElapsedColumn, but with addition of style parameter.
    """

    def __init__(self, style: Union[str, StyleType]):
        super().__init__()
        self.style = style

    def render(self, task: "Task") -> Text:
        """Show time elapsed."""
        elapsed = task.finished_time if task.finished else task.elapsed
        if elapsed is None:
            return Text("-:--:--", style=self.style)
        delta = timedelta(seconds=max(0, int(elapsed)))
        return Text(str(delta), style=self.style)


class TimeRemainingColumnWithStyle(ProgressColumn):
    """
    Renders estimated time remaining.

    Similar to TimeRemainingColumn, but with addition of style parameter.
    """

    # Only refresh twice a second to prevent jitter
    max_refresh = 0.5

    def __init__(
        self,
        style: Union[str, StyleType],
        compact: bool = False,
        elapsed_when_finished: bool = False,
        table_column: Optional[Column] = None,
    ):
        self.style = style
        self.compact = compact
        self.elapsed_when_finished = elapsed_when_finished
        super().__init__(table_column=table_column)

    def render(self, task: "Task") -> Text:
        """Show time remaining."""
        if self.elapsed_when_finished and task.finished:
            task_time = task.finished_time
        else:
            task_time = task.time_remaining

        if task.total is None:
            return Text("", style=self.style)

        if task_time is None:
            return Text("--:--" if self.compact else "-:--:--", style=self.style)

        # Based on https://github.com/tqdm/tqdm/blob/master/tqdm/std.py
        minutes, seconds = divmod(int(task_time), 60)
        hours, minutes = divmod(minutes, 60)

        if self.compact and not hours:
            formatted = f"{minutes:02d}:{seconds:02d}"
        else:
            formatted = f"{hours:d}:{minutes:02d}:{seconds:02d}"

        return Text(formatted, style=self.style)


class track:
    def __init__(
        self,
        sequence: Optional[Union[Sequence[ProgressType], Iterable[ProgressType]]] = None,
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
    ):
        """
        Track progress by iterating over a sequence.

        This function is very similar to rich.progress.track(), but with some customizations.

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
        :return: An iterable of the values in the sequence.
        """

        self.sequence = sequence
        self.total = total
        self.description = description
        self.update_period = update_period
        self.task = None

        text_style = f"rgb({INTEL_BLUE_COLOR[0]},{INTEL_BLUE_COLOR[1]},{INTEL_BLUE_COLOR[2]})"

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
                IterationsColumn(style=text_style),
                SeparatorColumn(),
                TimeElapsedColumnWithStyle(style=text_style),
                SeparatorColumn(disable_if_no_total=True),  # disable because time remaining will be empty
                TimeRemainingColumnWithStyle(style=text_style),
            )
        )

        disable = disable or (hasattr(sequence, "__len__") and len(sequence) == 0)

        self.progress = Progress(
            *self.columns,
            auto_refresh=auto_refresh,
            console=console,
            transient=transient,
            get_time=get_time,
            refresh_per_second=refresh_per_second or 10,
            disable=disable,
        )

    def __iter__(self) -> Iterable[ProgressType]:
        with self.progress:
            yield from self.progress.track(
                self.sequence, total=self.total, description=self.description, update_period=self.update_period
            )

    def __enter__(self):
        self.progress.start()
        self.task = self.progress.add_task(self.description, total=self.total)
        return self

    def __exit__(self, *args):
        self.task = None
        self.progress.stop()
