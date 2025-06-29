"""Module for user interface funcionality."""

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import plotext
import rich.progress as rp
from rich.ansi import AnsiDecoder
from rich.console import Group
from rich.jupyter import JupyterMixin
from rich.layout import Layout
from rich.live import Live
from rich.text import Text


def _interpolate_to_fixed_length(data, target_length) -> np.ndarray:
    if len(data) == target_length:
        return data
    old_indices = np.linspace(0, len(data) - 1, len(data))
    new_indices = np.linspace(0, len(data) - 1, target_length)
    return np.interp(new_indices, old_indices, data)


def cli_metric_plot(
    width: int,
    height: int,
    metrics: dict[str, tuple[list[float], list[float]]],
    theme: str = "pro",
):
    """Return a plotext canvas with curves of metrics.

    Parameters
    ----------
    width:
        Desired width of the canvas.
    height:
        Desired height of the canvas.
    metrics:
        Maps metric names to lists of values.
    theme:
        plottext theme to use.

    Returns
    -------
    plotext canvas instance.

    """
    plotext.clf()
    plotext.theme(theme)

    for metric, (indices, values) in metrics.items():
        plotext.plot(indices, values, label=metric)

    plotext.plotsize(width, height)
    return plotext.build()


class MetricPlotMixin(JupyterMixin):
    """Helper class to make `cli_metric_plot` work with rich.

    Attributes
    ----------
    handle:
        Identifier of this plot.
    indices:
        Indices to plot, i.e. x-values.
    values:
        Values to plot, i.e. y-values.

    """

    def __init__(self, handle: str, indices: list[float], values: list[float]):  # noqa: D107
        self._decoder = AnsiDecoder()
        self.handle = handle
        self.indices = indices
        self.values = values

    def __rich_console__(self, console, options):  # noqa: D105
        self.width = options.max_width or console.width
        self.height = options.height or console.height
        canvas = cli_metric_plot(
            self.width,
            self.height,
            {self.handle: (self.indices, self.values)},
        )
        self.rich_canvas = Group(*self._decoder.decode(canvas))

        yield self.rich_canvas


@dataclass
class BaseCLIMetricsCallback:
    """Base class to for callbacks that display metrics over time.

    Make sure to call `.stop()` afterwards to release rich's grip on
    the terminal.

    Values of metrics will only be saved sparsely for performance. At most
    1000 values will be saved, as this is sufficient for console-based plots.

    Attributes
    ----------
    metrics:
        Names of the metrics.
    total_steps:
        Amount of optimisation steps, used for progress display.
    description:
        Textual description of what is being worked on.
    plot_height:
        Height of the area the metric plot is in. Passed to rich's `Layout`.

    """

    metrics: tuple[str, ...]
    total_steps: int
    description: str = "Working..."
    plot_size: int = 15

    _items: dict[str, list[tuple[float, float]]] = field(
        default_factory=lambda: defaultdict(list), init=False
    )
    _steps_done: int = 0
    _is_setup: bool = field(init=False, default=False)

    def setup(self):
        """Prepare the callback for display."""
        self.layout = Layout()
        self.layout.split(
            Layout(name="plot", size=self.plot_size),
            Layout(name="progress", size=3),
        )
        self.layout["plot"].split_row(*[Layout(name=m) for m in self.metrics])

        progress_text = " / ".join(
            ("[bold yellow]" + f + ": {task.fields[" + f + "]:.4f}")
            for f in self.metrics
        )
        self.progress = rp.Progress(
            rp.TextColumn("[bold blue]{task.description}"),
            rp.BarColumn(),
            rp.TaskProgressColumn(),
            rp.TimeElapsedColumn(),
            rp.TimeRemainingColumn(),
            rp.TextColumn(progress_text),
        )
        self.task_id = self.progress.add_task(
            f"[green]{self.description}[/green]",
            total=self.total_steps,
            **{k: float("inf") for k in self.metrics},  # type: ignore
        )

        self.layout["progress"].update(self.progress)
        self.layout["plot"].update(Text("not data yet"))
        self.live = Live(self.layout, refresh_per_second=4)
        self.live.start()
        self._is_setup = True

    def _update(self, i_update: int, **values):
        if not all(m in values for m in self.metrics):
            raise ValueError("need to provide values for all metrics")

        for key in values:
            values[key] = (
                float(values[key]) if values[key] is not None else None
            )

        if not self._is_setup:
            self.setup()

        _steps_done = int(self.progress._tasks[self.task_id].completed)

        # Pick an unusual number--I decided for a prime--such that it does not
        # align by accident with episode starts and then hides some of the
        # performance. This happens for example if always the first chunk of a
        # chunked episode is used for reporting.
        max_console_width = 1009
        for metric in values:
            if values[metric] is None:
                continue
            self._items[metric].append((i_update, values[metric]))
            indices_all = [i for i, j in self._items[metric]]
            values_all = [j for i, j in self._items[metric]]
            indices_subsampled = _interpolate_to_fixed_length(
                indices_all, max_console_width
            )
            values_subsampled = _interpolate_to_fixed_length(
                values_all, max_console_width
            )
            self.layout["plot"][metric].update(
                MetricPlotMixin(
                    metric,
                    indices=[float(i) for i in indices_subsampled],
                    values=[float(i) for i in values_subsampled],
                )
            )

        last_values = {
            key: self._items[key][-1][1] for key in values if self._items[key]
        }
        self.progress.update(self.task_id, completed=i_update, **last_values)  # type: ignore

    def teardown(self):
        """Tear down the callback from display."""
        if self._is_setup:
            self.live.stop()
