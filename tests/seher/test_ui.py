"""Tests for the user interface module."""

import time

from seher import ui


def test_cli_metrics_callback():
    """Test whether cli metrics callback does not throw errors."""
    n_steps = 10
    cmc = ui.BaseCLIMetricsCallback(
        metrics=("cost", "loss"), total_steps=n_steps
    )

    for i in range(n_steps):
        cmc._update(i, cost=10 / (i + 1), loss=0.9**i)
        time.sleep(0.0001)

    cmc.teardown()
