"""Unit tests for policy search callbacks."""

import datetime
import os
import pathlib
import unittest.mock

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import pytest

from seher.apx_arch import MLP, StaticMLPPolicy
from seher.control.policy_search import (
    ActorCriticAuxiliary,
    ActorCriticCLIProgressCallback,
    BaseProgressCallback,
    MCPSCLIProgressCallback,
    SaveRolloutGif,
    WriteEvalCostsToFile,
    apx_cost_to_go_from_aux,
    cost_from_aux_history,
    cost_from_eval_history,
    cost_from_train_history,
    critic_loss_from_aux,
)
from seher.control.solvers import PolicySearchSolver
from seher.simulate import History, simulate
from seher.systems.mujoco_playground import MujocoPlaygroundMDP
from seher.systems.pendulum import Pendulum


def test_write_eval_costs_to_file_writes_cost_to_file(tmp_path):
    """Test WriteEvalCostsToFile writes costs to file correctly."""
    # Setup
    file_path = tmp_path / "test_costs.txt"
    callback = WriteEvalCostsToFile(file_name=file_path)

    # Create mock eval history with known costs
    eval_history = History(
        states=jnp.array([1.0, 2.0]),
        controls=jnp.array([0.5]),
        costs=jnp.array([[1.5, 2.5], [3.0, 4.0]]),  # mean = 2.75
        policy_carries=jnp.array([0.0]),
    )

    # Mock datetime to get predictable output
    with unittest.mock.patch("datetime.datetime") as mock_datetime:
        mock_datetime.now.return_value.isoformat.return_value = (
            "2024-01-01T12:00:00"
        )

        # Call the callback
        callback(i_update=42, eval_history=eval_history)

    # Verify file contents
    content = file_path.read_text()
    expected_line = "2024-01-01T12:00:00 42 2.7500\n"
    assert content == expected_line


def test_write_eval_costs_to_file_does_nothing_when_eval_history_is_none(
    tmp_path,
):
    """Test WriteEvalCostsToFile does nothing when eval_history is None."""
    file_path = tmp_path / "test_costs.txt"
    callback = WriteEvalCostsToFile(file_name=file_path)

    # Call with None eval_history
    callback(i_update=1, eval_history=None)

    # File should not exist
    assert not file_path.exists()


def test_write_eval_costs_to_file_appends_multiple_calls(tmp_path):
    """Test WriteEvalCostsToFile appends multiple calls to same file."""
    file_path = tmp_path / "test_costs.txt"
    callback = WriteEvalCostsToFile(file_name=file_path)

    eval_history1 = History(
        states=jnp.array([1.0]),
        controls=jnp.array([0.5]),
        costs=jnp.array([[1.0]]),
        policy_carries=jnp.array([0.0]),
    )

    eval_history2 = History(
        states=jnp.array([2.0]),
        controls=jnp.array([0.7]),
        costs=jnp.array([[2.0]]),
        policy_carries=jnp.array([0.0]),
    )

    with unittest.mock.patch("datetime.datetime") as mock_datetime:
        mock_datetime.now.return_value.isoformat.return_value = (
            "2024-01-01T12:00:00"
        )
        callback(i_update=1, eval_history=eval_history1)
        callback(i_update=2, eval_history=eval_history2)

    lines = file_path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert "1 1.0000" in lines[0]
    assert "2 2.0000" in lines[1]


@pytest.mark.slow
def test_save_rollout_gif_saves_gif_when_eval_history_provided(tmp_path):
    """Test that SaveRolloutGif saves gif when eval_history is provided."""
    # Setup MDP and callback
    mdp = MujocoPlaygroundMDP.from_registry("CartpoleBalance")
    callback = SaveRolloutGif(mdp=mdp, filename_prefix="test_")

    # Create a simple policy
    # Note: MDP state structures are dynamic and vary by environment type.
    # Static typing cannot handle this variability properly.
    policy = StaticMLPPolicy(
        obs_to_array=lambda state: state.obs,  # type: ignore
        array_to_control=lambda x: x,
        mlp=MLP.make(
            inpt_size=mdp.init(jr.PRNGKey(0)).obs.shape[0],  # type: ignore
            output_size=mdp.empty_control().shape[0],
            layer_sizes=[8],
            activations=[jnn.soft_sign, lambda x: x],
            key=jr.PRNGKey(1),
        ),
    )

    # Simulate to get history
    history = simulate(
        policy=policy,
        mdp=mdp,
        n_steps=5,
        key=jr.PRNGKey(2),
    )

    # Add batch dimension to simulate batched evaluation
    eval_history = jax.tree.map(lambda x: x[None, ...], history)

    # Change to tmp directory for gif output
    original_cwd = pathlib.Path.cwd()
    try:
        os.chdir(tmp_path)

        # Call callback
        callback(i_update=123, eval_history=eval_history)

        # Check gif was created
        expected_file = tmp_path / "test_rollout-123.gif"
        assert expected_file.exists()

    finally:
        os.chdir(original_cwd)


def test_save_rollout_gif_does_nothing_when_eval_history_is_none(tmp_path):
    """Test SaveRolloutGif callback does nothing when eval_history is None."""
    mdp = MujocoPlaygroundMDP.from_registry("CartpoleBalance")
    callback = SaveRolloutGif(mdp=mdp)

    original_cwd = pathlib.Path.cwd()
    try:
        os.chdir(tmp_path)

        callback(i_update=1, eval_history=None)

        # No gif files should be created
        gif_files = list(tmp_path.glob("*.gif"))
        assert len(gif_files) == 0

    finally:
        os.chdir(original_cwd)


def test_mcps_cli_progress_callback_updates_with_both_histories():
    """Test MCPSCLIProgressCallback with both histories."""
    callback = MCPSCLIProgressCallback(total_steps=100)

    train_history = History(
        states=jnp.array([1.0]),
        controls=jnp.array([0.5]),
        costs=jnp.array([[1.0, 2.0]]),  # mean = 1.5
        policy_carries=jnp.array([0.0]),
    )

    eval_history = History(
        states=jnp.array([2.0]),
        controls=jnp.array([0.7]),
        costs=jnp.array([[3.0, 4.0]]),  # mean = 3.5
        policy_carries=jnp.array([0.0]),
    )

    # Mock the _update method to capture calls
    with unittest.mock.patch.object(callback, "_update") as mock_update:
        callback(
            i_update=10,
            train_history=train_history,
            eval_history=eval_history,
        )

        mock_update.assert_called_once_with(
            i_update=10,
            train_cost=1.5,
            eval_cost=3.5,
        )


def test_mcps_cli_progress_callback_handles_none_histories():
    """Test MCPSCLIProgressCallback handles None histories."""
    callback = MCPSCLIProgressCallback(total_steps=100)

    with unittest.mock.patch.object(callback, "_update") as mock_update:
        callback(i_update=5, train_history=None, eval_history=None)

        mock_update.assert_called_once_with(
            i_update=5,
            train_cost=None,
            eval_cost=None,
        )


def test_actor_critic_cli_progress_callback_updates_with_aux_data():
    """Test ActorCriticCLIProgressCallback extracts metrics from aux data."""
    callback = ActorCriticCLIProgressCallback(total_steps=100)

    # Create mock auxiliary data
    history = History(
        states=jnp.array([1.0]),
        controls=jnp.array([0.5]),
        costs=jnp.array([[2.0, 3.0]]),  # mean = 2.5
        policy_carries=jnp.array([0.0]),
    )

    aux = ActorCriticAuxiliary(
        history=history,
        loss=jnp.array(1.0),
        apx_cost_to_go=jnp.array([4.0, 5.0]),  # mean = 4.5
        critic_loss=jnp.array([0.5, 0.7]),  # mean = 0.6
    )

    with unittest.mock.patch.object(callback, "_update") as mock_update:
        callback(
            i_update=15,
            train_history=history,
            eval_history=history,
            aux=aux,
        )

        # Check that mock was called once
        assert mock_update.call_count == 1
        call_args = mock_update.call_args[1]  # Get keyword arguments
        assert call_args["i_update"] == 15
        assert call_args["cost"] == pytest.approx(2.5)
        assert call_args["critic_loss"] == pytest.approx(0.6)
        assert call_args["apx_cost_to_go"] == pytest.approx(4.5)


@pytest.mark.slow
def test_write_eval_costs_integration_with_pendulum_solver(tmp_path):
    """Integration test: WriteEvalCostsToFile with PolicySearchSolver."""
    # Setup solver with callback
    file_path = tmp_path / "integration_costs.txt"
    callback = WriteEvalCostsToFile(file_name=file_path)

    solver = PolicySearchSolver(
        obs_to_array=lambda state: state.cos_sin_repr(),
        n_simulations=2,
        steps_per_update=4,
        episode_length=8,
        max_updates=3,
        updates_per_eval=1,  # Evaluate every update
        eval_n_simulations=1,
    )
    solver.callbacks.append(callback)

    # Solve on Pendulum
    mdp = Pendulum()
    solver.solve(mdp, key=jr.PRNGKey(42))

    # Verify file was created and has expected number of entries
    assert file_path.exists()
    lines = file_path.read_text().strip().split("\n")
    # Should have 3 lines (one per update due to updates_per_eval=1)
    assert len(lines) == 3

    # Verify format of each line
    for i, line in enumerate(lines):
        parts = line.split()
        assert len(parts) == 3
        # Check date format (ISO)
        datetime.datetime.fromisoformat(parts[0])
        # Check update number
        assert int(parts[1]) == i
        # Check cost is a float
        float(parts[2])


def test_base_progress_callback_with_custom_extractors():
    """Test BaseProgressCallback with custom metric extractors."""

    def custom_extractor(train_history, eval_history, aux):
        del train_history
        del eval_history
        del aux
        return 42.0

    callback = BaseProgressCallback(
        total_steps=100, metric_extractors={"custom_metric": custom_extractor}
    )

    with unittest.mock.patch.object(callback, "_update") as mock_update:
        callback(i_update=5, train_history=None, eval_history=None, aux=None)

        mock_update.assert_called_once_with(
            i_update=5,
            custom_metric=42.0,
        )


def test_base_progress_callback_handles_extractor_errors():
    """Test how BaseProgressCallback handles extractor errors."""

    def failing_extractor(train_history, eval_history, aux):
        del train_history
        del eval_history
        del aux
        raise AttributeError("Simulated failure")

    callback = BaseProgressCallback(
        total_steps=100,
        metric_extractors={"failing_metric": failing_extractor},
    )

    with unittest.mock.patch.object(callback, "_update") as mock_update:
        callback(i_update=5, train_history=None, eval_history=None, aux=None)

        mock_update.assert_called_once_with(
            i_update=5,
            failing_metric=None,
        )


def test_actor_critic_cli_progress_callback():
    """Test ActorCriticCLIProgressCallback."""
    callback = ActorCriticCLIProgressCallback(total_steps=100)

    # Create mock auxiliary data
    history = History(
        states=jnp.array([1.0]),
        controls=jnp.array([0.5]),
        costs=jnp.array([[2.0, 3.0]]),  # mean = 2.5
        policy_carries=jnp.array([0.0]),
    )

    aux = ActorCriticAuxiliary(
        history=history,
        loss=jnp.array(1.0),
        apx_cost_to_go=jnp.array([4.0, 5.0]),  # mean = 4.5
        critic_loss=jnp.array([0.5, 0.7]),  # mean = 0.6
    )

    with unittest.mock.patch.object(callback, "_update") as mock_update:
        callback(
            i_update=10,
            train_history=history,
            eval_history=history,
            aux=aux,
        )

        # Check that mock was called once
        assert mock_update.call_count == 1
        call_args = mock_update.call_args[1]  # Get keyword arguments
        assert call_args["i_update"] == 10
        assert call_args["cost"] == pytest.approx(2.5)
        assert call_args["critic_loss"] == pytest.approx(0.6)
        assert call_args["apx_cost_to_go"] == pytest.approx(4.5)


def test_mcps_cli_progress_callback():
    """Test MCPSCLIProgressCallback."""
    callback = MCPSCLIProgressCallback(total_steps=100)

    train_history = History(
        states=jnp.array([1.0]),
        controls=jnp.array([0.5]),
        costs=jnp.array([[1.0, 2.0]]),  # mean = 1.5
        policy_carries=jnp.array([0.0]),
    )

    eval_history = History(
        states=jnp.array([2.0]),
        controls=jnp.array([0.7]),
        costs=jnp.array([[3.0, 4.0]]),  # mean = 3.5
        policy_carries=jnp.array([0.0]),
    )

    with unittest.mock.patch.object(callback, "_update") as mock_update:
        callback(
            i_update=15,
            train_history=train_history,
            eval_history=eval_history,
        )

        mock_update.assert_called_once_with(
            i_update=15,
            train_cost=1.5,
            eval_cost=3.5,
        )


def test_metric_extractors_work_independently():
    """Test individual metric extractor functions work correctly."""
    # Create test data
    train_history = History(
        states=jnp.array([1.0]),
        controls=jnp.array([0.5]),
        costs=jnp.array([[10.0, 20.0]]),  # mean = 15.0
        policy_carries=jnp.array([0.0]),
    )

    eval_history = History(
        states=jnp.array([2.0]),
        controls=jnp.array([0.7]),
        costs=jnp.array([[30.0, 40.0]]),  # mean = 35.0
        policy_carries=jnp.array([0.0]),
    )

    aux_history = History(
        states=jnp.array([3.0]),
        controls=jnp.array([0.8]),
        costs=jnp.array([[50.0, 60.0]]),  # mean = 55.0
        policy_carries=jnp.array([0.0]),
    )

    aux = ActorCriticAuxiliary(
        history=aux_history,
        loss=jnp.array(1.0),
        apx_cost_to_go=jnp.array([70.0, 80.0]),  # mean = 75.0
        critic_loss=jnp.array([0.1, 0.2]),  # mean = 0.15
    )

    # Test individual extractors (use approximate equality for floating point)
    assert cost_from_train_history(
        train_history, eval_history, aux
    ) == pytest.approx(15.0)
    assert cost_from_eval_history(
        train_history, eval_history, aux
    ) == pytest.approx(35.0)
    assert cost_from_aux_history(
        train_history, eval_history, aux
    ) == pytest.approx(55.0)
    assert critic_loss_from_aux(
        train_history, eval_history, aux
    ) == pytest.approx(0.15)
    assert apx_cost_to_go_from_aux(
        train_history, eval_history, aux
    ) == pytest.approx(75.0)

    # Test with None inputs
    assert cost_from_train_history(None, eval_history, aux) is None
    assert cost_from_eval_history(train_history, None, aux) is None
    assert cost_from_aux_history(train_history, eval_history, None) is None
    assert critic_loss_from_aux(train_history, eval_history, None) is None
    assert apx_cost_to_go_from_aux(train_history, eval_history, None) is None
