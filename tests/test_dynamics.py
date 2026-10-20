import numpy as np

from careflow.eval.dynamics import run_lengths, summarize, trajectory, transition_matrix


def test_transition_matrix_rows_normalized():
    tm = transition_matrix([["a", "a", "b"], ["b", "a"]], ["a", "b", "c"])
    np.testing.assert_allclose(tm[0], [0.5, 0.5, 0])
    np.testing.assert_allclose(tm[1], [1, 0, 0])
    assert tm[2].sum() == 0  # unseen state


def test_run_lengths():
    assert run_lengths(["neg", "neg", "pos", "neg"], "neg") == [2, 1]
    assert run_lengths(["pos"], "neg") == []


def test_trajectory_flags():
    fast = trajectory(["neg", "pos", "pos"], fast_k=1, stuck_k=3)
    assert fast.too_fast and fast.deescalated and not fast.stuck and fast.first_pos_turn == 1
    slow = trajectory(["neg", "neg", "neu", "pos"], fast_k=1, stuck_k=3)
    assert not slow.too_fast and slow.polarity_shifts == 2 and slow.longest_neg_run == 2
    stuck = trajectory(["neu", "neg", "neg", "neg"], fast_k=1, stuck_k=3)
    assert stuck.stuck and stuck.escalated
    # positive before any negative is not a "recovery"
    assert trajectory(["pos", "neg"], 1, 3).first_pos_turn is None


def test_summarize():
    s = summarize([trajectory(["neg", "pos"], 1, 3), trajectory(["neg", "neg", "neg"], 1, 3)])
    assert s["deescalation_rate"] == 0.5 and s["stuck_rate"] == 0.5 and s["too_fast_rate"] == 0.5
