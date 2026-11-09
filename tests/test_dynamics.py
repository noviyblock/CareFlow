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


def test_state_windows_skip_rituals():
    from careflow.eval.state_signals import llm_intensity, spearman_ci, substantive, windows

    texts = ["Hello", "I lost my job last week", "I feel really scared now", "it keeps me awake",
             "maybe I can call them", "I will try that tomorrow", "thank you so much, bye"]
    assert substantive(texts, 4) == [1, 2, 3, 4, 5]
    assert windows(texts, 2, 4) == ([1, 2], [4, 5])
    assert windows(texts, 3, 4) is None
    assert llm_intensity(lambda p: " 4\n", ["x"]) == 4 and llm_intensity(lambda p: "n/a", ["x"]) is None
    r = spearman_ci([1, 2, 3, 4, 5], [2, 4, 6, 8, 10], 50, 0)
    assert abs(r["rho"] - 1.0) < 1e-9 and r["n"] == 5


def test_paired_rho_diff():
    from careflow.eval.state_signals import paired_rho_diff

    y = list(range(50))
    good = [v + (i % 3) for i, v in enumerate(y)]
    noise = [(i * 7919) % 50 for i in range(50)]
    r = paired_rho_diff(good, noise, y, 200, 0)
    assert r["diff"] > 0.5 and r["ci_low"] > 0 and r["p_le_0"] == 0.0
