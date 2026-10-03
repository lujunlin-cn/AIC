import itertools, json, platform, hashlib, time
from pathlib import Path
import numpy as np
import torch
from expected_f import expected_f

def enumerate_expected(logits, costs, gains, gt, empty_value=1.0):
    acts = torch.tensor(list(itertools.product((0., 1.), repeat=len(costs))), dtype=logits.dtype)
    p = logits.sigmoid()[None, :]
    probabilities = (acts * p + (1 - acts) * (1 - p)).prod(1)
    k = acts @ torch.tensor(costs, dtype=logits.dtype)
    m = acts @ gains
    rewards = 2 * m / (k + gt).clamp_min(1)
    rewards = torch.where((k + gt) == 0, rewards.new_full(rewards.shape, empty_value), rewards)
    return (probabilities * rewards).sum()

rng = np.random.default_rng(20261003)
torch.set_num_threads(1)
err_r, err_g = [], []
start = time.perf_counter()
for case in range(120):
    n = int(rng.integers(1, 10))
    costs = list(map(int, rng.integers(1, 8, size=n)))
    positives = rng.integers(0, np.array(costs) + 1)
    gt = int(positives.sum())
    gains = torch.tensor(positives * rng.random(n), dtype=torch.float64)
    logits = torch.tensor(rng.normal(size=n) * 2, dtype=torch.float64, requires_grad=True)
    exact = expected_f(logits, costs, gains, gt)
    brute = enumerate_expected(logits, costs, gains, gt)
    eg = torch.autograd.grad(exact, logits)[0]
    bg = torch.autograd.grad(brute, logits)[0]
    err_r.append(float(abs(exact.detach() - brute.detach())))
    err_g.append(float((eg.detach() - bg.detach()).abs().max()))
    assert torch.allclose(exact, brute, atol=1e-11, rtol=1e-11)
    assert torch.allclose(eg, bg, atol=1e-11, rtol=1e-11)
for n in (1, 4, 8):
    for empty in (0., 1.):
        x = torch.zeros(n, dtype=torch.float64, requires_grad=True)
        y = expected_f(x, [2] * n, torch.zeros(n, dtype=torch.float64), 0, empty)
        assert abs(float(y.detach()) - empty * 0.5 ** n) < 1e-12

numbers = {
    'p': 0.33,
    'all_keep_temporal_homogeneous': 2 * 0.33 / 1.33,
    'minimum_macro_p_if_all_keep_f_is_06237': 0.6237 / (2 - 0.6237),
    'homogeneous_score_all_keep_k09_m076': 100 * .9 * .76 * 2 * .33 / 1.33,
    'conditional_rank_oracle_proxy_score': 100 * .9 * .76 * .7804,
    'conditional_current_proxy_score': 100 * .9 * .76 * .6648,
    'anchor_scaled_prefix_oracle_not_forecast': 34.88 * .7804 / .6648,
    'decision_gap': .7804 - .6648,
    'temporal_required_at_45_k09_m076': .45 / (.9 * .76),
    'temporal_required_at_50_k09_m076': .50 / (.9 * .76),
    'temporal_required_at_60_k09_m076': .60 / (.9 * .76),
    'binary_gain_001_to_points_k09_m076': 100 * .9 * .76 * .01,
}
root = Path(__file__).parent
result = {
 'status': 'PASS', 'scope': 'synthetic DP value and gradient only; no AIC training or platform replication',
 'random_cases_value_and_gradient': 120, 'explicit_empty_cases': 6,
 'max_reward_abs_error': max(err_r), 'max_logit_gradient_abs_error': max(err_g),
 'seed': 20261003, 'python': platform.python_version(), 'torch': torch.__version__,
 'elapsed_seconds': time.perf_counter() - start,
 'implementation_sha256': hashlib.sha256((root / 'expected_f.py').read_bytes()).hexdigest(),
 'numbers': numbers,
}
(root / 'numerical_checks.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
