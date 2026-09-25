import torch
from aic.ranking_loss import pairwise_logistic


def call(z, y, mask=None):
    return pairwise_logistic(z, y, torch.ones_like(y, dtype=torch.bool) if mask is None else mask,
                             torch.Generator().manual_seed(13), draws=1024, min_gap=.1)


def test_preference_gradient_and_loss_are_directional():
    y = torch.tensor([.1, .9]); z = torch.zeros(2, requires_grad=True)
    loss, n = call(z, y); loss.backward()
    assert n > 0 and z.grad[0] > 0 and z.grad[1] < 0
    good, _ = call(torch.tensor([-2., 2.]), y)
    bad, _ = call(torch.tensor([2., -2.]), y)
    assert good < loss < bad


def test_padding_and_global_rng_are_unchanged():
    y = torch.tensor([.1, .9]); z = torch.tensor([-.3, .2])
    rng = torch.random.get_rng_state().clone()
    reference, n = call(z, y)
    padded, pn = call(torch.tensor([-.3, .2, 999.]), torch.tensor([.1, .9, 0.]), torch.tensor([1, 1, 0]))
    assert padded == reference and n == pn
    assert torch.equal(rng, torch.random.get_rng_state())


def test_ties_and_tiny_gaps_have_zero_differentiable_loss():
    z = torch.tensor([.4, .5], requires_grad=True)
    value, n = call(z, torch.tensor([.5, .51])); value.backward()
    assert n == 0 and value.item() == 0 and torch.equal(z.grad, torch.zeros_like(z))


def test_draws_reproducible_and_bounded():
    z = torch.linspace(-1, 1, 100000); y = torch.sigmoid(z)
    first, n = call(z, y); second, m = call(z, y)
    assert first == second and n == m and 0 < n <= 1024
