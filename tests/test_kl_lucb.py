import pytest
from Anchor.kl_lucb import KL_LUCB


def test_best_candidate():
    ...


def test_dup_bernoulli():
    um = KL_LUCB.dup_bernoulli(0.2, 1)
    assert um == 0.839887952608281


def test_dlow_bernoulli():
    lm = KL_LUCB.dlow_bernoulli(0.2, 1)
    assert lm == 0.0005531907081604004


def test_kl_bernoulli():
    kl = KL_LUCB.kl_bernoulli(1, 0.2)
    assert kl == 1.609437912434096
