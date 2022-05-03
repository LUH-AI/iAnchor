from ianchor.bandit import KL_LUCB


def test_dup_bernoulli():
    um = KL_LUCB.dup_bernoulli(0.5, 1)
    assert um == 0.9649367481470108


def test_dlow_bernoulli():
    lm = KL_LUCB.dlow_bernoulli(0.5, 1)
    assert lm == 0.0350632518529892


def test_kl_bernoulli():
    kl = KL_LUCB.kl_bernoulli(1, 0.5)
    assert kl == 0.6931471805599411


def test_compute_beta_bernoulli():
    beta = KL_LUCB.compute_beta(5, 1, 0.5)
    assert beta == 10.424889480332546
