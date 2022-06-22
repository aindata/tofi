"""
Current tests are written as a comparison from
https://github.com/rmunro/pytorch_active_learning/blob/master/uncertainty_sampling.py

We expect the same output from our functions as with Robert Munro's repo

Please note that methods from Robert Munro's repo is denoted with mu_ prefix
# TODO: validate better testing
"""


def test_least_confidence():
    """
    A test case scenario for an artificial input:

    mu_least_confidence:
     - input: tensor([[0.8254, 0.1746]])
     - output: 0.34910380840301514

    """
    pass


def test_margin_confidence():
    """
    A test case scenario for an artificial input:

    mu_margin_confidence:
     - input: tensor([0.8254, 0.1746])
     - output: 0.3491038680076599

    """
    pass


def test_ratio_confidence():
    """
    A test case scenario for ratio confidence
    mu_ratio_confidence:
     - input: tensor([0.8254, 0.1746])
     - output: 0.21146325767040253
    """
    pass


def test_entropy_based_confidence():
    """
    A test case scenario for entropy-based confidence

    mu_entropy_based:
     - input: tensor([0.8254, 0.1746])
     - output: 0.6680124998092651

    """
    pass

