"""
Current tests are written as a comparison from
https://github.com/rmunro/pytorch_active_learning/blob/master/uncertainty_sampling.py

We expect the same output from our functions as with Robert Munro's repo

Please note that methods from Robert Munro's repo is denoted with mu_ prefix

"""
import sys
sys.path.append('../..')  # TODO: find better way to import files from src
from src.tofi.sampling import uncertainty_sampling


# TODO: validate better testing

def test_least_confidence(proba):
    """
    A test case scenario for an artificial input:

    mu_least_confidence:
     - proba: tensor([0.8305774  0.16942263])
     - output: 0.33884525299072266

    """
    assert np.isclose(uncertainty_sampling.least_confidence(proba), 0.33884525299072266)


def test_margin_confidence(proba):
    """
    A test case scenario for an artificial input:

    mu_margin_confidence:
     - proba: tensor([0.8305774  0.16942263])
     - output: 0.33884525299072266

    """
    assert np.isclose(uncertainty_sampling.margin_confidence(proba), 0.33884525299072266)


def test_ratio_confidence(proba):
    """
    A test case scenario for ratio confidence
    mu_ratio_confidence:
     - proba: tensor([0.8305774  0.16942263])
     - output: 0.20398175716400146
    """
    assert np.isclose(uncertainty_sampling.ratio_confidence(proba), 0.20398175716400146)



def test_entropy_confidence(proba):
    """
    A test case scenario for entropy-based confidence

    mu_entropy_based:
     - proba: tensor([0.8305774  0.16942263])
     - output: 0.6563822627067566

    """
    np.isclose(uncertainty_sampling.entropy_confidence(proba), 0.6563822627067566)



if __name__ == '__main__':
    import numpy as np
    test_input = np.array([0.8305774 , 0.16942263], dtype=np.float32)
    test_least_confidence(test_input)
    test_margin_confidence(test_input)
    test_ratio_confidence(test_input)
    test_entropy_confidence(test_input)
    print("Uncertainty sampling tests passed!")