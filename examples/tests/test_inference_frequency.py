import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.evaluate_pi0_real import _inference_publish_period_steps as pi0_period
from examples.evaluate_policy_real import _inference_publish_period_steps as dsrl_period


def test_inference_frequency_default_publishes_every_control_step():
    assert pi0_period(10, None) == 1
    assert dsrl_period(10, None) == 1


def test_inference_frequency_uses_integer_control_step_period():
    assert pi0_period(10, 5.0) == 2
    assert dsrl_period(10, 5.0) == 2


def test_inference_frequency_never_exceeds_requested_rate():
    period = pi0_period(10, 3.0)

    assert period == 4
    assert 10 / period <= 3.0
    assert dsrl_period(10, 3.0) == period
