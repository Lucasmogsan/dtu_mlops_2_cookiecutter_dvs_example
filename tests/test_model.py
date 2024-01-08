from dtu_mlops_cookiecutter_example.models.model import myawesomemodel as model
from tests import _PATH_DATA, _PROJECT_ROOT, _TEST_ROOT
import torch
import pytest

@pytest.mark.parametrize("batch_size", [1, 8, 16])
def test_model(batch_size):
    # Test of model (with random input)
    test_input = torch.randn(batch_size, 2, 28, 28)
    test_output = model(test_input)
    assert test_output.shape == torch.Size([batch_size, 10])


if __name__ == "__main__":
    test_model()