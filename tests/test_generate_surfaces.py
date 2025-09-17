from ipykernel.pickleutil import istype
from pandas.tests.reshape.test_melt import value_name
from toy_functions import ResponseFunction
import pytest
import torch



class TestBasicResponseFunction:


    def test_function_callable(self) -> None:
        func = lambda x: x.sum(dim=-1)
        response = ResponseFunction(func, n_dim=2)
        assert response.function == func
        assert response.n_dim == 2



    def test_invalid_n_dim(self) -> None:
        resp = lambda n_dim: ResponseFunction(lambda x: x, n_dim=n_dim)

        with pytest.raises(TypeError):
            resp("two")

        with pytest.raises(TypeError):
            resp(2.5)

        with pytest.raises(ValueError):
            resp(0)

        with pytest.raises(ValueError):
            resp(-1)


    def test_valid_function(self) -> None:
        func = lambda x: x.prod(dim=-1)
        response = ResponseFunction(func, n_dim=3)
        assert callable(response.function)

    def test_raise_invalid_function(self) -> None:
        with pytest.raises(TypeError):
            ResponseFunction("not a function", n_dim=2)

    def test_evaluate(self) -> None:
        func = lambda x: x.sum(dim=-1)
        response = ResponseFunction(func, n_dim=2)
        coord = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = response.evaluate(coord)
        expected = torch.tensor([3.0, 7.0])
        assert torch.allclose(result, expected)

    def test_evaluate_invalid_coord(self) -> None:
        func = lambda x: x.sum(dim=-1)
        response = ResponseFunction(func, n_dim=2)
        with pytest.raises(ValueError):
            response.evaluate(torch.tensor([1.0, 2.0, 3.0]))

