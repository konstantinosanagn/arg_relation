from typing import Any, Literal
from typing_extensions import Self

from sympy.core.numbers import Integer, Rational
from sympy.liealgebras.cartan_type import Standard_Cartan
from sympy.matrices import Matrix

class TypeF(Standard_Cartan):
    def __new__(cls, n) -> Self: ...
    def dimension(self) -> Literal[4]: ...
    def basic_root(self, i, j): ...
    def simple_root(self, i) -> list[int] | list[Rational | Any | Integer] | None: ...
    def positive_roots(self) -> dict[Any, Any]: ...
    def roots(self) -> Literal[48]: ...
    def cartan_matrix(self) -> Matrix: ...
    def basis(self) -> Literal[52]: ...
    def dynkin_diagram(self) -> str: ...