from typing import Any

from numerical import LinAlgError as LinAlgError
from numerical import cholesky as cholesky
from numerical import cond as cond
from numerical import det as det
from numerical import diagonal as diagonal
from numerical import eig as eig
from numerical import eigh as eigh
from numerical import eigvals as eigvals
from numerical import eigvalsh as eigvalsh
from numerical import inv as inv
from numerical import lstsq as lstsq
from numerical import matmul as matmul
from numerical import matrix_norm as matrix_norm
from numerical import matrix_power as matrix_power
from numerical import matrix_rank as matrix_rank
from numerical import matrix_transpose as matrix_transpose
from numerical import multi_dot as multi_dot
from numerical import norm as norm
from numerical import pinv as pinv
from numerical import qr as qr
from numerical import slogdet as slogdet
from numerical import solve as solve
from numerical import svd as svd
from numerical import svdvals as svdvals
from numerical import tensordot as tensordot
from numerical import tensorinv as tensorinv
from numerical import tensorsolve as tensorsolve
from numerical import trace as trace
from numerical import vecdot as vecdot
from numerical import vector_norm as vector_norm
from numerical._numerical import Tensor

def cross(x1: Any, x2: Any, /, *, axis: int = -1) -> Tensor: ...
def outer(x1: Any, x2: Any, /) -> Tensor: ...

__all__ = ["matrix_power", "solve", "tensorsolve", "tensorinv", "inv", "cholesky", "eigvals", "eigvalsh", "pinv",
           "slogdet", "det", "svd", "svdvals", "eig", "eigh", "lstsq", "norm", "qr", "cond", "matrix_rank",
           "LinAlgError", "multi_dot", "trace", "diagonal", "cross", "outer", "tensordot", "matmul", "matrix_transpose",
           "matrix_norm", "vector_norm", "vecdot"]
