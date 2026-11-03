Python
======

.. module:: numerical

Everything is importable from ``numerical``; the FFT lives in ``numerical.fft``, and ``numpy.linalg``'s names in
``numerical.linalg`` too.

Tensors
-------

.. autoclass:: numerical.Tensor
   :members:

.. autofunction:: numerical.asarray
.. autofunction:: numerical.zeros
.. autofunction:: numerical.ones
.. autofunction:: numerical.full
.. autofunction:: numerical.eye
.. autofunction:: numerical.tri
.. autofunction:: numerical.meshgrid
.. autofunction:: numerical.arange
.. autofunction:: numerical.linspace
.. autofunction:: numerical.logspace
.. autofunction:: numerical.geomspace
.. autofunction:: numerical.identity
.. autofunction:: numerical.vander
.. autofunction:: numerical.zeros_like
.. autofunction:: numerical.ones_like
.. autofunction:: numerical.empty_like
.. autofunction:: numerical.full_like
.. autofunction:: numerical.copy
.. autofunction:: numerical.astype
.. autoclass:: numerical.finfo
.. autoclass:: numerical.iinfo
.. autodata:: numerical.pi
.. autodata:: numerical.e
.. autodata:: numerical.inf
.. autodata:: numerical.nan
.. autodata:: numerical.newaxis

Shapes and joins
----------------

.. autofunction:: numerical.reshape
.. autofunction:: numerical.ravel
.. autofunction:: numerical.transpose
.. autofunction:: numerical.permute_dims
.. autofunction:: numerical.matrix_transpose
.. autofunction:: numerical.squeeze
.. autofunction:: numerical.expand_dims
.. autofunction:: numerical.atleast_1d
.. autofunction:: numerical.atleast_2d
.. autofunction:: numerical.atleast_3d
.. autofunction:: numerical.shape
.. autofunction:: numerical.ndim
.. autofunction:: numerical.size
.. autofunction:: numerical.flip
.. autofunction:: numerical.flipud
.. autofunction:: numerical.fliplr
.. autofunction:: numerical.swapaxes
.. autofunction:: numerical.moveaxis
.. autofunction:: numerical.concatenate
.. autofunction:: numerical.concat
.. autofunction:: numerical.stack
.. autofunction:: numerical.vstack
.. autofunction:: numerical.hstack
.. autofunction:: numerical.column_stack
.. autofunction:: numerical.dstack
.. autofunction:: numerical.block
.. autofunction:: numerical.pad
.. autofunction:: numerical.trim_zeros
.. autofunction:: numerical.append
.. autofunction:: numerical.insert
.. autofunction:: numerical.delete
.. autofunction:: numerical.split
.. autofunction:: numerical.array_split
.. autofunction:: numerical.hsplit
.. autofunction:: numerical.vsplit
.. autofunction:: numerical.dsplit
.. autofunction:: numerical.unstack
.. autofunction:: numerical.tile
.. autofunction:: numerical.repeat
.. autofunction:: numerical.roll
.. autofunction:: numerical.rot90
.. autofunction:: numerical.take
.. autofunction:: numerical.take_along_axis
.. autofunction:: numerical.broadcast_to
.. autofunction:: numerical.broadcast_arrays
.. autofunction:: numerical.broadcast_shapes
.. autofunction:: numerical.tril
.. autofunction:: numerical.triu
.. autofunction:: numerical.diag
.. autofunction:: numerical.diagonal

Element-wise arithmetic
-----------------------

.. autofunction:: numerical.abs
.. autofunction:: numerical.absolute
.. autofunction:: numerical.negative
.. autofunction:: numerical.positive
.. autofunction:: numerical.square
.. autofunction:: numerical.add
.. autofunction:: numerical.subtract
.. autofunction:: numerical.multiply
.. autofunction:: numerical.reciprocal
.. autofunction:: numerical.fabs
.. autofunction:: numerical.divide
.. autofunction:: numerical.true_divide
.. autofunction:: numerical.floor_divide
.. autofunction:: numerical.remainder
.. autofunction:: numerical.mod
.. autofunction:: numerical.divmod
.. autofunction:: numerical.power
.. autofunction:: numerical.maximum
.. autofunction:: numerical.minimum
.. autofunction:: numerical.fmax
.. autofunction:: numerical.fmin
.. autofunction:: numerical.fmod
.. autofunction:: numerical.float_power
.. autofunction:: numerical.copysign
.. autofunction:: numerical.nextafter
.. autofunction:: numerical.clip
.. autofunction:: numerical.real
.. autofunction:: numerical.imag
.. autofunction:: numerical.conj
.. autofunction:: numerical.conjugate
.. autofunction:: numerical.angle
.. autofunction:: numerical.sign
.. autofunction:: numerical.round
.. autofunction:: numerical.around
.. autofunction:: numerical.floor
.. autofunction:: numerical.ceil
.. autofunction:: numerical.trunc
.. autofunction:: numerical.rint

Element-wise functions
----------------------

.. autofunction:: numerical.exp
.. autofunction:: numerical.log
.. autofunction:: numerical.sqrt
.. autofunction:: numerical.sin
.. autofunction:: numerical.cos
.. autofunction:: numerical.tan
.. autofunction:: numerical.tanh
.. autofunction:: numerical.arcsin
.. autofunction:: numerical.arccos
.. autofunction:: numerical.arctan
.. autofunction:: numerical.sinh
.. autofunction:: numerical.cosh
.. autofunction:: numerical.arcsinh
.. autofunction:: numerical.arccosh
.. autofunction:: numerical.arctanh
.. autofunction:: numerical.acos
.. autofunction:: numerical.asin
.. autofunction:: numerical.atan
.. autofunction:: numerical.acosh
.. autofunction:: numerical.asinh
.. autofunction:: numerical.atanh
.. autofunction:: numerical.pow
.. autofunction:: numerical.log2
.. autofunction:: numerical.log10
.. autofunction:: numerical.log1p
.. autofunction:: numerical.expm1
.. autofunction:: numerical.cbrt
.. autofunction:: numerical.deg2rad
.. autofunction:: numerical.rad2deg
.. autofunction:: numerical.radians
.. autofunction:: numerical.degrees
.. autofunction:: numerical.hypot
.. autofunction:: numerical.arctan2
.. autofunction:: numerical.atan2
.. autofunction:: numerical.logaddexp
.. autofunction:: numerical.logaddexp2
.. autofunction:: numerical.heaviside
.. autofunction:: numerical.unwrap
.. autofunction:: numerical.nan_to_num

Comparisons and logic
---------------------

.. autofunction:: numerical.equal
.. autofunction:: numerical.not_equal
.. autofunction:: numerical.less
.. autofunction:: numerical.less_equal
.. autofunction:: numerical.greater
.. autofunction:: numerical.greater_equal
.. autofunction:: numerical.logical_and
.. autofunction:: numerical.logical_or
.. autofunction:: numerical.logical_xor
.. autofunction:: numerical.logical_not
.. autofunction:: numerical.isnan
.. autofunction:: numerical.isinf
.. autofunction:: numerical.isfinite
.. autofunction:: numerical.isposinf
.. autofunction:: numerical.isneginf
.. autofunction:: numerical.signbit
.. autofunction:: numerical.iscomplex
.. autofunction:: numerical.isreal
.. autofunction:: numerical.iscomplexobj
.. autofunction:: numerical.isrealobj
.. autofunction:: numerical.isscalar
.. autofunction:: numerical.isclose
.. autofunction:: numerical.allclose
.. autofunction:: numerical.array_equal
.. autofunction:: numerical.where
.. autofunction:: numerical.select
.. autofunction:: numerical.any
.. autofunction:: numerical.all

Reductions, scans and sorting
-----------------------------

.. autofunction:: numerical.sum
.. autofunction:: numerical.mean
.. autofunction:: numerical.max
.. autofunction:: numerical.min
.. autofunction:: numerical.amax
.. autofunction:: numerical.amin
.. autofunction:: numerical.prod
.. autofunction:: numerical.cumsum
.. autofunction:: numerical.cumprod
.. autofunction:: numerical.cumulative_sum
.. autofunction:: numerical.cumulative_prod
.. autofunction:: numerical.ediff1d
.. autofunction:: numerical.ptp
.. autofunction:: numerical.var
.. autofunction:: numerical.std
.. autofunction:: numerical.nansum
.. autofunction:: numerical.nanprod
.. autofunction:: numerical.nancumsum
.. autofunction:: numerical.nancumprod
.. autofunction:: numerical.nanmax
.. autofunction:: numerical.nanmin
.. autofunction:: numerical.nanargmax
.. autofunction:: numerical.nanargmin
.. autofunction:: numerical.nanmean
.. autofunction:: numerical.nanvar
.. autofunction:: numerical.nanstd
.. autofunction:: numerical.median
.. autofunction:: numerical.nanmedian
.. autofunction:: numerical.quantile
.. autofunction:: numerical.percentile
.. autofunction:: numerical.nanquantile
.. autofunction:: numerical.nanpercentile
.. autofunction:: numerical.average
.. autofunction:: numerical.cov
.. autofunction:: numerical.corrcoef
.. autofunction:: numerical.sort
.. autofunction:: numerical.unique
.. autofunction:: numerical.unique_values
.. autofunction:: numerical.unique_counts
.. autofunction:: numerical.unique_inverse
.. autofunction:: numerical.unique_all
.. autofunction:: numerical.intersect1d
.. autofunction:: numerical.union1d
.. autofunction:: numerical.setdiff1d
.. autofunction:: numerical.setxor1d
.. autofunction:: numerical.isin
.. autofunction:: numerical.digitize
.. autofunction:: numerical.diff
.. autofunction:: numerical.argmax
.. autofunction:: numerical.argmin
.. autofunction:: numerical.argsort
.. autofunction:: numerical.nonzero
.. autofunction:: numerical.tril_indices
.. autofunction:: numerical.triu_indices
.. autofunction:: numerical.tril_indices_from
.. autofunction:: numerical.triu_indices_from
.. autofunction:: numerical.diag_indices
.. autofunction:: numerical.diag_indices_from
.. autofunction:: numerical.flatnonzero
.. autofunction:: numerical.argwhere
.. autofunction:: numerical.count_nonzero
.. autofunction:: numerical.searchsorted
.. autofunction:: numerical.trace

Products
--------

.. autofunction:: numerical.matmul
.. autofunction:: numerical.matvec
.. autofunction:: numerical.vecmat
.. autofunction:: numerical.dot
.. autofunction:: numerical.inner
.. autofunction:: numerical.vdot
.. autofunction:: numerical.outer
.. autofunction:: numerical.vecdot
.. autofunction:: numerical.tensordot
.. autofunction:: numerical.einsum
.. autofunction:: numerical.multi_dot
.. autofunction:: numerical.cross
.. autofunction:: numerical.kron

numpy.linalg
------------

The names and signatures of NumPy 2's ``numpy.linalg``; those SciPy shares take its arguments too.

.. autofunction:: numerical.solve
.. autofunction:: numerical.determinant
.. autofunction:: numerical.det
.. autofunction:: numerical.slogdet
.. autofunction:: numerical.inverse
.. autofunction:: numerical.inv
.. autofunction:: numerical.lstsq
.. autofunction:: numerical.pinv
.. autofunction:: numerical.matrix_rank
.. autofunction:: numerical.cholesky
.. autofunction:: numerical.qr
.. autofunction:: numerical.eig
.. autofunction:: numerical.eigvals
.. autofunction:: numerical.eigh
.. autofunction:: numerical.eigvalsh
.. autofunction:: numerical.svd
.. autofunction:: numerical.svdvals
.. autofunction:: numerical.norm
.. autofunction:: numerical.vector_norm
.. autofunction:: numerical.matrix_norm
.. autofunction:: numerical.cond
.. autofunction:: numerical.matrix_power
.. autofunction:: numerical.tensorsolve
.. autofunction:: numerical.tensorinv

scipy.linalg: solvers and factors
---------------------------------

.. autofunction:: numerical.solve_triangular
.. autofunction:: numerical.solve_banded
.. autofunction:: numerical.solveh_banded
.. autofunction:: numerical.solve_toeplitz
.. autofunction:: numerical.matmul_toeplitz
.. autofunction:: numerical.solve_circulant
.. autofunction:: numerical.lu
.. autofunction:: numerical.lu_factor
.. autofunction:: numerical.lu_solve
.. autoclass:: numerical.LUFactor
   :members:

.. autofunction:: numerical.cho_factor
.. autofunction:: numerical.cho_solve
.. autoclass:: numerical.CholeskyFactor
   :members:

.. autofunction:: numerical.cholesky_banded
.. autofunction:: numerical.cho_solve_banded
.. autofunction:: numerical.ldl

scipy.linalg: decompositions
----------------------------

.. autofunction:: numerical.schur
.. autofunction:: numerical.rsf2csf
.. autofunction:: numerical.qz
.. autofunction:: numerical.ordqz
.. autofunction:: numerical.hessenberg
.. autofunction:: numerical.rq
.. autofunction:: numerical.polar
.. autofunction:: numerical.cdf2rdf
.. autofunction:: numerical.matrix_balance
.. autofunction:: numerical.eigh_tridiagonal
.. autofunction:: numerical.eigvalsh_tridiagonal
.. autofunction:: numerical.eig_banded
.. autofunction:: numerical.eigvals_banded

scipy.linalg: matrix functions
------------------------------

.. autofunction:: numerical.expm
.. autofunction:: numerical.expm_multiply
.. autofunction:: numerical.expm_frechet
.. autofunction:: numerical.expm_cond
.. autofunction:: numerical.logm
.. autofunction:: numerical.sqrtm
.. autofunction:: numerical.fractional_matrix_power
.. autofunction:: numerical.funm
.. autofunction:: numerical.cosm
.. autofunction:: numerical.sinm
.. autofunction:: numerical.tanm
.. autofunction:: numerical.coshm
.. autofunction:: numerical.sinhm
.. autofunction:: numerical.tanhm
.. autofunction:: numerical.signm

scipy.linalg: matrix equations
------------------------------

.. autofunction:: numerical.solve_sylvester
.. autofunction:: numerical.solve_continuous_lyapunov
.. autofunction:: numerical.solve_lyapunov
.. autofunction:: numerical.solve_discrete_lyapunov
.. autofunction:: numerical.solve_continuous_are
.. autofunction:: numerical.solve_discrete_are

scipy.linalg: special matrices
------------------------------

.. autofunction:: numerical.toeplitz
.. autofunction:: numerical.hankel
.. autofunction:: numerical.circulant
.. autofunction:: numerical.block_diag
.. autofunction:: numerical.hilbert
.. autofunction:: numerical.invhilbert
.. autofunction:: numerical.pascal
.. autofunction:: numerical.invpascal
.. autofunction:: numerical.hadamard
.. autofunction:: numerical.helmert
.. autofunction:: numerical.leslie
.. autofunction:: numerical.companion
.. autofunction:: numerical.fiedler
.. autofunction:: numerical.fiedler_companion
.. autofunction:: numerical.convolution_matrix
.. autofunction:: numerical.dft
.. autofunction:: numerical.khatri_rao
.. autofunction:: numerical.diagsvd

scipy.linalg: subspaces and tests
---------------------------------

.. autofunction:: numerical.null_space
.. autofunction:: numerical.orth
.. autofunction:: numerical.subspace_angles
.. autofunction:: numerical.orthogonal_procrustes
.. autofunction:: numerical.pinvh
.. autofunction:: numerical.bandwidth
.. autofunction:: numerical.issymmetric
.. autofunction:: numerical.ishermitian

Errors and warnings
-------------------

.. autoexception:: numerical.error

.. autoexception:: numerical.LinAlgError

.. autoexception:: numerical.LinAlgWarning

.. autoexception:: numerical.LogmRankWarning

.. autoexception:: numerical.LogmExactlySingularWarning

.. autoexception:: numerical.LogmNearlySingularWarning


Threads, kernels and memory
---------------------------

.. autofunction:: numerical.set_num_threads
.. autofunction:: numerical.get_num_threads
.. autofunction:: numerical.gemm_path
.. autofunction:: numerical.limit_gemm_path
.. autofunction:: numerical.release_memory
.. autofunction:: numerical.get_include
.. autofunction:: numerical.get_config
.. autofunction:: numerical.blas_library

numerical.linalg
----------------

``numpy.linalg``'s names, each the function of that name at the root (the same object), but for ``cross`` and
``outer``, which take ``numpy.linalg``'s operands alone: 3-vectors, and vectors.

.. module:: numerical.linalg

.. autofunction:: numerical.linalg.cross
.. autofunction:: numerical.linalg.outer

numerical.fft
-------------

``numpy.fft``'s names, and ``scipy.fft``'s discrete cosine and sine transforms.

.. module:: numerical.fft

.. autofunction:: numerical.fft.fft
.. autofunction:: numerical.fft.ifft
.. autofunction:: numerical.fft.fft2
.. autofunction:: numerical.fft.ifft2
.. autofunction:: numerical.fft.fftn
.. autofunction:: numerical.fft.ifftn
.. autofunction:: numerical.fft.rfft
.. autofunction:: numerical.fft.irfft
.. autofunction:: numerical.fft.rfft2
.. autofunction:: numerical.fft.irfft2
.. autofunction:: numerical.fft.rfftn
.. autofunction:: numerical.fft.irfftn
.. autofunction:: numerical.fft.hfft
.. autofunction:: numerical.fft.ihfft
.. autofunction:: numerical.fft.fftfreq
.. autofunction:: numerical.fft.rfftfreq
.. autofunction:: numerical.fft.fftshift
.. autofunction:: numerical.fft.ifftshift
.. autofunction:: numerical.fft.dct
.. autofunction:: numerical.fft.idct
.. autofunction:: numerical.fft.dctn
.. autofunction:: numerical.fft.idctn
.. autofunction:: numerical.fft.dst
.. autofunction:: numerical.fft.idst
.. autofunction:: numerical.fft.dstn
.. autofunction:: numerical.fft.idstn
