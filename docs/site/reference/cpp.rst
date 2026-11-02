C++
===

``#include <numerical/numerical.hpp>`` brings in everything below, in namespace ``numerical``. The free functions
(``solve``, ``inverse``, ``determinant``, ``einsum``, ``expm``, ``fft`` and the rest) are listed, with every overload,
in the `Doxygen reference <../api/index.html>`_.

Containers
----------

.. doxygenclass:: numerical::matrix
   :members:

.. doxygenclass:: numerical::vector
   :members:

.. doxygenclass:: numerical::tensor
   :members:

Decompositions
--------------

.. doxygenclass:: numerical::lu_decomposition
   :members:

.. doxygenclass:: numerical::cholesky_decomposition
   :members:

.. doxygenclass:: numerical::ldl_decomposition
   :members:

.. doxygenclass:: numerical::qr_decomposition
   :members:

.. doxygenclass:: numerical::pivoted_qr_decomposition
   :members:

.. doxygenclass:: numerical::svd_decomposition
   :members:

.. doxygenclass:: numerical::symmetric_eigen
   :members:

.. doxygenclass:: numerical::hermitian_eigen
   :members:

.. doxygenclass:: numerical::eigen_decomposition
   :members:

.. doxygenclass:: numerical::schur_decomposition
   :members:

.. doxygenclass:: numerical::generalized_schur
   :members:

.. doxygenclass:: numerical::real_generalized_schur
   :members:

Exceptions
----------

.. doxygenclass:: numerical::dimension_error
.. doxygenclass:: numerical::singular_matrix_error
.. doxygenclass:: numerical::not_positive_definite_error
.. doxygenclass:: numerical::non_finite_error
.. doxygenclass:: numerical::convergence_error
.. doxygenclass:: numerical::einsum_error
