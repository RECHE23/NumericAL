// NumericAL — Python extension module: the element-wise arithmetic and functions, the reductions and the norms (their
// helpers in binding.hpp).

#include "binding.hpp"

namespace numerical_binding {

// _binary(op, a, b) -> Tensor: a op b element by element, broadcast as NumPy's; op one of "+-*/", or "^"
// for a ** b.
PyObject* module_binary(PyObject* /*self*/, PyObject* args)
{
    const char* op = nullptr;
    PyObject*   x  = nullptr;
    PyObject*   y  = nullptr;
    if (PyArg_ParseTuple(args, "sOO", &op, &x, &y) == 0) {
        return nullptr;
    }
    if (std::string_view {op}.size() != 1 || std::string_view {"+-*/^"}.find(op[0]) == std::string_view::npos) {
        PyErr_SetString(PyExc_ValueError, "_binary: op is one of + - * / ^ (power)");
        return nullptr;
    }
    const any_tensor* a = nullptr;
    const any_tensor* b = nullptr;
    if (!same_dtype_pair(x, y, a, b)) {
        return nullptr;
    }
    try {
        return visit_numeric([&](const auto& x) { return binary_on(op[0], x, std::get<std::decay_t<decltype(x)>>(*b)); }, *a);
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _power_shortcut(kind, a) -> Tensor: kind "2" (square), "r" (reciprocal) or "h" (square root) of a complex tensor.
PyObject* module_power_shortcut(PyObject* /*self*/, PyObject* args)
{
    const char* kind = nullptr;
    PyObject*   x    = nullptr;
    if (PyArg_ParseTuple(args, "sO", &kind, &x) == 0) {
        return nullptr;
    }
    if (std::string_view {kind}.size() != 1 || std::string_view {"2rh"}.find(kind[0]) == std::string_view::npos) {
        PyErr_SetString(PyExc_ValueError, "_power_shortcut: kind is 2, r or h");
        return nullptr;
    }
    const any_tensor* a = numeric_tensor(x);
    if (a == nullptr) {
        return nullptr;
    }
    try {
        return visit_numeric([&](const auto& t) { return power_shortcut_on(kind[0], t); }, *a);
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _negate(a) -> Tensor.
PyObject* module_negate(PyObject* /*self*/, PyObject* args)
{
    PyObject* x = nullptr;
    if (PyArg_ParseTuple(args, "O", &x) == 0) {
        return nullptr;
    }
    const any_tensor* a = numeric_tensor(x);
    if (a == nullptr) {
        return nullptr;
    }
    try { // an allocation the system refuses is a Python exception, not an abort
        return visit_numeric([](const auto& t) { return wrap(any_tensor {-t}); }, *a);
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _equal(a, b) -> bool: whether two tensors of one dtype and one size hold equal elements, in order (a NaN equals
// nothing). The caller checks the shapes and promotes the dtypes.
PyObject* module_equal(PyObject* /*self*/, PyObject* args)
{
    PyObject* x = nullptr;
    PyObject* y = nullptr;
    if (PyArg_ParseTuple(args, "OO", &x, &y) == 0) {
        return nullptr;
    }
    const any_tensor* a = as_tensor(x);
    const any_tensor* b = a == nullptr ? nullptr : as_tensor(y);
    if (b == nullptr) {
        return nullptr;
    }
    if (a->index() != b->index() || count_of(*a) != count_of(*b)) {
        PyErr_SetString(sciforge_module_error(), "_equal: the tensors differ in dtype or size");
        return nullptr;
    }
    const bool equal {std::visit([&](const auto& x) {
                                     const auto& y {std::get<std::decay_t<decltype(x)>>(*b)};
                                     return std::equal(x.data().begin(), x.data().end(), y.data().begin());
                                 }, *a)};
    return Py_NewRef(equal ? Py_True : Py_False);
}

// _compare(op, a, b) -> bool Tensor.
PyObject* module_compare(PyObject* /*self*/, PyObject* args)
{
    const char* op = nullptr;
    PyObject*   x  = nullptr;
    PyObject*   y  = nullptr;
    if (PyArg_ParseTuple(args, "sOO", &op, &x, &y) == 0) {
        return nullptr;
    }
    static constexpr std::array<std::string_view, 6> names {"<", "<=", "==", "!=", ">", ">="};
    const auto found {std::find(names.begin(), names.end(), std::string_view {op})};
    if (found == names.end()) {
        PyErr_SetString(PyExc_ValueError, "_compare: op is one of < <= == != > >=");
        return nullptr;
    }
    const auto        which {static_cast<std::size_t>(found - names.begin())};
    const any_tensor* a = as_tensor(x);
    const any_tensor* b = a == nullptr ? nullptr : as_tensor(y);
    if (b == nullptr) {
        return nullptr;
    }
    if (a->index() != b->index()) {
        PyErr_SetString(sciforge_module_error(), "_compare: the operands differ in dtype");
        return nullptr;
    }
    if (is_complex(*a) && which != 2 && which != 3) {
        PyErr_SetString(PyExc_TypeError, "a complex tensor has no order: compare with == or !=, or compare the real parts");
        return nullptr;
    }
    try {
        std::optional<bool_tensor> result;
        {
            const sb::scoped_gil_release unlocked {releases_gil(std::max(count_of(*a), count_of(*b)))};
            std::visit([&](const auto& p) {
                using tensor_type = std::decay_t<decltype(p)>;
                using T           = element_of_t<tensor_type>;
                const T* const u {p.data().data()};
                const T* const v {std::get<tensor_type>(*b).data().data()};
                const std::array shapes {p.shape(), shape_of(*b)};
                const auto       compared {[&](const auto& relation) {
                    result.emplace(broadcast_elements<boolean>(shapes, [&](const std::array<std::size_t, 2>& at) {
                        if constexpr (std::is_same_v<T, boolean>) {
                            return boolean {relation(u[at[0]].value, v[at[1]].value)};
                        }
                        else {
                            return boolean {relation(u[at[0]], v[at[1]])};
                        }
                    }));
                }};
                switch (which) {
                    case 2: compared(std::equal_to<> {}); break;
                    case 3: compared(std::not_equal_to<> {}); break;
                    default:
                        if constexpr (!complex_scalar<T>) {
                            switch (which) {
                                case 0: compared(std::less<> {}); break;
                                case 1: compared(std::less_equal<> {}); break;
                                case 4: compared(std::greater<> {}); break;
                                default: compared(std::greater_equal<> {}); break;
                            }
                        }
                        break;
                }
            }, *a);
        }
        return wrap(any_tensor {std::move(*result)});
    }
    catch (...) {
        return set_cpp_error();
    }
}

// The bool tensor of a Tensor argument, or nullptr with TypeError set for another dtype (numerical.error for
// anything else, as as_tensor).
inline const bool_tensor* bool_operand(PyObject* obj, const char* name)
{
    const any_tensor* v = as_tensor(obj);
    if (v == nullptr) {
        return nullptr;
    }
    if (!is_bool(*v)) {
        PyErr_Format(PyExc_TypeError, "%s takes bool tensors, not %s: compare first, as t != 0", name, dtype_name(*v));
        return nullptr;
    }
    return &std::get<bool_tensor>(*v);
}

// _logical(op, a, b) -> bool Tensor.
PyObject* module_logical(PyObject* /*self*/, PyObject* args)
{
    const char* op = nullptr;
    PyObject*   x  = nullptr;
    PyObject*   y  = nullptr;
    if (PyArg_ParseTuple(args, "sOO", &op, &x, &y) == 0) {
        return nullptr;
    }
    const std::string_view name {op};
    if (name != "&" && name != "|" && name != "^") {
        PyErr_SetString(PyExc_ValueError, "_logical: op is one of & | ^");
        return nullptr;
    }
    const bool_tensor* a = bool_operand(x, op);
    const bool_tensor* b = a == nullptr ? nullptr : bool_operand(y, op);
    if (b == nullptr) {
        return nullptr;
    }
    try {
        const boolean* const u {a->data().data()};
        const boolean* const v {b->data().data()};
        const std::array     shapes {a->shape(), b->shape()};
        std::optional<bool_tensor> result;
        {
            const sb::scoped_gil_release unlocked {releases_gil(std::max(a->size(), b->size()))};
            result.emplace(broadcast_elements<boolean>(shapes, [&](const std::array<std::size_t, 2>& at) {
                const bool p {u[at[0]].value};
                const bool q {v[at[1]].value};
                return boolean {name == "&" ? (p && q) : name == "|" ? (p || q) : (p != q)};
            }));
        }
        return wrap(any_tensor {std::move(*result)});
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _invert(a) -> bool Tensor.
PyObject* module_invert(PyObject* /*self*/, PyObject* args)
{
    PyObject* x = nullptr;
    if (PyArg_ParseTuple(args, "O", &x) == 0) {
        return nullptr;
    }
    const bool_tensor* a = bool_operand(x, "~");
    if (a == nullptr) {
        return nullptr;
    }
    try {
        return wrap(any_tensor {numerical::detail::parallel_map(*a, [](const boolean& p) { return boolean {!p.value}; })});
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _where(condition, x, y) -> Tensor.
PyObject* module_where(PyObject* /*self*/, PyObject* args)
{
    PyObject* c_obj = nullptr;
    PyObject* x_obj = nullptr;
    PyObject* y_obj = nullptr;
    if (PyArg_ParseTuple(args, "OOO", &c_obj, &x_obj, &y_obj) == 0) {
        return nullptr;
    }
    const bool_tensor* c = bool_operand(c_obj, "where's condition");
    const any_tensor*  x = c == nullptr ? nullptr : as_tensor(x_obj);
    const any_tensor*  y = x == nullptr ? nullptr : as_tensor(y_obj);
    if (y == nullptr) {
        return nullptr;
    }
    if (x->index() != y->index()) {
        PyErr_SetString(sciforge_module_error(), "_where: x and y differ in dtype");
        return nullptr;
    }
    try {
        std::optional<any_tensor> result;
        {
            const sb::scoped_gil_release unlocked {releases_gil(std::max({c->size(), count_of(*x), count_of(*y)}))};
            std::visit([&](const auto& p) {
                using tensor_type = std::decay_t<decltype(p)>;
                using T           = element_of_t<tensor_type>;
                const boolean* const holds {c->data().data()};
                const T* const       u {p.data().data()};
                const T* const       v {std::get<tensor_type>(*y).data().data()};
                const std::array     shapes {c->shape(), p.shape(), shape_of(*y)};
                result.emplace(broadcast_elements<T>(shapes, [&](const std::array<std::size_t, 3>& at) {
                    return holds[at[0]].value ? u[at[1]] : v[at[2]];
                }));
            }, *x);
        }
        return wrap(std::move(*result));
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _classify(name, a) -> bool Tensor.
PyObject* module_classify(PyObject* /*self*/, PyObject* args)
{
    const char* op = nullptr;
    PyObject*   x  = nullptr;
    if (PyArg_ParseTuple(args, "sO", &op, &x) == 0) {
        return nullptr;
    }
    const std::string_view name {op};
    if (name != "isnan" && name != "isinf" && name != "isfinite") {
        PyErr_SetString(PyExc_ValueError, "_classify: name is isnan, isinf or isfinite");
        return nullptr;
    }
    const any_tensor* a = as_tensor(x);
    if (a == nullptr) {
        return nullptr;
    }
    try {
        return wrap(any_tensor {std::visit([&](const auto& t) {
            using T = element_of_t<std::decay_t<decltype(t)>>;
            return numerical::detail::parallel_map(t, [name](const T& value) {
                if constexpr (std::is_same_v<T, boolean>) {
                    return boolean {name == "isfinite"};
                }
                else {
                    const auto parts {[&](const auto& test) {
                        if constexpr (complex_scalar<T>) {
                            return test(value.real()) || test(value.imag());
                        }
                        else {
                            return test(value);
                        }
                    }};
                    if (name == "isnan") {
                        return boolean {parts([](auto part) { return std::isnan(part); })};
                    }
                    if (name == "isinf") {
                        return boolean {parts([](auto part) { return std::isinf(part); })};
                    }
                    return boolean {!parts([](auto part) { return !std::isfinite(part); })};
                }
            });
        }, *a)});
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _unary(name, a) -> Tensor: the function `name` of every element.
PyObject* module_unary(PyObject* /*self*/, PyObject* args)
{
    const char* name = nullptr;
    PyObject*   x    = nullptr;
    if (PyArg_ParseTuple(args, "sO", &name, &x) == 0) {
        return nullptr;
    }
    const any_tensor* a = numeric_tensor(x);
    if (a == nullptr) {
        return nullptr;
    }
    try { // an allocation the system refuses is a Python exception, not an abort
        return visit_numeric([&](const auto& t) { return unary_on(name, t); }, *a);
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _reduce(op, a, axis) -> number or Tensor: op one of "sum", "mean", "max", "min"; axis -1 for every
// element (a number), else a non-negative axis (a Tensor without it). max and min take float64 only.
PyObject* module_reduce(PyObject* /*self*/, PyObject* args)
{
    const char* op   = nullptr;
    PyObject*   x    = nullptr;
    Py_ssize_t  axis = -1;
    if (PyArg_ParseTuple(args, "sOn", &op, &x, &axis) == 0) {
        return nullptr;
    }
    const std::string_view name {op};
    const any_tensor*      a = as_tensor(x);
    if (a == nullptr) {
        return nullptr;
    }
    const bool truth {name == "any" || name == "all"};
    try {
        if (truth || is_bool(*a)) {
            // any and all of a number's truth (NumPy's: nonzero), or a mask's: the sum and the mean count its trues, its
            // maximum is any and its minimum all, of which an empty one has none (as of numbers)
            if (!is_bool(*a)) {
                return truth_of(name == "any", visit_numeric([](const auto& t) { return converted<boolean>(t); }, *a), axis);
            }
            const bool_tensor& mask {std::get<bool_tensor>(*a)};
            if (truth) {
                return truth_of(name == "any", mask, axis);
            }
            if (name == "sum" || name == "mean") {
                return sum_or_mean(name == "mean", converted<double>(mask), axis);
            }
            if (name == "max" || name == "min") {
                const bool empty {axis < 0 ? mask.size() == 0
                                           : static_cast<std::size_t>(axis) < mask.rank() && mask.shape()[static_cast<std::size_t>(axis)] == 0};
                if (empty) {
                    throw numerical::dimension_error("a zero-size reduction to a maximum or a minimum has no identity");
                }
                return truth_of(name == "max", mask, axis);
            }
        }
        else if (name == "sum" || name == "mean") {
            return visit_numeric([&](const auto& t) { return sum_or_mean(name == "mean", t, axis); }, *a);
        }
        else if (name == "max" || name == "min") {
            if (is_complex(*a)) {
                PyErr_SetString(PyExc_TypeError, "max and min take a real tensor: a complex one has no order");
                return nullptr;
            }
            if (is_single(*a)) {
                return extreme_of(name == "max", std::get<single_tensor>(*a), axis);
            }
            return extreme_of(name == "max", std::get<real_tensor>(*a), axis);
        }
        PyErr_SetString(PyExc_ValueError, "_reduce: op is sum, mean, max, min, any or all");
        return nullptr;
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _vector_norm(x, ord, axes, keepdims) -> Tensor or number: the ord norm over the axes `axes` (distinct, non-negative)
// of every other position, as numpy.linalg.vector_norm; real, of x's precision; a number when every axis goes and
// keepdims is false.
PyObject* module_vector_norm(PyObject* /*self*/, PyObject* args)
{
    PyObject* obj      = nullptr;
    double    ord      = 2.0;
    PyObject* axes_obj = nullptr;
    int       keepdims = 0;
    if (PyArg_ParseTuple(args, "OdOp", &obj, &ord, &axes_obj, &keepdims) == 0) {
        return nullptr;
    }
    const any_tensor*       h = numeric_tensor(obj);
    std::vector<Py_ssize_t> axes;
    if (h == nullptr || !read_integers(axes_obj, axes)) {
        return nullptr;
    }
    const std::vector<std::size_t>& shape {shape_of(*h)};
    std::vector<bool>               reduced(shape.size(), false);
    for (const Py_ssize_t axis : axes) {
        if (axis < 0 || static_cast<std::size_t>(axis) >= shape.size() || reduced[static_cast<std::size_t>(axis)]) {
            PyErr_SetString(sciforge_module_error(), "_vector_norm: the axes are distinct axes of the tensor");
            return nullptr;
        }
        reduced[static_cast<std::size_t>(axis)] = true;
    }
    try {
        std::vector<std::size_t> strides(shape.size(), 1);
        for (std::size_t a = shape.size(); a-- > 1;) {
            strides[a - 1] = strides[a] * shape[a];
        }
        std::vector<std::size_t> kept_shape;
        std::vector<std::size_t> out_shape;
        std::size_t              kept_count {1};
        std::size_t              reduced_count {1};
        for (std::size_t a = 0; a < shape.size(); ++a) {
            if (reduced[a]) {
                reduced_count *= shape[a];
                if (keepdims != 0) {
                    out_shape.push_back(1);
                }
            }
            else {
                kept_shape.push_back(shape[a]);
                out_shape.push_back(shape[a]);
            }
        }
        // The positions kept: none where one of their extents is 0, whatever the others; else their checked
        // product, which with the reduced count's makes the operand's.
        kept_count = numerical::detail::element_count(kept_shape.begin(), kept_shape.end());
        // The 2-norm over trailing axes, each position's elements one run: their squares summed where they lie.
        bool trailing {true};
        for (std::size_t a = 0; a + 1 < shape.size(); ++a) {
            trailing = trailing && !(reduced[a] && !reduced[a + 1]);
        }
        if (ord == 2.0 && trailing) {
            numerical::buffer<double> out(kept_count, 0.0);
            visit_numeric([&](const auto& t) {
                for (std::size_t k = 0; k < kept_count; ++k) {
                    const auto* const run {t.data().data() + (k * reduced_count)};
                    if (const std::optional<double> plain {plain_two_norm(run, reduced_count)}) {
                        out[k] = *plain;
                        continue;
                    }
                    std::vector<double> values(reduced_count);
                    std::transform(run, run + reduced_count, values.begin(), [](const auto& x) { return static_cast<double>(std::abs(x)); });
                    out[k] = norm_of(values, ord);
                }
            }, *h);
            if (out_shape.empty() && keepdims == 0) {
                return PyFloat_FromDouble(is_single(*h) ? static_cast<double>(static_cast<float>(out[0])) : out[0]);
            }
            any_tensor result {real_tensor {std::move(out_shape), std::move(out)}};
            return wrap(is_single(*h) ? narrowed(result) : std::move(result));
        }
        const auto magnitudes {visit_numeric([](const auto& t) {
                                              std::vector<double> m(t.size());
                                              std::transform(t.data().begin(), t.data().end(), m.begin(), [](const auto& x) {
                                                  return static_cast<double>(std::abs(x));
                                              });
                                              return m;
                                          }, *h)};
        // The offset of a multi-index over the axes `which` (kept or reduced) of position `n` in their C order.
        const auto offset_of {[&](std::size_t n, bool which) {
            std::size_t offset {0};
            for (std::size_t a = shape.size(); a-- > 0;) {
                if (reduced[a] == which) {
                    offset += (n % shape[a]) * strides[a];
                    n /= shape[a];
                }
            }
            return offset;
        }};
        numerical::buffer<double> out(kept_count, 0.0);
        std::vector<double>       values(reduced_count);
        for (std::size_t k = 0; k < kept_count; ++k) {
            const std::size_t base {offset_of(k, false)};
            for (std::size_t r = 0; r < reduced_count; ++r) {
                values[r] = magnitudes[base + offset_of(r, true)];
            }
            out[k] = norm_of(values, ord);
        }
        if (out_shape.empty() && keepdims == 0) {
            return PyFloat_FromDouble(is_single(*h) ? static_cast<double>(static_cast<float>(out[0])) : out[0]);
        }
        any_tensor result {real_tensor {std::move(out_shape), std::move(out)}};
        return wrap(is_single(*h) ? narrowed(result) : std::move(result));
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _borrowed_norm(x) -> float or NotImplemented: the 2-norm of every element of a Tensor or of a NumPy array read where it
// lies (borrow_operand; a column-major matrix in its C order, plain_two_norm_columns), its squares summed where
// plain_two_norm takes them; NotImplemented for any other operand and
// for a sum it leaves to the scaled norm, which vector_norm then computes from a copy as before. A float32 or complex64
// operand's norm rounded to single precision, as vector_norm gives it. A NumPy array was copied into a Tensor first:
// six AVX2 cores, 371 × 208 doubles, 39.5 of the 56 µs.
PyObject* module_borrowed_norm(PyObject* /*self*/, PyObject* obj)
{
    borrowed_operand x;
    if (!borrow_operand(obj, x, true)) {
        return Py_NewRef(Py_NotImplemented);
    }
    std::size_t count {1};
    for (const std::size_t extent : x.shape) {
        count *= extent;
    }
    std::optional<double> norm;
    if (x.transposed) {
        const std::size_t rows {x.shape[0]};
        const std::size_t cols {x.shape[1]};
        switch (x.dtype) {
        case 0: norm = plain_two_norm_columns(static_cast<const double*>(x.data), rows, cols); break;
        case 1: norm = plain_two_norm_columns(static_cast<const std::complex<double>*>(x.data), rows, cols); break;
        case 2: norm = plain_two_norm_columns(static_cast<const float*>(x.data), rows, cols); break;
        default: norm = plain_two_norm_columns(static_cast<const std::complex<float>*>(x.data), rows, cols); break;
        }
        return norm ? PyFloat_FromDouble(x.dtype >= 2 ? static_cast<double>(static_cast<float>(*norm)) : *norm) : Py_NewRef(Py_NotImplemented);
    }
    switch (x.dtype) {
    case 0: norm = plain_two_norm(static_cast<const double*>(x.data), count); break;
    case 1: norm = plain_two_norm(static_cast<const std::complex<double>*>(x.data), count); break;
    case 2: norm = plain_two_norm(static_cast<const float*>(x.data), count); break;
    default: norm = plain_two_norm(static_cast<const std::complex<float>*>(x.data), count); break;
    }
    if (!norm) {
        return Py_NewRef(Py_NotImplemented);
    }
    return PyFloat_FromDouble(x.dtype >= 2 ? static_cast<double>(static_cast<float>(*norm)) : *norm);
}

// _borrowed_reduce(op, x) -> number or NotImplemented: sum, mean, max or min of every element of a NumPy array read
// where it lies (borrow_operand: its elements in C order, as its Tensor's), by the functions numerical::sum, mean, max
// and min call on a Tensor's storage, so their bits; NotImplemented for any other operand, and for max or min of a
// complex or an empty one, which the converting path then takes (and refuses, as for a Tensor). The array was copied into a Tensor first, three
// to four times the reduction's own time: 10^6 doubles summed 1.51 ms against 0.35 from a Tensor.
PyObject* module_borrowed_reduce(PyObject* /*self*/, PyObject* args)
{
    const char* op  = nullptr;
    PyObject*   obj = nullptr;
    if (PyArg_ParseTuple(args, "sO", &op, &obj) == 0) {
        return nullptr;
    }
    const std::string_view name {op};
    borrowed_operand       x;
    if ((name != "sum" && name != "mean" && name != "max" && name != "min") || !borrow_operand(obj, x)) {
        return Py_NewRef(Py_NotImplemented);
    }
    std::size_t count {1};
    for (const std::size_t extent : x.shape) {
        count *= extent;
    }
    try {
        const auto reduce = [&]<typename T>(std::type_identity<T>) -> PyObject* {
            const T* const data {static_cast<const T*>(x.data)};
            if (name == "sum" || name == "mean") {
                const T total {numerical::detail::parallel_pairwise_sum(data, count)};
                return from_scalar<T>(name == "mean" ? numerical::detail::divided_by_count(total, count) : total);
            }
            if constexpr (numerical::detail::split_complex<T>) {
                return Py_NewRef(Py_NotImplemented);
            }
            else if (count == 0) { // no maximum: the converting path raises it, as for a Tensor
                return Py_NewRef(Py_NotImplemented);
            }
            else if (name == "max") {
                return from_scalar<T>(numerical::detail::extreme(data, count, 1, std::greater<T> {}));
            }
            else {
                return from_scalar<T>(numerical::detail::extreme(data, count, 1, std::less<T> {}));
            }
        };
        switch (x.dtype) {
        case 0: return reduce(std::type_identity<double> {});
        case 1: return reduce(std::type_identity<std::complex<double>> {});
        case 2: return reduce(std::type_identity<float> {});
        default: return reduce(std::type_identity<std::complex<float>> {});
        }
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _borrowed_unary(name, x) -> Tensor or NotImplemented: exp, log, sqrt, sin, cos, tan, tanh of every element of a NumPy
// array read where it lies (borrow_operand), or abs of a real one, by the element functions numerical::exp and the
// rest map over a Tensor (detail::parallel_map), so its answer to the bit; NotImplemented for any other operand or name
// (abs of a complex array goes through NumPy's loop, conj through its Tensor), which the converting path takes. The
// array was copied into a Tensor first: 10^6 doubles' exp 2.63 ms, 1.06 from a Tensor.
PyObject* module_borrowed_unary(PyObject* /*self*/, PyObject* args)
{
    const char* op  = nullptr;
    PyObject*   obj = nullptr;
    if (PyArg_ParseTuple(args, "sO", &op, &obj) == 0) {
        return nullptr;
    }
    const std::string_view name {op};
    const bool             known {name == "exp" || name == "log" || name == "sqrt" || name == "sin" || name == "cos" || name == "tan"
                                  || name == "tanh" || name == "abs"};
    borrowed_operand       x;
    if (!known || !borrow_operand(obj, x) || (name == "abs" && (x.dtype == 1 || x.dtype == 3))) {
        return Py_NewRef(Py_NotImplemented);
    }
    std::size_t count {1};
    for (const std::size_t extent : x.shape) {
        count *= extent;
    }
    try {
        std::optional<any_tensor> result;
        {
            const sb::scoped_gil_release unlocked {releases_gil(count)};
            const auto map = [&]<typename T>(std::type_identity<T>) {
                namespace d = numerical::detail;
                const T* const data {static_cast<const T*>(x.data)};
                if (name == "exp") {
                    result.emplace(d::parallel_map(data, x.shape, [](const T& v) { return d::elementary_exp(v); }));
                }
                else if (name == "log") {
                    result.emplace(d::parallel_map(data, x.shape, [](const T& v) { return d::elementary_log(v); }));
                }
                else if (name == "sqrt") {
                    result.emplace(d::parallel_map(data, x.shape, [](const T& v) { return d::root(v); }));
                }
                else if (name == "sin") {
                    result.emplace(d::parallel_map(data, x.shape, [](const T& v) { return d::elementary_sin(v); }));
                }
                else if (name == "cos") {
                    result.emplace(d::parallel_map(data, x.shape, [](const T& v) { return d::elementary_cos(v); }));
                }
                else if (name == "tan") {
                    result.emplace(d::parallel_map(data, x.shape, [](const T& v) { return d::elementary_tan(v); }));
                }
                else if (name == "tanh") {
                    result.emplace(d::parallel_map(data, x.shape, [](const T& v) { return d::elementary_tanh(v); }));
                }
                else if constexpr (!complex_scalar<T>) { // abs of a real array: unary_on's
                    result.emplace(d::parallel_map(data, x.shape, [](const T& v) { return static_cast<real_of_t<T>>(std::abs(v)); }));
                }
            };
            switch (x.dtype) {
            case 0: map(std::type_identity<double> {}); break;
            case 1: map(std::type_identity<std::complex<double>> {}); break;
            case 2: map(std::type_identity<float> {}); break;
            default: map(std::type_identity<std::complex<float>> {}); break;
            }
        }
        return wrap(std::move(*result));
    }
    catch (...) {
        return set_cpp_error();
    }
}

} // namespace numerical_binding
