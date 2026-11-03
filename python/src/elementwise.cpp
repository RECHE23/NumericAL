// NumericAL — Python extension module: the element-wise arithmetic and functions, the reductions and the norms (their
// helpers in binding.hpp).

#include <numbers>

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
    if (std::string_view {op}.size() != 1 || std::string_view {"+-*/^|%"}.find(op[0]) == std::string_view::npos) {
        PyErr_SetString(PyExc_ValueError, "_binary: op is one of + - * / ^ (power) | (floor division) %");
        return nullptr;
    }
    const any_tensor* a = nullptr;
    const any_tensor* b = nullptr;
    if (!same_dtype_pair(x, y, a, b)) {
        return nullptr;
    }
    try {
        return visit_arithmetic([&](const auto& x) { return binary_on(op[0], x, std::get<std::decay_t<decltype(x)>>(*b)); }, *a);
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

// _extreme_pair(larger, a, b) -> Tensor: numpy.maximum (larger) or numpy.minimum of two real tensors of one dtype,
// broadcast, in one pass (a NaN wins, the zeros ordered as IEEE 754-2019); the caller promotes them.
PyObject* module_extreme_pair(PyObject* /*self*/, PyObject* args)
{
    int       larger = 0;
    PyObject* x      = nullptr;
    PyObject* y      = nullptr;
    if (PyArg_ParseTuple(args, "pOO", &larger, &x, &y) == 0) {
        return nullptr;
    }
    const any_tensor* a = as_tensor(x);
    const any_tensor* b = a == nullptr ? nullptr : as_tensor(y);
    if (b == nullptr) {
        return nullptr;
    }
    try {
        std::optional<any_tensor> result;
        {
            const sb::scoped_gil_release unlocked {releases_gil(std::max(count_of(*a), count_of(*b)))};
            if (const auto* wide {std::get_if<real_tensor>(a)}; wide != nullptr && b->index() == a->index()) {
                const real_tensor& other {std::get<real_tensor>(*b)};
                result.emplace(larger != 0 ? numerical::maximum(*wide, other) : numerical::minimum(*wide, other));
            }
            else if (const auto* single {std::get_if<single_tensor>(a)}; single != nullptr && b->index() == a->index()) {
                const single_tensor& other {std::get<single_tensor>(*b)};
                result.emplace(larger != 0 ? numerical::maximum(*single, other) : numerical::minimum(*single, other));
            }
            else if (const auto* integers {std::get_if<int_tensor>(a)}; integers != nullptr && b->index() == a->index()) {
                const int_tensor& other {std::get<int_tensor>(*b)};
                result.emplace(larger != 0 ? numerical::maximum(*integers, other) : numerical::minimum(*integers, other));
            }
        }
        if (!result) {
            PyErr_SetString(PyExc_TypeError, "_extreme_pair: two float64, two float32 or two int64 tensors");
            return nullptr;
        }
        return wrap(std::move(*result));
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _axis_op(name, a, axis) -> number or Tensor, as NumPy's: "prod" (axis -1: every element, a number), or along a
// non-negative axis "prod", "cumsum", "cumprod" or "sort" (ascending, NaNs last, equal elements in their order).
PyObject* module_axis_op(PyObject* /*self*/, PyObject* args)
{
    const char* op   = nullptr;
    PyObject*   x    = nullptr;
    Py_ssize_t  axis = -1;
    if (PyArg_ParseTuple(args, "sOn", &op, &x, &axis) == 0) {
        return nullptr;
    }
    const std::string_view name {op};
    if (name != "prod" && name != "cumsum" && name != "cumprod" && name != "sort") {
        PyErr_SetString(PyExc_ValueError, "_axis_op: name is prod, cumsum, cumprod or sort");
        return nullptr;
    }
    const any_tensor* a = arithmetic_tensor(x);
    if (a == nullptr) {
        return nullptr;
    }
    const numerical::detail::numpy_loops loops {numpy_loops_beside()}; // read while the GIL is held
    try {
        return visit_arithmetic([&](const auto& t) -> PyObject* {
                                 if (axis < 0) {
                                     if (name != "prod") {
                                         PyErr_SetString(PyExc_ValueError, "_axis_op: an axis for cumsum, cumprod and sort");
                                         return nullptr;
                                     }
                                     return from_scalar(prod_as_numpy(t, loops));
                                 }
                                 const auto                along {static_cast<std::size_t>(axis)};
                                 std::optional<any_tensor> result;
                                 {
                                     const sb::scoped_gil_release unlocked {releases_gil(t.size())};
                                     if (name == "prod") {
                                         result.emplace(prod_as_numpy(t, along, loops));
                                     }
                                     else if (name == "cumsum") {
                                         result.emplace(numerical::cumsum(t, along));
                                     }
                                     else if (name == "cumprod") {
                                         result.emplace(cumprod_as_numpy(t, along, loops));
                                     }
                                     else {
                                         result.emplace(numerical::sort(t, along));
                                     }
                                 }
                                 return wrap(std::move(*result));
                             }, *a);
    }
    catch (...) {
        return set_cpp_error();
    }
}

namespace {
    //! NumPy's sign of a real: ±1, +0 for either zero, a NaN as it is.
    template <typename R>
    R sign_of(R x)
    {
        if (x > R {0}) {
            return R {1};
        }
        if (x < R {0}) {
            return R {-1};
        }
        return x == R {0} ? R {0} : x;
    }

    //! The real elementary functions by NumPy's names, C's where it has one (rint: half to even, as numpy.rint and
    //! numpy.round; deg2rad and rad2deg NumPy's products by π/180 and 180/π).
    template <typename R>
    const std::array<std::pair<std::string_view, R (*)(R)>, 20> real_functions {{
        {"floor", [](R x) { return std::floor(x); }},
        {"ceil", [](R x) { return std::ceil(x); }},
        {"trunc", [](R x) { return std::trunc(x); }},
        {"rint", [](R x) { return std::nearbyint(x); }},
        {"sign", &sign_of<R>},
        {"arcsin", [](R x) { return std::asin(x); }},
        {"arccos", [](R x) { return std::acos(x); }},
        {"arctan", [](R x) { return std::atan(x); }},
        {"sinh", [](R x) { return std::sinh(x); }},
        {"cosh", [](R x) { return std::cosh(x); }},
        {"arcsinh", [](R x) { return std::asinh(x); }},
        {"arccosh", [](R x) { return std::acosh(x); }},
        {"arctanh", [](R x) { return std::atanh(x); }},
        {"log2", [](R x) { return std::log2(x); }},
        {"log10", [](R x) { return std::log10(x); }},
        {"log1p", [](R x) { return std::log1p(x); }},
        {"expm1", [](R x) { return std::expm1(x); }},
        {"cbrt", [](R x) { return std::cbrt(x); }},
        {"deg2rad", [](R x) { return x * (std::numbers::pi_v<R> / R {180}); }},
        {"rad2deg", [](R x) { return x * (R {180} / std::numbers::pi_v<R>); }},
    }};

    //! The function \p name of real_functions<R>, or null.
    template <typename R>
    R (*real_function(std::string_view name))(R)
    {
        const auto& table {real_functions<R>};
        const auto  found {std::find_if(table.begin(), table.end(), [name](const auto& entry) { return entry.first == name; })};
        return found == table.end() ? nullptr : found->second;
    }

    //! The real function \p name (one of real_functions<R>) of every element at \p data, of shape \p shape. Those
    //! the compiler maps to an instruction or two (a rounding, a product, a comparison) each go through a lambda of
    //! their own, which the loop inlines and vectorises (10^5 floors 96 µs through the pointer, 17 so); the others,
    //! calls to C's library, through the pointer.
    template <typename R>
    numerical::tensor<R> real_map(std::string_view name, const R* data, const std::vector<std::size_t>& shape)
    {
        using numerical::detail::parallel_map;
        if (name == "floor") {
            return parallel_map(data, shape, [](R x) { return std::floor(x); });
        }
        if (name == "ceil") {
            return parallel_map(data, shape, [](R x) { return std::ceil(x); });
        }
        if (name == "trunc") {
            return parallel_map(data, shape, [](R x) { return std::trunc(x); });
        }
        if (name == "rint") {
            return parallel_map(data, shape, [](R x) { return std::nearbyint(x); });
        }
        if (name == "sign") {
            return parallel_map(data, shape, [](R x) { return sign_of(x); });
        }
        if (name == "deg2rad") {
            return parallel_map(data, shape, [](R x) { return x * (std::numbers::pi_v<R> / R {180}); });
        }
        if (name == "rad2deg") {
            return parallel_map(data, shape, [](R x) { return x * (R {180} / std::numbers::pi_v<R>); });
        }
        return parallel_map(data, shape, real_function<R>(name));
    }
} // namespace

// _real_unary(name, a) -> Tensor: the real elementary function name of every element of a float64 or float32 tensor,
// as NumPy's (C's functions: within an ulp or two of NumPy's own, which differ by platform).
PyObject* module_real_unary(PyObject* /*self*/, PyObject* args)
{
    const char* op = nullptr;
    PyObject*   x  = nullptr;
    if (PyArg_ParseTuple(args, "sO", &op, &x) == 0) {
        return nullptr;
    }
    const std::string_view name {op};
    if (real_function<double>(name) == nullptr) {
        PyErr_SetString(PyExc_ValueError, "_real_unary: an unknown function");
        return nullptr;
    }
    const any_tensor* a = as_tensor(x);
    if (a == nullptr) {
        return nullptr;
    }
    try {
        std::optional<any_tensor> result;
        {
            const sb::scoped_gil_release unlocked {releases_gil(count_of(*a))};
            if (const auto* wide {std::get_if<real_tensor>(a)}) {
                result.emplace(real_map(name, wide->data().data(), wide->shape()));
            }
            else if (const auto* single {std::get_if<single_tensor>(a)}) {
                result.emplace(real_map(name, single->data().data(), single->shape()));
            }
        }
        if (!result) {
            PyErr_SetString(PyExc_TypeError, "_real_unary: a float64 or float32 tensor");
            return nullptr;
        }
        return wrap(std::move(*result));
    }
    catch (...) {
        return set_cpp_error();
    }
}

namespace {
    //! NumPy's logaddexp of two reals, `log(eˣ + eʸ)` without overflow: `x + log1p(e^(y − x))` from the larger;
    //! equal arguments (infinities of one sign among them) `x + log 2`.
    template <typename R>
    R log_add_exp(R x, R y)
    {
        if (x == y) {
            return x + std::numbers::ln2_v<R>;
        }
        const R difference {x - y};
        if (difference > R {0}) {
            return x + std::log1p(std::exp(-difference));
        }
        if (difference <= R {0}) {
            return y + std::log1p(std::exp(difference));
        }
        return difference; // a NaN
    }

    //! NumPy's logaddexp2, `log₂(2ˣ + 2ʸ)` as log_add_exp in base 2. The product by log₂ e is a statement of its own,
    //! rounded before the sum as NumPy's (a fused multiply-add moves the last bit: logaddexp2(-1, 0.5)).
    template <typename R>
    R log_add_exp2(R x, R y)
    {
        if (x == y) {
            return x + R {1};
        }
        const R difference {x - y};
        if (difference > R {0}) {
            const R above {std::numbers::log2e_v<R> * std::log1p(std::exp2(-difference))};
            return x + above;
        }
        if (difference <= R {0}) {
            const R above {std::numbers::log2e_v<R> * std::log1p(std::exp2(difference))};
            return y + above;
        }
        return difference; // a NaN
    }

    //! numpy.fmax (\p larger) or numpy.fmin of two reals: a NaN loses to a number (the first's where both), the
    //! zeros ordered as IEEE 754-2019's maximum and minimum (numpy.maximum's).
    template <bool larger, typename R>
    R number_extreme(R x, R y)
    {
        if (std::isnan(y)) {
            return x;
        }
        if (std::isnan(x)) {
            return y;
        }
        return larger ? numerical::detail::ieee_maximum(x, y) : numerical::detail::ieee_minimum(x, y);
    }

    //! numpy.heaviside: 0 below 0, \p at_zero at either zero, 1 above, NaN (C's quiet NaN, as NumPy's) for a NaN.
    template <typename R>
    R heaviside_step(R x, R at_zero)
    {
        if (x < R {0}) {
            return R {0};
        }
        if (x > R {0}) {
            return R {1};
        }
        return x == R {0} ? at_zero : std::numeric_limits<R>::quiet_NaN();
    }

    constexpr std::array<std::string_view, 10> real_binary_names {"hypot", "arctan2", "copysign", "nextafter", "fmod",
                                                                  "logaddexp", "logaddexp2", "fmax", "fmin", "heaviside"};

    //! The real binary function \p name (one of real_binary_names) of \p a and \p b element by element, broadcast;
    //! each through a lambda of its own, which the loop inlines.
    template <typename R>
    numerical::tensor<R> real_binary_map(std::string_view name, const numerical::tensor<R>& a, const numerical::tensor<R>& b)
    {
        using numerical::broadcast_apply;
        if (name == "hypot") {
            return broadcast_apply(a, b, [](R x, R y) { return std::hypot(x, y); });
        }
        if (name == "arctan2") {
            return broadcast_apply(a, b, [](R x, R y) { return std::atan2(x, y); });
        }
        if (name == "copysign") {
            return broadcast_apply(a, b, [](R x, R y) { return std::copysign(x, y); });
        }
        if (name == "nextafter") {
            return broadcast_apply(a, b, [](R x, R y) { return std::nextafter(x, y); });
        }
        if (name == "fmod") {
            return broadcast_apply(a, b, [](R x, R y) { return std::fmod(x, y); });
        }
        if (name == "logaddexp") {
            return broadcast_apply(a, b, [](R x, R y) { return log_add_exp(x, y); });
        }
        if (name == "logaddexp2") {
            return broadcast_apply(a, b, [](R x, R y) { return log_add_exp2(x, y); });
        }
        if (name == "fmax") {
            return broadcast_apply(a, b, [](R x, R y) { return number_extreme<true>(x, y); });
        }
        if (name == "fmin") {
            return broadcast_apply(a, b, [](R x, R y) { return number_extreme<false>(x, y); });
        }
        return broadcast_apply(a, b, [](R x, R y) { return heaviside_step(x, y); });
    }
} // namespace

// _real_binary(name, a, b) -> Tensor: the real binary function name of two float64 or two float32 tensors element by
// element, broadcast, as NumPy's (C's functions where NumPy calls them: hypot, atan2, copysign, nextafter, fmod; its
// own formulas for logaddexp, logaddexp2, fmax, fmin and heaviside); the caller promotes them.
PyObject* module_real_binary(PyObject* /*self*/, PyObject* args)
{
    const char* op = nullptr;
    PyObject*   x  = nullptr;
    PyObject*   y  = nullptr;
    if (PyArg_ParseTuple(args, "sOO", &op, &x, &y) == 0) {
        return nullptr;
    }
    const std::string_view name {op};
    if (std::ranges::find(real_binary_names, name) == real_binary_names.end()) {
        PyErr_SetString(PyExc_ValueError, "_real_binary: an unknown function");
        return nullptr;
    }
    const any_tensor* a = as_tensor(x);
    const any_tensor* b = a == nullptr ? nullptr : as_tensor(y);
    if (b == nullptr) {
        return nullptr;
    }
    try {
        std::optional<any_tensor> result;
        {
            const sb::scoped_gil_release unlocked {releases_gil(std::max(count_of(*a), count_of(*b)))};
            if (const auto* wide {std::get_if<real_tensor>(a)}; wide != nullptr && b->index() == a->index()) {
                result.emplace(real_binary_map(name, *wide, std::get<real_tensor>(*b)));
            }
            else if (const auto* single {std::get_if<single_tensor>(a)}; single != nullptr && b->index() == a->index()) {
                result.emplace(real_binary_map(name, *single, std::get<single_tensor>(*b)));
            }
        }
        if (!result) {
            PyErr_SetString(PyExc_TypeError, "_real_binary: two float64 or two float32 tensors");
            return nullptr;
        }
        return wrap(std::move(*result));
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _nan_to_num(a, nan, posinf, neginf) -> Tensor: a with each NaN, +inf and -inf replaced by the given numbers (in a's
// precision), a complex element's parts apart, as numpy.nan_to_num.
PyObject* module_nan_to_num(PyObject* /*self*/, PyObject* args)
{
    PyObject* x        = nullptr;
    double    for_nan  = 0.0;
    double    for_up   = 0.0;
    double    for_down = 0.0;
    if (PyArg_ParseTuple(args, "Oddd", &x, &for_nan, &for_up, &for_down) == 0) {
        return nullptr;
    }
    const any_tensor* a = numeric_tensor(x);
    if (a == nullptr) {
        return nullptr;
    }
    try {
        std::optional<any_tensor> result;
        {
            const sb::scoped_gil_release unlocked {releases_gil(count_of(*a))};
            visit_numeric([&]<typename T>(const numerical::tensor<T>& t) {
                using R = real_of_t<T>;
                const auto finite = [nan = static_cast<R>(for_nan), up = static_cast<R>(for_up),
                                     down = static_cast<R>(for_down)](R v) { return std::isnan(v) ? nan : std::isinf(v) ? (v > R {0} ? up : down) : v; };
                if constexpr (complex_scalar<T>) {
                    result.emplace(numerical::detail::parallel_map(t.data().data(), t.shape(), [finite](const T& v) {
                        return T {finite(v.real()), finite(v.imag())};
                    }));
                }
                else {
                    result.emplace(numerical::detail::parallel_map(t.data().data(), t.shape(), finite));
                }
            }, *a);
        }
        return wrap(std::move(*result));
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _arg_extreme(largest, a, axis) -> int or Tensor.
PyObject* module_arg_extreme(PyObject* /*self*/, PyObject* args)
{
    int        largest = 0;
    PyObject*  x       = nullptr;
    Py_ssize_t axis    = -1;
    if (PyArg_ParseTuple(args, "pOn", &largest, &x, &axis) == 0) {
        return nullptr;
    }
    const any_tensor* a = arithmetic_tensor(x);
    if (a == nullptr) {
        return nullptr;
    }
    if (is_complex(*a)) {
        PyErr_SetString(PyExc_TypeError, "argmax and argmin take a real tensor: a complex one has no order here");
        return nullptr;
    }
    try {
        const auto extreme = [&](const auto& t) -> PyObject* {
            if (axis < 0) {
                std::int64_t index {0};
                {
                    const sb::scoped_gil_release unlocked {releases_gil(t.size())};
                    index = largest != 0 ? numerical::argmax(t) : numerical::argmin(t);
                }
                return PyLong_FromLongLong(index);
            }
            const auto                along {static_cast<std::size_t>(axis)};
            std::optional<any_tensor> result;
            {
                const sb::scoped_gil_release unlocked {releases_gil(t.size())};
                result.emplace(largest != 0 ? numerical::argmax(t, along) : numerical::argmin(t, along));
            }
            return wrap(std::move(*result));
        };
        if (const auto* integers {std::get_if<int_tensor>(a)}) {
            return extreme(*integers);
        }
        if (const auto* single {std::get_if<single_tensor>(a)}) {
            return extreme(*single);
        }
        return extreme(std::get<real_tensor>(*a));
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _argsort(a, axis) -> Tensor.
PyObject* module_argsort(PyObject* /*self*/, PyObject* args)
{
    PyObject*  x    = nullptr;
    Py_ssize_t axis = 0;
    if (PyArg_ParseTuple(args, "On", &x, &axis) == 0) {
        return nullptr;
    }
    const any_tensor* a = arithmetic_tensor(x);
    if (a == nullptr) {
        return nullptr;
    }
    try {
        std::optional<any_tensor> result;
        {
            const sb::scoped_gil_release unlocked {releases_gil(count_of(*a))};
            result.emplace(visit_arithmetic([axis](const auto& t) { return any_tensor {numerical::argsort(t, static_cast<std::size_t>(axis))}; }, *a));
        }
        return wrap(std::move(*result));
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _nonzero(a) -> tuple of Tensors.
PyObject* module_nonzero(PyObject* /*self*/, PyObject* args)
{
    PyObject* x = nullptr;
    if (PyArg_ParseTuple(args, "O", &x) == 0) {
        return nullptr;
    }
    const any_tensor* a = as_tensor(x);
    if (a == nullptr) {
        return nullptr;
    }
    try {
        std::vector<int_tensor> indices;
        {
            const sb::scoped_gil_release unlocked {releases_gil(count_of(*a))};
            indices = std::visit([](const auto& t) { return numerical::nonzero(t); }, *a);
        }
        PyObject* tuple {PyTuple_New(static_cast<Py_ssize_t>(indices.size()))};
        if (tuple == nullptr) {
            return nullptr;
        }
        for (std::size_t k {0}; k < indices.size(); ++k) {
            PyObject* along {wrap(any_tensor {indices[k]})};
            if (along == nullptr) {
                Py_DECREF(tuple);
                return nullptr;
            }
            PyTuple_SetItem(tuple, static_cast<Py_ssize_t>(k), along); // steals the reference
        }
        return tuple;
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _count_nonzero(a, axis) -> int or Tensor.
PyObject* module_count_nonzero(PyObject* /*self*/, PyObject* args)
{
    PyObject*  x    = nullptr;
    Py_ssize_t axis = -1;
    if (PyArg_ParseTuple(args, "On", &x, &axis) == 0) {
        return nullptr;
    }
    const any_tensor* a = as_tensor(x);
    if (a == nullptr) {
        return nullptr;
    }
    try {
        std::optional<any_tensor> along;
        std::size_t               total {0};
        {
            const sb::scoped_gil_release unlocked {releases_gil(count_of(*a))};
            std::visit([&](const auto& t) {
                using T = element_of_t<std::decay_t<decltype(t)>>;
                if (axis < 0) {
                    total = static_cast<std::size_t>(std::ranges::count_if(t.data(), [](const T& value) { return value != T {}; }));
                }
                else {
                    along.emplace(numerical::count_nonzero(t, static_cast<std::size_t>(axis)));
                }
            }, *a);
        }
        return along.has_value() ? wrap(std::move(*along)) : PyLong_FromSize_t(total);
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _take_along_axis(a, indices, axis) -> Tensor: numerical::take_along_axis, the indices int64 of a's shape but along
// the axis (the caller broadcasts and wraps them).
PyObject* module_take_along_axis(PyObject* /*self*/, PyObject* args)
{
    PyObject*  x    = nullptr;
    PyObject*  y    = nullptr;
    Py_ssize_t axis = 0;
    if (PyArg_ParseTuple(args, "OOn", &x, &y, &axis) == 0) {
        return nullptr;
    }
    const any_tensor* a = as_tensor(x);
    const any_tensor* b = a == nullptr ? nullptr : as_tensor(y);
    if (b == nullptr) {
        return nullptr;
    }
    const auto* indices = std::get_if<int_tensor>(b);
    if (indices == nullptr || axis < 0) {
        PyErr_SetString(PyExc_ValueError, "_take_along_axis: int64 indices and a non-negative axis");
        return nullptr;
    }
    try {
        std::optional<any_tensor> result;
        {
            const sb::scoped_gil_release unlocked {releases_gil(indices->size())};
            result.emplace(std::visit([&](const auto& t) {
                return any_tensor {numerical::take_along_axis(t, *indices, static_cast<std::size_t>(axis))};
            }, *a));
        }
        return wrap(std::move(*result));
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _correlate(a, v, first, count) -> Tensor: numerical::correlate of two vectors of one dtype over count lags from
// first.
PyObject* module_correlate(PyObject* /*self*/, PyObject* args)
{
    PyObject*  x     = nullptr;
    PyObject*  y     = nullptr;
    Py_ssize_t first = 0;
    Py_ssize_t count = 0;
    if (PyArg_ParseTuple(args, "OOnn", &x, &y, &first, &count) == 0) {
        return nullptr;
    }
    const any_tensor* a = nullptr;
    const any_tensor* v = nullptr;
    if (!same_dtype_pair(x, y, a, v)) {
        return nullptr;
    }
    if (count < 0 || shape_of(*a).size() != 1 || shape_of(*v).size() != 1) {
        PyErr_SetString(PyExc_ValueError, "_correlate: two vectors and a non-negative count");
        return nullptr;
    }
    try {
        std::optional<any_tensor> result;
        {
            const sb::scoped_gil_release unlocked {releases_gil(count_of(*a) * count_of(*v))};
            result.emplace(visit_arithmetic([&](const auto& t) {
                return any_tensor {numerical::correlate(t, std::get<std::decay_t<decltype(t)>>(*v), first, static_cast<std::size_t>(count))};
            }, *a));
        }
        return wrap(std::move(*result));
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _complex_from_parts(re, im) -> Tensor: re + im·i of two real tensors of one shape and dtype (float32 or float64),
// exactly (each zero's sign kept), in one pass.
PyObject* module_complex_from_parts(PyObject* /*self*/, PyObject* args)
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
    if (a->index() != b->index() || shape_of(*a) != shape_of(*b) || !(std::holds_alternative<real_tensor>(*a) || std::holds_alternative<single_tensor>(*a))) {
        PyErr_SetString(PyExc_ValueError, "_complex_from_parts: two float tensors of one shape and dtype");
        return nullptr;
    }
    const auto join = [](const auto& re, const auto& im) {
        using R = element_of_t<std::decay_t<decltype(re)>>;
        return any_tensor {numerical::detail::broadcast_into(re.data().data(), re.shape(), im.data().data(), im.shape(),
                                                             [](R r, R i) { return std::complex<R> {r, i}; })};
    };
    try {
        std::optional<any_tensor> result;
        {
            const sb::scoped_gil_release unlocked {releases_gil(count_of(*a))};
            if (const auto* single {std::get_if<single_tensor>(a)}) {
                result.emplace(join(*single, std::get<single_tensor>(*b)));
            }
            else {
                result.emplace(join(std::get<real_tensor>(*a), std::get<real_tensor>(*b)));
            }
        }
        return wrap(std::move(*result));
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _bincount(x, weights, length) -> Tensor: how often each value 0 … length − 1 occurs in the int64 vector x (none
// negative, none past length − 1: the caller's checks), an int64 tensor; or with float64 weights (x's length), each
// value's weights summed in x's order, as NumPy's loop adds them, a float64 one.
PyObject* module_bincount(PyObject* /*self*/, PyObject* args)
{
    PyObject*  x       = nullptr;
    PyObject*  w       = nullptr;
    Py_ssize_t length  = 0;
    if (PyArg_ParseTuple(args, "OOn", &x, &w, &length) == 0) {
        return nullptr;
    }
    const any_tensor* a = as_tensor(x);
    if (a == nullptr) {
        return nullptr;
    }
    const auto* values = std::get_if<numerical::tensor<std::int64_t>>(a);
    if (values == nullptr || values->rank() != 1 || length < 0) {
        PyErr_SetString(PyExc_ValueError, "_bincount: an int64 vector and a non-negative length");
        return nullptr;
    }
    const auto size = static_cast<std::size_t>(length);
    try {
        if (w == Py_None) {
            numerical::buffer<std::int64_t> counts(size, 0);
            {
                const sb::scoped_gil_release unlocked {releases_gil(values->size())};
                for (const std::int64_t v : values->data()) {
                    ++counts[static_cast<std::size_t>(v)];
                }
            }
            return wrap(any_tensor {numerical::tensor<std::int64_t> {{size}, std::move(counts)}});
        }
        const any_tensor* b       = as_tensor(w);
        const auto*       weights = b == nullptr ? nullptr : std::get_if<numerical::tensor<double>>(b);
        if (weights == nullptr || weights->size() != values->size()) {
            if (b != nullptr) {
                PyErr_SetString(PyExc_ValueError, "_bincount: float64 weights, one per value");
            }
            return nullptr;
        }
        numerical::buffer<double> sums(size, 0.0);
        {
            const sb::scoped_gil_release unlocked {releases_gil(values->size())};
            for (std::size_t i {0}; i < values->size(); ++i) {
                sums[static_cast<std::size_t>(values->data()[i])] += weights->data()[i];
            }
        }
        return wrap(any_tensor {numerical::tensor<double> {{size}, std::move(sums)}});
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _searchsorted(a, v, right) -> Tensor.
PyObject* module_searchsorted(PyObject* /*self*/, PyObject* args)
{
    PyObject* x     = nullptr;
    PyObject* y     = nullptr;
    int       right = 0;
    if (PyArg_ParseTuple(args, "OOp", &x, &y, &right) == 0) {
        return nullptr;
    }
    const any_tensor* a = nullptr;
    const any_tensor* v = nullptr;
    if (!same_dtype_pair(x, y, a, v)) {
        return nullptr;
    }
    try {
        std::optional<any_tensor> result;
        {
            const sb::scoped_gil_release unlocked {releases_gil(count_of(*v))};
            result.emplace(visit_arithmetic([&](const auto& sorted) {
                return any_tensor {numerical::searchsorted(sorted, std::get<std::decay_t<decltype(sorted)>>(*v), right != 0)};
            }, *a));
        }
        return wrap(std::move(*result));
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _round(a, decimals) -> Tensor: numerical::round of a float or complex tensor (decimals clamped by the caller).
PyObject* module_round(PyObject* /*self*/, PyObject* args)
{
    PyObject* x        = nullptr;
    int       decimals = 0;
    if (PyArg_ParseTuple(args, "Oi", &x, &decimals) == 0) {
        return nullptr;
    }
    const any_tensor* a = numeric_tensor(x);
    if (a == nullptr) {
        return nullptr;
    }
    try {
        std::optional<any_tensor> result;
        {
            const sb::scoped_gil_release unlocked {releases_gil(count_of(*a))};
            result.emplace(visit_numeric([decimals](const auto& t) { return any_tensor {numerical::round(t, decimals)}; }, *a));
        }
        return wrap(std::move(*result));
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
    const any_tensor* a = arithmetic_tensor(x);
    if (a == nullptr) {
        return nullptr;
    }
    try { // an allocation the system refuses is a Python exception, not an abort
        return visit_arithmetic([](const auto& t) { return wrap(any_tensor {-t}); }, *a);
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
                const tensor_type& q {std::get<tensor_type>(*b)};
                const auto         compared {[&](const auto& relation) { // broadcast_into's loops, the relation inlined in each
                    result.emplace(numerical::detail::broadcast_into(p.data().data(), p.shape(), q.data().data(), q.shape(), [&relation](const T& x, const T& y) {
                        if constexpr (std::is_same_v<T, boolean>) {
                            return boolean {relation(x.value, y.value)};
                        }
                        else {
                            return boolean {relation(x, y)};
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
        std::optional<bool_tensor> result;
        {
            const sb::scoped_gil_release unlocked {releases_gil(std::max(a->size(), b->size()))};
            // The operator chosen once, outside the loops; on the bytes (each 0 or 1), which vectorise as such.
            const auto applied {[&](const auto& bits) {
                result.emplace(numerical::detail::broadcast_into(a->data().data(), a->shape(), b->data().data(), b->shape(), [&bits](boolean p, boolean q) {
                    return boolean {bits(static_cast<unsigned char>(p.value), static_cast<unsigned char>(q.value)) != 0};
                }));
            }};
            if (name == "&") {
                applied(std::bit_and<> {});
            }
            else if (name == "|") {
                applied(std::bit_or<> {});
            }
            else {
                applied(std::bit_xor<> {});
            }
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
        std::optional<bool_tensor> result;
        {
            const sb::scoped_gil_release unlocked {releases_gil(a->size())};
            // on the bytes (each 0 or 1), which vectorises as such
            result.emplace(numerical::detail::parallel_map(*a, [](boolean p) { return boolean {(static_cast<unsigned char>(p.value) ^ 1U) != 0}; }));
        }
        return wrap(any_tensor {std::move(*result)});
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
        std::optional<any_tensor> result;
        {
        const sb::scoped_gil_release unlocked {releases_gil(count_of(*a))};
        result.emplace(any_tensor {std::visit([&](const auto& t) {
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
        return wrap(std::move(*result));
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
                return visit_arithmetic([&](const auto& t) { return truth_of(name == "any", t, axis); }, *a);
            }
            const bool_tensor& mask {std::get<bool_tensor>(*a)};
            if (truth) {
                return truth_of(name == "any", mask, axis);
            }
            if (name == "sum" || name == "mean") {
                return counted_trues(name == "mean", mask, axis);
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
            if (const auto* integers {std::get_if<int_tensor>(a)}) { // NumPy's: a sum in int64, wrapped; a mean in float64
                return name == "sum" ? sum_or_mean(false, *integers, axis) : sum_or_mean(true, converted<double>(*integers), axis);
            }
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
            if (is_int(*a)) {
                return extreme_of(name == "max", std::get<int_tensor>(*a), axis);
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
                const sb::scoped_gil_release unlocked {releases_gil(t.size())};
                numerical::detail::parallel_for(kept_count, reduced_count, [&](std::size_t k) { // each position its own run
                    const auto* const run {t.data().data() + (k * reduced_count)};
                    if (const std::optional<double> plain {plain_two_norm(run, reduced_count)}) {
                        out[k] = *plain;
                        return;
                    }
                    std::vector<double> values(reduced_count);
                    std::transform(run, run + reduced_count, values.begin(), [](const auto& x) { return static_cast<double>(std::abs(x)); });
                    out[k] = norm_of(values, ord);
                });
            }, *h);
            if (out_shape.empty() && keepdims == 0) {
                return PyFloat_FromDouble(is_single(*h) ? static_cast<double>(static_cast<float>(out[0])) : out[0]);
            }
            any_tensor result {real_tensor {std::move(out_shape), std::move(out)}};
            return wrap(is_single(*h) ? narrowed(result) : std::move(result));
        }
        // One axis that is not the last, of order 1, 2 or ±∞: the lines along it accumulated row by row over bands of
        // 256 of the positions after it (contiguous), the bands on the threads, where gathering each line strode across
        // the rows. A 2-norm whose sum of squares leaves [2^-900, ∞) is the scaled one of its gathered line, as below.
        if (axes.size() == 1 && static_cast<std::size_t>(axes[0]) + 1 < shape.size() && kept_count != 0
            && (ord == 1.0 || ord == 2.0 || std::isinf(ord))) {
            const numerical::detail::axis_walk walk {numerical::detail::walk_along(shape, static_cast<std::size_t>(axes[0]))};
            numerical::buffer<double>          out(kept_count, 0.0);
            visit_numeric([&](const auto& t) {
                const sb::scoped_gil_release unlocked {releases_gil(t.size())};
                constexpr std::size_t        band {256};
                const std::size_t            per_outer {(walk.inner + band - 1) / band};
                numerical::detail::parallel_for(walk.outer * per_outer, band * walk.extent, [&](std::size_t task) {
                    const std::size_t outer {task / per_outer};
                    const std::size_t first {(task % per_outer) * band};
                    const std::size_t last {std::min(walk.inner, first + band)};
                    double* const     to {out.data() + (outer * walk.inner)};
                    const bool        largest {ord > 0.0};
                    for (std::size_t w = first; w < last; ++w) {
                        to[w] = std::isinf(ord) ? (largest ? 0.0 : std::numeric_limits<double>::infinity()) : 0.0;
                    }
                    for (std::size_t j = 0; j < walk.extent; ++j) {
                        const auto* const row {t.data().data() + (((outer * walk.extent) + j) * walk.inner)};
                        for (std::size_t w = first; w < last; ++w) {
                            const auto magnitude {static_cast<double>(std::abs(row[w]))};
                            if (ord == 1.0) {
                                to[w] += magnitude;
                            }
                            else if (ord == 2.0) {
                                add_square(to[w], magnitude);
                            }
                            else if (std::isnan(magnitude) || std::isnan(to[w])) { // a NaN stays
                                to[w] = std::numeric_limits<double>::quiet_NaN();
                            }
                            else {
                                to[w] = largest ? std::max(to[w], magnitude) : std::min(to[w], magnitude);
                            }
                        }
                    }
                    for (std::size_t w = first; w < last && ord == 2.0; ++w) {
                        if (const std::optional<double> plain {root_of_squares(to[w])}) {
                            to[w] = *plain;
                            continue;
                        }
                        std::vector<double> values(walk.extent);
                        for (std::size_t j = 0; j < walk.extent; ++j) {
                            values[j] = static_cast<double>(std::abs(t.data().data()[(((outer * walk.extent) + j) * walk.inner) + w]));
                        }
                        to[w] = norm_of(values, ord);
                    }
                });
            }, *h);
            if (out_shape.empty() && keepdims == 0) {
                return PyFloat_FromDouble(is_single(*h) ? static_cast<double>(static_cast<float>(out[0])) : out[0]);
            }
            any_tensor result {real_tensor {std::move(out_shape), std::move(out)}};
            return wrap(is_single(*h) ? narrowed(result) : std::move(result));
        }
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
        // Each kept position's magnitudes gathered where they lie (the reduced offsets computed once), its norm on the
        // threads.
        std::vector<std::size_t> within(kept_count == 0 ? 0 : reduced_count);
        for (std::size_t r = 0; r < within.size(); ++r) {
            within[r] = offset_of(r, true);
        }
        numerical::buffer<double> out(kept_count, 0.0);
        visit_numeric([&](const auto& t) {
            const sb::scoped_gil_release unlocked {releases_gil(t.size())};
            numerical::detail::parallel_for(kept_count, reduced_count, [&](std::size_t k) {
                const auto* const   base {t.data().data() + offset_of(k, false)};
                std::vector<double> values(reduced_count);
                for (std::size_t r = 0; r < reduced_count; ++r) {
                    values[r] = static_cast<double>(std::abs(base[within[r]]));
                }
                out[k] = norm_of(values, ord);
            });
        }, *h);
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
                const T total {numerical::detail::parallel_pairwise_sum(data, count) + T {}}; // from +0, as NumPy's
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
// array read where it lies (borrow_operand), or abs or a real function (real_map) of a real one, by the element functions numerical::exp and the
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
    const bool             real_only {real_function<double>(name) != nullptr};
    const bool             known {real_only || name == "exp" || name == "log" || name == "sqrt" || name == "sin" || name == "cos"
                                  || name == "tan" || name == "tanh" || name == "abs"};
    borrowed_operand       x;
    if (!known || !borrow_operand(obj, x) || ((name == "abs" || real_only) && (x.dtype == 1 || x.dtype == 3))) {
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
                if constexpr (!complex_scalar<T>) {
                    if (real_only) {
                        result.emplace(real_map(name, data, x.shape));
                        return;
                    }
                }
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
