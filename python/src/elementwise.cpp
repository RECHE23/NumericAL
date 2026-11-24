// NumericAL — Python extension module: the element-wise arithmetic and functions, the reductions and the norms (their
// helpers in binding.hpp).

#include <bit>
#include <numbers>
#include <numeric>
#include <type_traits>

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

namespace {

// A Python float or int (not a bool) as the arithmetic operators take one beside a tensor of a float or complex dtype
// (asarray(number, dtype)'s double), into `value`; false for anything else, or an int no double holds, no error set:
// the converting path then raises the error.
bool weak_number(PyObject* obj, double& value)
{
    if (PyFloat_CheckExact(obj) == 0 && PyLong_CheckExact(obj) == 0) {
        return false;
    }
    value = PyFloat_AsDouble(obj);
    if (PyErr_Occurred() != nullptr) {
        PyErr_Clear();
        return false;
    }
    return true;
}

} // namespace

// _clip(a, low, high) -> Tensor or NotImplemented: numerical::clip of a float64, float32 or int64 tensor between two
// rank-0 tensors of its dtype, or of a float64 or float32 one between two Python floats or ints (weak: asarray(number,
// dtype)'s elements), in one pass; NotImplemented for two Python numbers beside another dtype, or an int no double
// holds, which the converting path answers. The numbers' tensors took 1.5 of the 2.9 µs of a clip of 75 doubles.
PyObject* module_clip(PyObject* /*self*/, PyObject* args)
{
    PyObject* x    = nullptr;
    PyObject* low  = nullptr;
    PyObject* high = nullptr;
    if (PyArg_ParseTuple(args, "OOO", &x, &low, &high) == 0) {
        return nullptr;
    }
    const any_tensor* a = as_tensor(x);
    if (a == nullptr) {
        return nullptr;
    }
    // Two Python floats or ints beside a float Tensor, weak: asarray(number, dtype)'s elements, without their tensors.
    double     low_value {0.0};
    double     high_value {0.0};
    const bool exact {(PyFloat_CheckExact(low) != 0 || PyLong_CheckExact(low) != 0) && (PyFloat_CheckExact(high) != 0 || PyLong_CheckExact(high) != 0)};
    const bool numbers {(a->index() == 0 || a->index() == 2) && weak_number(low, low_value) && weak_number(high, high_value)};
    if (exact && !numbers) { // an int no double holds, or a tensor of another dtype: the converting path's answer
        return Py_NewRef(Py_NotImplemented);
    }
    const any_tensor* l = numbers ? nullptr : as_tensor(low);
    const any_tensor* h = numbers || l == nullptr ? nullptr : as_tensor(high);
    if (!numbers && h == nullptr) {
        return nullptr;
    }
    try {
        std::optional<any_tensor> result;
        {
            const sb::scoped_gil_release unlocked {releases_gil(count_of(*a))};
            const auto                   clipped = [&]<typename T>(std::type_identity<T>) {
                const auto* values {std::get_if<numerical::tensor<T>>(a)};
                if (values != nullptr && numbers) {
                    if constexpr (std::is_floating_point_v<T>) {
                        result.emplace(numerical::clip(*values, static_cast<T>(low_value), static_cast<T>(high_value)));
                    }
                }
                else if (values != nullptr && l->index() == a->index() && h->index() == a->index()) {
                    const auto& lower {std::get<numerical::tensor<T>>(*l)};
                    const auto& upper {std::get<numerical::tensor<T>>(*h)};
                    if (lower.size() == 1 && upper.size() == 1) {
                        result.emplace(numerical::clip(*values, lower.data()[0], upper.data()[0]));
                    }
                }
            };
            clipped(std::type_identity<double> {});
            clipped(std::type_identity<float> {});
            clipped(std::type_identity<std::int64_t> {});
        }
        if (!result) {
            PyErr_SetString(PyExc_TypeError, "_clip: a float64, float32 or int64 tensor and two numbers of its dtype");
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

    //! The cube root within half an ulp and a little: C's, then one Newton step whose residual y³ − x two fused
    //! products give to the last bit (C's cbrt is up to 3 ulps off in glibc). Below 2^-600 (float: 2^-60) the
    //! residual would fall among the subnormals, so x is scaled by 2^768 (2^96) and the root back by 2^-256 (2^-32).
    template <typename R>
    R cube_root(R x)
    {
        R scale {1};
        if constexpr (std::is_same_v<R, double>) {
            if (std::fabs(x) < 0x1p-600) {
                x *= 0x1p768;
                scale = 0x1p-256;
            }
        }
        else {
            if (std::fabs(x) < 0x1p-60F) {
                x *= 0x1p96F;
                scale = 0x1p-32F;
            }
        }
        const R y {std::cbrt(x)};
        if (!std::isfinite(y) || y == R {0}) {
            return y * scale;
        }
        const R square {y * y};
        const R square_error {std::fma(y, y, -square)}; // y² = square + square_error exactly
        const R residual {std::fma(square_error, y, std::fma(square, y, -x))};
        return (y - residual / (R {3} * square)) * scale;
    }

    //! The real elementary functions by NumPy's names, C's where it has one (rint: half to even, as numpy.rint and
    //! numpy.round; deg2rad and rad2deg NumPy's products by π/180 and 180/π).
    template <typename R>
    const std::array<std::pair<std::string_view, R (*)(R)>, 21> real_functions {{
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
        {"exp2", [](R x) { return std::exp2(x); }},
        {"cbrt", &cube_root<R>},
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

    //! The real function \p name (one of real_functions<R>) of every element of \p source: a tensor, read where it lies
    //! (a view too), or the elements at a pointer and their shape (a NumPy array's, borrowed). Those the compiler maps
    //! to an instruction or two (a rounding, a product, a comparison) each go through a lambda of their own, which the
    //! loop inlines and vectorises (10^5 floors 96 µs through the pointer, 17 so); the table functions by their
    //! kernels in vectors; the others, calls to C's library, through the pointer.
    template <typename R, typename... Source>
    numerical::tensor<R> real_map(std::string_view name, const Source&... source)
    {
        namespace d = numerical::detail;
        using d::parallel_map;
        constexpr bool whole {sizeof...(Source) == 1}; // a tensor, not a pointer and a shape
        if (name == "floor") {
            return parallel_map(source..., [](R x) { return std::floor(x); });
        }
        if (name == "ceil") {
            return parallel_map(source..., [](R x) { return std::ceil(x); });
        }
        if (name == "trunc") {
            return parallel_map(source..., [](R x) { return std::trunc(x); });
        }
        if (name == "rint") {
            return parallel_map(source..., [](R x) { return std::nearbyint(x); });
        }
        if (name == "sign") {
            return parallel_map(source..., [](R x) { return sign_of(x); });
        }
        if (name == "deg2rad") {
            return parallel_map(source..., [](R x) { return x * (std::numbers::pi_v<R> / R {180}); });
        }
        if (name == "rad2deg") {
            return parallel_map(source..., [](R x) { return x * (R {180} / std::numbers::pi_v<R>); });
        }
        if (name == "expm1") { // a double's by the table, in vectors (numerical::detail::expm1_into)
            if constexpr (whole) {
                return d::expm1_viewed(source...);
            }
            else {
                return d::expm1_of(source...);
            }
        }
        if (name == "arctan") { // a double's by the table, in vectors (numerical::detail::atan_into)
            if constexpr (whole) {
                return d::atan_viewed(source...);
            }
            else {
                return d::atan_of(source...);
            }
        }
        if (name == "sinh") { // a double's by exp's table, in vectors (numerical::detail::sinh_into)
            if constexpr (whole) {
                return d::sinh_viewed(source...);
            }
            else {
                return d::sinh_of(source...);
            }
        }
        if (name == "cosh") { // a double's by exp's table, in vectors (numerical::detail::cosh_into)
            if constexpr (whole) {
                return d::cosh_viewed(source...);
            }
            else {
                return d::cosh_of(source...);
            }
        }
        if (name == "exp2") { // a double's by exp's table, in vectors (numerical::detail::exp2_into)
            if constexpr (whole) {
                return d::exp2_viewed(source...);
            }
            else {
                return d::exp2_of(source...);
            }
        }
        if (name == "log1p") { // a double's by the table, in vectors (numerical::detail::log1p_into)
            if constexpr (whole) {
                return d::log1p_viewed(source...);
            }
            else {
                return d::log1p_of(source...);
            }
        }
        return parallel_map(source..., real_function<R>(name));
    }
} // namespace

namespace {
    //! The complex function `name` (NumPy's name) of a complex number, or nullptr for one this module does not take.
    template <typename R>
    using complex_function = std::complex<R> (*)(const std::complex<R>&);

    template <typename R>
    complex_function<R> complex_named(std::string_view name)
    {
        namespace d = numerical::detail;
        static constexpr std::array<std::pair<std::string_view, complex_function<R>>, 14> table {{
            {"arcsin", &d::complex_asin<R>},   {"arccos", &d::complex_acos<R>},   {"arctan", &d::complex_atan<R>},
            {"arcsinh", &d::complex_asinh<R>}, {"arccosh", &d::complex_acosh<R>}, {"arctanh", &d::complex_atanh<R>},
            {"sinh", &d::complex_sinh<R>},     {"cosh", &d::complex_cosh<R>},     {"log2", &d::complex_log2<R>},
            {"log10", &d::complex_log10<R>},   {"log1p", &d::complex_log1p<R>},   {"expm1", &d::complex_expm1<R>},
            {"exp2", &d::complex_exp2<R>},     {"sign", &d::complex_sign<R>},
        }};
        const auto found {std::ranges::find(table, name, &std::pair<std::string_view, complex_function<R>>::first)}; // an iterator, not a pointer under MSVC
        return found == table.end() ? nullptr : found->second;
    }
} // namespace

// _complex_unary(name, a) -> Tensor: the elementary function name of every element of a complex128 or complex64
// tensor: glibc's inverse functions (NumPy's on Linux), NumPy's formulas for the rest (numerical/complex_math.hpp).
PyObject* module_complex_unary(PyObject* /*self*/, PyObject* args)
{
    const char* op = nullptr;
    PyObject*   x  = nullptr;
    if (PyArg_ParseTuple(args, "sO", &op, &x) == 0) {
        return nullptr;
    }
    const std::string_view name {op};
    if (complex_named<double>(name) == nullptr) {
        PyErr_SetString(PyExc_ValueError, "_complex_unary: an unknown function");
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
            if (const auto* wide {std::get_if<cplx_tensor>(a)}) {
                result.emplace(numerical::detail::parallel_map(*wide, complex_named<double>(name)));
            }
            else if (const auto* single {std::get_if<csingle_tensor>(a)}) {
                result.emplace(numerical::detail::parallel_map(*single, complex_named<float>(name)));
            }
        }
        if (!result) {
            PyErr_SetString(PyExc_TypeError, "_complex_unary: a complex128 or complex64 tensor");
            return nullptr;
        }
        return wrap(std::move(*result));
    }
    catch (...) {
        return set_cpp_error();
    }
}

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
                result.emplace(real_map<double>(name, *wide));
            }
            else if (const auto* single {std::get_if<single_tensor>(a)}) {
                result.emplace(real_map<float>(name, *single));
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

    constexpr std::array<std::string_view, 11> real_binary_names {"hypot", "arctan2", "copysign", "nextafter", "fmod",
                                                                  "logaddexp", "logaddexp2", "fmax", "fmin", "heaviside",
                                                                  "ldexp"};

    //! \p x · 2^\p e, \p e an integer held by the real (bounded first: past ±2^17 every finite \p x over- or
    //! underflows alike), as C's ldexp rounds it once.
    template <typename R>
    R scaled_by_power_of_two(R x, R e)
    {
        constexpr R bound {131072};
        return std::ldexp(x, static_cast<int>(std::fmax(std::fmin(e, bound), -bound)));
    }

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
        if (name == "ldexp") {
            return broadcast_apply(a, b, [](R x, R y) { return scaled_by_power_of_two(x, y); });
        }
        return broadcast_apply(a, b, [](R x, R y) { return heaviside_step(x, y); });
    }
} // namespace

namespace {
    //! NumPy's shifts of int64: by 64 places or more (a negative count among them, as its unsigned reading),
    //! every bit shifted out (0, or −1 to the right of a negative number); computed on the unsigned bits, which
    //! wrap as NumPy's.
    std::int64_t shifted_left(std::int64_t x, std::int64_t n)
    {
        return static_cast<std::uint64_t>(n) < 64 ? static_cast<std::int64_t>(static_cast<std::uint64_t>(x) << n) : 0;
    }

    std::int64_t shifted_right(std::int64_t x, std::int64_t n)
    {
        if (static_cast<std::uint64_t>(n) < 64) {
            return x >> n; // arithmetic, as C++20 defines it
        }
        return x < 0 ? -1 : 0;
    }

    //! The magnitude of \p x as an unsigned number: INT64_MIN's 2⁶³ held.
    std::uint64_t magnitude(std::int64_t x)
    {
        return x < 0 ? std::uint64_t {0} - static_cast<std::uint64_t>(x) : static_cast<std::uint64_t>(x);
    }

    //! NumPy's gcd of int64: of the magnitudes, wrapped back (gcd(INT64_MIN, 0) is INT64_MIN, as NumPy's).
    std::int64_t greatest_divisor(std::int64_t x, std::int64_t y)
    {
        return static_cast<std::int64_t>(std::gcd(magnitude(x), magnitude(y)));
    }

    //! NumPy's lcm of int64: |x / gcd · y|, wrapped; 0 where either is.
    std::int64_t least_multiple(std::int64_t x, std::int64_t y)
    {
        const std::uint64_t divisor {std::gcd(magnitude(x), magnitude(y))};
        return divisor == 0 ? 0 : static_cast<std::int64_t>((magnitude(x) / divisor) * magnitude(y));
    }

    constexpr std::string_view bitwise_names {"&|^<>glc"};
} // namespace

// _bitwise(op, a, b) -> Tensor: two int64 tensors element by element, broadcast, as NumPy's: op & | ^, < (left
// shift), > (right shift), g (gcd), l (lcm), or c (bitwise_count: the 1 bits of a's magnitude, b unread).
PyObject* module_bitwise(PyObject* /*self*/, PyObject* args)
{
    const char* op = nullptr;
    PyObject*   x  = nullptr;
    PyObject*   y  = nullptr;
    if (PyArg_ParseTuple(args, "sOO", &op, &x, &y) == 0) {
        return nullptr;
    }
    if (std::string_view {op}.size() != 1 || bitwise_names.find(op[0]) == std::string_view::npos) {
        PyErr_SetString(PyExc_ValueError, "_bitwise: op is one of & | ^ < > g l c");
        return nullptr;
    }
    const any_tensor* a = as_tensor(x);
    const any_tensor* b = a == nullptr ? nullptr : as_tensor(y);
    if (b == nullptr) {
        return nullptr;
    }
    const auto* left  = std::get_if<int_tensor>(a);
    const auto* right = std::get_if<int_tensor>(b);
    if (left == nullptr || right == nullptr) {
        PyErr_SetString(PyExc_TypeError, "_bitwise: two int64 tensors");
        return nullptr;
    }
    try {
        std::optional<int_tensor> result;
        {
            const sb::scoped_gil_release unlocked {releases_gil(std::max(left->size(), right->size()))};
            using I = std::int64_t;
            switch (op[0]) {
            case '&':
                result.emplace(numerical::broadcast_apply(*left, *right, [](I p, I q) { return p & q; }));
                break;
            case '|':
                result.emplace(numerical::broadcast_apply(*left, *right, [](I p, I q) { return p | q; }));
                break;
            case '^':
                result.emplace(numerical::broadcast_apply(*left, *right, [](I p, I q) { return p ^ q; }));
                break;
            case '<':
                result.emplace(numerical::broadcast_apply(*left, *right, &shifted_left));
                break;
            case '>':
                result.emplace(numerical::broadcast_apply(*left, *right, &shifted_right));
                break;
            case 'g':
                result.emplace(numerical::broadcast_apply(*left, *right, &greatest_divisor));
                break;
            case 'l':
                result.emplace(numerical::broadcast_apply(*left, *right, &least_multiple));
                break;
            default:
                result.emplace(numerical::detail::parallel_map(*left, [](I p) {
                    return static_cast<I>(std::popcount(magnitude(p)));
                }));
                break;
            }
        }
        return wrap(any_tensor {std::move(*result)});
    }
    catch (...) {
        return set_cpp_error();
    }
}

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
                    result.emplace(numerical::detail::parallel_map(t, [finite](const T& v) {
                        return T {finite(v.real()), finite(v.imag())};
                    }));
                }
                else {
                    result.emplace(numerical::detail::parallel_map(t, finite));
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

// _partition(a, kths, axis, indices) -> Tensor: numerical::partition of a's lines along axis at the non-negative
// positions kths (a sequence of ints, each below the extent: the caller's checks), or with indices its argpartition.
PyObject* module_partition(PyObject* /*self*/, PyObject* args)
{
    PyObject*  x       = nullptr;
    PyObject*  k       = nullptr;
    Py_ssize_t axis    = 0;
    int        indices = 0;
    if (PyArg_ParseTuple(args, "OOnp", &x, &k, &axis, &indices) == 0) {
        return nullptr;
    }
    const any_tensor* a = arithmetic_tensor(x);
    if (a == nullptr) {
        return nullptr;
    }
    const Py_ssize_t count {PySequence_Size(k)};
    if (count < 0) {
        return nullptr;
    }
    std::vector<std::size_t> kths;
    for (Py_ssize_t i {0}; i < count; ++i) {
        PyObject* item = PySequence_GetItem(k, i);
        if (item == nullptr) {
            return nullptr;
        }
        const std::size_t position {PyLong_AsSize_t(item)};
        Py_DECREF(item);
        if (PyErr_Occurred() != nullptr) {
            return nullptr;
        }
        kths.push_back(position);
    }
    try {
        std::optional<any_tensor> result;
        {
            const sb::scoped_gil_release unlocked {releases_gil(count_of(*a))};
            const auto                   along {static_cast<std::size_t>(axis)};
            result.emplace(visit_arithmetic([&](const auto& t) {
                return indices != 0 ? any_tensor {numerical::argpartition(t, kths, along)}
                                    : any_tensor {numerical::partition(t, kths, along)};
            }, *a));
        }
        return wrap(std::move(*result));
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _vdot(a, b) -> float or complex: numerical::vdot of two tensors of one float or complex dtype and one count (the
// caller's checks), their elements in row-major order.
PyObject* module_vdot(PyObject* /*self*/, PyObject* args)
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
    try {
        const auto take = [&]<typename T>(std::type_identity<T>) -> PyObject* {
            const auto* left  = std::get_if<numerical::tensor<T>>(a);
            const auto* right = std::get_if<numerical::tensor<T>>(b);
            if (left == nullptr || right == nullptr) {
                return nullptr;
            }
            T value {};
            {
                const sb::scoped_gil_release unlocked {releases_gil(left->size())};
                value = numerical::vdot(*left, *right);
            }
            return from_scalar<T>(value);
        };
        if (PyObject* found = take(std::type_identity<double> {})) {
            return found;
        }
        if (PyObject* found = take(std::type_identity<std::complex<double>> {})) {
            return found;
        }
        if (PyObject* found = take(std::type_identity<float> {})) {
            return found;
        }
        if (PyObject* found = take(std::type_identity<std::complex<float>> {})) {
            return found;
        }
        PyErr_SetString(PyExc_TypeError, "_vdot: two tensors of one float or complex dtype");
        return nullptr;
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _argsort_by(order, key, axis) -> Tensor: the int64 order's lines sorted stably by the key's elements they index.
PyObject* module_argsort_by(PyObject* /*self*/, PyObject* args)
{
    PyObject*  o    = nullptr;
    PyObject*  x    = nullptr;
    Py_ssize_t axis = 0;
    if (PyArg_ParseTuple(args, "OOn", &o, &x, &axis) == 0) {
        return nullptr;
    }
    const any_tensor* indices = as_tensor(o);
    const any_tensor* key     = indices == nullptr ? nullptr : arithmetic_tensor(x);
    if (key == nullptr) {
        return nullptr;
    }
    const auto* order = std::get_if<int_tensor>(indices);
    if (order == nullptr) {
        PyErr_SetString(PyExc_ValueError, "_argsort_by: an int64 order");
        return nullptr;
    }
    try {
        std::optional<any_tensor> result;
        {
            const sb::scoped_gil_release unlocked {releases_gil(count_of(*key))};
            result.emplace(visit_arithmetic([&](const auto& t) {
                return any_tensor {numerical::argsort_by(*order, t, static_cast<std::size_t>(axis))};
            }, *key));
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
                if (axis < 0) {
                    total = numerical::count_nonzero(t);
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

namespace {

// An index's integers read where they lie: an int64 Tensor's in C order, or a C-contiguous NumPy array of 4- or 8-byte
// signed integers. `owner` (a new reference: the Tensor, or the capsule, which holds the array) keeps them alive.
struct borrowed_indices
{
    PyObject*                owner {nullptr};
    const void*              data {nullptr};
    int                      bytes {8}; // each integer's size
    std::vector<std::size_t> shape;

    borrowed_indices() = default;
    borrowed_indices(const borrowed_indices&) = delete;
    borrowed_indices& operator=(const borrowed_indices&) = delete;
    ~borrowed_indices() { Py_XDECREF(owner); }

    [[nodiscard]] std::int64_t at(std::size_t i) const noexcept
    {
        return bytes == 8 ? static_cast<const std::int64_t*>(data)[i] : static_cast<const std::int32_t*>(data)[i];
    }
};

// Whether `obj` is an int64 Tensor or such a NumPy array, read into `out`; false, no error set, otherwise.
bool borrow_indices(PyObject* obj, borrowed_indices& out)
{
    if (PyObject_TypeCheck(obj, reinterpret_cast<PyTypeObject*>(sb::class_type<any_tensor>())) != 0) {
        const any_tensor* v {reinterpret_cast<sb::wrapper<any_tensor>*>(obj)->held};
        if (v == nullptr || !is_int(*v)) {
            return false;
        }
        out.owner = Py_NewRef(obj);
        out.data  = data_of(*v);
        out.shape = shape_of(*v);
        return true;
    }
    PyObject* capsule = PyObject_GetAttr(obj, interned_name<array_struct_name>()); // new ref
    if (capsule == nullptr) {
        PyErr_Clear();
        return false;
    }
    const auto* view {static_cast<const array_struct*>(PyCapsule_GetPointer(capsule, nullptr))};
    if (view == nullptr) {
        PyErr_Clear();
    }
    constexpr int needed {array_c_contiguous | array_aligned | array_not_swapped};
    if (view == nullptr || view->two != 2 || view->data == nullptr || view->typekind != 'i' || (view->itemsize != 8 && view->itemsize != 4)
        || (view->flags & needed) != needed) {
        Py_DECREF(capsule);
        return false;
    }
    out.owner = capsule;
    out.data  = view->data;
    out.bytes = view->itemsize;
    for (int axis = 0; axis < view->nd; ++axis) {
        out.shape.push_back(static_cast<std::size_t>(view->shape[axis]));
    }
    return true;
}

} // namespace

// _take_indices(a, indices, axis, mode) -> Tensor or NotImplemented: numerical::take, an index outside the axis an
// IndexError in NumPy's words; the indices an int64 Tensor, or a NumPy array of 4- or 8-byte integers read where it
// lies (borrow_indices), NotImplemented for any other array, which take converts. Its conversion into a Tensor took
// half of a take of 7 indices along an axis (1.66 µs, NumPy 0.79).
PyObject* module_take_indices(PyObject* /*self*/, PyObject* args)
{
    PyObject*  x    = nullptr;
    PyObject*  y    = nullptr;
    Py_ssize_t axis = 0;
    int        mode = 0;
    if (PyArg_ParseTuple(args, "OOni", &x, &y, &axis, &mode) == 0) {
        return nullptr;
    }
    const any_tensor* a = as_tensor(x);
    if (a == nullptr) {
        return nullptr;
    }
    std::optional<int_tensor> read; // a NumPy array's indices, copied: a few, beside the gather
    const int_tensor*         indices {nullptr};
    if (PyObject_TypeCheck(y, reinterpret_cast<PyTypeObject*>(sb::class_type<any_tensor>())) != 0) {
        const any_tensor* b {as_tensor(y)};
        if (b == nullptr) {
            return nullptr;
        }
        indices = std::get_if<int_tensor>(b);
    }
    else {
        borrowed_indices borrowed;
        if (!borrow_indices(y, borrowed)) {
            return Py_NewRef(Py_NotImplemented);
        }
        numerical::buffer<std::int64_t> values(numerical::detail::element_count(borrowed.shape.begin(), borrowed.shape.end()));
        for (std::size_t i {0}; i < values.size(); ++i) {
            values[i] = borrowed.at(i);
        }
        read.emplace(borrowed.shape, std::move(values));
        indices = &*read;
    }
    if (indices == nullptr || axis < 0 || mode < 0 || mode > 2) {
        PyErr_SetString(PyExc_ValueError, "_take_indices: int64 indices, a non-negative axis and a mode of 0, 1 or 2");
        return nullptr;
    }
    try {
        std::optional<any_tensor> result;
        {
            const sb::scoped_gil_release unlocked {releases_gil(count_of(*a))};
            result.emplace(std::visit([&](const auto& t) {
                return any_tensor {numerical::take(t, *indices, static_cast<std::size_t>(axis), static_cast<numerical::take_mode>(mode))};
            }, *a));
        }
        return wrap(std::move(*result));
    }
    catch (const std::out_of_range& outside) {
        PyErr_SetString(PyExc_IndexError, outside.what());
        return nullptr;
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _select(a, positions) -> Tensor: numerical::select_along_last, the elements a stable sort of each line along a's
// last axis puts at the int64 positions (one row of them for every line, or one per line).
PyObject* module_select(PyObject* /*self*/, PyObject* args)
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
    const auto* positions = std::get_if<int_tensor>(b);
    if (positions == nullptr || std::holds_alternative<bool_tensor>(*a)) {
        PyErr_SetString(PyExc_ValueError, "_select: a tensor of numbers and int64 positions");
        return nullptr;
    }
    try {
        std::optional<any_tensor> result;
        {
            const sb::scoped_gil_release unlocked {releases_gil(count_of(*a))};
            result.emplace(std::visit([&](const auto& t) {
                if constexpr (std::is_same_v<std::decay_t<decltype(t)>, bool_tensor>) {
                    return any_tensor {t}; // refused above: a bool has no order the median reads
                }
                else {
                    return any_tensor {numerical::select_along_last(t, *positions)};
                }
            }, *a));
        }
        return wrap(std::move(*result));
    }
    catch (...) {
        return set_cpp_error();
    }
}

namespace {

// One line's linear quantiles, as _quantiles computes them (NumPy's _quantile, method "linear"): each q's virtual index
// (n - 1) q, its neighbours floor and floor + 1 (both the last value from n - 1 up, both the first below 0), gamma the
// index less the first neighbour, and _lerp of the two values a stable sort puts there; the line's last value (a NaN,
// the stable sort's last) where it holds one. Each operation rounded alone, as the tensor operations round them: a in
// T, b - a in T, the rest in R (T, or float64 for a float32 line beside float64 quantiles), a weak q's gamma and 1 -
// gamma (computed in float64) each rounded to R. `at` and `picked` hold 2 m + 1 positions and values of scratch.
template <typename T, typename R>
void linear_quantiles_of_line(const T* row, std::size_t n, const double* q, std::size_t m, T* work, std::int64_t* at,
                              T* picked, R* out)
{
    namespace d = numerical::detail;
    const auto   last {static_cast<double>(n - 1)};
    const auto   to_position {[n](double index) {
        const auto position {static_cast<std::int64_t>(index)};
        return position < 0 ? position + static_cast<std::int64_t>(n) : position;
    }};
    bool nan {false};
    for (std::size_t i {0}; i < n && !nan; ++i) {
        nan = std::isnan(row[i]);
    }
    for (std::size_t j {0}; j < m; ++j) {
        const double index {last * q[j]};
        double       previous {std::floor(index)};
        double       following {previous + 1.0};
        if (index < 0.0) {
            previous  = 0.0;
            following = 0.0;
        }
        else if (index >= last) {
            previous  = -1.0;
            following = -1.0;
        }
        at[j]     = to_position(previous);
        at[m + j] = to_position(following);
    }
    const std::size_t count {nan ? (2 * m) + 1 : 2 * m};
    at[2 * m] = static_cast<std::int64_t>(n - 1);
    if (count <= d::select_positions_sorted) {
        d::select_row(row, n, at, count, work, picked);
    }
    else { // more than a selection takes: the line sorted whole, as select_row would
        std::copy(row, row + n, work);
        d::sort_run(work, work + n);
        for (std::size_t j {0}; j < count; ++j) {
            picked[j] = work[static_cast<std::size_t>(at[j])];
        }
    }
    for (std::size_t j {0}; j < m; ++j) {
        if (nan) {
            out[j] = static_cast<R>(picked[2 * m]);
            continue;
        }
        const double index {last * q[j]};
        double       previous {std::floor(index)};
        if (index < 0.0) {
            previous = 0.0;
        }
        else if (index >= last) {
            previous = -1.0;
        }
        const double gamma {index - previous};
        const T      a {picked[j]};
        const T      b {picked[m + j]};
        const T      difference {b - a};
        if (gamma >= 0.5) {
            const auto weight {static_cast<R>(1.0 - gamma)};
            const R    product {static_cast<R>(difference) * weight};
            out[j] = static_cast<R>(b) - product;
        }
        else {
            const auto weight {static_cast<R>(gamma)};
            const R    product {static_cast<R>(difference) * weight};
            out[j] = static_cast<R>(a) + product;
        }
    }
}

} // namespace

// _quantile_linear(a, q, weak) -> Tensor or NotImplemented: the linear quantiles at the float64 q (a vector) of each
// line along a's last axis (float64 or float32, lines not empty), of shape a's lines and len(q): _quantiles' answer
// to the bit for method "linear" without NaNs ignored (linear_quantiles_of_line), in a's dtype, or float64 for a
// float32 a beside q not weak (a Tensor). NotImplemented for anything else, which _quantiles computes as before. It
// took some 150 tensor operations of one element: 45 µs for 100 doubles, where NumPy takes 26.
PyObject* module_quantile_linear(PyObject* /*self*/, PyObject* args)
{
    PyObject* x    = nullptr;
    PyObject* y    = nullptr;
    int       weak = 0;
    if (PyArg_ParseTuple(args, "OOp", &x, &y, &weak) == 0) {
        return nullptr;
    }
    auto* const type {reinterpret_cast<PyTypeObject*>(sb::class_type<any_tensor>())};
    if (PyObject_TypeCheck(x, type) == 0 || PyObject_TypeCheck(y, type) == 0) {
        return Py_NewRef(Py_NotImplemented);
    }
    const any_tensor* a {reinterpret_cast<sb::wrapper<any_tensor>*>(x)->held};
    const any_tensor* b {reinterpret_cast<sb::wrapper<any_tensor>*>(y)->held};
    const auto*       q {b == nullptr ? nullptr : std::get_if<real_tensor>(b)};
    if (a == nullptr || q == nullptr || q->rank() != 1 || (a->index() != 0 && a->index() != 2)) {
        return Py_NewRef(Py_NotImplemented);
    }
    const std::vector<std::size_t> shape {shape_of(*a)};
    if (shape.empty() || shape.back() == 0) {
        return Py_NewRef(Py_NotImplemented);
    }
    try {
        std::optional<any_tensor> result;
        const auto                compute = [&]<typename T, typename R>(const numerical::tensor<T>& t, std::type_identity<R>) {
            const std::size_t n {shape.back()};
            const std::size_t m {q->size()};
            const std::size_t lines {t.size() / n};
            std::vector<std::size_t> out_shape(shape.begin(), shape.end() - 1);
            out_shape.push_back(m);
            numerical::buffer<R> out(lines * m);
            const T* const       in {t.data().data()};
            const double* const  qs {q->data().data()};
            {
                const sb::scoped_gil_release unlocked {releases_gil(t.size())};
                const std::size_t            block {std::max<std::size_t>(1, 4096 / n)};
                numerical::detail::parallel_for((lines + block - 1) / block, block * n * 16, [&](std::size_t chunk) {
                    numerical::buffer<T>            work(n);
                    std::vector<std::int64_t>       at((2 * m) + 1);
                    numerical::buffer<T>            picked((2 * m) + 1);
                    for (std::size_t line {chunk * block}; line < std::min(lines, (chunk + 1) * block); ++line) {
                        linear_quantiles_of_line(in + (line * n), n, qs, m, work.data(), at.data(), picked.data(), out.data() + (line * m));
                    }
                });
            }
            result.emplace(numerical::tensor<R> {std::move(out_shape), std::move(out)});
        };
        if (const auto* t {std::get_if<real_tensor>(a)}) {
            compute(*t, std::type_identity<double> {});
        }
        else if (weak != 0) {
            compute(std::get<single_tensor>(*a), std::type_identity<float> {});
        }
        else {
            compute(std::get<single_tensor>(*a), std::type_identity<double> {});
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

namespace {
    //! NumPy's binary_search_with_guess (compiled_base.c): the index i with arr[i] <= key < arr[i + 1], -1 below
    //! arr[0], len above arr[len − 1]; the values about the guess tried first, a linear search at four or fewer.
    std::ptrdiff_t interp_search(double key, const double* arr, std::ptrdiff_t len, std::ptrdiff_t guess)
    {
        constexpr std::ptrdiff_t likely_in_cache {8};
        std::ptrdiff_t           imin {0};
        std::ptrdiff_t           imax {len};
        if (key > arr[len - 1]) {
            return len;
        }
        if (key < arr[0]) {
            return -1;
        }
        if (len <= 4) {
            std::ptrdiff_t i {1};
            while (i < len && key >= arr[i]) {
                ++i;
            }
            return i - 1;
        }
        guess = std::clamp<std::ptrdiff_t>(guess, 1, len - 3);
        if (key < arr[guess]) {
            if (key >= arr[guess - 1]) {
                return guess - 1;
            }
            imax = guess - 1;
            if (guess > likely_in_cache && key >= arr[guess - likely_in_cache]) {
                imin = guess - likely_in_cache;
            }
        }
        else {
            if (key < arr[guess + 1]) {
                return guess;
            }
            if (key < arr[guess + 2]) {
                return guess + 1;
            }
            imin = guess + 2;
            if (guess < len - likely_in_cache - 1 && key < arr[guess + likely_in_cache]) {
                imax = guess + likely_in_cache;
            }
        }
        while (imin < imax) {
            const std::ptrdiff_t imid {imin + ((imax - imin) >> 1)};
            if (key >= arr[imid]) {
                imin = imid + 1;
            }
            else {
                imax = imid;
            }
        }
        return imin - 1;
    }

    //! `slope · (x − x₀) + y₀`, one fused multiply-add where `fused` (NumPy's macOS arm64 wheels contract it).
    double interp_line(double slope, double x, double x0, double y0, bool fused)
    {
        if (fused) {
            return std::fma(slope, x - x0, y0);
        }
        const double rise {slope * (x - x0)};
        return rise + y0;
    }

    //! The points a part of an interpolation takes, and the work of one, in multiply-adds (a search about the
    //! previous point's interval and a line).
    constexpr std::size_t interp_chunk {4096};
    constexpr std::size_t interp_work {16};

    //! One part (or a real value) of NumPy's interpolation between points j and j + 1: through the first, or the
    //! second where that is NaN, or the shared value where both ends are equal.
    double interp_part(double slope, double x, double x0, double x1, double y0, double y1, bool fused)
    {
        double value {interp_line(slope, x, x0, y0, fused)};
        if (std::isnan(value)) {
            value = interp_line(slope, x, x1, y1, fused);
            if (std::isnan(value) && y0 == y1) {
                value = y0;
            }
        }
        return value;
    }
} // namespace

// _interp(x, xp, fp, left, right, fused) -> Tensor: numpy.interp's loop (compiled_base.c arr_interp and
// arr_interp_complex) over float64 x, the increasing float64 xp and float64 or complex128 fp (one value per point),
// left and right the fills below and above; each value through NumPy's search and line, fused as `fused`.
PyObject* module_interp(PyObject* /*self*/, PyObject* args)
{
    PyObject* x     = nullptr;
    PyObject* xp    = nullptr;
    PyObject* fp    = nullptr;
    PyObject* left  = nullptr;
    PyObject* right = nullptr;
    int       fused = 0;
    if (PyArg_ParseTuple(args, "OOOOOp", &x, &xp, &fp, &left, &right, &fused) == 0) {
        return nullptr;
    }
    const any_tensor* a = as_tensor(x);
    const any_tensor* b = a == nullptr ? nullptr : as_tensor(xp);
    const any_tensor* c = b == nullptr ? nullptr : as_tensor(fp);
    const any_tensor* l = c == nullptr ? nullptr : as_tensor(left);
    const any_tensor* r = l == nullptr ? nullptr : as_tensor(right);
    if (r == nullptr) {
        return nullptr;
    }
    const auto* points = std::get_if<real_tensor>(a);
    const auto* knots  = std::get_if<real_tensor>(b);
    if (points == nullptr || knots == nullptr || knots->rank() != 1 || knots->size() == 0 || shape_of(*c) != shape_of(*b)
        || c->index() != l->index() || c->index() != r->index() || !(std::holds_alternative<real_tensor>(*c) || std::holds_alternative<cplx_tensor>(*c))) {
        PyErr_SetString(PyExc_ValueError, "_interp: float64 x and xp, one fp per xp, fills of fp's dtype");
        return nullptr;
    }
    try {
        std::optional<any_tensor> result;
        {
            const sb::scoped_gil_release unlocked {releases_gil(points->size())};
            result.emplace(std::visit([&](const auto& values) -> any_tensor {
                using T = element_of_t<std::decay_t<decltype(values)>>;
                if constexpr (std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>) {
                    const auto&    fill_left {std::get<numerical::tensor<T>>(*l).data()[0]};
                    const auto&    fill_right {std::get<numerical::tensor<T>>(*r).data()[0]};
                    const double*  dx {knots->data().data()};
                    const T*       dy {values.data().data()};
                    const auto     len {static_cast<std::ptrdiff_t>(knots->size())};
                    const std::size_t    size {points->size()};
                    numerical::buffer<T> out(size);
                    // the search starts from the previous point's interval, and finds the one index with xp[j] <= key <
                    // xp[j + 1] whatever it starts from only where xp is non-decreasing and holds no NaN: the points
                    // split among the threads only there, each part searching from the first interval
                    const bool split {size > interp_chunk && std::is_sorted(dx, dx + len)
                                      && std::none_of(dx, dx + len, [](double v) { return std::isnan(v); })};
                    const std::size_t share {split ? interp_chunk : std::max<std::size_t>(size, 1)};
                    numerical::detail::parallel_for((size + share - 1) / share, share * interp_work, [&](std::size_t part) {
                        std::ptrdiff_t j {0};
                        for (std::size_t i {part * share}; i < std::min(size, (part + 1) * share); ++i) {
                            const double key {points->data()[i]};
                            if (len == 1) {
                                out[i] = key < dx[0] ? fill_left : (key > dx[0] ? fill_right : dy[0]);
                                continue;
                            }
                            if (std::isnan(key)) {
                                out[i] = T {key};
                                continue;
                            }
                            j = interp_search(key, dx, len, j);
                            if (j == -1) {
                                out[i] = fill_left;
                            }
                            else if (j == len) {
                                out[i] = fill_right;
                            }
                            else if (j == len - 1 || dx[j] == key) {
                                out[i] = dy[j];
                            }
                            else if constexpr (std::is_same_v<T, double>) {
                                const double slope {(dy[j + 1] - dy[j]) / (dx[j + 1] - dx[j])};
                                out[i] = interp_part(slope, key, dx[j], dx[j + 1], dy[j], dy[j + 1], fused != 0);
                            }
                            else {
                                const double inverse {1.0 / (dx[j + 1] - dx[j])};
                                const double re {(dy[j + 1].real() - dy[j].real()) * inverse};
                                const double im {(dy[j + 1].imag() - dy[j].imag()) * inverse};
                                out[i] = T {interp_part(re, key, dx[j], dx[j + 1], dy[j].real(), dy[j + 1].real(), fused != 0),
                                            interp_part(im, key, dx[j], dx[j + 1], dy[j].imag(), dy[j + 1].imag(), fused != 0)};
                            }
                        }
                    });
                    return any_tensor {numerical::tensor<T> {points->shape(), std::move(out)}};
                }
                else {
                    throw std::runtime_error("_interp: float64 or complex128 values");
                }
            }, *c));
        }
        return wrap(std::move(*result));
    }
    catch (...) {
        return set_cpp_error();
    }
}

namespace {
    //! The bin of `x` among `count` equal bins, as NumPy's histogram finds it: the offset from the first edge in
    //! `Offset` (`x − first`), over the span and times the count in `Quotient`, truncated; then moved down where `x`
    //! lies below that bin's lower edge, up where it reaches the next edge (but the last bin, closed).
    template <typename Offset, typename Quotient, typename D>
    std::size_t histogram_bin(D x, Offset first, Quotient span, std::size_t count, const D* edges)
    {
        const Offset   offset {static_cast<Offset>(x) - first};
        const Quotient scaled {(static_cast<Quotient>(offset) / span) * static_cast<Quotient>(count)};
        auto           index {static_cast<std::size_t>(static_cast<std::int64_t>(scaled))};
        index -= static_cast<std::size_t>(index == count);
        index -= static_cast<std::size_t>(x < edges[index]);
        index += static_cast<std::size_t>(x >= edges[index + 1] && index != count - 1);
        return index;
    }

    //! How many of the `size` elements fall in each of `count` bins, `bin(i)` element i's (or `count` to leave it
    //! out): on the threads, each part counted into bins of its own, then the parts' added; integers, so the counts
    //! are a sequential pass's. Split only so far as the parts' bins add up to no more than the elements.
    template <typename Bin>
    numerical::buffer<std::int64_t> bin_counts(std::size_t size, std::size_t count, const Bin& bin)
    {
        const std::size_t parts {std::max<std::size_t>(1, std::min(numerical::detail::parallel_threads(size, 1), size / (count + 1)))};
        const std::size_t share {(size + parts - 1) / parts};
        numerical::buffer<std::int64_t> local(parts * (count + 1), 0);
        numerical::detail::parallel_for(parts, share, [&](std::size_t p) {
            std::int64_t* const own {local.data() + (p * (count + 1))};
            const std::size_t   last {std::min(size, (p + 1) * share)};
            for (std::size_t i {p * share}; i < last; ++i) {
                ++own[bin(i)];
            }
        });
        numerical::buffer<std::int64_t> counts(count, 0);
        for (std::size_t p {0}; p < parts; ++p) {
            const std::int64_t* const own {local.data() + (p * (count + 1))};
            for (std::size_t b {0}; b < count; ++b) {
                counts[b] += own[b];
            }
        }
        return counts;
    }

    //! Every block of `block` elements' weights summed into its own bins in float64 in the elements' order (NumPy's
    //! bincount), each block's sums then made `W`s and added to the totals in `W`, as NumPy's histogram adds them.
    template <typename W, typename D, typename Bin>
    numerical::buffer<W> histogram_sums(const D* values, std::size_t size, const W* weights, std::size_t count,
                                        D low, D high, const Bin& bin)
    {
        constexpr std::size_t block {65536};
        numerical::buffer<W>  total(count, W {});
        std::vector<double>   sums(count);
        for (std::size_t start {0}; start < size; start += block) {
            std::fill(sums.begin(), sums.end(), 0.0);
            for (std::size_t i {start}; i < std::min(size, start + block); ++i) {
                if (values[i] >= low && values[i] <= high) {
                    sums[bin(values[i])] += static_cast<double>(weights[i]);
                }
            }
            for (std::size_t k {0}; k < count; ++k) {
                if constexpr (std::is_same_v<W, std::int64_t>) { // truncated, then added wrapping, as NumPy's int64
                    total[k] = static_cast<std::int64_t>(static_cast<std::uint64_t>(total[k])
                                                         + static_cast<std::uint64_t>(static_cast<std::int64_t>(sums[k])));
                }
                else {
                    total[k] = static_cast<W>(total[k] + static_cast<W>(sums[k]));
                }
            }
        }
        return total;
    }
} // namespace

// _histogram_uniform(a, edges, low, high, first, single_offset, span, single_quotient, weights) -> Tensor: the counts
// (int64) or the weights' sums (the weights' dtype: float64, float32 or int64) of a's elements (float64 or float32,
// edges' dtype) between low and high (a's dtype) over len(edges) − 1 equal bins, as numpy.histogram's loop over
// blocks finds and adds them: the offsets from `first` in float32 where single_offset (else float64), the quotients
// by `span` in float32 where single_quotient. The caller has found those dtypes as NumPy's arithmetic gives them.
PyObject* module_histogram_uniform(PyObject* /*self*/, PyObject* args)
{
    PyObject* x              = nullptr;
    PyObject* e              = nullptr;
    double    low            = 0.0;
    double    high           = 0.0;
    double    first          = 0.0;
    int       single_offset  = 0;
    double    span           = 0.0;
    int       single_divider = 0;
    PyObject* w              = nullptr;
    if (PyArg_ParseTuple(args, "OOdddpdpO", &x, &e, &low, &high, &first, &single_offset, &span, &single_divider, &w) == 0) {
        return nullptr;
    }
    const any_tensor* a     = as_tensor(x);
    const any_tensor* edges = a == nullptr ? nullptr : as_tensor(e);
    const any_tensor* wt    = (edges == nullptr || w == Py_None) ? nullptr : as_tensor(w);
    if (edges == nullptr || (w != Py_None && wt == nullptr)) {
        return nullptr;
    }
    if (a->index() != edges->index() || (!std::holds_alternative<real_tensor>(*a) && !std::holds_alternative<single_tensor>(*a))
        || count_of(*edges) < 2
        || (wt != nullptr && (count_of(*wt) != count_of(*a) || (!std::holds_alternative<real_tensor>(*wt)
                                                                && !std::holds_alternative<single_tensor>(*wt)
                                                                && !std::holds_alternative<int_tensor>(*wt))))) {
        PyErr_SetString(PyExc_ValueError, "_histogram_uniform: float64 or float32 values and edges, real weights");
        return nullptr;
    }
    try {
        std::optional<any_tensor> result;
        {
            const sb::scoped_gil_release unlocked {releases_gil(count_of(*a))};
            const auto run {[&](const auto& values) {
                using D = element_of_t<std::decay_t<decltype(values)>>;
                const auto&       bounds {std::get<numerical::tensor<D>>(*edges)};
                const std::size_t count {bounds.size() - 1};
                const D*          edge {bounds.data().data()};
                const D           lo {static_cast<D>(low)};
                const D           hi {static_cast<D>(high)};
                const auto        with_types {[&](auto offset_type, auto quotient_type) {
                    using Offset   = decltype(offset_type);
                    using Quotient = decltype(quotient_type);
                    const auto bin {[&](D v) {
                        return histogram_bin(v, static_cast<Offset>(first), static_cast<Quotient>(span), count, edge);
                    }};
                    const D* const    in {values.data().data()};
                    const std::size_t size {values.size()};
                    if (wt == nullptr) {
                        result.emplace(int_tensor {{count}, bin_counts(size, count, [&](std::size_t i) {
                            return in[i] >= lo && in[i] <= hi ? bin(in[i]) : count;
                        })});
                        return;
                    }
                    std::visit([&](const auto& weights) {
                        using W = element_of_t<std::decay_t<decltype(weights)>>;
                        if constexpr (std::is_same_v<W, double> || std::is_same_v<W, float> || std::is_same_v<W, std::int64_t>) {
                            result.emplace(numerical::tensor<W> {{count}, histogram_sums(in, size, weights.data().data(), count, lo, hi, bin)});
                        }
                    }, *wt);
                }};
                if (single_offset != 0) {
                    single_divider != 0 ? with_types(float {}, float {}) : with_types(float {}, double {});
                }
                else {
                    single_divider != 0 ? with_types(double {}, float {}) : with_types(double {}, double {});
                }
            }};
            if (const auto* wide {std::get_if<real_tensor>(a)}) {
                run(*wide);
            }
            else {
                run(std::get<single_tensor>(*a));
            }
        }
        return wrap(std::move(*result));
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _inverse_permutation(order) -> Tensor: the int64 permutation q with q[order[k]] = k, of an int64 vector order
// holding each of 0 … n − 1 once (argsort's answer), where a second argsort found it in O(n log n).
PyObject* module_inverse_permutation(PyObject* /*self*/, PyObject* arg)
{
    const any_tensor* a = as_tensor(arg);
    if (a == nullptr) {
        return nullptr;
    }
    const auto* order = std::get_if<int_tensor>(a);
    if (order == nullptr || order->rank() != 1) {
        PyErr_SetString(PyExc_ValueError, "_inverse_permutation: an int64 vector");
        return nullptr;
    }
    const std::size_t               n {order->size()};
    numerical::buffer<std::int64_t> inverse(n, -1);
    for (std::size_t k {0}; k < n; ++k) {
        const std::int64_t at {order->data()[k]};
        if (at < 0 || static_cast<std::size_t>(at) >= n || inverse[static_cast<std::size_t>(at)] >= 0) {
            PyErr_SetString(PyExc_ValueError, "_inverse_permutation: not a permutation of 0 … n − 1");
            return nullptr;
        }
        inverse[static_cast<std::size_t>(at)] = static_cast<std::int64_t>(k);
    }
    return wrap(any_tensor {int_tensor {{n}, std::move(inverse)}});
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
            numerical::buffer<std::int64_t> counts;
            {
                const sb::scoped_gil_release unlocked {releases_gil(values->size())};
                const std::int64_t* const    in {values->data().data()};
                counts = bin_counts(values->size(), size, [in](std::size_t i) { return static_cast<std::size_t>(in[i]); });
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
                                     // Equal element by element in any order: two transposes of the same order and shape compared
                                     // where they lie.
                                     if (x.shape() == y.shape()) {
                                         const auto x_laid {x.laid_out()};
                                         const auto y_laid {y.laid_out()};
                                         if (x_laid && y_laid && x_laid->second == y_laid->second) {
                                             const auto& u {x_laid->first.data()};
                                             return std::equal(u.begin(), u.end(), y_laid->first.data().begin());
                                         }
                                     }
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
                    result.emplace(numerical::broadcast_apply(p, q, [&relation](const T& x, const T& y) { // views read in place
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
                result.emplace(numerical::broadcast_apply(*a, *b, [&bits](boolean p, boolean q) { // views read in place
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

// Whether a condition's element holds: a bool Tensor's, or a NumPy bool array's byte (nonzero, as asarray reads it).
inline bool holds_at(boolean b)
{
    return b.value;
}

inline bool holds_at(unsigned char b)
{
    return b != 0;
}

// The one shape where(c, x, y) takes without broadcasting's walk, of operands of the shapes `shapes`: each one element
// of no more axes than it, or of that shape. Null where there is none, or every operand is one element.
inline const std::vector<std::size_t>* alike_shape(const std::array<const std::vector<std::size_t>*, 3>& shapes)
{
    const auto elements {[](const std::vector<std::size_t>& each) { return numerical::detail::element_count(each.begin(), each.end()); }};
    const auto found {std::ranges::find_if(shapes, [&](const std::vector<std::size_t>* each) { return elements(*each) != 1; })};
    if (found == shapes.end()) {
        return nullptr;
    }
    const std::vector<std::size_t>& shape {**found};
    const bool alike {std::ranges::all_of(shapes, [&](const std::vector<std::size_t>* each) {
        return *each == shape || (elements(*each) == 1 && each->size() <= shape.size());
    })};
    return alike ? &shape : nullptr;
}

// `holds ? x : y` by a mask of the bits, with no branch to mispredict on a condition of no pattern: a real's (and a
// complex's parts) by masked_select, which vectorises; another element as it is.
template <std::floating_point T>
T chosen_one(bool holds, T x, T y)
{
    return numerical::detail::masked_select(holds, x, y);
}

template <numerical::detail::split_complex C>
C chosen_one(bool holds, const C& x, const C& y)
{
    return {numerical::detail::masked_select(holds, x.real(), y.real()), numerical::detail::masked_select(holds, x.imag(), y.imag())};
}

template <typename T>
    requires(!std::floating_point<T> && !numerical::detail::split_complex<T>)
T chosen_one(bool holds, const T& x, const T& y)
{
    return holds ? x : y;
}

// `holds[i · at_c] ? u[i · at_x] : v[i · at_y]` for each of `count` elements, by chunks among the threads: an operand
// of one element read at a step of 0.
template <typename T, typename C>
numerical::buffer<T> chosen_in_order(const C* holds, std::size_t at_c, const T* u, std::size_t at_x, const T* v, std::size_t at_y,
                                     std::size_t count)
{
    numerical::buffer<T>  out(count);
    constexpr std::size_t chunk {numerical::detail::elementwise_chunk};
    numerical::detail::parallel_for((count + chunk - 1) / chunk, chunk, [&](std::size_t k) {
        const std::size_t last {std::min(count, (k + 1) * chunk)};
        for (std::size_t i {k * chunk}; i < last; ++i) {
            out[i] = chosen_one(holds_at(holds[i * at_c]), u[i * at_x], v[i * at_y]);
        }
    });
    return out;
}

// where(c, x, y) without broadcasting's walk, where it has none to do (alike_shape); those all row-major, or all a
// transpose of the same order, read where they lie (the result put back in that order). Nothing otherwise. Each
// element chosen alone: the same either way.
template <typename T>
std::optional<numerical::tensor<T>> where_alike(const bool_tensor& c, const numerical::tensor<T>& x, const numerical::tensor<T>& y)
{
    const std::vector<std::size_t>* shape {alike_shape({&c.shape(), &x.shape(), &y.shape()})};
    if (shape == nullptr) {
        return std::nullopt;
    }
    const auto single {[shape](const auto& t) { return t.size() == 1 && t.rank() <= shape->size(); }};
    // Each operand's elements and its step through them: 0 for one element.
    const boolean*                   holds {c.origin()};
    const T*                         u {x.origin()};
    const T*                         v {y.origin()};
    std::vector<std::size_t>         lying {*shape};
    std::optional<std::vector<std::size_t>> order;
    const bool                       row_major {(single(c) || c.contiguous()) && (single(x) || x.contiguous()) && (single(y) || y.contiguous())};
    if (!row_major) {
        const auto laid_as {[&](const auto& t, auto& at) {
            if (single(t)) {
                return true;
            }
            const auto laid {t.laid_out()};
            if (!laid || (order && *order != laid->second)) {
                return false;
            }
            order = laid->second;
            lying = laid->first.shape();
            at    = laid->first.origin();
            return true;
        }};
        if (!laid_as(c, holds) || !laid_as(x, u) || !laid_as(y, v)) {
            return std::nullopt;
        }
    }
    const std::size_t    count {numerical::detail::element_count(lying.begin(), lying.end())};
    numerical::tensor<T> chosen {std::move(lying), chosen_in_order(holds, single(c) ? 0U : 1U, u, single(x) ? 0U : 1U, v, single(y) ? 0U : 1U, count)};
    if (order) {
        return chosen.transposed(numerical::detail::view_axes(*order));
    }
    return chosen;
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
                const tensor_type& q {std::get<tensor_type>(*y)};
                if (auto chosen {where_alike(*c, p, q)}) {
                    result.emplace(std::move(*chosen));
                    return;
                }
                const boolean* const holds {c->data().data()};
                const T* const       u {p.data().data()};
                const T* const       v {q.data().data()};
                const std::array     shapes {c->shape(), p.shape(), q.shape()};
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

namespace {
// The reduction `name` of a, of every element (axis -1) or along an axis (\ref module_reduce).
PyObject* reduced(std::string_view name, const any_tensor* a, Py_ssize_t axis)
{
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
} // namespace

// _reduce(op, a, axis) -> number or Tensor: op one of "sum", "mean", "max", "min", "any", "all"; axis -1 for every
// element (a number), else a non-negative axis (a Tensor without it). The library's reductions read a transpose where
// it lies; the truths and a mask's counts here alike: of every element in any order the same answer, along an axis
// each line of its elements as they lie along the matching axis, the result's axes put back in its order.
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
    if (name != "any" && name != "all" && !is_bool(*a)) { // the library's
        return reduced(name, a, axis);
    }
    std::optional<any_tensor> laid;
    std::vector<std::size_t>  back; // the result's axes, from the laid out one's
    try {
        std::visit([&](const auto& t) {
            if (axis >= 0 && static_cast<std::size_t>(axis) >= t.rank()) {
                return;
            }
            auto found {t.laid_out()};
            if (!found) {
                return;
            }
            auto& [elements, order] = *found;
            laid.emplace(std::move(elements));
            if (axis >= 0) {
                back = numerical::detail::reduced_axes(order, static_cast<std::size_t>(axis));
                axis = static_cast<Py_ssize_t>(std::ranges::find(order, static_cast<std::size_t>(axis)) - order.begin());
            }
        }, *a);
    }
    catch (...) {
        return set_cpp_error();
    }
    PyObject* result = reduced(name, laid ? &*laid : a, axis);
    if (result == nullptr || !laid || axis < 0) {
        return result;
    }
    const any_tensor* r = as_tensor(result); // along an axis of two or more: a Tensor
    PyObject*         put_back = nullptr;
    try {
        put_back = std::visit([&](const auto& t) { return wrap(any_tensor {t.transposed(back)}); }, *r);
    }
    catch (...) {
        put_back = set_cpp_error();
    }
    Py_DECREF(result);
    return put_back;
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
    // a complex array's function where this module has one (complex_named), else NotImplemented
    const bool complex_taken {complex_named<double>(name) != nullptr};
    if (!known || !borrow_operand(obj, x) || ((name == "abs" || (real_only && !complex_taken)) && (x.dtype == 1 || x.dtype == 3))) {
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
                        result.emplace(real_map<T>(name, data, x.shape));
                        return;
                    }
                }
                else {
                    if (real_only) { // the complex functions complex_named has, in one pass
                        result.emplace(d::parallel_map(data, x.shape, complex_named<real_of_t<T>>(name)));
                        return;
                    }
                }
                if (name == "exp") {
                    result.emplace(d::exp_of(data, x.shape));
                }
                else if (name == "log") {
                    result.emplace(d::parallel_map(data, x.shape, [](const T& v) { return d::elementary_log(v); }));
                }
                else if (name == "sqrt") {
                    result.emplace(d::parallel_map(data, x.shape, [](const T& v) { return d::root(v); }));
                }
                else if (name == "sin") {
                    result.emplace(d::sin_of(data, x.shape));
                }
                else if (name == "cos") {
                    result.emplace(d::cos_of(data, x.shape));
                }
                else if (name == "tan") {
                    result.emplace(d::parallel_map(data, x.shape, [](const T& v) { return d::elementary_tan(v); }));
                }
                else if (name == "tanh") {
                    result.emplace(d::tanh_of(data, x.shape));
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

namespace {


// A borrowed operand's elements, or a weak number's, and their shape: the number one element of no axis.
template <typename T>
struct binary_operand
{
    const T*                        data;
    const std::vector<std::size_t>& shape;
};

template <typename T>
std::size_t elements_of(const binary_operand<T>& operand)
{
    return numerical::detail::element_count(operand.shape.begin(), operand.shape.end());
}

// `x op y` of two operands of one dtype, by the functions the Tensor operators apply (binary_on): the bits of
// _binary of the same operands as Tensors.
template <typename T>
any_tensor borrowed_arithmetic(char op, const binary_operand<T>& x, const binary_operand<T>& y,
                               const numerical::detail::numpy_loops& loops)
{
    namespace d = numerical::detail;
    switch (op) {
    case '+': return any_tensor {d::broadcast_rows(x.data, x.shape, y.data, y.shape, typename d::arithmetic<T>::plus {})};
    case '-': return any_tensor {d::broadcast_rows(x.data, x.shape, y.data, y.shape, typename d::arithmetic<T>::minus {})};
    case '*':
        if constexpr (d::split_complex<T>) { // NumPy's loops (numpy_multiply's)
            const bool fused {d::numpy_loop_is_scalar(x.shape, y.shape) ? loops.scalar_products : loops.products};
            return any_tensor {d::broadcast_rows(x.data, x.shape, y.data, y.shape,
                                                 [fused](const T& p, const T& q) { return d::numpy_multiply(p, q, fused); })};
        }
        else {
            return any_tensor {d::broadcast_rows(x.data, x.shape, y.data, y.shape, typename d::arithmetic<T>::multiplies {})};
        }
    default:
        if constexpr (d::split_complex<T>) { // NumPy's loop (numpy_divide's)
            const bool fused {loops.quotients};
            return any_tensor {d::broadcast_rows(x.data, x.shape, y.data, y.shape,
                                                 [fused](const T& p, const T& q) { return d::numpy_divide(p, q, fused); })};
        }
        else {
            return any_tensor {d::broadcast_rows(x.data, x.shape, y.data, y.shape, std::divides<T> {})};
        }
    }
}

// The array operands of an element-wise pair, read where they lie, and the order they are read in: both row-major, or,
// where each array operand is a matrix whose elements run column-major (Fortran order, a transpose) of one shape and
// the other operand a number, both as they lie, `columns` set: the answer then computed on the elements in that order
// (of the reversed shape) and transposed back, as NumPy answers a Fortran-ordered pair in that order. A number's operand
// stays empty. False where an array operand is read neither way.
struct borrowed_pair
{
    std::unique_ptr<borrowed_operand> x;
    std::unique_ptr<borrowed_operand> y;
    bool                              columns {false};

    // An operand's shape as the computation reads it: the reversed one where `columns`.
    [[nodiscard]] std::vector<std::size_t> read_shape(const borrowed_operand& operand) const
    {
        return columns ? std::vector<std::size_t>(operand.shape.rbegin(), operand.shape.rend()) : operand.shape;
    }
};

bool borrow_pair(PyObject* x, PyObject* y, bool x_number, bool y_number, borrowed_pair& out)
{
    const auto borrow = [&](bool columns) {
        out.x = std::make_unique<borrowed_operand>();
        out.y = std::make_unique<borrowed_operand>();
        return (x_number || borrow_operand(x, *out.x, columns)) && (y_number || borrow_operand(y, *out.y, columns));
    };
    if (borrow(true)) {
        const bool x_columns {!x_number && out.x->transposed};
        const bool y_columns {!y_number && out.y->transposed};
        if (!x_columns && !y_columns) {
            return true; // both row-major (or one a number)
        }
        if ((x_number || x_columns) && (y_number || y_columns) && (x_number || y_number || out.x->shape == out.y->shape)) {
            out.columns = true;
            return true;
        }
    }
    out.columns = false;
    return borrow(false); // a column-major matrix beside a row-major array: both read in C order (or copied)
}

} // namespace

// _borrowed_binary(op, x, y) -> Tensor or NotImplemented: x op y for op one of + - * /, each operand a Tensor or a NumPy
// array read where it lies (borrow_operand), or one of them a Python float or int (weak: of the other's dtype), the two
// of one float or complex dtype; by the functions _binary applies to Tensors, so its answer to the bit and its errors.
// NotImplemented for any other op or operand, or two dtypes, which the converting path then promotes. A NumPy array was
// copied into a Tensor first: 10^5 doubles added, M1 Pro, 93 µs where two Tensors take 25.
PyObject* module_borrowed_binary(PyObject* /*self*/, PyObject* args)
{
    const char* op = nullptr;
    PyObject*   x  = nullptr;
    PyObject*   y  = nullptr;
    if (PyArg_ParseTuple(args, "sOO", &op, &x, &y) == 0) {
        return nullptr;
    }
    const std::string_view name {op};
    if (name.size() != 1 || std::string_view {"+-*/"}.find(name[0]) == std::string_view::npos) {
        return Py_NewRef(Py_NotImplemented);
    }
    double           x_value {0.0};
    double           y_value {0.0};
    const bool       x_number {weak_number(x, x_value)};
    const bool       y_number {!x_number && weak_number(y, y_value)};
    borrowed_pair pair;
    if (!borrow_pair(x, y, x_number, y_number, pair) || (!x_number && !y_number && pair.x->dtype != pair.y->dtype)) {
        return Py_NewRef(Py_NotImplemented);
    }
    const borrowed_operand&              a {*pair.x};
    const borrowed_operand&              b {*pair.y};
    const std::size_t                    dtype {x_number ? b.dtype : a.dtype};
    const numerical::detail::numpy_loops loops {numpy_loops_beside()}; // read while the GIL is held
    const std::vector<std::size_t>       no_axes;
    const std::vector<std::size_t>       x_shape {x_number ? no_axes : pair.read_shape(a)};
    const std::vector<std::size_t>       y_shape {y_number ? no_axes : pair.read_shape(b)};
    try {
        std::optional<any_tensor> result;
        const auto                apply = [&]<typename T>(std::type_identity<T>) {
            const T                 x_scalar {static_cast<T>(x_value)}; // asarray(number, dtype)'s element
            const T                 y_scalar {static_cast<T>(y_value)};
            const binary_operand<T> left {x_number ? &x_scalar : static_cast<const T*>(a.data), x_shape};
            const binary_operand<T> right {y_number ? &y_scalar : static_cast<const T*>(b.data), y_shape};
            const sb::scoped_gil_release unlocked {releases_gil(std::max(elements_of(left), elements_of(right)))};
            any_tensor answer {borrowed_arithmetic(name[0], left, right, loops)};
            if (pair.columns) { // the elements as the operands lie: transposed back to their shape
                std::visit([](auto& t) { t = t.transposed(); }, answer);
            }
            result.emplace(std::move(answer));
        };
        switch (dtype) {
        case 0: apply(std::type_identity<double> {}); break;
        case 1: apply(std::type_identity<std::complex<double>> {}); break;
        case 2: apply(std::type_identity<float> {}); break;
        default: apply(std::type_identity<std::complex<float>> {}); break;
        }
        return wrap(std::move(*result));
    }
    catch (...) {
        return set_cpp_error();
    }
}


namespace {

// A condition read where it lies: a bool Tensor's bytes in C order, or a C-contiguous NumPy bool array's. `owner` (a
// new reference: the Tensor, or the capsule, which holds the array) keeps them alive.
struct borrowed_condition
{
    PyObject*                owner {nullptr};
    const unsigned char*     data {nullptr};
    std::vector<std::size_t> shape;

    borrowed_condition() = default;
    borrowed_condition(const borrowed_condition&) = delete;
    borrowed_condition& operator=(const borrowed_condition&) = delete;
    ~borrowed_condition() { Py_XDECREF(owner); }
};

// Whether `obj` is a bool Tensor or a C-contiguous NumPy bool array, read into `out`; false, no error set, otherwise.
bool borrow_condition(PyObject* obj, borrowed_condition& out)
{
    if (PyObject_TypeCheck(obj, reinterpret_cast<PyTypeObject*>(sb::class_type<any_tensor>())) != 0) {
        const any_tensor* v {reinterpret_cast<sb::wrapper<any_tensor>*>(obj)->held};
        if (v == nullptr || !is_bool(*v)) {
            return false;
        }
        out.owner = Py_NewRef(obj);
        out.data  = static_cast<const unsigned char*>(data_of(*v)); // a boolean is its byte
        out.shape = shape_of(*v);
        return true;
    }
    PyObject* capsule = PyObject_GetAttr(obj, interned_name<array_struct_name>()); // new ref
    if (capsule == nullptr) {
        PyErr_Clear();
        return false;
    }
    const auto* view {static_cast<const array_struct*>(PyCapsule_GetPointer(capsule, nullptr))};
    if (view == nullptr) {
        PyErr_Clear();
    }
    if (view == nullptr || view->two != 2 || view->data == nullptr || view->typekind != 'b' || view->itemsize != 1
        || (view->flags & array_c_contiguous) == 0) {
        Py_DECREF(capsule);
        return false;
    }
    out.owner = capsule;
    out.data  = static_cast<const unsigned char*>(view->data);
    for (int axis = 0; axis < view->nd; ++axis) {
        out.shape.push_back(static_cast<std::size_t>(view->shape[axis]));
    }
    return true;
}

} // namespace

// _borrowed_where(condition, x, y) -> Tensor or NotImplemented: where of a bool Tensor or C-contiguous NumPy bool array
// and two operands, each a Tensor or a NumPy array read where it lies (borrow_operand) or one of them a Python float or
// int (weak: of the other's dtype), of one float or complex dtype; each element chosen as _where chooses it.
// NotImplemented for any other operand, which the converting path then reads. The arrays were copied into Tensors first:
// 10^5 doubles, M1 Pro, 252 µs where NumPy takes 76.
PyObject* module_borrowed_where(PyObject* /*self*/, PyObject* args)
{
    PyObject* c_obj = nullptr;
    PyObject* x     = nullptr;
    PyObject* y     = nullptr;
    if (PyArg_ParseTuple(args, "OOO", &c_obj, &x, &y) == 0) {
        return nullptr;
    }
    borrowed_condition c;
    double             x_value {0.0};
    double             y_value {0.0};
    const bool         x_number {weak_number(x, x_value)};
    const bool         y_number {!x_number && weak_number(y, y_value)};
    borrowed_operand   a;
    borrowed_operand   b;
    if (!borrow_condition(c_obj, c) || (!x_number && !borrow_operand(x, a)) || (!y_number && !borrow_operand(y, b))
        || (!x_number && !y_number && a.dtype != b.dtype)) {
        return Py_NewRef(Py_NotImplemented);
    }
    const std::size_t              dtype {x_number ? b.dtype : a.dtype};
    const std::vector<std::size_t> no_axes;
    try {
        std::optional<any_tensor> result;
        const auto                choose = [&]<typename T>(std::type_identity<T>) {
            const T                         x_scalar {static_cast<T>(x_value)}; // asarray(number, dtype)'s element
            const T                         y_scalar {static_cast<T>(y_value)};
            const T* const                  u {x_number ? &x_scalar : static_cast<const T*>(a.data)};
            const T* const                  v {y_number ? &y_scalar : static_cast<const T*>(b.data)};
            const std::vector<std::size_t>& x_shape {x_number ? no_axes : a.shape};
            const std::vector<std::size_t>& y_shape {y_number ? no_axes : b.shape};
            const auto elements {[](const std::vector<std::size_t>& shape) { return numerical::detail::element_count(shape.begin(), shape.end()); }};
            const sb::scoped_gil_release unlocked {releases_gil(std::max({elements(c.shape), elements(x_shape), elements(y_shape)}))};
            if (const std::vector<std::size_t>* shape {alike_shape({&c.shape, &x_shape, &y_shape})}) {
                const auto step {[shape](const std::vector<std::size_t>& each) { return each == *shape ? std::size_t {1} : std::size_t {0}; }};
                result.emplace(numerical::tensor<T> {*shape, chosen_in_order(c.data, step(c.shape), u, step(x_shape), v, step(y_shape), elements(*shape))});
                return;
            }
            const std::array shapes {c.shape, x_shape, y_shape};
            result.emplace(broadcast_elements<T>(shapes, [&](const std::array<std::size_t, 3>& at) {
                return holds_at(c.data[at[0]]) ? u[at[1]] : v[at[2]];
            }));
        };
        switch (dtype) {
        case 0: choose(std::type_identity<double> {}); break;
        case 1: choose(std::type_identity<std::complex<double>> {}); break;
        case 2: choose(std::type_identity<float> {}); break;
        default: choose(std::type_identity<std::complex<float>> {}); break;
        }
        return wrap(std::move(*result));
    }
    catch (...) {
        return set_cpp_error();
    }
}


// _fast_binary(op, x, y) -> Tensor or NotImplemented: x op y, op one of + - * / ^ | %, of two Tensors of one dtype other
// than bool (_binary's), or of a Tensor of a float or complex dtype and a Python float or int on either side, the
// number weak: a rank-0 tensor of the Tensor's dtype (asarray(number, dtype)'s element) and binary_on, as _binary
// computes them after the operator read the number through asarray. NotImplemented for anything else (two dtypes, a
// bool Tensor, an int64 Tensor beside a number, an int no double holds), which the operator then reads as before. Of a
// Tensor of 16 doubles, t * 2.0 took 1.0 µs, most of it asarray, and t + t 0.48, half of it the operator's checks.
PyObject* module_fast_binary(PyObject* /*self*/, PyObject* args)
{
    const char* op = nullptr;
    PyObject*   x  = nullptr;
    PyObject*   y  = nullptr;
    if (PyArg_ParseTuple(args, "sOO", &op, &x, &y) == 0) {
        return nullptr;
    }
    const std::string_view name {op};
    if (name.size() != 1 || std::string_view {"+-*/^|%"}.find(name[0]) == std::string_view::npos) {
        return Py_NewRef(Py_NotImplemented);
    }
    auto* const       type {reinterpret_cast<PyTypeObject*>(sb::class_type<any_tensor>())};
    const auto        held {[type](PyObject* obj) -> const any_tensor* {
        return PyObject_TypeCheck(obj, type) != 0 ? reinterpret_cast<sb::wrapper<any_tensor>*>(obj)->held : nullptr;
    }};
    const any_tensor* a {held(x)};
    const any_tensor* b {held(y)};
    double            value {0.0};
    const bool        pair {a != nullptr && b != nullptr && a->index() == b->index() && !is_bool(*a)};
    const bool        x_number {!pair && b != nullptr && weak_number(x, value)};
    const bool        y_number {!pair && !x_number && a != nullptr && weak_number(y, value)};
    const any_tensor* t {x_number ? b : a};
    if (!pair && ((!x_number && !y_number) || t->index() > 3)) { // a float or complex Tensor beside a number
        return Py_NewRef(Py_NotImplemented);
    }
    try {
        if (pair) {
            return visit_arithmetic([&](const auto& p) { return binary_on(name[0], p, std::get<std::decay_t<decltype(p)>>(*b)); }, *a);
        }
        return visit_numeric([&](const auto& tensor) {
            using T = element_of_t<std::decay_t<decltype(tensor)>>;
            const numerical::tensor<T> number {{}, numerical::buffer<T> {static_cast<T>(value)}};
            return x_number ? binary_on(name[0], number, tensor) : binary_on(name[0], tensor, number);
        }, *t);
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _fast_compare(op, x, y) -> bool Tensor or NotImplemented: _compare of the Tensor x and y, a Tensor of x's dtype or,
// x of a float or complex dtype, a Python float or int (weak: a rank-0 tensor of x's dtype, asarray(number, dtype)'s
// element), its answer and errors; NotImplemented for anything else, which the operator then promotes as before. Of a
// Tensor of 16 doubles, t > 0.5 took 1.3 µs, most of it the promotion's Python.
PyObject* module_fast_compare(PyObject* self, PyObject* args)
{
    const char* op = nullptr;
    PyObject*   x  = nullptr;
    PyObject*   y  = nullptr;
    if (PyArg_ParseTuple(args, "sOO", &op, &x, &y) == 0) {
        return nullptr;
    }
    auto* const type {reinterpret_cast<PyTypeObject*>(sb::class_type<any_tensor>())};
    if (PyObject_TypeCheck(x, type) == 0) {
        return Py_NewRef(Py_NotImplemented);
    }
    const any_tensor* a {reinterpret_cast<sb::wrapper<any_tensor>*>(x)->held};
    double            value {0.0};
    if (a != nullptr && PyObject_TypeCheck(y, type) != 0) {
        const any_tensor* b {reinterpret_cast<sb::wrapper<any_tensor>*>(y)->held};
        return b != nullptr && b->index() == a->index() ? module_compare(self, args) : Py_NewRef(Py_NotImplemented);
    }
    if (a == nullptr || a->index() > 3 || !weak_number(y, value)) {
        return Py_NewRef(Py_NotImplemented);
    }
    try {
        PyObject* number {visit_numeric([&](const auto& t) {
            using T = element_of_t<std::decay_t<decltype(t)>>;
            return wrap(any_tensor {numerical::tensor<T> {{}, numerical::buffer<T> {static_cast<T>(value)}}});
        }, *a)};
        if (number == nullptr) {
            return nullptr;
        }
        PyObject* const pair {Py_BuildValue("(sON)", op, x, number)}; // steals number
        if (pair == nullptr) {
            return nullptr;
        }
        PyObject* const answer {module_compare(self, pair)};
        Py_DECREF(pair);
        return answer;
    }
    catch (...) {
        return set_cpp_error();
    }
}


// _gather_index(t, index, flat=False) -> Tensor or NotImplemented: t[index] (with `flat`, of t's elements in C order, as
// take(t, index) reads them) for the two advanced indexes most taken, read where they lie
// and gathered in C: a bool Tensor or C-contiguous NumPy bool array over t's leading axes (their true elements' blocks
// in C order, of shape (count,) and the axes after), or an int64 Tensor or C-contiguous NumPy integer array of rank 1 or
// more along the first axis (the rows at those indices, a negative one counted from the end, of the index's shape and
// the axes after). NotImplemented for anything else, a mask of another shape or an index out of the axis included,
// which _getitem_advanced then takes (and refuses in NumPy's words). It took 5 to 9 µs of Python on a small tensor.
PyObject* module_gather_index(PyObject* /*self*/, PyObject* args)
{
    PyObject* x     = nullptr;
    PyObject* index = nullptr;
    int       flat  = 0;
    if (PyArg_ParseTuple(args, "OO|p", &x, &index, &flat) == 0) {
        return nullptr;
    }
    if (PyObject_TypeCheck(x, reinterpret_cast<PyTypeObject*>(sb::class_type<any_tensor>())) == 0) {
        return Py_NewRef(Py_NotImplemented);
    }
    const any_tensor* v {reinterpret_cast<sb::wrapper<any_tensor>*>(x)->held};
    if (v == nullptr) {
        return Py_NewRef(Py_NotImplemented);
    }
    const std::vector<std::size_t> shape {flat != 0 ? std::vector<std::size_t> {count_of(*v)} : shape_of(*v)}; // its elements in C order
    const auto                     product {[](auto first, auto last) { return std::accumulate(first, last, std::size_t {1}, std::multiplies<> {}); }};
    std::vector<std::size_t>       rows;        // the first element of each block taken, in blocks
    std::vector<std::size_t>       out_shape;
    std::size_t                    leading {0}; // the axes the index takes
    borrowed_condition             mask;
    borrowed_indices               integers;
    bool                           by_integers {false}; // gathered in one pass from `integers`, `rows` unused
    if (borrow_condition(index, mask)) {
        leading = mask.shape.size();
        if (leading == 0 || leading > shape.size() || !std::equal(mask.shape.begin(), mask.shape.end(), shape.begin())) {
            return Py_NewRef(Py_NotImplemented);
        }
        const std::size_t cells {product(mask.shape.begin(), mask.shape.end())};
        for (std::size_t i = 0; i < cells; ++i) {
            if (mask.data[i] != 0) {
                rows.push_back(i);
            }
        }
        out_shape.push_back(rows.size());
    }
    else if (borrow_indices(index, integers)) {
        if (integers.shape.empty() || shape.empty()) {
            return Py_NewRef(Py_NotImplemented);
        }
        leading     = 1;
        out_shape   = integers.shape;
        by_integers = true;
    }
    else {
        return Py_NewRef(Py_NotImplemented);
    }
    out_shape.insert(out_shape.end(), shape.begin() + static_cast<std::ptrdiff_t>(leading), shape.end());
    const std::size_t inner {product(shape.begin() + static_cast<std::ptrdiff_t>(leading), shape.end())};
    try {
        if (by_integers) {
            // The indices read and their blocks copied in one pass, an index outside the axis ending it: a pass that
            // listed the rows first, then copied each through copy_n (a call an element for a flattened take), took
            // twice NumPy's time (take of 4 · 10⁵ random indices: 1.56 ms, NumPy 0.75).
            const std::size_t count {product(integers.shape.begin(), integers.shape.end())};
            const auto        extent {static_cast<std::uint64_t>(shape[0])};
            bool              inside {true};
            PyObject* const   answer {std::visit([&](const auto& t) -> PyObject* {
                using T = element_of_t<std::decay_t<decltype(t)>>;
                const T* const       source {static_cast<const T*>(data_of(*v))};
                numerical::buffer<T> out(count * inner);
                T* const             to {out.data()};
                const auto           gather = [&]<typename I>(const I* indices) {
                    for (std::size_t i = 0; i < count; ++i) {
                        const auto at {static_cast<std::int64_t>(indices[i])};
                        // a negative index counted from the end; either side outside the axis wraps past `extent`
                        const auto row {static_cast<std::uint64_t>(at) + (at < 0 ? extent : 0)};
                        if (row >= extent) { // refused by the general path, in NumPy's words
                            inside = false;
                            return;
                        }
                        if (inner == 1) {
                            to[i] = source[row];
                        }
                        else {
                            std::copy_n(source + (row * inner), inner, to + (i * inner));
                        }
                    }
                };
                if (integers.bytes == 8) {
                    gather(static_cast<const std::int64_t*>(integers.data));
                }
                else {
                    gather(static_cast<const std::int32_t*>(integers.data));
                }
                return inside ? wrap(any_tensor {numerical::tensor<T> {out_shape, std::move(out)}}) : nullptr;
            }, *v)};
            return inside ? answer : Py_NewRef(Py_NotImplemented);
        }
        return std::visit([&](const auto& t) {
            using T = element_of_t<std::decay_t<decltype(t)>>;
            const T* const       source {static_cast<const T*>(data_of(*v))};
            numerical::buffer<T> out(rows.size() * inner);
            for (std::size_t r = 0; r < rows.size(); ++r) {
                std::copy_n(source + (rows[r] * inner), inner, out.data() + (r * inner));
            }
            return wrap(any_tensor {numerical::tensor<T> {out_shape, std::move(out)}});
        }, *v);
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _borrowed_compare(op, x, y) -> bool Tensor or NotImplemented: x op y for op one of < <= == != > >=, each operand a
// Tensor or a NumPy array read where they lie (borrow_operand), or one of them a Python float or int (weak: of the
// other's dtype), the two of one float or complex dtype (a complex one compared by == and != alone); by the relations
// _compare applies, through broadcast_rows, so its answer and its errors. NotImplemented for anything else, which the
// converting path then takes. A NumPy array was copied into a Tensor first: greater of a 9 × 9 array and 0, M1 Pro,
// 1.4 µs where NumPy takes 0.5.
PyObject* module_borrowed_compare(PyObject* /*self*/, PyObject* args)
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
        return Py_NewRef(Py_NotImplemented);
    }
    const auto       which {static_cast<std::size_t>(found - names.begin())};
    double           x_value {0.0};
    double           y_value {0.0};
    const bool       x_number {weak_number(x, x_value)};
    const bool       y_number {!x_number && weak_number(y, y_value)};
    borrowed_pair pair;
    if (!borrow_pair(x, y, x_number, y_number, pair) || (!x_number && !y_number && pair.x->dtype != pair.y->dtype)) {
        return Py_NewRef(Py_NotImplemented);
    }
    const borrowed_operand& a {*pair.x};
    const borrowed_operand& b {*pair.y};
    const std::size_t       dtype {x_number ? b.dtype : a.dtype};
    if ((dtype == 1 || dtype == 3) && which != 2 && which != 3) { // no order: the converting path refuses it
        return Py_NewRef(Py_NotImplemented);
    }
    const std::vector<std::size_t> no_axes;
    try {
        std::optional<bool_tensor> result;
        const auto                 compare = [&]<typename T>(std::type_identity<T>) {
            namespace d = numerical::detail;
            const T                         x_scalar {static_cast<T>(x_value)}; // asarray(number, dtype)'s element
            const T                         y_scalar {static_cast<T>(y_value)};
            const T* const                  u {x_number ? &x_scalar : static_cast<const T*>(a.data)};
            const T* const                  v {y_number ? &y_scalar : static_cast<const T*>(b.data)};
            const std::vector<std::size_t>  x_shape {x_number ? no_axes : pair.read_shape(a)};
            const std::vector<std::size_t>  y_shape {y_number ? no_axes : pair.read_shape(b)};
            const auto elements {[](const std::vector<std::size_t>& shape) { return d::element_count(shape.begin(), shape.end()); }};
            const sb::scoped_gil_release unlocked {releases_gil(std::max(elements(x_shape), elements(y_shape)))};
            const auto                   compared {[&](const auto& relation) {
                result.emplace(d::broadcast_rows(u, x_shape, v, y_shape, [&relation](const T& p, const T& q) { return boolean {relation(p, q)}; }));
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
        };
        switch (dtype) {
        case 0: compare(std::type_identity<double> {}); break;
        case 1: compare(std::type_identity<std::complex<double>> {}); break;
        case 2: compare(std::type_identity<float> {}); break;
        default: compare(std::type_identity<std::complex<float>> {}); break;
        }
        if (pair.columns) { // the elements as the operands lie: transposed back to their shape
            *result = result->transposed();
        }
        return wrap(any_tensor {std::move(*result)});
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _isin_table(a, b, invert) -> bool Tensor or NotImplemented: whether each element of the int64 Tensor `a` (in C order)
// is among those of the int64 Tensor `b`, by NumPy's table method: a flag for each value from b's least to its largest,
// where that range is no more than 6 times the two counts together (NumPy's memory bound for its default), `invert` the
// opposite; NotImplemented otherwise (another dtype, an empty b, a wider range), which the sorting method then answers
// alike. Sorting took 2.5 ms for 10^5 elements among 10^4 in a range of 2.5 × 10^4, NumPy's table 0.27.
PyObject* module_isin_table(PyObject* /*self*/, PyObject* args)
{
    PyObject* x      = nullptr;
    PyObject* y      = nullptr;
    int       invert = 0;
    if (PyArg_ParseTuple(args, "OOp", &x, &y, &invert) == 0) {
        return nullptr;
    }
    auto* const       type {reinterpret_cast<PyTypeObject*>(sb::class_type<any_tensor>())};
    const auto        held {[type](PyObject* obj) -> const any_tensor* {
        return PyObject_TypeCheck(obj, type) != 0 ? reinterpret_cast<sb::wrapper<any_tensor>*>(obj)->held : nullptr;
    }};
    const any_tensor* a {held(x)};
    const any_tensor* b {held(y)};
    if (a == nullptr || b == nullptr || !is_int(*a) || !is_int(*b) || count_of(*b) == 0) {
        return Py_NewRef(Py_NotImplemented);
    }
    const auto* const elements {static_cast<const std::int64_t*>(data_of(*a))};
    const auto* const tests {static_cast<const std::int64_t*>(data_of(*b))};
    const std::size_t count {count_of(*a)};
    const std::size_t test_count {count_of(*b)};
    const auto [least, largest] {std::minmax_element(tests, tests + test_count)};
    // the range in unsigned arithmetic, exact for any two int64 (NumPy refuses a range past INT64_MAX: its sort then)
    const std::uint64_t range {static_cast<std::uint64_t>(*largest) - static_cast<std::uint64_t>(*least)};
    if (range > std::uint64_t {INT64_MAX} || range > 6 * (static_cast<std::uint64_t>(count) + test_count)) {
        return Py_NewRef(Py_NotImplemented);
    }
    try {
        std::vector<unsigned char> table(static_cast<std::size_t>(range) + 1, 0);
        const std::int64_t         low {*least};
        const std::int64_t         high {*largest};
        for (std::size_t j = 0; j < test_count; ++j) {
            table[static_cast<std::size_t>(static_cast<std::uint64_t>(tests[j]) - static_cast<std::uint64_t>(low))] = 1;
        }
        numerical::buffer<boolean> out(count);
        const bool                 flip {invert != 0};
        for (std::size_t i = 0; i < count; ++i) {
            const std::int64_t value {elements[i]};
            const bool inside {value >= low && value <= high
                               && table[static_cast<std::size_t>(static_cast<std::uint64_t>(value) - static_cast<std::uint64_t>(low))] != 0};
            out[i] = boolean {inside != flip};
        }
        return wrap(any_tensor {bool_tensor {{count}, std::move(out)}});
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _isin_sorted(a, b, invert) -> bool Tensor or NotImplemented: whether each element of the Tensor `a` (in C order) equals
// one of the Tensor `b`'s, the two of one dtype, float64, float32 or int64: b's elements sorted (its NaNs left out, as
// no NaN equals one) and each of a's searched among them, so that -0 finds +0 and a NaN finds nothing, as NumPy's
// sorting method answers; `invert` the opposite. NotImplemented for anything else, which isin then answers in Python.
// The Python steps (unique, the promotion, searchsorted, the comparison) took 14 µs on a handful of elements, NumPy 3.
PyObject* module_isin_sorted(PyObject* /*self*/, PyObject* args)
{
    PyObject* x      = nullptr;
    PyObject* y      = nullptr;
    int       invert = 0;
    if (PyArg_ParseTuple(args, "OOp", &x, &y, &invert) == 0) {
        return nullptr;
    }
    auto* const       type {reinterpret_cast<PyTypeObject*>(sb::class_type<any_tensor>())};
    const auto        held {[type](PyObject* obj) -> const any_tensor* {
        return PyObject_TypeCheck(obj, type) != 0 ? reinterpret_cast<sb::wrapper<any_tensor>*>(obj)->held : nullptr;
    }};
    const any_tensor* a {held(x)};
    const any_tensor* b {held(y)};
    if (a == nullptr || b == nullptr || a->index() != b->index() || (a->index() != 0 && a->index() != 2 && a->index() != int_index)) {
        return Py_NewRef(Py_NotImplemented);
    }
    try {
        const auto searched = [&]<typename T>(std::type_identity<T>) {
            const T* const    elements {static_cast<const T*>(data_of(*a))};
            const T* const    tests {static_cast<const T*>(data_of(*b))};
            const std::size_t count {count_of(*a)};
            std::vector<T>    sorted;
            sorted.reserve(count_of(*b));
            std::copy_if(tests, tests + count_of(*b), std::back_inserter(sorted), [](T v) { return v == v; }); // no NaN
            std::ranges::sort(sorted);
            numerical::buffer<boolean> out(count);
            const bool                 flip {invert != 0};
            for (std::size_t i = 0; i < count; ++i) {
                const T value {elements[i]};
                // a NaN finds nothing: binary_search would take it for every element, none being less or greater
                out[i] = boolean {(value == value && std::ranges::binary_search(sorted, value)) != flip};
            }
            return wrap(any_tensor {bool_tensor {{count}, std::move(out)}});
        };
        switch (a->index()) {
        case 0: return searched(std::type_identity<double> {});
        case 2: return searched(std::type_identity<float> {});
        default: return searched(std::type_identity<std::int64_t> {});
        }
    }
    catch (...) {
        return set_cpp_error();
    }
}
} // namespace numerical_binding
