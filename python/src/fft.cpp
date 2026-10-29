// NumericAL — Python extension module: the Fourier, cosine and sine transforms (their helpers in binding.hpp).

#include "binding.hpp"

namespace numerical_binding {

// eigvalsh(a, upper) -> Tensor, as numpy.linalg.eigvalsh: the eigenvalues alone, real and ascending.
// _fft(a, n, axis, norm, inverse) -> Tensor: numpy.fft.fft (or ifft) of n points along axis, norm 0 backward, 1 ortho,
// 2 forward; float64 to complex128, float32 to complex64, complex as itself. The caller (numerical.fft) checks n and
// axis as NumPy does.
any_tensor fft_axis(const any_tensor& v, std::size_t n, std::size_t axis, int norm, bool inverse)
{
    const auto scaling {static_cast<numerical::fft_norm>(norm)};
    return std::visit([&](const auto& t) -> any_tensor {
        const sb::scoped_gil_release unlocked {releases_gil(t.data().size() + n)};
        const auto                   index {static_cast<std::ptrdiff_t>(axis)};
        return any_tensor {inverse ? numerical::ifft(t, n, index, scaling) : numerical::fft(t, n, index, scaling)};
    }, v);
}

// _fftshift(a, axes, inverse) -> Tensor: numpy.fft.fftshift (or ifftshift) along each of axes (from 0), in one copy.
PyObject* module_fftshift(PyObject* /*self*/, PyObject* args)
{
    PyObject* obj     = nullptr;
    PyObject* axes    = nullptr;
    int       inverse = 0;
    if (PyArg_ParseTuple(args, "OOp", &obj, &axes, &inverse) == 0) {
        return nullptr;
    }
    const any_tensor* h = as_tensor(obj);
    if (h == nullptr) {
        return nullptr;
    }
    std::vector<Py_ssize_t> raw;
    if (!read_integers(axes, raw)) {
        return nullptr;
    }
    std::vector<std::ptrdiff_t> axis_v(raw.begin(), raw.end());
    try {
        return std::visit([&](const auto& t) { return wrap(any_tensor {numerical::fftshift(t, axis_v, inverse != 0)}); }, *h);
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _rfft(a, n, axis, norm, hermitian) -> Tensor: numpy.fft.rfft (or, hermitian, ihfft) of a real tensor along axis.
any_tensor rfft_axis(const any_tensor& v, std::size_t n, std::size_t axis, int norm, bool hermitian)
{
    const auto scaling {static_cast<numerical::fft_norm>(norm)};
    return std::visit([&](const auto& t) -> any_tensor {
        using T = std::remove_cvref_t<decltype(t.data()[0])>;
        if constexpr (std::is_floating_point_v<T>) {
            const sb::scoped_gil_release unlocked {releases_gil(t.data().size() + n)};
            const auto                   index {static_cast<std::ptrdiff_t>(axis)};
            return any_tensor {hermitian ? numerical::ihfft(t, n, index, scaling) : numerical::rfft(t, n, index, scaling)};
        }
        else {
            throw std::invalid_argument("rfft takes a real tensor");
        }
    }, v);
}

// _trig(a, sine, type, n, axis, norm, orthogonalize, inverse) -> Tensor: scipy.fft's dct, idct, dst or idst along axis
// (from 0), orthogonalize -1 for its default; a complex tensor's real and imaginary parts each transformed, as SciPy's.
any_tensor trig_axis(const any_tensor& v, bool sine, int type, std::size_t n, std::size_t axis, int norm, int orthogonalize, bool inverse)
{
    using numerical::detail::trig_family;
    const auto                scaling {static_cast<numerical::fft_norm>(norm)};
    const std::optional<bool> weighted {orthogonalize < 0 ? std::nullopt : std::optional<bool> {orthogonalize != 0}};
    const trig_family         family {sine ? trig_family::sine : trig_family::cosine};
    const auto                index {static_cast<std::ptrdiff_t>(axis)};
    return std::visit([&](const auto& t) -> any_tensor {
        using T = std::remove_cvref_t<decltype(t.data()[0])>;
        const sb::scoped_gil_release unlocked {releases_gil(t.data().size() + n)};
        if constexpr (std::is_floating_point_v<T>) {
            return any_tensor {numerical::detail::trig_transform(t, family, type, n, index, scaling, weighted, inverse)};
        }
        else {
            using R = typename T::value_type;
            const numerical::tensor<R> re {numerical::detail::trig_transform(numerical::detail::parallel_map(t, [](const T& z) { return z.real(); }),
                                                                             family, type, n, index, scaling, weighted, inverse)};
            const numerical::tensor<R> im {numerical::detail::trig_transform(numerical::detail::parallel_map(t, [](const T& z) { return z.imag(); }),
                                                                             family, type, n, index, scaling, weighted, inverse)};
            numerical::buffer<T>       joined(re.data().size());
            for (std::size_t e {0}; e < joined.size(); ++e) {
                joined[e] = T {re.data()[e], im.data()[e]};
            }
            return any_tensor {numerical::tensor<T> {re.shape(), std::move(joined)}};
        }
    }, v);
}

// _irfft(a, n, axis, norm, hermitian) -> Tensor: numpy.fft.irfft (or, hermitian, hfft) of a complex tensor along axis.
any_tensor irfft_axis(const any_tensor& v, std::size_t n, std::size_t axis, int norm, bool hermitian)
{
    const auto scaling {static_cast<numerical::fft_norm>(norm)};
    return std::visit([&](const auto& t) -> any_tensor {
        using T = std::remove_cvref_t<decltype(t.data()[0])>;
        if constexpr (numerical::detail::split_complex<T>) {
            const sb::scoped_gil_release unlocked {releases_gil(t.data().size() + n)};
            const auto                   index {static_cast<std::ptrdiff_t>(axis)};
            return any_tensor {hermitian ? numerical::hfft(t, n, index, scaling) : numerical::irfft(t, n, index, scaling)};
        }
        else {
            throw std::invalid_argument("irfft takes a complex tensor");
        }
    }, v);
}

// _borrowed_fft(kind, a, n, axis, norm, inverse) -> Tensor or NotImplemented: a one-axis transform of a Tensor or NumPy
// array read where it lies (borrow_operand), kind 0 fft (or, inverse, ifft) of real or complex elements, 1 rfft of
// real ones, 2 irfft of complex ones; the bits of _fft, _rfft and _irfft, which read the elements from a copy.
// NotImplemented for an operand not so read, or rfft of complex elements and irfft of real ones, which the caller
// (numerical.fft) refuses or widens. A NumPy array was copied first: 1000 lines of 1024 complex128, 4.1 ms, M1 Pro, 2.9
// read where they lie.
PyObject* module_borrowed_fft(PyObject* /*self*/, PyObject* args)
{
    int        kind    = 0;
    PyObject*  obj     = nullptr;
    Py_ssize_t n       = 0;
    Py_ssize_t axis    = 0;
    int        norm    = 0;
    int        inverse = 0;
    if (PyArg_ParseTuple(args, "iOnnip", &kind, &obj, &n, &axis, &norm, &inverse) == 0) {
        return nullptr;
    }
    borrowed_operand x;
    if (!borrow_operand(obj, x)) {
        return Py_NewRef(Py_NotImplemented);
    }
    const bool complex_elements {x.dtype == 1 || x.dtype == 3};
    if ((kind == 1 && complex_elements) || (kind == 2 && !complex_elements)) {
        return Py_NewRef(Py_NotImplemented);
    }
    try {
        std::optional<any_tensor> result;
        {
            std::size_t count {1};
            for (const std::size_t extent : x.shape) {
                count *= extent;
            }
            const sb::scoped_gil_release unlocked {releases_gil(count + static_cast<std::size_t>(n))};
            const auto                   points {static_cast<std::size_t>(n)};
            const auto                   scaling {static_cast<numerical::fft_norm>(norm)};
            const auto transform = [&]<typename T>(std::type_identity<T>) {
                const numerical::detail::fft_source<T> source {static_cast<const T*>(x.data), x.shape};
                if constexpr (numerical::detail::split_complex<T>) {
                    if (kind == 2) {
                        result.emplace(numerical::detail::irfft_of(source, points, axis, scaling));
                    }
                    else {
                        result.emplace(numerical::detail::transform_axis(source, points, axis, scaling, inverse != 0));
                    }
                }
                else if (kind == 1) {
                    result.emplace(numerical::detail::rfft_of(source, points, axis, scaling));
                }
                else {
                    result.emplace(numerical::detail::real_transform_of(source, points, axis, scaling, inverse != 0));
                }
            };
            switch (x.dtype) {
            case 0: transform(std::type_identity<double> {}); break;
            case 1: transform(std::type_identity<std::complex<double>> {}); break;
            case 2: transform(std::type_identity<float> {}); break;
            default: transform(std::type_identity<std::complex<float>> {}); break;
            }
        }
        return wrap(std::move(*result));
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _borrowed_trig(a, sine, type, n, axis, norm, orthogonalize, inverse) -> Tensor or NotImplemented: _trig of a Tensor or
// NumPy array of real elements read where it lies (borrow_operand), its bits. NotImplemented for an operand not so read,
// or of complex elements, which the caller converts.
PyObject* module_borrowed_trig(PyObject* /*self*/, PyObject* args)
{
    PyObject*  obj           = nullptr;
    int        sine          = 0;
    int        type          = 0;
    Py_ssize_t n             = 0;
    Py_ssize_t axis          = 0;
    int        norm          = 0;
    int        orthogonalize = 0;
    int        inverse       = 0;
    if (PyArg_ParseTuple(args, "Opinniip", &obj, &sine, &type, &n, &axis, &norm, &orthogonalize, &inverse) == 0) {
        return nullptr;
    }
    borrowed_operand x;
    if (!borrow_operand(obj, x) || x.dtype == 1 || x.dtype == 3) {
        return Py_NewRef(Py_NotImplemented);
    }
    try {
        std::optional<any_tensor> result;
        {
            std::size_t count {1};
            for (const std::size_t extent : x.shape) {
                count *= extent;
            }
            using numerical::detail::trig_family;
            const sb::scoped_gil_release unlocked {releases_gil(count + static_cast<std::size_t>(n))};
            const auto                   scaling {static_cast<numerical::fft_norm>(norm)};
            const std::optional<bool>    weighted {orthogonalize < 0 ? std::nullopt : std::optional<bool> {orthogonalize != 0}};
            const trig_family            family {sine != 0 ? trig_family::sine : trig_family::cosine};
            const auto transform = [&]<typename R>(std::type_identity<R>) {
                const numerical::detail::fft_source<R> source {static_cast<const R*>(x.data), x.shape};
                result.emplace(numerical::detail::trig_transform(source, family, type, static_cast<std::size_t>(n), static_cast<std::ptrdiff_t>(axis), scaling,
                                                                 weighted, inverse != 0));
            };
            if (x.dtype == 0) {
                transform(std::type_identity<double> {});
            }
            else {
                transform(std::type_identity<float> {});
            }
        }
        return wrap(std::move(*result));
    }
    catch (...) {
        return set_cpp_error();
    }
}

} // namespace numerical_binding
