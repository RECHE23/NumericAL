// NumericAL — Python extension module (CPython Limited API, abi3).
//
// Exposes one unifying object — numerical.Tensor (rank-1 is a vector, rank-2 a
// matrix, rank-0 a scalar) — plus the algebra over it:
//   numerical.Tensor(data, shape=None, dtype="float64")  # float64 or complex128
//   numerical.einsum(spec, *operands) -> Tensor          # Einstein summation
//   numerical.solve(a, b) -> Tensor                       # LU linear solve
//   numerical.asarray(obj, dtype=None) -> Tensor          # from __array_interface__
//   numerical.error                                       # raised on bad input
//
// A Tensor interoperates zero-copy with the scientific Python stack:
//   * __array_interface__  → np.asarray(t) shares the buffer (read-only view)
//   * __dlpack__ / __dlpack_device__ → torch/jax/cupy/np.from_dlpack(t)
// Storage is row-major contiguous, matching both protocols natively.

#define PY_SSIZE_T_CLEAN
#define Py_LIMITED_API 0x030A0000
#include <Python.h>

#include <numerical/numerical.hpp>
#include <sciforge/binding/class.hpp>
#include <sciforge/binding/convert.hpp>
#include <sciforge/binding/dispatch.hpp>
#include <sciforge/binding/error.hpp>
#include <sciforge/binding/gil.hpp>
#include <sciforge/binding/module.hpp>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <new>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

// The module error type (numerical.error) is created by SCIFORGE_MODULE at the bottom and
// reached through this fixed-name getter; the helpers below route errors through it.
SCIFORGE_BINDING_ERROR_GETTER;

namespace {

namespace sb = sciforge::binding;

using real_tensor = numerical::tensor<double>;
using cplx_tensor = numerical::tensor<std::complex<double>>;
using any_tensor  = std::variant<real_tensor, cplx_tensor>;

// Bridge the in-flight C++ exception to a Python error via the shared substrate:
// bad_alloc -> MemoryError, other std::exception -> numerical.error, else internal error.
PyObject* set_cpp_error() { return sb::set_cpp_error(sciforge_module_error()); }

// --- a minimal DLPack ABI (CPU, classic DLManagedTensor) -------------------
constexpr int          kDLCPU     = 1;
constexpr std::uint8_t kDLFloat   = 2;
constexpr std::uint8_t kDLComplex = 5;

struct DLDevice
{
    int device_type;
    int device_id;
};
struct DLDataType
{
    std::uint8_t  code;
    std::uint8_t  bits;
    std::uint16_t lanes;
};
struct DLTensor
{
    void*         data;
    DLDevice      device;
    int           ndim;
    DLDataType    dtype;
    std::int64_t* shape;
    std::int64_t* strides;
    std::uint64_t byte_offset;
};
struct DLManagedTensor
{
    DLTensor dl_tensor;
    void*    manager_ctx;
    void (*deleter)(DLManagedTensor* self);
};

// --- the wrapped value -----------------------------------------------------
// The Python object is class_<any_tensor>: wrapper<any_tensor>{PyObject_HEAD; any_tensor*
// held}, identical to the former PyTensorObject. The variant lives directly in the wrapped
// value (the old vestigial holder is gone). caster<any_tensor> (SCIFORGE_WRAPPED below)
// unwraps an argument / wraps a return; as_tensor is the manual unwrap for the .raw functions.
const std::vector<std::size_t>& shape_of(const any_tensor& v)
{
    return v.index() == 0 ? std::get<real_tensor>(v).shape()
                          : std::get<cplx_tensor>(v).shape();
}

std::size_t count_of(const any_tensor& v)
{
    return v.index() == 0 ? std::get<real_tensor>(v).size()
                          : std::get<cplx_tensor>(v).size();
}

const void* data_of(const any_tensor& v)
{
    if (v.index() == 0) {
        return std::get<real_tensor>(v).data().data();
    }
    return std::get<cplx_tensor>(v).data().data();
}

bool is_complex(const any_tensor& v)
{
    return v.index() == 1;
}

const char* dtype_name(const any_tensor& v)
{
    return is_complex(v) ? "complex128" : "float64";
}

// The native byte-order character used by the array-interface typestr.
char byte_order()
{
    const std::uint16_t one {1};
    return (*reinterpret_cast<const char*>(&one) == 1) ? '<' : '>';
}

// Reverses the bytes of each of the \p count doubles at \p bytes: a source in the other byte order.
void swap_doubles(unsigned char* bytes,
                  std::size_t    count)
{
    for (std::size_t i = 0; i < count; ++i) {
        std::reverse(bytes + (i * sizeof(double)), bytes + ((i + 1) * sizeof(double)));
    }
}

std::string typestr_of(const any_tensor& v)
{
    std::string out(1, byte_order());
    out += is_complex(v) ? "c16" : "f8";
    return out;
}

// The wrapped any_tensor of a Tensor argument, or nullptr (with numerical.error set) — the
// manual unwrap for the .raw functions (the m.def functions use caster<any_tensor> instead,
// which raises TypeError on a non-Tensor; the .raw path keeps the historical numerical.error).
any_tensor* as_tensor(PyObject* obj)
{
    const int ok = PyObject_IsInstance(obj, sb::class_type<any_tensor>());
    if (ok < 0) {
        return nullptr;
    }
    if (ok == 0) {
        PyErr_SetString(sciforge_module_error(), "expected a numerical.Tensor");
        return nullptr;
    }
    any_tensor* v = reinterpret_cast<sb::wrapper<any_tensor>*>(obj)->held;
    if (v == nullptr) {
        PyErr_SetString(sciforge_module_error(), "uninitialized numerical.Tensor");
        return nullptr;
    }
    return v;
}

// Wraps a freshly computed tensor in a new Python Tensor (for the .raw functions).
PyObject* wrap(any_tensor value)
{
    return sb::class_wrap<any_tensor>(std::move(value));
}

// --- reading Python data into typed buffers --------------------------------
template <typename T>
T to_scalar(PyObject* item);

template <>
double to_scalar<double>(PyObject* item)
{
    return PyFloat_AsDouble(item);
}

template <>
std::complex<double> to_scalar<std::complex<double>>(PyObject* item)
{
    // Py_complex / PyComplex_AsCComplex are outside the Limited API; these are in.
    const double real = PyComplex_RealAsDouble(item);
    const double imag = PyComplex_ImagAsDouble(item);
    return {real, imag};
}

template <typename T>
PyObject* from_scalar(T value);

template <>
PyObject* from_scalar<double>(double value)
{
    return PyFloat_FromDouble(value);
}

template <>
PyObject* from_scalar<std::complex<double>>(std::complex<double> value)
{
    return PyComplex_FromDoubles(value.real(), value.imag());
}

template <typename T>
bool read_buffer(PyObject* sequence, std::vector<T>& out)
{
    const Py_ssize_t size = PySequence_Size(sequence);
    if (size < 0) {
        return false;
    }
    out.reserve(static_cast<std::size_t>(size));
    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject* item = PySequence_GetItem(sequence, i); // new ref
        if (item == nullptr) {
            return false;
        }
        const T value = to_scalar<T>(item);
        Py_DECREF(item);
        if (PyErr_Occurred() != nullptr) {
            return false;
        }
        out.push_back(value);
    }
    return true;
}

bool read_shape(PyObject* shape_obj, std::vector<std::size_t>& out)
{
    const Py_ssize_t size = PySequence_Size(shape_obj);
    if (size < 0) {
        return false;
    }
    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject* item = PySequence_GetItem(shape_obj, i); // new ref
        if (item == nullptr) {
            return false;
        }
        const long extent = PyLong_AsLong(item);
        Py_DECREF(item);
        if (PyErr_Occurred() != nullptr) {
            return false;
        }
        if (extent < 0) {
            PyErr_SetString(sciforge_module_error(), "shape extents must be non-negative");
            return false;
        }
        out.push_back(static_cast<std::size_t>(extent));
    }
    return true;
}

// Reads a contiguous buffer of T from `data_obj` and builds the variant value, throwing on
// a bad element (the dispatched tp_init turns the throw into numerical.error).
template <typename T>
any_tensor build_any(PyObject* data_obj, const std::vector<std::size_t>& shape)
{
    std::vector<T> buffer;
    if (!read_buffer<T>(data_obj, buffer)) {
        throw std::runtime_error("Tensor: could not read the data buffer");
    }
    return any_tensor {numerical::tensor<T> {shape, std::move(buffer)}};
}

// --- the constructor (def_init factory) ------------------------------------
// Tensor(data, shape=None, dtype="float64"): shape None infers a 1-D shape from the data
// length. A free function returning the wrapped value; domain failures throw -> numerical.error.
any_tensor tensor_factory(PyObject*                 data,
                          std::optional<PyObject*>  shape_obj,
                          std::string               dtype)
{
    std::vector<std::size_t> shape;
    if (!shape_obj.has_value() || *shape_obj == Py_None) {
        const Py_ssize_t length = PySequence_Size(data);
        if (length < 0) {
            throw std::runtime_error("Tensor: data has no length to infer a shape from");
        }
        shape.push_back(static_cast<std::size_t>(length));
    }
    else if (!read_shape(*shape_obj, shape)) {
        throw std::runtime_error("Tensor: invalid shape");
    }
    if (dtype == "float64") {
        return build_any<double>(data, shape);
    }
    if (dtype == "complex128") {
        return build_any<std::complex<double>>(data, shape);
    }
    throw std::runtime_error("dtype must be 'float64' or 'complex128'");
}

// __repr__ via def_repr (the tp_repr slot): a free function returning the repr string.
std::string tensor_repr(const any_tensor& v)
{
    return "numerical.Tensor(ndim=" + std::to_string(shape_of(v).size())
           + ", size=" + std::to_string(count_of(v))
           + ", dtype=" + dtype_name(v) + ")";
}

PyObject* shape_tuple(const any_tensor& v)
{
    const std::vector<std::size_t>& shape = shape_of(v);
    PyObject*                       tuple = PyTuple_New(static_cast<Py_ssize_t>(shape.size()));
    if (tuple == nullptr) {
        return nullptr;
    }
    for (std::size_t i = 0; i < shape.size(); ++i) {
        PyObject* extent = PyLong_FromSize_t(shape[i]);
        if (extent == nullptr) {
            Py_DECREF(tuple);
            return nullptr;
        }
        PyTuple_SetItem(tuple, static_cast<Py_ssize_t>(i), extent); // steals ref
    }
    return tuple;
}

// def_prop_ro getters take the unwrapped value. ndim/size/dtype return a C++ value (the
// value-caster builds the int/str); shape and __array_interface__ return a computed PyObject*
// (a dynamic tuple / dict) which def_prop_ro passes straight through (sciforge-build 2026.6.15).
PyObject* get_shape(const any_tensor& v)
{
    return shape_tuple(v);
}

std::size_t get_ndim(const any_tensor& v)
{
    return shape_of(v).size();
}

std::size_t get_size(const any_tensor& v)
{
    return count_of(v);
}

std::string get_dtype(const any_tensor& v)
{
    return dtype_name(v);
}

PyObject* get_array_interface(const any_tensor& v)
{
    PyObject* shape = shape_tuple(v);
    if (shape == nullptr) {
        return nullptr;
    }
    const std::string typestr = typestr_of(v);
    // 'data' is (address, read-only): the view shares our immutable buffer.
    return Py_BuildValue("{s:N,s:s,s:(NO),s:i}",
                         "shape", shape,
                         "typestr", typestr.c_str(),
                         "data", PyLong_FromVoidPtr(const_cast<void*>(data_of(v))), Py_True,
                         "version", 3);
}

// --- DLPack export ---------------------------------------------------------
struct dlpack_ctx
{
    PyObject*     owner; // the Tensor, kept alive while the capsule lives
    std::int64_t* shape;
};

void dlpack_deleter(DLManagedTensor* managed)
{
    auto* ctx = static_cast<dlpack_ctx*>(managed->manager_ctx);
    if (ctx != nullptr) {
        Py_XDECREF(ctx->owner);
        std::free(ctx->shape);
        delete ctx;
    }
    std::free(managed);
}

void dlpack_capsule_destructor(PyObject* capsule)
{
    if (PyCapsule_IsValid(capsule, "dltensor") == 0) {
        return; // already consumed (renamed to "used_dltensor")
    }
    auto* managed = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule, "dltensor"));
    if (managed != nullptr && managed->deleter != nullptr) {
        managed->deleter(managed);
    }
}

PyObject* tensor_dlpack(PyObject* self, PyObject* /*args*/, PyObject* /*kwds*/)
{
    const any_tensor* h = as_tensor(self);
    if (h == nullptr) {
        return nullptr;
    }
    const std::vector<std::size_t>& shape = shape_of(*h);
    const int                       ndim  = static_cast<int>(shape.size());

    auto* managed = static_cast<DLManagedTensor*>(std::malloc(sizeof(DLManagedTensor)));
    auto* axes    = static_cast<std::int64_t*>(std::malloc(sizeof(std::int64_t) * (ndim > 0 ? static_cast<std::size_t>(ndim) : 1)));
    auto* ctx     = new (std::nothrow) dlpack_ctx {Py_NewRef(self), axes};
    if (managed == nullptr || axes == nullptr || ctx == nullptr) {
        std::free(managed);
        std::free(axes);
        delete ctx;
        Py_DECREF(self); // undo Py_NewRef when ctx was not constructed
        return PyErr_NoMemory();
    }
    for (int i = 0; i < ndim; ++i) {
        axes[i] = static_cast<std::int64_t>(shape[static_cast<std::size_t>(i)]);
    }

    managed->dl_tensor.data        = const_cast<void*>(data_of(*h));
    managed->dl_tensor.device      = DLDevice {kDLCPU, 0};
    managed->dl_tensor.ndim        = ndim;
    managed->dl_tensor.dtype       = is_complex(*h) ? DLDataType {kDLComplex, 128, 1}
                                                    : DLDataType {kDLFloat, 64, 1};
    managed->dl_tensor.shape       = axes;
    managed->dl_tensor.strides     = nullptr; // compact, row-major
    managed->dl_tensor.byte_offset = 0;
    managed->manager_ctx           = ctx;
    managed->deleter               = dlpack_deleter;

    PyObject* capsule = PyCapsule_New(managed, "dltensor", dlpack_capsule_destructor);
    if (capsule == nullptr) {
        dlpack_deleter(managed);
        return nullptr;
    }
    return capsule;
}

// __dlpack_device__ as a .def method (free function + self-unwrap): always (kDLCPU, 0),
// returned through the tuple-caster.
std::tuple<int, int> tensor_dlpack_device(const any_tensor& /*v*/)
{
    return {kDLCPU, 0};
}

// --- tolist (nested, numpy-style; rank-0 yields a bare scalar) -------------
template <typename T>
PyObject* nested(const T* data, const std::vector<std::size_t>& shape,
                 std::size_t axis, std::size_t& offset)
{
    if (axis == shape.size()) {
        return from_scalar<T>(data[offset++]);
    }
    PyObject* list = PyList_New(static_cast<Py_ssize_t>(shape[axis]));
    if (list == nullptr) {
        return nullptr;
    }
    for (std::size_t i = 0; i < shape[axis]; ++i) {
        PyObject* child = nested<T>(data, shape, axis + 1, offset);
        if (child == nullptr) {
            Py_DECREF(list);
            return nullptr;
        }
        PyList_SetItem(list, static_cast<Py_ssize_t>(i), child); // steals ref
    }
    return list;
}

PyObject* tensor_tolist(PyObject* self, PyObject* /*ignored*/)
{
    const any_tensor* h = as_tensor(self);
    if (h == nullptr) {
        return nullptr;
    }
    std::size_t offset = 0;
    if (is_complex(*h)) {
        const auto& tensor = std::get<cplx_tensor>(*h);
        return nested<std::complex<double>>(tensor.data().data(), tensor.shape(), 0, offset);
    }
    const auto& tensor = std::get<real_tensor>(*h);
    return nested<double>(tensor.data().data(), tensor.shape(), 0, offset);
}

// --- module functions ------------------------------------------------------
// Whether a computation over this many elements releases the GIL. Re-acquiring it may wait out another
// thread's switch interval (5 ms by default), which a small matrix would pay many times over; from 64 × 64
// up, the computation dominates. The operands stay readable meanwhile: a Tensor never changes once built,
// and the call holds a reference to each.
bool releases_gil(std::size_t elements)
{
    return elements >= std::size_t {4096};
}

// einsum(spec, *operands): all operands must share a dtype.
PyObject* module_einsum(PyObject* /*self*/, PyObject* args)
{
    const Py_ssize_t count = PyTuple_Size(args);
    if (count < 2) {
        PyErr_SetString(sciforge_module_error(), "einsum(spec, *operands) needs a spec and ≥1 operand");
        return nullptr;
    }
    PyObject* spec_bytes = PyUnicode_AsUTF8String(PyTuple_GetItem(args, 0)); // new ref
    if (spec_bytes == nullptr) {
        return nullptr;
    }
    char*      spec_data = nullptr;
    Py_ssize_t spec_size = 0;
    if (PyBytes_AsStringAndSize(spec_bytes, &spec_data, &spec_size) < 0) {
        Py_DECREF(spec_bytes);
        return nullptr;
    }
    const std::string spec {spec_data, static_cast<std::size_t>(spec_size)};
    Py_DECREF(spec_bytes);

    std::vector<const any_tensor*> holders;
    holders.reserve(static_cast<std::size_t>(count - 1));
    bool any_complex = false;
    for (Py_ssize_t i = 1; i < count; ++i) {
        const any_tensor* h = as_tensor(PyTuple_GetItem(args, i));
        if (h == nullptr) {
            return nullptr;
        }
        any_complex = any_complex || is_complex(*h);
        holders.push_back(h);
    }
    for (const any_tensor* h : holders) {
        if (is_complex(*h) != any_complex) {
            PyErr_SetString(sciforge_module_error(), "einsum operands must share one dtype");
            return nullptr;
        }
    }

    std::size_t elements {0};
    for (const any_tensor* h : holders) {
        elements += any_complex ? std::get<cplx_tensor>(*h).data().size() : std::get<real_tensor>(*h).data().size();
    }
    try {
        std::optional<any_tensor> result;
        {
            const sb::scoped_gil_release unlocked {releases_gil(elements)};
            if (any_complex) {
                std::vector<const cplx_tensor*> operands;
                operands.reserve(holders.size());
                for (const any_tensor* h : holders) {
                    operands.push_back(&std::get<cplx_tensor>(*h));
                }
                result.emplace(numerical::detail::einsum_impl<std::complex<double>>(spec, operands));
            }
            else {
                std::vector<const real_tensor*> operands;
                operands.reserve(holders.size());
                for (const any_tensor* h : holders) {
                    operands.push_back(&std::get<real_tensor>(*h));
                }
                result.emplace(numerical::detail::einsum_impl<double>(spec, operands));
            }
        }
        return wrap(std::move(*result));
    }
    catch (...) {
        return set_cpp_error();
    }
}

// --- the linear-algebra functions (m.def: any_tensor in/out through caster<any_tensor>) ---
// Validation failures throw -> numerical.error; the variant dispatch (is_complex / std::get)
// stays here in the consumer (the substrate grows no variant machinery).
template <typename T>
numerical::matrix<T> matrix_from(const numerical::tensor<T>& t)
{
    return numerical::matrix<T> {t.shape()[0], t.shape()[1], t.data()};
}

template <typename T>
any_tensor do_solve(const numerical::tensor<T>& a, const numerical::tensor<T>& b)
{
    const sb::scoped_gil_release unlocked {releases_gil(a.data().size())};
    numerical::matrix<T>         coeff {a.shape()[0], a.shape()[1], a.data()};
    numerical::vector<T>         rhs {b.data()};
    numerical::vector<T>         answer {numerical::solve(coeff, rhs)};
    std::vector<T>    out;
    out.reserve(answer.size());
    for (std::size_t i = 0; i < answer.size(); ++i) {
        out.push_back(answer[i]);
    }
    return any_tensor {numerical::tensor<T> {{answer.size()}, std::move(out)}};
}

// solve(a, b): a a rank-2 tensor, b a rank-1 tensor, sharing a dtype.
any_tensor solve(const any_tensor& a, const any_tensor& b)
{
    if (shape_of(a).size() != 2 || shape_of(b).size() != 1) {
        throw std::runtime_error("solve(a, b) needs a rank-2 a and a rank-1 b");
    }
    if (is_complex(a) != is_complex(b)) {
        throw std::runtime_error("solve operands must share one dtype");
    }
    if (is_complex(a)) {
        return do_solve<std::complex<double>>(std::get<cplx_tensor>(a), std::get<cplx_tensor>(b));
    }
    return do_solve<double>(std::get<real_tensor>(a), std::get<real_tensor>(b));
}

// determinant(a) -> float or complex: a dtype-dependent scalar (a variant return no single
// caster covers), so it stays a manual .raw function.
PyObject* module_determinant(PyObject* /*self*/, PyObject* args)
{
    PyObject* obj = nullptr;
    if (PyArg_ParseTuple(args, "O", &obj) == 0) {
        return nullptr;
    }
    const any_tensor* h = as_tensor(obj);
    if (h == nullptr) {
        return nullptr;
    }
    if (shape_of(*h).size() != 2) {
        PyErr_SetString(sciforge_module_error(), "determinant expects a matrix (rank-2 tensor)");
        return nullptr;
    }
    try {
        if (is_complex(*h)) {
            const cplx_tensor&   a     {std::get<cplx_tensor>(*h)};
            std::complex<double> value;
            {
                const sb::scoped_gil_release unlocked {releases_gil(a.data().size())};
                value = numerical::determinant(matrix_from(a));
            }
            return PyComplex_FromDoubles(value.real(), value.imag());
        }
        const real_tensor& a {std::get<real_tensor>(*h)};
        double             value {};
        {
            const sb::scoped_gil_release unlocked {releases_gil(a.data().size())};
            value = numerical::determinant(matrix_from(a));
        }
        return PyFloat_FromDouble(value);
    }
    catch (...) {
        return set_cpp_error();
    }
}

template <typename T>
any_tensor do_inverse(const numerical::tensor<T>& a)
{
    const sb::scoped_gil_release unlocked {releases_gil(a.data().size())};
    return any_tensor {numerical::tensor<T> {numerical::inverse(matrix_from(a))}};
}

// inverse(a) -> Tensor (the matrix inverse).
any_tensor inverse(const any_tensor& v)
{
    if (shape_of(v).size() != 2) {
        throw std::runtime_error("inverse expects a matrix (rank-2 tensor)");
    }
    if (is_complex(v)) {
        return do_inverse<std::complex<double>>(std::get<cplx_tensor>(v));
    }
    return do_inverse<double>(std::get<real_tensor>(v));
}

template <typename T>
any_tensor do_lstsq(const numerical::tensor<T>& a, const numerical::tensor<T>& b)
{
    const sb::scoped_gil_release unlocked {releases_gil(a.data().size())};
    const numerical::vector<T>   answer {numerical::lstsq(matrix_from(a), numerical::vector<T> {b.data()})};
    std::vector<T>          out;
    out.reserve(answer.size());
    for (std::size_t i = 0; i < answer.size(); ++i) {
        out.push_back(answer[i]);
    }
    return any_tensor {numerical::tensor<T> {{answer.size()}, std::move(out)}};
}

// lstsq(a, b) -> Tensor (the minimum-norm least-squares solution of a x ≈ b, from the SVD).
any_tensor lstsq(const any_tensor& a, const any_tensor& b)
{
    if (shape_of(a).size() != 2 || shape_of(b).size() != 1) {
        throw std::runtime_error("lstsq(a, b) needs a rank-2 a and a rank-1 b");
    }
    if (is_complex(a) != is_complex(b)) {
        throw std::runtime_error("lstsq operands must share one dtype");
    }
    if (is_complex(a)) {
        return do_lstsq<std::complex<double>>(std::get<cplx_tensor>(a), std::get<cplx_tensor>(b));
    }
    return do_lstsq<double>(std::get<real_tensor>(a), std::get<real_tensor>(b));
}

template <typename T>
any_tensor do_cholesky(const numerical::tensor<T>& a)
{
    const sb::scoped_gil_release unlocked {releases_gil(a.data().size())};
    return any_tensor {numerical::tensor<T> {numerical::cholesky_decomposition<T> {matrix_from(a)}.lower()}};
}

// cholesky(a) -> Tensor (the lower factor L of A = L Lᴴ).
any_tensor cholesky(const any_tensor& v)
{
    if (shape_of(v).size() != 2) {
        throw std::runtime_error("cholesky expects a matrix (rank-2 tensor)");
    }
    if (is_complex(v)) {
        return do_cholesky<std::complex<double>>(std::get<cplx_tensor>(v));
    }
    return do_cholesky<double>(std::get<real_tensor>(v));
}

template <typename T>
std::tuple<any_tensor, any_tensor> do_qr(const numerical::tensor<T>& a)
{
    const sb::scoped_gil_release         unlocked {releases_gil(a.data().size())};
    const numerical::qr_decomposition<T> factored {matrix_from(a)};
    return {any_tensor {numerical::tensor<T> {factored.q()}}, any_tensor {numerical::tensor<T> {factored.r()}}};
}

// qr(a) -> (Q, R), through the tuple-caster.
std::tuple<any_tensor, any_tensor> qr(const any_tensor& v)
{
    if (shape_of(v).size() != 2) {
        throw std::runtime_error("qr expects a matrix (rank-2 tensor)");
    }
    if (is_complex(v)) {
        return do_qr<std::complex<double>>(std::get<cplx_tensor>(v));
    }
    return do_qr<double>(std::get<real_tensor>(v));
}

// The real float64 matrix of a Tensor, or throws (eigh is real-only).
const real_tensor& real_matrix(const any_tensor& v, const char* name)
{
    if (shape_of(v).size() != 2) {
        throw std::runtime_error("expected a matrix (rank-2 tensor)");
    }
    if (is_complex(v)) {
        throw std::runtime_error(name);
    }
    return std::get<real_tensor>(v);
}

// eigh(a) -> (eigenvalues, eigenvectors) for a real symmetric matrix.
std::tuple<any_tensor, any_tensor> eigh(const any_tensor& v)
{
    const real_tensor&            a = real_matrix(v, "eigh supports float64 matrices only");
    const sb::scoped_gil_release     unlocked {releases_gil(a.data().size())};
    const numerical::symmetric_eigen decomposition {matrix_from(a)};
    return {any_tensor {real_tensor {decomposition.values()}},
            any_tensor {real_tensor {decomposition.vectors()}}};
}

template <typename T>
std::tuple<any_tensor, any_tensor, any_tensor> do_svd(const numerical::tensor<T>& a)
{
    const sb::scoped_gil_release          unlocked {releases_gil(a.data().size())};
    const numerical::svd_decomposition<T> decomposition {matrix_from(a)};
    return {any_tensor {numerical::tensor<T> {decomposition.u()}},
            any_tensor {real_tensor {decomposition.singular_values()}},
            any_tensor {numerical::tensor<T> {decomposition.vh()}}};
}

// svd(a) -> (U, singular_values, Vh) for an m×n matrix, as numpy.linalg.svd(a, full_matrices=False).
std::tuple<any_tensor, any_tensor, any_tensor> svd(const any_tensor& v)
{
    if (shape_of(v).size() != 2) {
        throw std::runtime_error("svd expects a matrix (rank-2 tensor)");
    }
    if (is_complex(v)) {
        return do_svd<std::complex<double>>(std::get<cplx_tensor>(v));
    }
    return do_svd<double>(std::get<real_tensor>(v));
}

// Widen a contiguous real buffer to a complex tensor (zero imaginary part) — used
// when dtype='complex128' is requested of real source data.
any_tensor promote_to_complex(std::vector<std::size_t> shape, const double* src, std::size_t n)
{
    std::vector<std::complex<double>> buffer(n);
    for (std::size_t i = 0; i < n; ++i) {
        buffer[i] = src[i];
    }
    return any_tensor {cplx_tensor {std::move(shape), std::move(buffer)}};
}

// asarray(obj, dtype=None): an existing Tensor (returned as-is when its dtype
// already matches, else converted), or a copy built from obj's __array_interface__
// (a C-contiguous float64/complex128 array). dtype, when given, is honoured:
// real→complex128 widens with a zero imaginary part; complex128→float64 is refused
// (it would discard the imaginary part).
PyObject* module_asarray(PyObject* /*self*/, PyObject* args, PyObject* kwds)
{
    static const char* keywords[] = {"obj", "dtype", nullptr};
    PyObject*          obj         = nullptr;
    const char*        dtype_str   = nullptr;
    if (PyArg_ParseTupleAndKeywords(args, kwds, "O|z", const_cast<char**>(keywords),
                                    &obj, &dtype_str) == 0) {
        return nullptr;
    }

    // Resolve the requested dtype up front (fail fast on an unknown name).
    const bool dtype_given      = dtype_str != nullptr;
    bool       want_complex_out = false;
    if (dtype_given) {
        if (std::strcmp(dtype_str, "float64") == 0) {
            want_complex_out = false;
        }
        else if (std::strcmp(dtype_str, "complex128") == 0) {
            want_complex_out = true;
        }
        else {
            PyErr_SetString(sciforge_module_error(), "asarray: dtype must be 'float64' or 'complex128'");
            return nullptr;
        }
    }

    const int already = PyObject_IsInstance(obj, sb::class_type<any_tensor>());
    if (already < 0) {
        return nullptr;
    }
    if (already == 1) {
        const any_tensor* const h = as_tensor(obj);
        if (h == nullptr) {
            return nullptr;
        }
        if (!dtype_given || is_complex(*h) == want_complex_out) {
            return Py_NewRef(obj); // already the requested dtype (or none requested)
        }
        if (is_complex(*h)) { // complex source, float64 requested
            PyErr_SetString(sciforge_module_error(),
                            "asarray: refusing to cast complex128 to float64 (discards the imaginary part)");
            return nullptr;
        }
        try { // real source, complex128 requested: widen with a zero imaginary part
            return wrap(promote_to_complex(shape_of(*h), static_cast<const double*>(data_of(*h)), count_of(*h)));
        }
        catch (...) {
            return set_cpp_error();
        }
    }

    PyObject* ai = PyObject_GetAttrString(obj, "__array_interface__"); // new ref
    if (ai == nullptr) {
        PyErr_SetString(sciforge_module_error(), "asarray: object exposes no __array_interface__");
        return nullptr;
    }
    if (PyDict_Check(ai) == 0) {
        Py_DECREF(ai);
        PyErr_SetString(sciforge_module_error(), "asarray: __array_interface__ is not a dict");
        return nullptr;
    }

    PyObject* strides = PyDict_GetItemString(ai, "strides"); // borrowed
    if (strides != nullptr && strides != Py_None) {
        Py_DECREF(ai);
        PyErr_SetString(sciforge_module_error(), "asarray: only C-contiguous arrays are supported");
        return nullptr;
    }
    PyObject* typestr_obj = PyDict_GetItemString(ai, "typestr"); // borrowed
    PyObject* shape_obj   = PyDict_GetItemString(ai, "shape");   // borrowed
    PyObject* data_obj    = PyDict_GetItemString(ai, "data");    // borrowed
    if (typestr_obj == nullptr || shape_obj == nullptr || data_obj == nullptr
        || PyTuple_Check(data_obj) == 0 || PyTuple_Size(data_obj) < 1) {
        Py_DECREF(ai);
        PyErr_SetString(sciforge_module_error(), "asarray: malformed __array_interface__");
        return nullptr;
    }

    PyObject* typestr_bytes = PyUnicode_AsUTF8String(typestr_obj); // new ref
    if (typestr_bytes == nullptr) {
        Py_DECREF(ai);
        return nullptr;
    }
    const char*       typestr_chars = PyBytes_AsString(typestr_bytes);
    const std::string typestr {typestr_chars != nullptr ? typestr_chars : ""};
    Py_DECREF(typestr_bytes);

    // A typestr is a byte order then a kind and size: `<f8`, `>c16`. Any other byte order than '<',
    // '>' or '=' (native), and any other kind, is refused rather than read as native float64.
    const char order        = typestr.empty() ? '\0' : typestr[0];
    const bool want_complex = typestr.size() == 4 && typestr.substr(1) == "c16";
    const bool want_real    = typestr.size() == 3 && typestr.substr(1) == "f8";
    const bool swapped      = order != '=' && order != byte_order();
    if ((order != '<' && order != '>' && order != '=') || (!want_complex && !want_real)) {
        Py_DECREF(ai);
        PyErr_SetString(sciforge_module_error(), "asarray: only float64 and complex128 are supported");
        return nullptr;
    }
    // The output dtype is the requested one, or the source dtype when none is given.
    const bool out_complex = dtype_given ? want_complex_out : want_complex;
    if (want_complex && !out_complex) {
        Py_DECREF(ai);
        PyErr_SetString(sciforge_module_error(),
                        "asarray: refusing to cast complex128 to float64 (discards the imaginary part)");
        return nullptr;
    }

    std::vector<std::size_t> shape;
    if (!read_shape(shape_obj, shape)) {
        Py_DECREF(ai);
        return nullptr;
    }
    std::size_t total = 1;
    for (const std::size_t extent : shape) {
        if (extent != 0 && total > SIZE_MAX / sizeof(std::complex<double>) / extent) {
            Py_DECREF(ai);
            PyErr_SetString(sciforge_module_error(), "asarray: the shape's element count overflows");
            return nullptr;
        }
        total *= extent;
    }

    void* address = PyLong_AsVoidPtr(PyTuple_GetItem(data_obj, 0));
    if (PyErr_Occurred() != nullptr) {
        Py_DECREF(ai);
        return nullptr;
    }
    // A null address holds no element: reading `total` of them from it would crash. An empty array's
    // address is never read.
    if (address == nullptr && total > 0) {
        Py_DECREF(ai);
        PyErr_SetString(sciforge_module_error(), "asarray: __array_interface__ gives a null data address");
        return nullptr;
    }

    PyObject* result = nullptr;
    try {
        if (want_complex) { // complex source; narrowing to float64 was refused above
            std::vector<std::complex<double>> buffer(total);
            if (total > 0) {
                std::memcpy(buffer.data(), address, total * sizeof(std::complex<double>));
            }
            if (swapped) {
                swap_doubles(reinterpret_cast<unsigned char*>(buffer.data()), 2 * total);
            }
            result = wrap(any_tensor {cplx_tensor {shape, std::move(buffer)}});
        }
        else { // real source: keep as float64, or widen to complex128 if requested
            std::vector<double> buffer(total);
            if (total > 0) {
                std::memcpy(buffer.data(), address, total * sizeof(double));
            }
            if (swapped) {
                swap_doubles(reinterpret_cast<unsigned char*>(buffer.data()), total);
            }
            result = out_complex ? wrap(promote_to_complex(shape, buffer.data(), total))
                                 : wrap(any_tensor {real_tensor {shape, std::move(buffer)}});
        }
    }
    catch (...) {
        result = set_cpp_error();
    }
    Py_DECREF(ai);
    return result;
}

} // namespace

// caster<any_tensor> <-> the numerical.Tensor heap type (replaces the hand-rolled as_holder/wrap).
SCIFORGE_WRAPPED(any_tensor)

SCIFORGE_MODULE(_numerical, "numerical.error", m)
{
    // The Tensor type: a def_init constructor, the DLPack/list/repr methods, and the
    // introspection properties (shape/__array_interface__ are computed PyObject* getters).
    // init_once: np.asarray, __array_interface__ and __dlpack__ view the held buffer in place, so a
    // second __init__ would free what those views read.
    m.type<any_tensor>("numerical.Tensor")
    .init_once()
    .def_init<&tensor_factory>(sb::arg("data"), sb::arg("shape") = sb::none, sb::arg("dtype") = "float64")
    .def_repr<&tensor_repr>()
    .def<&tensor_dlpack_device>("__dlpack_device__",
                                "__dlpack_device__() -> (device_type, device_id)\nThe DLPack device (always CPU).")
    .raw("__dlpack__", reinterpret_cast<PyCFunction>(reinterpret_cast<void*>(tensor_dlpack)),
         METH_VARARGS | METH_KEYWORDS,
         "__dlpack__(stream=None) -> PyCapsule\nExport a DLPack capsule sharing this tensor's buffer.")
    .raw("tolist", tensor_tolist, METH_NOARGS,
         "tolist() -> list\nThe elements as nested Python lists (a bare scalar when rank-0).")
    .def_prop_ro<&get_shape>("shape", "The size of each axis, as a tuple.")
    .def_prop_ro<&get_ndim>("ndim", "The number of axes.")
    .def_prop_ro<&get_size>("size", "The total number of elements.")
    .def_prop_ro<&get_dtype>("dtype", "'float64' or 'complex128'.")
    .def_prop_ro<&get_array_interface>("__array_interface__",
                                       "NumPy array-interface (v3) describing the shared, read-only buffer.");
    // The 7 linear-algebra functions whose args/return are caster-able -> m.def.
    m.def<&solve>("solve", "solve(a, b) -> Tensor\nSolve the linear system a @ x = b (a rank-2, b rank-1).");
    m.def<&inverse>("inverse", "inverse(a) -> Tensor\nThe inverse of a square matrix.");
    m.def<&lstsq>("lstsq", "lstsq(a, b) -> Tensor\nThe minimum-norm least-squares solution of a @ x ≈ b, as numpy.linalg.lstsq (SVD; a rank-2, b rank-1).");
    m.def<&cholesky>("cholesky", "cholesky(a) -> Tensor\nThe lower factor L of A = L Lᴴ (Hermitian positive-definite).");
    m.def<&qr>("qr", "qr(a) -> (Q, R)\nThe QR factorisation of an m×n matrix (m ≥ n).");
    m.def<&eigh>("eigh", "eigh(a) -> (values, vectors)\nEigenpairs of a real symmetric matrix (ascending values).");
    m.def<&svd>("svd", "svd(a) -> (U, S, Vh)\nThe singular value decomposition a = U @ diag(S) @ Vh of an m×n matrix, as numpy.linalg.svd(a, full_matrices=False).");
    // The 3 that stay manual: determinant (dtype-dependent scalar), einsum (variadic),
    // asarray (returns the same object when the dtype matches — Py_NewRef identity).
    m.raw("determinant", module_determinant, METH_VARARGS,
          "determinant(a) -> number\nThe determinant of a square matrix (float or complex).");
    m.raw("einsum", module_einsum, METH_VARARGS,
          "einsum(spec, *operands) -> Tensor\n"
          "Einstein summation, e.g. 'ij,jk->ik' (matmul), 'i,i->' (dot), 'ij->ji'.");
    m.raw("asarray", reinterpret_cast<PyCFunction>(reinterpret_cast<void*>(module_asarray)),
          METH_VARARGS | METH_KEYWORDS,
          "asarray(obj, dtype=None) -> Tensor\n"
          "A Tensor (converted if dtype differs), or a copy of a C-contiguous array via\n"
          "__array_interface__. dtype is honoured: real->complex128 widens (zero imaginary\n"
          "part); complex128->float64 is refused.");
}
