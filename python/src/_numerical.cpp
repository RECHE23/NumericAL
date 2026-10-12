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
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <new>
#include <optional>
#include <string>
#include <string_view>
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

// An empty tensor's buffer has no address; the exchange protocols get one that is never read: NumPy 2.2
// refuses a null data pointer and falls back to the sequence protocol, which an empty tensor cannot serve.
const void* data_of(const any_tensor& v)
{
    static constexpr std::complex<double> placeholder {};
    const void* data = v.index() == 0 ? static_cast<const void*>(std::get<real_tensor>(v).data().data())
                                      : static_cast<const void*>(std::get<cplx_tensor>(v).data().data());
    return data != nullptr ? data : &placeholder;
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

std::string typestr_of(const any_tensor& v)
{
    std::string out(1, byte_order());
    out += is_complex(v) ? "c16" : "f8";
    return out;
}

// The wrapped any_tensor of a Tensor argument, or nullptr (with numerical.error set) — the
// manual unwrap for the .raw functions (the m.def functions use caster<any_tensor> instead,
// which raises TypeError on a non-Tensor; the .raw path keeps the historical numerical.error).
// The object's real type is checked, never its `__class__`: a mock that claims to be a Tensor holds
// no any_tensor, and reading one out of it crashed the interpreter.
any_tensor* as_tensor(PyObject* obj)
{
    if (PyObject_TypeCheck(obj, reinterpret_cast<PyTypeObject*>(sb::class_type<any_tensor>())) == 0) {
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

// The answer of a solver for a rank-1 b (a vector) or a rank-2 b (one right-hand side per column).
template <typename T, typename VectorSolver, typename MatrixSolver>
any_tensor solve_for(const numerical::tensor<T>& b, VectorSolver by_vector, MatrixSolver by_matrix)
{
    if (b.shape().size() == 2) {
        return any_tensor {numerical::tensor<T> {by_matrix(matrix_from(b))}};
    }
    const numerical::vector<T> answer {by_vector(numerical::vector<T> {b.data()})};
    return any_tensor {numerical::tensor<T> {{answer.size()}, std::vector<T>(answer.begin(), answer.end())}};
}

template <typename T>
any_tensor do_solve(const numerical::tensor<T>& a, const numerical::tensor<T>& b)
{
    const sb::scoped_gil_release unlocked {releases_gil(a.data().size())};
    const numerical::matrix<T>   coeff {matrix_from(a)};
    return solve_for(b, [&coeff](const numerical::vector<T>& rhs) { return numerical::solve(coeff, rhs); },
                     [&coeff](const numerical::matrix<T>& rhs) { return numerical::solve(coeff, rhs); });
}

// solve(a, b): a a rank-2 tensor, b a rank-1 or rank-2 tensor, sharing a dtype.
any_tensor solve(const any_tensor& a, const any_tensor& b)
{
    if (shape_of(a).size() != 2 || (shape_of(b).size() != 1 && shape_of(b).size() != 2)) {
        throw std::runtime_error("solve(a, b) needs a rank-2 a and a rank-1 or rank-2 b");
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

// The integers of a sequence (a list from the Python indexing code), or false with an error set.
bool read_integers(PyObject* seq, std::vector<Py_ssize_t>& out)
{
    const Py_ssize_t n = PySequence_Size(seq); // the Limited API has no PySequence_Fast accessors
    if (n < 0) {
        return false;
    }
    out.resize(static_cast<std::size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* item = PySequence_GetItem(seq, i);
        if (item == nullptr) {
            return false;
        }
        const Py_ssize_t value = PyNumber_AsSsize_t(item, PyExc_OverflowError);
        Py_DECREF(item);
        if (value == -1 && PyErr_Occurred() != nullptr) {
            return false;
        }
        out[static_cast<std::size_t>(i)] = value;
    }
    return true;
}

// The elements `starts[a] + k·steps[a]`, `k < counts[a]`, on every axis `a` of `t`, in row-major order: the
// axes with `keep[a]` are the result's, the others are indexed by one integer and dropped.
template <typename T>
PyObject* take_from(const numerical::tensor<T>& t, const std::vector<std::size_t>& starts,
                    const std::vector<std::ptrdiff_t>& steps, const std::vector<std::size_t>& counts,
                    const std::vector<std::size_t>& keep)
{
    const std::vector<std::size_t>& shape {t.shape()};
    const std::size_t               rank  {shape.size()};
    std::vector<std::size_t>        strides(rank, 1);
    for (std::size_t a = rank; a-- > 1;) {
        strides[a - 1] = strides[a] * shape[a];
    }
    std::vector<std::size_t> out_shape;
    std::size_t              total {1};
    for (std::size_t a = 0; a < rank; ++a) {
        total *= counts[a];
        if (keep[a] != 0) {
            out_shape.push_back(counts[a]);
        }
    }
    std::vector<T>           out;
    out.reserve(total);
    std::vector<std::size_t> k(rank, 0);
    for (std::size_t n = 0; n < total; ++n) {
        std::size_t offset {0};
        for (std::size_t a = 0; a < rank; ++a) {
            offset += static_cast<std::size_t>(static_cast<std::ptrdiff_t>(starts[a])
                                               + (static_cast<std::ptrdiff_t>(k[a]) * steps[a])) * strides[a];
        }
        out.push_back(t.data()[offset]);
        for (std::size_t a = rank; a-- > 0;) { // the next multi-index, last axis fastest
            if (++k[a] < counts[a]) {
                break;
            }
            k[a] = 0;
        }
    }
    if (out_shape.empty()) {
        return from_scalar<T>(out.front());
    }
    return wrap(any_tensor {numerical::tensor<T> {std::move(out_shape), std::move(out)}});
}

// _take(t, starts, steps, counts, keep) -> Tensor or number: the indexing behind Tensor.__getitem__, whose
// Python side resolves every key to these per-axis ranges (the bounds are its job; the copy is this one's).
PyObject* module_take(PyObject* /*self*/, PyObject* args)
{
    PyObject* obj    = nullptr;
    PyObject* starts = nullptr;
    PyObject* steps  = nullptr;
    PyObject* counts = nullptr;
    PyObject* keep   = nullptr;
    if (PyArg_ParseTuple(args, "OOOOO", &obj, &starts, &steps, &counts, &keep) == 0) {
        return nullptr;
    }
    const any_tensor* h = as_tensor(obj);
    if (h == nullptr) {
        return nullptr;
    }
    std::vector<Py_ssize_t> start_raw;
    std::vector<Py_ssize_t> step_raw;
    std::vector<Py_ssize_t> count_raw;
    std::vector<Py_ssize_t> keep_raw;
    if (!read_integers(starts, start_raw) || !read_integers(steps, step_raw) || !read_integers(counts, count_raw)
        || !read_integers(keep, keep_raw)) {
        return nullptr;
    }
    const std::size_t rank {shape_of(*h).size()};
    if (start_raw.size() != rank || step_raw.size() != rank || count_raw.size() != rank || keep_raw.size() != rank) {
        PyErr_SetString(sciforge_module_error(), "_take: one range per axis");
        return nullptr;
    }
    std::vector<std::size_t>    start_v(rank);
    std::vector<std::ptrdiff_t> step_v(rank);
    std::vector<std::size_t>    count_v(rank);
    std::vector<std::size_t>    keep_v(rank);
    for (std::size_t a = 0; a < rank; ++a) {
        const Py_ssize_t dim {static_cast<Py_ssize_t>(shape_of(*h)[a])};
        const Py_ssize_t last {start_raw[a] + ((count_raw[a] - 1) * step_raw[a])};
        // The Python side resolves every key in bounds; a direct call that did not is refused, not read.
        if (count_raw[a] < 0 || (count_raw[a] > 0 && (start_raw[a] < 0 || start_raw[a] >= dim || last < 0 || last >= dim))) {
            PyErr_SetString(PyExc_IndexError, "_take: a range leaves its axis");
            return nullptr;
        }
        start_v[a] = static_cast<std::size_t>(start_raw[a]);
        step_v[a]  = static_cast<std::ptrdiff_t>(step_raw[a]);
        count_v[a] = static_cast<std::size_t>(count_raw[a]);
        keep_v[a]  = keep_raw[a] != 0 ? 1U : 0U;
    }
    try {
        if (is_complex(*h)) {
            return take_from(std::get<cplx_tensor>(*h), start_v, step_v, count_v, keep_v);
        }
        return take_from(std::get<real_tensor>(*h), start_v, step_v, count_v, keep_v);
    }
    catch (...) {
        return set_cpp_error();
    }
}

// --- element-wise arithmetic, reductions, reshaping and creation (tensor_ops.hpp) ----------------------
// The Python side promotes the operands to one dtype and resolves negative axes and a -1 extent; these take
// what it hands over and refuse anything else.

// Two tensors of one dtype, or an error set and nullptr.
bool same_dtype_pair(PyObject* x, PyObject* y, const any_tensor*& a, const any_tensor*& b)
{
    a = as_tensor(x);
    b = a == nullptr ? nullptr : as_tensor(y);
    if (b == nullptr) {
        return false;
    }
    if (is_complex(*a) != is_complex(*b)) {
        PyErr_SetString(sciforge_module_error(), "element-wise operands must share one dtype");
        return false;
    }
    return true;
}

template <typename T>
PyObject* binary_on(char op, const numerical::tensor<T>& a, const numerical::tensor<T>& b)
{
    switch (op) {
        case '+': return wrap(any_tensor {a + b});
        case '-': return wrap(any_tensor {a - b});
        case '*': return wrap(any_tensor {a * b});
        case '^': return wrap(any_tensor {numerical::pow(a, b)});
        default: return wrap(any_tensor {a / b});
    }
}

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
        if (is_complex(*a)) {
            return binary_on(op[0], std::get<cplx_tensor>(*a), std::get<cplx_tensor>(*b));
        }
        return binary_on(op[0], std::get<real_tensor>(*a), std::get<real_tensor>(*b));
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
    const any_tensor* a = as_tensor(x);
    if (a == nullptr) {
        return nullptr;
    }
    if (is_complex(*a)) {
        return wrap(any_tensor {-std::get<cplx_tensor>(*a)});
    }
    return wrap(any_tensor {-std::get<real_tensor>(*a)});
}

template <typename T>
PyObject* unary_on(std::string_view name, const numerical::tensor<T>& t)
{
    using numerical::map;
    if (name == "exp") {
        return wrap(any_tensor {map(t, [](const T& x) { return std::exp(x); })});
    }
    if (name == "log") {
        return wrap(any_tensor {map(t, [](const T& x) { return std::log(x); })});
    }
    if (name == "sqrt") {
        return wrap(any_tensor {map(t, [](const T& x) { return std::sqrt(x); })});
    }
    if (name == "sin") {
        return wrap(any_tensor {map(t, [](const T& x) { return std::sin(x); })});
    }
    if (name == "cos") {
        return wrap(any_tensor {map(t, [](const T& x) { return std::cos(x); })});
    }
    if (name == "tan") {
        return wrap(any_tensor {map(t, [](const T& x) { return std::tan(x); })});
    }
    if (name == "tanh") {
        return wrap(any_tensor {map(t, [](const T& x) { return std::tanh(x); })});
    }
    if (name == "abs") { // a real result for either dtype, as NumPy's absolute
        return wrap(any_tensor {map(t, [](const T& x) { return static_cast<double>(std::abs(x)); })});
    }
    PyErr_SetString(PyExc_ValueError, "_unary: name is exp, log, sqrt, sin, cos, tan, tanh or abs");
    return nullptr;
}

// _unary(name, a) -> Tensor: the function `name` of every element.
PyObject* module_unary(PyObject* /*self*/, PyObject* args)
{
    const char* name = nullptr;
    PyObject*   x    = nullptr;
    if (PyArg_ParseTuple(args, "sO", &name, &x) == 0) {
        return nullptr;
    }
    const any_tensor* a = as_tensor(x);
    if (a == nullptr) {
        return nullptr;
    }
    if (is_complex(*a)) {
        return unary_on(name, std::get<cplx_tensor>(*a));
    }
    return unary_on(name, std::get<real_tensor>(*a));
}

template <typename T>
PyObject* sum_or_mean(bool mean, const numerical::tensor<T>& t, Py_ssize_t axis)
{
    if (axis < 0) {
        return from_scalar<T>(mean ? numerical::mean(t) : numerical::sum(t));
    }
    const auto a {static_cast<std::size_t>(axis)};
    return wrap(any_tensor {mean ? numerical::mean(t, a) : numerical::sum(t, a)});
}

PyObject* extreme_of(bool largest, const real_tensor& t, Py_ssize_t axis)
{
    if (axis < 0) {
        return from_scalar<double>(largest ? numerical::max(t) : numerical::min(t));
    }
    const auto a {static_cast<std::size_t>(axis)};
    return wrap(any_tensor {largest ? numerical::max(t, a) : numerical::min(t, a)});
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
    const any_tensor* a = as_tensor(x);
    if (a == nullptr) {
        return nullptr;
    }
    const std::string_view name {op};
    try {
        if (name == "sum" || name == "mean") {
            if (is_complex(*a)) {
                return sum_or_mean(name == "mean", std::get<cplx_tensor>(*a), axis);
            }
            return sum_or_mean(name == "mean", std::get<real_tensor>(*a), axis);
        }
        if (name == "max" || name == "min") {
            if (is_complex(*a)) {
                PyErr_SetString(PyExc_TypeError, "max and min take a float64 tensor: complex128 has no order");
                return nullptr;
            }
            return extreme_of(name == "max", std::get<real_tensor>(*a), axis);
        }
        PyErr_SetString(PyExc_ValueError, "_reduce: op is sum, mean, max or min");
        return nullptr;
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _reshape(a, shape) -> Tensor.
PyObject* module_reshape(PyObject* /*self*/, PyObject* args)
{
    PyObject* x   = nullptr;
    PyObject* seq = nullptr;
    if (PyArg_ParseTuple(args, "OO", &x, &seq) == 0) {
        return nullptr;
    }
    const any_tensor*        a = as_tensor(x);
    std::vector<std::size_t> shape;
    if (a == nullptr || !read_shape(seq, shape)) {
        return nullptr;
    }
    try {
        if (is_complex(*a)) {
            return wrap(any_tensor {numerical::reshape(std::get<cplx_tensor>(*a), std::move(shape))});
        }
        return wrap(any_tensor {numerical::reshape(std::get<real_tensor>(*a), std::move(shape))});
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _full(shape, value, complex) -> Tensor: float64, or complex128 when complex is true.
PyObject* module_full(PyObject* /*self*/, PyObject* args)
{
    PyObject* seq        = nullptr;
    PyObject* value      = nullptr;
    int       as_complex = 0;
    if (PyArg_ParseTuple(args, "OOp", &seq, &value, &as_complex) == 0) {
        return nullptr;
    }
    std::vector<std::size_t> shape;
    if (!read_shape(seq, shape)) {
        return nullptr;
    }
    try {
        if (as_complex != 0) {
            const double re = PyComplex_RealAsDouble(value); // the Limited API has no Py_complex accessor
            const double im = PyComplex_ImagAsDouble(value);
            if (PyErr_Occurred() != nullptr) {
                return nullptr;
            }
            return wrap(any_tensor {numerical::full(std::move(shape), std::complex<double> {re, im})});
        }
        const double v = PyFloat_AsDouble(value);
        if (v == -1.0 && PyErr_Occurred() != nullptr) {
            return nullptr;
        }
        return wrap(any_tensor {numerical::full(std::move(shape), v)});
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _eye(n, complex) -> Tensor.
PyObject* module_eye(PyObject* /*self*/, PyObject* args)
{
    Py_ssize_t n          = 0;
    int        as_complex = 0;
    if (PyArg_ParseTuple(args, "np", &n, &as_complex) == 0) {
        return nullptr;
    }
    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "negative dimensions are not allowed");
        return nullptr;
    }
    try {
        if (as_complex != 0) {
            return wrap(any_tensor {numerical::eye<std::complex<double>>(static_cast<std::size_t>(n))});
        }
        return wrap(any_tensor {numerical::eye<double>(static_cast<std::size_t>(n))});
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _arange(start, stop, step) -> Tensor (float64).
PyObject* module_arange(PyObject* /*self*/, PyObject* args)
{
    double start = 0.0;
    double stop  = 0.0;
    double step  = 1.0;
    if (PyArg_ParseTuple(args, "ddd", &start, &stop, &step) == 0) {
        return nullptr;
    }
    try {
        return wrap(any_tensor {numerical::arange(start, stop, step)});
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _linspace(start, stop, num) -> Tensor (float64).
PyObject* module_linspace(PyObject* /*self*/, PyObject* args)
{
    double     start = 0.0;
    double     stop  = 0.0;
    Py_ssize_t num   = 0;
    if (PyArg_ParseTuple(args, "ddn", &start, &stop, &num) == 0) {
        return nullptr;
    }
    if (num < 0) {
        PyErr_SetString(PyExc_ValueError, "linspace: the number of samples must be non-negative");
        return nullptr;
    }
    return wrap(any_tensor {numerical::linspace(start, stop, static_cast<std::size_t>(num))});
}

// slogdet(a) -> (sign, logabsdet), as numpy.linalg.slogdet: the sign a float or complex like determinant.
PyObject* module_slogdet(PyObject* /*self*/, PyObject* args)
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
        PyErr_SetString(sciforge_module_error(), "slogdet expects a matrix (rank-2 tensor)");
        return nullptr;
    }
    try {
        if (is_complex(*h)) {
            const cplx_tensor&                       a {std::get<cplx_tensor>(*h)};
            std::pair<std::complex<double>, double> value;
            {
                const sb::scoped_gil_release unlocked {releases_gil(a.data().size())};
                value = numerical::lu_decomposition<std::complex<double>> {matrix_from(a)}.log_determinant();
            }
            PyObject* sign {PyComplex_FromDoubles(value.first.real(), value.first.imag())}; // new ref
            if (sign == nullptr) {
                return nullptr;
            }
            return Py_BuildValue("(Nd)", sign, value.second); // N: the tuple takes the reference
        }
        const real_tensor&         a {std::get<real_tensor>(*h)};
        std::pair<double, double> value;
        {
            const sb::scoped_gil_release unlocked {releases_gil(a.data().size())};
            value = numerical::lu_decomposition<double> {matrix_from(a)}.log_determinant();
        }
        return Py_BuildValue("(dd)", value.first, value.second);
    }
    catch (...) {
        return set_cpp_error();
    }
}

template <typename T>
std::size_t do_matrix_rank(const numerical::tensor<T>& a)
{
    const sb::scoped_gil_release unlocked {releases_gil(a.data().size())};
    return numerical::matrix_rank(matrix_from(a));
}

// matrix_rank(a) -> int, as numpy.linalg.matrix_rank.
std::size_t matrix_rank(const any_tensor& v)
{
    if (shape_of(v).size() != 2) {
        throw std::runtime_error("matrix_rank expects a matrix (rank-2 tensor)");
    }
    if (is_complex(v)) {
        return do_matrix_rank<std::complex<double>>(std::get<cplx_tensor>(v));
    }
    return do_matrix_rank<double>(std::get<real_tensor>(v));
}

template <typename T>
any_tensor do_pinv(const numerical::tensor<T>& a)
{
    const sb::scoped_gil_release unlocked {releases_gil(a.data().size())};
    return any_tensor {numerical::tensor<T> {numerical::pinv(matrix_from(a))}};
}

// pinv(a) -> Tensor, as numpy.linalg.pinv.
any_tensor pinv(const any_tensor& v)
{
    if (shape_of(v).size() != 2) {
        throw std::runtime_error("pinv expects a matrix (rank-2 tensor)");
    }
    if (is_complex(v)) {
        return do_pinv<std::complex<double>>(std::get<cplx_tensor>(v));
    }
    return do_pinv<double>(std::get<real_tensor>(v));
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
    const numerical::matrix<T>   coeff {matrix_from(a)};
    return solve_for(b, [&coeff](const numerical::vector<T>& rhs) { return numerical::lstsq(coeff, rhs); },
                     [&coeff](const numerical::matrix<T>& rhs) { return numerical::lstsq(coeff, rhs); });
}

// lstsq(a, b) -> Tensor (the minimum-norm least-squares solution of a x ≈ b, from the SVD).
any_tensor lstsq(const any_tensor& a, const any_tensor& b)
{
    if (shape_of(a).size() != 2 || (shape_of(b).size() != 1 && shape_of(b).size() != 2)) {
        throw std::runtime_error("lstsq(a, b) needs a rank-2 a and a rank-1 or rank-2 b");
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
// eigh(a) -> (eigenvalues, eigenvectors) for a real symmetric or a complex Hermitian matrix: the values are real.
std::tuple<any_tensor, any_tensor> eigh(const any_tensor& v)
{
    if (shape_of(v).size() != 2) {
        throw std::runtime_error("eigh expects a matrix (rank-2 tensor)");
    }
    if (is_complex(v)) {
        const cplx_tensor&               a = std::get<cplx_tensor>(v);
        const sb::scoped_gil_release     unlocked {releases_gil(a.data().size())};
        const numerical::hermitian_eigen decomposition {matrix_from(a)};
        return {any_tensor {real_tensor {decomposition.values()}},
                any_tensor {cplx_tensor {decomposition.vectors()}}};
    }
    const real_tensor&               a = std::get<real_tensor>(v);
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
// An element type asarray reads from an __array_interface__ typestr: a byte order ('<', '>', '=' native,
// or '|' where order does not apply), a kind (b bool, i signed, u unsigned, f float, c complex) and a size.
struct element_type
{
    char        kind;    // b, i, u, f or c
    std::size_t size;    // bytes per element
    bool        swapped; // stored in the other byte order than this machine's
};

std::optional<element_type> parse_typestr(const std::string& typestr)
{
    if (typestr.size() < 3 || typestr.find_first_not_of("0123456789", 2) != std::string::npos) {
        return std::nullopt;
    }
    const char        order {typestr[0]};
    const char        kind {typestr[1]};
    const std::size_t size {static_cast<std::size_t>(std::stoul(typestr.substr(2)))};
    const bool        known_size {(kind == 'b' && size == 1)
                                  || ((kind == 'i' || kind == 'u') && (size == 1 || size == 2 || size == 4 || size == 8))
                                  || (kind == 'f' && (size == 4 || size == 8))
                                  || (kind == 'c' && (size == 8 || size == 16))};
    const bool        known_order {order == '<' || order == '>' || order == '=' || (order == '|' && size == 1)};
    if (!known_size || !known_order) {
        return std::nullopt;
    }
    return element_type {kind, size, size > 1 && (order == '<' || order == '>') && order != byte_order()};
}

// Reads the scalar of type `bytes`'s element as `T` (a native copy, so any alignment reads).
template <typename T>
T read_scalar(const unsigned char* bytes)
{
    T value {};
    std::memcpy(&value, bytes, sizeof(T));
    return value;
}

// The element at `at`, as a complex number (a zero imaginary part for a real kind).
std::complex<double> read_element(const unsigned char* at, const element_type& type)
{
    unsigned char bytes[16];
    std::memcpy(bytes, at, type.size);
    if (type.swapped) { // each component in turn: a complex is two floats
        const std::size_t part {type.kind == 'c' ? type.size / 2 : type.size};
        for (std::size_t start = 0; start < type.size; start += part) {
            std::reverse(bytes + start, bytes + start + part);
        }
    }
    switch (type.kind) {
        case 'b': return {bytes[0] != 0 ? 1.0 : 0.0, 0.0};
        case 'i':
            switch (type.size) {
                case 1:  return {static_cast<double>(read_scalar<std::int8_t>(bytes)), 0.0};
                case 2:  return {static_cast<double>(read_scalar<std::int16_t>(bytes)), 0.0};
                case 4:  return {static_cast<double>(read_scalar<std::int32_t>(bytes)), 0.0};
                default: return {static_cast<double>(read_scalar<std::int64_t>(bytes)), 0.0};
            }
        case 'u':
            switch (type.size) {
                case 1:  return {static_cast<double>(read_scalar<std::uint8_t>(bytes)), 0.0};
                case 2:  return {static_cast<double>(read_scalar<std::uint16_t>(bytes)), 0.0};
                case 4:  return {static_cast<double>(read_scalar<std::uint32_t>(bytes)), 0.0};
                default: return {static_cast<double>(read_scalar<std::uint64_t>(bytes)), 0.0};
            }
        case 'f':
            return {type.size == 4 ? static_cast<double>(read_scalar<float>(bytes)) : read_scalar<double>(bytes), 0.0};
        default: // 'c'
            if (type.size == 8) {
                return {static_cast<double>(read_scalar<float>(bytes)), static_cast<double>(read_scalar<float>(bytes + 4))};
            }
            return {read_scalar<double>(bytes), read_scalar<double>(bytes + 8)};
    }
}

// Copies the elements at `base` (byte `strides` per axis) into `out` in C order, each converted from `Src`
// to `Out`: the type is resolved once for the whole copy, and the innermost axis runs as one strided loop.
template <typename Src, typename Out>
void gather_typed(const unsigned char*           base,
                  const std::vector<std::size_t>& shape,
                  const std::vector<Py_ssize_t>&  strides,
                  Out*                            out)
{
    const std::size_t rank {shape.size()};
    if (rank == 0) {
        *out = static_cast<Out>(read_scalar<Src>(base));
        return;
    }
    const std::size_t        inner  {shape[rank - 1]};
    const Py_ssize_t         step   {strides[rank - 1]};
    std::vector<std::size_t> index(rank - 1, 0);
    std::size_t              rows   {1};
    for (std::size_t axis = 0; axis + 1 < rank; ++axis) {
        rows *= shape[axis];
    }
    for (std::size_t row = 0; row < rows; ++row) {
        Py_ssize_t offset {0};
        for (std::size_t axis = 0; axis + 1 < rank; ++axis) {
            offset += static_cast<Py_ssize_t>(index[axis]) * strides[axis];
        }
        const unsigned char* at {base + offset};
        if (step == static_cast<Py_ssize_t>(sizeof(Src)) && std::is_same_v<Src, Out>) {
            std::memcpy(out, at, inner * sizeof(Src)); // a contiguous run of the output's own type
        }
        else {
            for (std::size_t i = 0; i < inner; ++i) {
                if constexpr (std::is_same_v<Src, bool>) {
                    out[i] = static_cast<Out>(at[static_cast<Py_ssize_t>(i) * step] != 0 ? 1.0 : 0.0);
                }
                else {
                    out[i] = static_cast<Out>(read_scalar<Src>(at + (static_cast<Py_ssize_t>(i) * step)));
                }
            }
        }
        out += inner;
        for (std::size_t axis = rank - 1; axis-- > 0;) {
            if (++index[axis] < shape[axis]) {
                break;
            }
            index[axis] = 0;
        }
    }
}

// gather_typed for the element type `type` names; false for a byte-swapped one, which takes read_element.
template <typename Out>
bool gather(const unsigned char*           base,
            const std::vector<std::size_t>& shape,
            const std::vector<Py_ssize_t>&  strides,
            const element_type&             type,
            Out*                            out)
{
    if (type.swapped) {
        return false;
    }
    switch (type.kind) {
        case 'b': gather_typed<bool>(base, shape, strides, out); return true;
        case 'i':
            switch (type.size) {
                case 1:  gather_typed<std::int8_t>(base, shape, strides, out); return true;
                case 2:  gather_typed<std::int16_t>(base, shape, strides, out); return true;
                case 4:  gather_typed<std::int32_t>(base, shape, strides, out); return true;
                default: gather_typed<std::int64_t>(base, shape, strides, out); return true;
            }
        case 'u':
            switch (type.size) {
                case 1:  gather_typed<std::uint8_t>(base, shape, strides, out); return true;
                case 2:  gather_typed<std::uint16_t>(base, shape, strides, out); return true;
                case 4:  gather_typed<std::uint32_t>(base, shape, strides, out); return true;
                default: gather_typed<std::uint64_t>(base, shape, strides, out); return true;
            }
        case 'f':
            if (type.size == 4) {
                gather_typed<float>(base, shape, strides, out);
            }
            else {
                gather_typed<double>(base, shape, strides, out);
            }
            return true;
        default: // 'c': only into a complex output (a complex source never reaches a real one)
            if constexpr (std::is_same_v<Out, std::complex<double>>) {
                if (type.size == 8) {
                    gather_typed<std::complex<float>>(base, shape, strides, out);
                }
                else {
                    gather_typed<std::complex<double>>(base, shape, strides, out);
                }
                return true;
            }
            else {
                return false;
            }
    }
}

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

    if (PyObject_TypeCheck(obj, reinterpret_cast<PyTypeObject*>(sb::class_type<any_tensor>())) != 0) {
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

    PyObject* strides_obj = PyDict_GetItemString(ai, "strides"); // borrowed
    PyObject* typestr_obj = PyDict_GetItemString(ai, "typestr");  // borrowed
    PyObject* shape_obj   = PyDict_GetItemString(ai, "shape");    // borrowed
    PyObject* data_obj    = PyDict_GetItemString(ai, "data");     // borrowed
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

    const std::optional<element_type> element {parse_typestr(typestr)};
    if (!element) {
        Py_DECREF(ai);
        PyErr_SetString(sciforge_module_error(),
                        "asarray: unsupported typestr (bool, integers, float32/64 and complex64/128 are read)");
        return nullptr;
    }
    const bool want_complex = element->kind == 'c';
    // The output dtype is the requested one, else complex128 for a complex source and float64 otherwise.
    const bool out_complex = dtype_given ? want_complex_out : want_complex;
    if (want_complex && !out_complex) {
        Py_DECREF(ai);
        PyErr_SetString(sciforge_module_error(),
                        "asarray: refusing to cast a complex array to float64 (discards the imaginary part)");
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

    // Byte strides: the array's own (a transposed, sliced or reversed view has them), else C order.
    std::vector<Py_ssize_t> strides(shape.size());
    if (strides_obj != nullptr && strides_obj != Py_None) {
        if (PySequence_Size(strides_obj) != static_cast<Py_ssize_t>(shape.size())) {
            Py_DECREF(ai);
            PyErr_SetString(sciforge_module_error(), "asarray: strides and shape differ in length");
            return nullptr;
        }
        for (std::size_t axis = 0; axis < shape.size(); ++axis) {
            PyObject* item = PySequence_GetItem(strides_obj, static_cast<Py_ssize_t>(axis)); // new ref
            strides[axis]  = item != nullptr ? PyLong_AsSsize_t(item) : -1;
            Py_XDECREF(item);
            if (PyErr_Occurred() != nullptr) {
                Py_DECREF(ai);
                return nullptr;
            }
        }
    }
    else {
        auto step = static_cast<Py_ssize_t>(element->size);
        for (std::size_t axis = shape.size(); axis-- > 0;) {
            strides[axis] = step;
            step         *= static_cast<Py_ssize_t>(shape[axis]);
        }
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
        const auto* base {static_cast<const unsigned char*>(address)};
        if (!element->swapped && (out_complex || element->kind != 'c')) {
            // The common case: the conversion picked once, the copy without the GIL (it reads only the buffer).
            if (out_complex) {
                std::vector<std::complex<double>> values(total);
                {
                    const sb::scoped_gil_release unlocked {releases_gil(total)};
                    gather(base, shape, strides, *element, values.data());
                }
                result = wrap(any_tensor {cplx_tensor {shape, std::move(values)}});
            }
            else {
                std::vector<double> values(total);
                {
                    const sb::scoped_gil_release unlocked {releases_gil(total)};
                    gather(base, shape, strides, *element, values.data());
                }
                result = wrap(any_tensor {real_tensor {shape, std::move(values)}});
            }
            Py_DECREF(ai);
            return result;
        }
        // A byte-swapped element: each read through read_element, in C order.
        std::vector<std::complex<double>> values;
        values.reserve(total);
        std::vector<std::size_t> index(shape.size(), 0);
        for (std::size_t n = 0; n < total; ++n) {
            Py_ssize_t offset = 0;
            for (std::size_t axis = 0; axis < shape.size(); ++axis) {
                offset += static_cast<Py_ssize_t>(index[axis]) * strides[axis];
            }
            values.push_back(read_element(base + offset, *element));
            for (std::size_t axis = shape.size(); axis-- > 0;) {
                if (++index[axis] < shape[axis]) {
                    break;
                }
                index[axis] = 0;
            }
        }
        if (out_complex) {
            result = wrap(any_tensor {cplx_tensor {shape, std::move(values)}});
        }
        else {
            std::vector<double> reals;
            reals.reserve(total);
            for (const std::complex<double>& value : values) {
                reals.push_back(value.real());
            }
            result = wrap(any_tensor {real_tensor {shape, std::move(reals)}});
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
    // The Tensor type: a def_init constructor, the DLPack/list methods, and the
    // introspection properties (shape/__array_interface__ are computed PyObject* getters).
    // init_once: np.asarray, __array_interface__ and __dlpack__ view the held buffer in place, so a
    // second __init__ would free what those views read.
    m.type<any_tensor>("numerical.Tensor")
    .init_once()
    .def_init<&tensor_factory>(sb::arg("data"), sb::arg("shape") = sb::none, sb::arg("dtype") = "float64")
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
    m.def<&solve>("solve", "solve(a, b) -> Tensor\nSolve the linear system a @ x = b (a rank-2; b rank-1, or rank-2 with one right-hand side per column).");
    m.def<&inverse>("inverse", "inverse(a) -> Tensor\nThe inverse of a square matrix.");
    m.def<&lstsq>("lstsq", "lstsq(a, b) -> Tensor\nThe minimum-norm least-squares solution of a @ x ≈ b, as numpy.linalg.lstsq (SVD; a rank-2; b rank-1, or rank-2 with one right-hand side per column).");
    m.def<&cholesky>("cholesky", "cholesky(a) -> Tensor\nThe lower factor L of A = L Lᴴ (Hermitian positive-definite).");
    m.def<&qr>("qr", "qr(a) -> (Q, R)\nThe QR factorisation of an m×n matrix, as numpy.linalg.qr (reduced): Q is m×k, R k×n, k = min(m, n).");
    m.def<&eigh>("eigh", "eigh(a) -> (values, vectors)\nEigenpairs of a real symmetric or complex Hermitian matrix (real values, ascending); its lower triangle is read, as numpy.linalg.eigh reads it.");
    m.def<&matrix_rank>("matrix_rank", "matrix_rank(a) -> int\nThe numerical rank of a matrix, as numpy.linalg.matrix_rank.");
    m.def<&pinv>("pinv", "pinv(a) -> Tensor\nThe Moore–Penrose pseudo-inverse of an m×n matrix, as numpy.linalg.pinv.");
    m.def<&svd>("svd", "svd(a) -> (U, S, Vh)\nThe singular value decomposition a = U @ diag(S) @ Vh of an m×n matrix, as numpy.linalg.svd(a, full_matrices=False).");
    // The 3 that stay manual: determinant (dtype-dependent scalar), einsum (variadic),
    // asarray (returns the same object when the dtype matches — Py_NewRef identity).
    m.raw("determinant", module_determinant, METH_VARARGS,
          "determinant(a) -> number\nThe determinant of a square matrix (float or complex).");
    m.raw("_take", module_take, METH_VARARGS,
          "_take(t, starts, steps, counts, keep) -> Tensor or number\nThe strided copy behind Tensor.__getitem__.");
    m.raw("_binary", module_binary, METH_VARARGS,
          "_binary(op, a, b) -> Tensor\nElement-wise a op b (op one of + - * /), broadcast; one dtype.");
    m.raw("_negate", module_negate, METH_VARARGS, "_negate(a) -> Tensor\nElement-wise negation.");
    m.raw("_unary", module_unary, METH_VARARGS,
          "_unary(name, a) -> Tensor\nexp, log, sqrt, sin, cos, tan, tanh or abs of every element.");
    m.raw("_reduce", module_reduce, METH_VARARGS,
          "_reduce(op, a, axis) -> number or Tensor\nsum, mean, max or min of every element (axis -1) or along an axis.");
    m.raw("_reshape", module_reshape, METH_VARARGS, "_reshape(a, shape) -> Tensor\nThe elements under another shape.");
    m.raw("_full", module_full, METH_VARARGS, "_full(shape, value, complex) -> Tensor\nA tensor of one value.");
    m.raw("_eye", module_eye, METH_VARARGS, "_eye(n, complex) -> Tensor\nThe n×n identity.");
    m.raw("_arange", module_arange, METH_VARARGS, "_arange(start, stop, step) -> Tensor\nAs numpy.arange (float64).");
    m.raw("_linspace", module_linspace, METH_VARARGS,
          "_linspace(start, stop, num) -> Tensor\nAs numpy.linspace with the end point (float64).");
    m.raw("slogdet", module_slogdet, METH_VARARGS,
          "slogdet(a) -> (sign, logabsdet)\nThe sign and the natural log of |det(a)|, as numpy.linalg.slogdet; it\n"
          "does not underflow or overflow where determinant(a) does.");
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
