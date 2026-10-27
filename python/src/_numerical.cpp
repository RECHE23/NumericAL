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
#include <cctype>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <map>
#include <new>
#include <array>
#include <optional>
#include <type_traits>
#include <stdexcept>
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

using real_tensor    = numerical::tensor<double>;
using cplx_tensor    = numerical::tensor<std::complex<double>>;
using single_tensor  = numerical::tensor<float>;
using csingle_tensor = numerical::tensor<std::complex<float>>;
// float64, complex128, float32, complex64: the index is the dtype (dtype_names).
using any_tensor = std::variant<real_tensor, cplx_tensor, single_tensor, csingle_tensor>;

// The dtypes by variant index, and the array-interface type codes of their elements.
constexpr std::array<const char*, 4> dtype_names {"float64", "complex128", "float32", "complex64"};
constexpr std::array<const char*, 4> type_codes {"f8", "c16", "f4", "c8"};

// The variant index of a dtype's name, or nothing.
std::optional<std::size_t> dtype_index(std::string_view name)
{
    for (std::size_t i = 0; i < dtype_names.size(); ++i) {
        if (name == dtype_names[i]) {
            return i;
        }
    }
    return std::nullopt;
}

// Whether T is a complex scalar, and its real counterpart.
template <typename T>
inline constexpr bool complex_scalar = false;
template <typename U>
inline constexpr bool complex_scalar<std::complex<U>> = true;

template <typename T>
struct real_of
{
    using type = T;
};
template <typename U>
struct real_of<std::complex<U>>
{
    using type = U;
};
template <typename T>
using real_of_t = typename real_of<T>::type;

// The element type of a tensor type.
template <typename Tensor>
struct element_of;
template <typename T>
struct element_of<numerical::tensor<T>>
{
    using type = T;
};
template <typename Tensor>
using element_of_t = typename element_of<Tensor>::type;

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
    return std::visit([](const auto& t) -> const std::vector<std::size_t>& { return t.shape(); }, v);
}

std::size_t count_of(const any_tensor& v)
{
    return std::visit([](const auto& t) { return t.size(); }, v);
}

// An empty tensor's buffer has no address; the exchange protocols get one that is never read: NumPy 2.2
// refuses a null data pointer and falls back to the sequence protocol, which an empty tensor cannot serve.
const void* data_of(const any_tensor& v)
{
    static constexpr std::complex<double> placeholder {};
    const void* data = std::visit([](const auto& t) { return static_cast<const void*>(t.data().data()); }, v);
    return data != nullptr ? data : &placeholder;
}

bool is_complex(const any_tensor& v)
{
    return v.index() == 1 || v.index() == 3;
}

// Whether a tensor holds single-precision elements, float32 or complex64.
bool is_single(const any_tensor& v)
{
    return v.index() >= 2;
}

const char* dtype_name(const any_tensor& v)
{
    return dtype_names.at(v.index());
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
    out += type_codes.at(v.index());
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
float to_scalar<float>(PyObject* item)
{
    return static_cast<float>(PyFloat_AsDouble(item));
}

template <>
std::complex<float> to_scalar<std::complex<float>>(PyObject* item)
{
    return {static_cast<float>(PyComplex_RealAsDouble(item)), static_cast<float>(PyComplex_ImagAsDouble(item))};
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

template <>
PyObject* from_scalar<float>(float value)
{
    return PyFloat_FromDouble(static_cast<double>(value));
}

template <>
PyObject* from_scalar<std::complex<float>>(std::complex<float> value)
{
    return PyComplex_FromDoubles(static_cast<double>(value.real()), static_cast<double>(value.imag()));
}

template <typename T>
bool read_buffer(PyObject* sequence, numerical::buffer<T>& out)
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

// `x` as an element of `Out`, a dtype's conversion: a complex `Out` takes each part in the type of its parts, where
// building it from a wider number narrowed that implicitly (MSVC's C4244, clang's -Wconversion).
template <typename Out, typename In>
Out converted(const In& x)
{
    if constexpr (complex_scalar<Out>) {
        using R = typename Out::value_type;
        if constexpr (complex_scalar<In>) {
            return Out {static_cast<R>(x.real()), static_cast<R>(x.imag())};
        }
        else {
            return Out {static_cast<R>(x)};
        }
    }
    else {
        return static_cast<Out>(x);
    }
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
        const Py_ssize_t extent = PyLong_AsSsize_t(item); // a pointer's width everywhere, where long is 32 bits under Windows
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
    numerical::buffer<T> buffer;
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
    switch (dtype_index(dtype).value_or(dtype_names.size())) {
        case 0: return build_any<double>(data, shape);
        case 1: return build_any<std::complex<double>>(data, shape);
        case 2: return build_any<float>(data, shape);
        case 3: return build_any<std::complex<float>>(data, shape);
        default: throw std::runtime_error("dtype must be 'float64', 'complex128', 'float32' or 'complex64'");
    }
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
    const auto bits                = static_cast<std::uint8_t>((is_single(*h) ? 32 : 64) * (is_complex(*h) ? 2 : 1));
    managed->dl_tensor.dtype       = DLDataType {is_complex(*h) ? kDLComplex : kDLFloat, bits, 1};
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
    return std::visit([&offset](const auto& tensor) { return nested(tensor.data().data(), tensor.shape(), 0, offset); }, *h);
}

// --- module functions ------------------------------------------------------
// Whether a computation over this many elements releases the GIL. Re-acquiring it may wait out another
// thread's switch interval (5 ms by default), which a small matrix would pay many times over; from 64 × 64
// up, the computation dominates. The operands stay readable meanwhile: a Tensor never changes once built,
// and the call holds a reference to each.
//
// A stack of matrices releases the GIL once, around the whole stack: the computations of each matrix, on this thread
// or on the pool's, then run with `gil_released` set, and release nothing more (a thread that does not hold the GIL
// cannot release it).
thread_local bool gil_released {false};

bool releases_gil(std::size_t elements)
{
    return elements >= std::size_t {4096} && !gil_released;
}

// Marks the calling thread's computations as running without the GIL, for its lifetime.
class without_gil
{
public:

    without_gil() : previous_(gil_released)
    {
        gil_released = true;
    }

    ~without_gil()
    {
        gil_released = previous_;
    }

    without_gil(const without_gil&)            = delete;
    without_gil& operator=(const without_gil&) = delete;
    without_gil(without_gil&&)                 = delete;
    without_gil& operator=(without_gil&&)      = delete;

private:

    bool previous_;
};

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
    std::size_t elements {0};
    for (Py_ssize_t i = 1; i < count; ++i) {
        const any_tensor* h = as_tensor(PyTuple_GetItem(args, i));
        if (h == nullptr) {
            return nullptr;
        }
        holders.push_back(h);
        elements += count_of(*h);
    }
    for (const any_tensor* h : holders) {
        if (h->index() != holders.front()->index()) {
            PyErr_SetString(sciforge_module_error(), "einsum operands must share one dtype");
            return nullptr;
        }
    }
    try {
        std::optional<any_tensor> result;
        {
            const sb::scoped_gil_release unlocked {releases_gil(elements)};
            std::visit([&](const auto& first) {
                           using tensor_type = std::decay_t<decltype(first)>;
                           std::vector<const tensor_type*> operands;
                           operands.reserve(holders.size());
                           for (const any_tensor* h : holders) {
                               operands.push_back(&std::get<tensor_type>(*h));
                           }
                           result.emplace(numerical::detail::einsum_impl<element_of_t<tensor_type>>(spec, operands));
                       }, *holders.front());
        }
        return wrap(std::move(*result));
    }
    catch (...) {
        return set_cpp_error();
    }
}

// The product of `a` and `b` read where they lie (a Tensor's elements, a NumPy array's), when both can be and it is a
// plain one (defined below, beside NumPy's __array_struct__): nullopt otherwise, the caller then converting.
std::optional<PyObject*> borrowed_product(PyObject* a, PyObject* b);

// The Python `_matmul` (numerical/__init__.py), set once at import: every product Tensor._matmul does not take
// itself (an operand to convert, a rank past 2, two dtypes) goes there, so its conversions and errors stay one.
PyObject* matmul_fallback = nullptr;

PyObject* module_set_matmul_fallback(PyObject* /*self*/, PyObject* fallback)
{
    PyObject* previous {matmul_fallback};
    matmul_fallback = Py_NewRef(fallback);
    Py_XDECREF(previous);
    Py_RETURN_NONE;
}

// `self @ other` for two Tensors of one dtype, each a vector or a matrix, as `_matmul` computes it (the same
// einsum), without its Python frame, attribute reads and spec lookup: a product of order 64 took 4.8 µs through
// it, 3.4 in C++.
PyObject* tensor_matmul(PyObject* self, PyObject* other)
{
    if (PyObject_TypeCheck(other, reinterpret_cast<PyTypeObject*>(sb::class_type<any_tensor>())) != 0) {
        const any_tensor* a = as_tensor(self);
        const any_tensor* b = as_tensor(other);
        if (a == nullptr || b == nullptr) {
            return nullptr;
        }
        const std::size_t ranks[2] {shape_of(*a).size(), shape_of(*b).size()};
        static constexpr std::array<std::array<const char*, 2>, 2> specs {{{"i,i->", "i,ij->j"}, {"ij,j->i", "ij,jk->ik"}}};
        if (a->index() == b->index() && ranks[0] >= 1 && ranks[0] <= 2 && ranks[1] >= 1 && ranks[1] <= 2) {
            const std::string spec {specs.at(ranks[0] - 1).at(ranks[1] - 1)};
            try {
                std::optional<any_tensor> result;
                {
                    const sb::scoped_gil_release unlocked {releases_gil(count_of(*a) + count_of(*b))};
                    std::visit([&](const auto& x) {
                                   using tensor_type = std::decay_t<decltype(x)>;
                                   const std::vector<const tensor_type*> operands {&x, &std::get<tensor_type>(*b)};
                                   result.emplace(numerical::detail::einsum_impl<element_of_t<tensor_type>>(spec, operands));
                               }, *a);
                }
                if (ranks[0] == 1 && ranks[1] == 1) { // two vectors give a number, as NumPy's
                    return std::visit([](const auto& t) { return from_scalar(t.data()[0]); }, *result);
                }
                return wrap(std::move(*result));
            }
            catch (...) {
                return set_cpp_error();
            }
        }
    }
    else if (const std::optional<PyObject*> product {borrowed_product(self, other)}) { // a NumPy array, read in place
        return *product;
    }
    if (matmul_fallback == nullptr) {
        PyErr_SetString(sciforge_module_error(), "Tensor._matmul: numerical/__init__.py set no fallback");
        return nullptr;
    }
    return PyObject_CallFunctionObjArgs(matmul_fallback, self, other, nullptr);
}

// _matmul_nd(a, b) -> Tensor: numerical::matmul, the product of two stacks of one dtype, their leading axes broadcast.
any_tensor matmul_nd(const any_tensor& a, const any_tensor& b)
{
    if (a.index() != b.index()) {
        throw std::runtime_error("_matmul_nd: the operands share one dtype");
    }
    const sb::scoped_gil_release unlocked {releases_gil(count_of(a) + count_of(b))};
    return std::visit([&](const auto& x) -> any_tensor {
                          using tensor_type = std::decay_t<decltype(x)>;
                          return numerical::matmul(x, std::get<tensor_type>(b));
                      }, a);
}

// --- the linear-algebra functions (m.def: any_tensor in/out through caster<any_tensor>) ---
// Validation failures throw -> numerical.error; the variant dispatch (is_complex / std::get)
// stays here in the consumer (the substrate grows no variant machinery).
// The rows × cols matrix of the row-major elements at `data`, copied on threads (its pages faulted in there).
template <typename T>
numerical::matrix<T> matrix_of_elements(const T* data, std::size_t rows, std::size_t cols)
{
    numerical::buffer<T> values(rows * cols);
    numerical::detail::parallel_copy(data, rows * cols, values.data());
    return numerical::matrix<T> {rows, cols, std::move(values)};
}

template <typename T>
numerical::matrix<T> matrix_from(const numerical::tensor<T>& t)
{
    return matrix_of_elements(t.data().data(), t.shape()[0], t.shape()[1]);
}

// A tensor's elements in another scalar type, converted one by one.
template <typename Out, typename In>
numerical::tensor<Out> converted(const numerical::tensor<In>& t)
{
    numerical::buffer<Out> values(t.size());
    std::transform(t.data().begin(), t.data().end(), values.begin(), [](const In& x) { return converted<Out>(x); });
    return numerical::tensor<Out> {t.shape(), std::move(values)};
}

// A float32 or complex64 tensor in double precision, float64 or complex128: the decompositions compute there, and
// \ref narrowed rounds their answers back. Any other, as it is.
any_tensor widened(const any_tensor& v)
{
    if (const auto* single {std::get_if<single_tensor>(&v)}) {
        return any_tensor {converted<double>(*single)};
    }
    if (const auto* csingle {std::get_if<csingle_tensor>(&v)}) {
        return any_tensor {converted<std::complex<double>>(*csingle)};
    }
    return v;
}

// A float64 or complex128 answer rounded to float32 or complex64 (a single-precision input's dtype, as NumPy's
// linalg returns it); an integer (a rank) as it is.
any_tensor narrowed(const any_tensor& v)
{
    if (const auto* real {std::get_if<real_tensor>(&v)}) {
        return any_tensor {converted<float>(*real)};
    }
    if (const auto* complex {std::get_if<cplx_tensor>(&v)}) {
        return any_tensor {converted<std::complex<float>>(*complex)};
    }
    return v;
}

std::size_t narrowed(std::size_t n)
{
    return n;
}

template <typename... Ts>
std::tuple<Ts...> narrowed(const std::tuple<Ts...>& values)
{
    return std::apply([](const auto&... each) { return std::tuple<Ts...> {narrowed(each)...}; }, values);
}

// --- stacks of matrices: (..., m, n), the leading axes a batch, as NumPy's linalg takes them --------------------

// The order up to which the matrices of a stack go to the threads together, one each, however few: a decomposition
// that small shares little of itself among threads. A larger one keeps its own threads, the stack going a matrix at a
// time, unless there are as many matrices as threads.
constexpr std::size_t stack_order {128};

// The work a stacked matrix costs past its arithmetic, in multiply-adds, for parallel_for's split: its slice copied in
// and its answer's tensor built, about 300 ns where a 2 × 2 inverse's arithmetic is ten (M1 Pro). Counted as its
// order cubed alone, a stack of 1000 matrices of order 4 stayed on one thread; with it, inv and det of 256 to 1000
// matrices of order 2 to 6 take 0.4 to 0.6 times as long, 16 and fewer the same (8192 doubled those).
constexpr std::size_t stack_item_work {2048};

// The leading axes of a stacked tensor (rank ≥ 2) and its matrices' rows and columns.
struct stack_shape
{
    std::vector<std::size_t> batch;
    std::size_t              count {1};
    std::size_t              rows {0};
    std::size_t              cols {0};
};

stack_shape stack_of(const any_tensor& v)
{
    const std::vector<std::size_t>& shape {shape_of(v)};
    stack_shape                     out {{shape.begin(), shape.end() - 2}, 1, shape[shape.size() - 2], shape.back()};
    out.count = numerical::detail::element_count(out.batch.begin(), out.batch.end()); // empty matrices: any batch
    return out;
}

// Matrix `i` of the stack `v` (its last two axes `rows` × `cols`), a rank-2 tensor of its dtype.
any_tensor matrix_of(const any_tensor& v, std::size_t rows, std::size_t cols, std::size_t i)
{
    return std::visit([&](const auto& t) -> any_tensor {
                          using T = element_of_t<std::decay_t<decltype(t)>>;
                          const auto first {t.data().begin() + static_cast<std::ptrdiff_t>(i * rows * cols)};
                          return numerical::tensor<T> {{rows, cols}, numerical::buffer<T>(first, first + static_cast<std::ptrdiff_t>(rows * cols))};
                      }, v);
}

// The `rows` × `cols` identity (ones on the diagonal) of `v`'s dtype: an empty stack's answers take their shapes and
// dtypes from those of a matrix every function takes.
any_tensor identity_like(const any_tensor& v, std::size_t rows, std::size_t cols)
{
    return std::visit([&](const auto& t) -> any_tensor {
                          using T = element_of_t<std::decay_t<decltype(t)>>;
                          numerical::buffer<T> values(rows * cols, T {});
                          for (std::size_t d = 0; d < std::min(rows, cols); ++d) {
                              values[(d * cols) + d] = T {1};
                          }
                          return numerical::tensor<T> {{rows, cols}, std::move(values)};
                      }, v);
}

// The answers `parts` (one per matrix, all of one shape and dtype; `probe` stands for them when there are none)
// stacked along the leading axes `batch`.
any_tensor stacked(const std::vector<any_tensor>& parts, const std::vector<std::size_t>& batch, const any_tensor& probe)
{
    const any_tensor& first {parts.empty() ? probe : parts.front()};
    return std::visit([&](const auto& t) -> any_tensor {
                          using tensor_type = std::decay_t<decltype(t)>;
                          using T           = element_of_t<tensor_type>;
                          std::vector<std::size_t> shape {batch};
                          shape.insert(shape.end(), t.shape().begin(), t.shape().end());
                          numerical::buffer<T> values;
                          values.reserve(parts.size() * t.size());
                          for (const any_tensor& part : parts) {
                              const auto& data {std::get<tensor_type>(part).data()};
                              values.insert(values.end(), data.begin(), data.end());
                          }
                          return tensor_type {std::move(shape), std::move(values)};
                      }, first);
}

template <typename... Ts>
std::tuple<Ts...> stacked(const std::vector<std::tuple<Ts...>>& parts, const std::vector<std::size_t>& batch,
                          const std::tuple<Ts...>& probe)
{
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
        const auto component {[&](auto index) {
            std::vector<any_tensor> each;
            each.reserve(parts.size());
            for (const auto& part : parts) {
                each.push_back(std::get<decltype(index)::value>(part));
            }
            return stacked(each, batch, std::get<decltype(index)::value>(probe));
        }};
        return std::tuple<Ts...> {component(std::integral_constant<std::size_t, I> {})...};
    }(std::index_sequence_for<Ts...> {});
}

// `answer(i)` for each of the `count` items of a batch of leading axes `batch`, stacked (`probe()` answers for an
// empty one): the GIL released once around them all, the items to the threads together where their matrices are no
// larger than `stack_order` or there are as many as threads (each one's own computation is then on one thread), else
// one at a time, each with the threads. Every item's answer is the same to the bit either way.
template <typename Result, typename Answer, typename Probe>
Result over_batch(const std::vector<std::size_t>& batch, std::size_t count, std::size_t order, std::size_t elements,
                  Answer answer, Probe probe)
{
    if (count == 0) {
        return stacked(std::vector<Result> {}, batch, probe());
    }
    std::vector<std::optional<Result>> parts(count);
    {
        const sb::scoped_gil_release unlocked {releases_gil(elements)};
        const without_gil            inside;
        if (order <= stack_order || count >= numerical::max_threads()) {
            numerical::detail::parallel_for(count, (order * order * order) + stack_item_work, [&](std::size_t i) {
                const without_gil worker;
                parts[i].emplace(answer(i));
            });
        }
        else {
            for (std::size_t i = 0; i < count; ++i) {
                parts[i].emplace(answer(i));
            }
        }
    }
    std::vector<Result> answers;
    answers.reserve(count);
    for (std::optional<Result>& part : parts) {
        answers.push_back(std::move(*part));
    }
    return stacked(answers, batch, answers.front()); // the probe stands in for an empty stack's answers alone
}

// `f` of every matrix of the stack `v` (rank > 2), stacked.
template <typename Result, typename F>
Result over_stack(const any_tensor& v, F f)
{
    const stack_shape s {stack_of(v)};
    return over_batch<Result>(s.batch, s.count, std::max(s.rows, s.cols), count_of(v),
                              [&](std::size_t i) { return f(matrix_of(v, s.rows, s.cols, i)); },
                              [&] { return f(identity_like(v, s.rows, s.cols)); });
}

// The relative cutoff of a float32 or complex64 matrix's rank where neither tolerance is given: NumPy's default at
// float32's epsilon (its LAPACK computes in it), where the double default would count its rounding as rank.
std::optional<double> single_rtol(const any_tensor& a, std::optional<double> rtol, std::optional<double> tol)
{
    if (rtol || tol) {
        return rtol;
    }
    const std::vector<std::size_t>& shape {shape_of(a)};
    return static_cast<double>(std::numeric_limits<float>::epsilon()) * static_cast<double>(std::max(shape[0], shape[1]));
}

// The answer of a solver for a rank-1 b (a vector) or a rank-2 b (one right-hand side per column).
template <typename T, typename VectorSolver, typename MatrixSolver>
any_tensor solve_for(const numerical::tensor<T>& b, VectorSolver by_vector, MatrixSolver by_matrix)
{
    if (b.shape().size() == 2) {
        return any_tensor {numerical::tensor<T> {by_matrix(matrix_from(b))}};
    }
    const numerical::vector<T> answer {by_vector(numerical::vector<T> {b.data()})};
    return any_tensor {numerical::tensor<T> {{answer.size()}, numerical::buffer<T>(answer.begin(), answer.end())}};
}

// A solved for b, `a` the copy of A the factorisation takes in place.
template <typename T>
any_tensor solve_copied(numerical::matrix<T> a, const numerical::tensor<T>& b)
{
    const numerical::lu_decomposition<T> factored {std::move(a)};
    return solve_for(b, [&factored](const numerical::vector<T>& rhs) { return factored.solve(rhs); },
                     [&factored](numerical::matrix<T>&& rhs) { return factored.solve(std::move(rhs)); });
}

template <typename T>
any_tensor do_solve(const numerical::tensor<T>& a, const numerical::tensor<T>& b)
{
    const sb::scoped_gil_release unlocked {releases_gil(a.data().size())};
    return solve_copied(matrix_from(a), b);
}

// The leading axes two stacks' answers take: `x` and `y` right-aligned, an axis of 1 (or a missing one) taking the
// other's extent, as NumPy broadcasts them.
std::vector<std::size_t> broadcast_batch(const std::vector<std::size_t>& x, const std::vector<std::size_t>& y)
{
    std::vector<std::size_t> out(std::max(x.size(), y.size()), 1);
    for (std::size_t k = 0; k < out.size(); ++k) {
        const std::size_t from_x {k < x.size() ? x[x.size() - 1 - k] : 1};
        const std::size_t from_y {k < y.size() ? y[y.size() - 1 - k] : 1};
        if (from_x != from_y && from_x != 1 && from_y != 1) {
            throw std::runtime_error("solve: the stacks' leading axes do not broadcast together");
        }
        out[out.size() - 1 - k] = from_x == 1 ? from_y : from_x;
    }
    return out;
}

// The index, in a stack of leading axes `own`, of item `i` of the broadcast leading axes `out`.
std::size_t broadcast_index(std::size_t i, const std::vector<std::size_t>& out, const std::vector<std::size_t>& own)
{
    std::size_t index {0};
    std::size_t stride {1};
    for (std::size_t k = 0; k < own.size(); ++k) {
        const std::size_t axis {out.size() - 1 - k};
        std::size_t       position {i};
        for (std::size_t later = out.size() - 1; later > axis; --later) {
            position /= out[later];
        }
        position %= out[axis];
        const std::size_t extent {own[own.size() - 1 - k]};
        index += (extent == 1 ? 0 : position) * stride;
        stride *= extent;
    }
    return index;
}

// solve(a, b), sharing a dtype: a a matrix or a stack of them (..., m, m); b a vector (m,), solved against every
// matrix, or matrices (..., m, k), the leading axes broadcast as NumPy 2 broadcasts them.
any_tensor solve(const any_tensor& a, const any_tensor& b)
{
    const std::size_t rank_a {shape_of(a).size()};
    const std::size_t rank_b {shape_of(b).size()};
    if (rank_a < 2 || rank_b < 1) {
        throw std::runtime_error("solve(a, b) needs a matrix or a stack of them, and a vector or matrices b");
    }
    if (a.index() != b.index()) {
        throw std::runtime_error("solve operands must share one dtype");
    }
    if (rank_a > 2 || rank_b > 2) {
        const stack_shape              sa {stack_of(a)};
        const bool                     vector_b {rank_b == 1};
        const std::vector<std::size_t> own_b {vector_b ? std::vector<std::size_t> {} : stack_of(b).batch};
        const std::vector<std::size_t> batch {broadcast_batch(sa.batch, own_b)};
        const std::size_t              count {numerical::detail::element_count(batch.begin(), batch.end())};
        const std::size_t b_rows {vector_b ? 0 : shape_of(b)[rank_b - 2]};
        const std::size_t b_cols {shape_of(b).back()};
        const auto        right_side {[&](std::size_t i) {
            return vector_b ? b : matrix_of(b, b_rows, b_cols, broadcast_index(i, batch, own_b));
        }};
        return over_batch<any_tensor>(batch, count, sa.rows, count_of(a) + count_of(b),
                                      [&](std::size_t i) { return solve(matrix_of(a, sa.rows, sa.cols, broadcast_index(i, batch, sa.batch)), right_side(i)); },
                                      [&] { return solve(identity_like(a, sa.rows, sa.cols), vector_b ? b : identity_like(b, b_rows, b_cols)); });
    }
    if (is_single(a)) {
        return narrowed(solve(widened(a), widened(b)));
    }
    if (is_complex(a)) {
        return do_solve<std::complex<double>>(std::get<cplx_tensor>(a), std::get<cplx_tensor>(b));
    }
    return do_solve<double>(std::get<real_tensor>(a), std::get<real_tensor>(b));
}

// The determinant of the matrix `m` as a rank-0 tensor of its dtype (a single-precision one's computed in double and
// rounded): a stack's determinants, stacked.
any_tensor determinant_tensor(const any_tensor& m)
{
    // A double matrix read as it is: widened() would copy it whole for nothing (8 MiB at order 1024).
    const auto of = [](const any_tensor& wide) {
        return is_complex(wide)
                   ? any_tensor {cplx_tensor {{}, numerical::buffer<std::complex<double>> {numerical::determinant(matrix_from(std::get<cplx_tensor>(wide)))}}}
                   : any_tensor {real_tensor {{}, numerical::buffer<double> {numerical::determinant(matrix_from(std::get<real_tensor>(wide)))}}};
    };
    return is_single(m) ? narrowed(of(widened(m))) : of(m);
}

// The sign (of `m`'s dtype) and the logarithm of the magnitude (real, of its precision) of the matrix `m`'s
// determinant, as rank-0 tensors: a stack's, stacked.
std::tuple<any_tensor, any_tensor> slogdet_tensors(const any_tensor& m)
{
    // As determinant_tensor: a double matrix read as it is.
    const auto of = [](const any_tensor& wide) -> std::tuple<any_tensor, any_tensor> {
        if (is_complex(wide)) {
            const auto [sign, logarithm] {numerical::lu_decomposition<std::complex<double>> {matrix_from(std::get<cplx_tensor>(wide))}.log_determinant()};
            return {cplx_tensor {{}, numerical::buffer<std::complex<double>> {sign}}, real_tensor {{}, numerical::buffer<double> {logarithm}}};
        }
        const auto [sign, logarithm] {numerical::lu_decomposition<double> {matrix_from(std::get<real_tensor>(wide))}.log_determinant()};
        return {real_tensor {{}, numerical::buffer<double> {sign}}, real_tensor {{}, numerical::buffer<double> {logarithm}}};
    };
    return is_single(m) ? narrowed(of(widened(m))) : of(m);
}

// determinant(a) -> float or complex: a dtype-dependent scalar (a variant return no single
// caster covers), so it stays a manual .raw function; a stack's determinants, a Tensor.
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
    if (shape_of(*h).size() < 2) {
        PyErr_SetString(sciforge_module_error(), "determinant expects a matrix or a stack of matrices (rank ≥ 2)");
        return nullptr;
    }
    try {
        if (shape_of(*h).size() > 2) {
            return wrap(over_stack<any_tensor>(*h, determinant_tensor));
        }
        std::optional<any_tensor> value; // a single-precision matrix's rounded to its precision, as a stack's
        {
            const sb::scoped_gil_release unlocked {releases_gil(count_of(*h))};
            value.emplace(determinant_tensor(*h));
        }
        return std::visit([](const auto& t) { return from_scalar(t.data()[0]); }, *value);
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
    numerical::buffer<T>           out;
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
        return std::visit([&](const auto& t) { return take_from(t, start_v, step_v, count_v, keep_v); }, *h);
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
    if (a->index() != b->index()) {
        PyErr_SetString(sciforge_module_error(), "element-wise operands must share one dtype");
        return false;
    }
    return true;
}

// NumPy's complex loops as the NumPy beside this process fuses them: its PyPI wheels' until the Python side has
// measured the NumPy it imported (numerical._numpy_loops). Read and written under the GIL.
numerical::detail::numpy_loops& numpy_loops_beside()
{
    static numerical::detail::numpy_loops loops {numerical::detail::numpy_wheel_loops()};
    return loops;
}

// _set_numpy_loops(products, scalar_products, quotients): which of NumPy's complex loops fuse, as measured.
void set_numpy_loops(bool products, bool scalar_products, bool quotients)
{
    numpy_loops_beside() = numerical::detail::numpy_loops {.products = products, .scalar_products = scalar_products, .quotients = quotients};
}

// `a * b` and `a / b` as NumPy's array loops compute them: a real type's are IEEE's, the same; a complex type's
// NumPy's own (numerical::detail::numpy_multiply, numpy_divide, fused as `loops`), where std::complex's operators recover
// infinities that NumPy's leave NaN, and fuse otherwise.
template <typename T>
numerical::tensor<T> product_as_numpy(const numerical::tensor<T>& a, const numerical::tensor<T>& b,
                                      const numerical::detail::numpy_loops& /*loops*/)
{
    return a * b;
}

template <numerical::detail::split_complex T>
numerical::tensor<T> product_as_numpy(const numerical::tensor<T>& a, const numerical::tensor<T>& b,
                                      const numerical::detail::numpy_loops& loops)
{
    return numerical::detail::numpy_multiply(a, b, loops);
}

template <typename T>
numerical::tensor<T> quotient_as_numpy(const numerical::tensor<T>& a, const numerical::tensor<T>& b,
                                       const numerical::detail::numpy_loops& /*loops*/)
{
    return a / b;
}

template <numerical::detail::split_complex T>
numerical::tensor<T> quotient_as_numpy(const numerical::tensor<T>& a, const numerical::tensor<T>& b,
                                       const numerical::detail::numpy_loops& loops)
{
    return numerical::detail::numpy_divide(a, b, loops);
}

// `a ** b` as NumPy's power computes it: a real type's as std::pow (the exponents NumPy takes apart, 2, 0.5 and -1,
// exactly); a complex type's as npymath's npy_cpow (numerical::detail::numpy_power).
template <typename T>
numerical::tensor<T> power_as_numpy(const numerical::tensor<T>& a, const numerical::tensor<T>& b,
                                    const numerical::detail::numpy_loops& /*loops*/)
{
    return numerical::pow(a, b);
}

template <numerical::detail::split_complex T>
numerical::tensor<T> power_as_numpy(const numerical::tensor<T>& a, const numerical::tensor<T>& b,
                                    const numerical::detail::numpy_loops& loops)
{
    return numerical::detail::numpy_power(a, b, loops);
}

template <typename T>
PyObject* binary_on(char op, const numerical::tensor<T>& a, const numerical::tensor<T>& b)
{
    const numerical::detail::numpy_loops loops {numpy_loops_beside()}; // read while the GIL is held
    std::optional<any_tensor>    result;
    {
        const sb::scoped_gil_release unlocked {releases_gil(std::max(a.size(), b.size()))};
        switch (op) {
            case '+': result.emplace(a + b); break;
            case '-': result.emplace(a - b); break;
            case '*': result.emplace(product_as_numpy(a, b, loops)); break;
            case '^': result.emplace(power_as_numpy(a, b, loops)); break;
            default: result.emplace(quotient_as_numpy(a, b, loops)); break;
        }
    }
    return wrap(std::move(*result));
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
        return std::visit([&](const auto& x) { return binary_on(op[0], x, std::get<std::decay_t<decltype(x)>>(*b)); }, *a);
    }
    catch (...) {
        return set_cpp_error();
    }
}

// A complex tensor to the Python integer 2 or -1, or the float 0.5, as NumPy takes an array there: its square (the
// product's vector loop), its reciprocal, its square root.
template <typename T>
PyObject* power_shortcut_on(char kind, const numerical::tensor<T>& t)
{
    if constexpr (numerical::detail::split_complex<T>) {
        const numerical::detail::numpy_loops loops {numpy_loops_beside()};
        std::optional<any_tensor>    result;
        {
            const sb::scoped_gil_release unlocked {releases_gil(t.size())};
            switch (kind) {
                case '2': result.emplace(numerical::detail::numpy_square(t, loops)); break;
                case 'r': result.emplace(numerical::detail::numpy_reciprocal(t, loops)); break;
                default: result.emplace(numerical::sqrt(t)); break;
            }
        }
        return wrap(std::move(*result));
    }
    else {
        PyErr_SetString(PyExc_TypeError, "_power_shortcut: a complex tensor");
        return nullptr;
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
    const any_tensor* a = as_tensor(x);
    if (a == nullptr) {
        return nullptr;
    }
    try {
        return std::visit([&](const auto& t) { return power_shortcut_on(kind[0], t); }, *a);
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _stack_last(parts) -> Tensor: the tensors of `parts` (a sequence of one shape and one dtype) side by side along a
// new last axis, as numpy.stack(parts, axis=-1): a copy, element by element, where assembling them by products with
// unit vectors multiplied each by zeros (an infinite part made the others NaN).
PyObject* module_stack_last(PyObject* /*self*/, PyObject* args)
{
    PyObject* seq = nullptr;
    if (PyArg_ParseTuple(args, "O", &seq) == 0) {
        return nullptr;
    }
    const Py_ssize_t count = PySequence_Size(seq);
    if (count <= 0) {
        if (count == 0) {
            PyErr_SetString(PyExc_ValueError, "_stack_last: at least one tensor");
        }
        return nullptr;
    }
    std::vector<const any_tensor*> parts;
    for (Py_ssize_t i = 0; i < count; ++i) {
        PyObject* item = PySequence_GetItem(seq, i); // new ref; the sequence keeps the tensor alive
        if (item == nullptr) {
            return nullptr;
        }
        const any_tensor* part = as_tensor(item);
        Py_DECREF(item);
        if (part == nullptr) {
            return nullptr;
        }
        if (!parts.empty() && (part->index() != parts.front()->index() || shape_of(*part) != shape_of(*parts.front()))) {
            PyErr_SetString(sciforge_module_error(), "_stack_last: the tensors differ in shape or dtype");
            return nullptr;
        }
        parts.push_back(part);
    }
    try {
        return std::visit([&](const auto& first) {
            using T = std::remove_cvref_t<decltype(first.data()[0])>;
            const std::size_t        k {parts.size()};
            const std::size_t        n {first.data().size()};
            std::vector<std::size_t> shape {first.shape()};
            shape.push_back(k);
            numerical::buffer<T> out(numerical::detail::checked_mul(n, k));
            for (std::size_t j = 0; j < k; ++j) {
                const auto& values {std::get<std::decay_t<decltype(first)>>(*parts[j]).data()};
                for (std::size_t i = 0; i < n; ++i) {
                    out[(i * k) + j] = values[i];
                }
            }
            return wrap(any_tensor {numerical::tensor<T> {std::move(shape), std::move(out)}});
        }, *parts.front());
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
    try { // an allocation the system refuses is a Python exception, not an abort
        return std::visit([](const auto& t) { return wrap(any_tensor {-t}); }, *a);
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

template <typename T>
PyObject* unary_on(std::string_view name, const numerical::tensor<T>& t)
{
    using numerical::detail::parallel_map;
    if (name != "exp" && name != "log" && name != "sqrt" && name != "sin" && name != "cos" && name != "tan"
        && name != "tanh" && name != "conj" && name != "abs") {
        PyErr_SetString(PyExc_ValueError, "_unary: name is exp, log, sqrt, sin, cos, tan, tanh, abs or conj");
        return nullptr;
    }
    const numerical::detail::numpy_loops loops {numpy_loops_beside()}; // read while the GIL is held
    std::optional<any_tensor>    result;
    {
        const sb::scoped_gil_release unlocked {releases_gil(t.size())};
        if (name == "exp") {
            result.emplace(numerical::exp(t));
        }
        else if (name == "log") {
            result.emplace(numerical::log(t));
        }
        else if (name == "sqrt") { // a complex number's elementary functions as glibc's, NumPy's (numerical/complex_math.hpp)
            result.emplace(numerical::sqrt(t));
        }
        else if (name == "sin") {
            result.emplace(numerical::sin(t));
        }
        else if (name == "cos") {
            result.emplace(numerical::cos(t));
        }
        else if (name == "tan") {
            result.emplace(numerical::tan(t));
        }
        else if (name == "tanh") {
            result.emplace(numerical::tanh(t));
        }
        else if (name == "conj") { // a real tensor's own elements
            if constexpr (complex_scalar<T>) {
                result.emplace(parallel_map(t, [](const T& x) { return std::conj(x); }));
            }
            else {
                result.emplace(t);
            }
        }
        else if constexpr (numerical::detail::split_complex<T>) { // abs: NumPy's absolute, by its vector loop
            result.emplace(numerical::detail::numpy_absolute(t, loops));
        }
        else { // abs: a real result of the same precision, as NumPy's absolute
            result.emplace(parallel_map(t, [](const T& x) { return static_cast<real_of_t<T>>(std::abs(x)); }));
        }
    }
    return wrap(std::move(*result));
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
    try { // an allocation the system refuses is a Python exception, not an abort
        return std::visit([&](const auto& t) { return unary_on(name, t); }, *a);
    }
    catch (...) {
        return set_cpp_error();
    }
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

template <typename T>
PyObject* extreme_of(bool largest, const numerical::tensor<T>& t, Py_ssize_t axis)
{
    if (axis < 0) {
        return from_scalar<T>(largest ? numerical::max(t) : numerical::min(t));
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
            return std::visit([&](const auto& t) { return sum_or_mean(name == "mean", t, axis); }, *a);
        }
        if (name == "max" || name == "min") {
            if (is_complex(*a)) {
                PyErr_SetString(PyExc_TypeError, "max and min take a real tensor: a complex one has no order");
                return nullptr;
            }
            if (is_single(*a)) {
                return extreme_of(name == "max", std::get<single_tensor>(*a), axis);
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
        return std::visit([&](const auto& t) { return wrap(any_tensor {numerical::reshape(t, std::move(shape))}); }, *a);
    }
    catch (...) {
        return set_cpp_error();
    }
}

// The variant index of the dtype named by `name`, or nothing with ValueError set.
std::optional<std::size_t> dtype_named(const char* name)
{
    const std::optional<std::size_t> index {dtype_index(name)};
    if (!index) {
        PyErr_SetString(PyExc_ValueError, "dtype is 'float64', 'complex128', 'float32' or 'complex64'");
    }
    return index;
}

// _full(shape, value, dtype) -> Tensor of that dtype.
PyObject* module_full(PyObject* /*self*/, PyObject* args)
{
    PyObject*   seq   = nullptr;
    PyObject*   value = nullptr;
    const char* name  = nullptr;
    if (PyArg_ParseTuple(args, "OOs", &seq, &value, &name) == 0) {
        return nullptr;
    }
    std::vector<std::size_t>         shape;
    const std::optional<std::size_t> dtype {dtype_named(name)};
    if (!dtype || !read_shape(seq, shape)) {
        return nullptr;
    }
    const double re = PyComplex_RealAsDouble(value); // the Limited API has no Py_complex accessor
    const double im = PyComplex_ImagAsDouble(value);
    if (PyErr_Occurred() != nullptr) {
        return nullptr;
    }
    try {
        switch (*dtype) {
            case 0: return wrap(any_tensor {numerical::full(std::move(shape), re)});
            case 1: return wrap(any_tensor {numerical::full(std::move(shape), std::complex<double> {re, im})});
            case 2: return wrap(any_tensor {numerical::full(std::move(shape), static_cast<float>(re))});
            default: return wrap(any_tensor {numerical::full(std::move(shape), std::complex<float> {static_cast<float>(re), static_cast<float>(im)})});
        }
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _eye(n, dtype) -> Tensor of that dtype.
PyObject* module_eye(PyObject* /*self*/, PyObject* args)
{
    Py_ssize_t  n    = 0;
    const char* name = nullptr;
    if (PyArg_ParseTuple(args, "ns", &n, &name) == 0) {
        return nullptr;
    }
    const std::optional<std::size_t> dtype {dtype_named(name)};
    if (!dtype) {
        return nullptr;
    }
    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "negative dimensions are not allowed");
        return nullptr;
    }
    const auto order {static_cast<std::size_t>(n)};
    try {
        switch (*dtype) {
            case 0: return wrap(any_tensor {numerical::eye<double>(order)});
            case 1: return wrap(any_tensor {numerical::eye<std::complex<double>>(order)});
            case 2: return wrap(any_tensor {numerical::eye<float>(order)});
            default: return wrap(any_tensor {numerical::eye<std::complex<float>>(order)});
        }
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
    try { // a count past what memory holds is a Python exception, not an abort
        return wrap(any_tensor {numerical::linspace(start, stop, static_cast<std::size_t>(num))});
    }
    catch (...) {
        return set_cpp_error();
    }
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
    if (shape_of(*h).size() < 2) {
        PyErr_SetString(sciforge_module_error(), "slogdet expects a matrix or a stack of matrices (rank ≥ 2)");
        return nullptr;
    }
    try {
        if (shape_of(*h).size() > 2) {
            auto [sign, logarithm] {over_stack<std::tuple<any_tensor, any_tensor>>(*h, slogdet_tensors)};
            return Py_BuildValue("(NN)", wrap(std::move(sign)), wrap(std::move(logarithm)));
        }
        std::optional<std::tuple<any_tensor, any_tensor>> value; // as determinant's
        {
            const sb::scoped_gil_release unlocked {releases_gil(count_of(*h))};
            value.emplace(slogdet_tensors(*h));
        }
        const auto scalar {[](const any_tensor& v) { return std::visit([](const auto& t) { return from_scalar(t.data()[0]); }, v); }};
        PyObject* sign {scalar(std::get<0>(*value))}; // new ref
        if (sign == nullptr) {
            return nullptr;
        }
        const double logarithm {std::visit([](const auto& t) { return static_cast<double>(std::real(t.data()[0])); }, std::get<1>(*value))};
        return Py_BuildValue("(Nd)", sign, logarithm); // N: the tuple takes the reference
    }
    catch (...) {
        return set_cpp_error();
    }
}

// The cutoff named by NumPy's tolerances: tol absolute, else rtol relative to the largest singular value, else NumPy's
// default.
numerical::rank_cutoff cutoff_named(std::optional<double> rtol, std::optional<double> tol)
{
    if (tol) {
        return numerical::rank_cutoff::absolute(*tol);
    }
    return rtol ? numerical::rank_cutoff::relative(*rtol) : numerical::rank_cutoff {};
}

// The matrix whose lower triangle is a's upper one, conjugated: what a routine reading the lower triangle reads for
// NumPy's UPLO="U".
template <typename T>
numerical::matrix<T> lower_read(numerical::matrix<T> m, bool upper)
{
    if (upper) {
        return numerical::detail::adjoint(m);
    }
    return m; // moved out: a conditional expression would copy it
}

template <typename T>
numerical::matrix<T> lower_read(const numerical::tensor<T>& a, bool upper)
{
    return lower_read(matrix_from(a), upper);
}

template <typename T>
std::size_t do_matrix_rank(const numerical::tensor<T>& a, numerical::rank_cutoff cutoff, bool hermitian)
{
    const sb::scoped_gil_release unlocked {releases_gil(a.data().size())};
    return numerical::matrix_rank(matrix_from(a), cutoff, hermitian);
}

// matrix_rank(a, rtol, tol, hermitian) -> int, as numpy.linalg.matrix_rank.
std::size_t matrix_rank(const any_tensor& v, std::optional<double> rtol, std::optional<double> tol, bool hermitian)
{
    if (shape_of(v).size() != 2) {
        throw std::runtime_error("matrix_rank expects a matrix (rank-2 tensor); a stack goes through _matrix_ranks");
    }
    if (is_single(v)) {
        return matrix_rank(widened(v), single_rtol(v, rtol, tol), tol, hermitian);
    }
    const numerical::rank_cutoff cutoff {cutoff_named(rtol, tol)};
    if (is_complex(v)) {
        return do_matrix_rank<std::complex<double>>(std::get<cplx_tensor>(v), cutoff, hermitian);
    }
    return do_matrix_rank<double>(std::get<real_tensor>(v), cutoff, hermitian);
}

// _matrix_ranks(a, rtol, tol, hermitian) -> Tensor: the rank of every matrix of the stack `a` (rank > 2), as float64
// numbers along its leading axes (the Python side makes them integers).
any_tensor matrix_ranks(const any_tensor& v, std::optional<double> rtol, std::optional<double> tol, bool hermitian)
{
    if (shape_of(v).size() <= 2) {
        throw std::runtime_error("_matrix_ranks expects a stack of matrices (rank > 2)");
    }
    return over_stack<any_tensor>(v, [&](const any_tensor& m) {
        return any_tensor {real_tensor {{}, numerical::buffer<double> {static_cast<double>(matrix_rank(m, rtol, tol, hermitian))}}};
    });
}

template <typename T>
any_tensor do_eigvalsh(const numerical::tensor<T>& a, bool upper)
{
    const sb::scoped_gil_release unlocked {releases_gil(a.data().size())};
    return any_tensor {real_tensor {numerical::symmetric_eigenvalues(lower_read(a, upper))}};
}

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

any_tensor eigvalsh(const any_tensor& v, bool upper)
{
    if (shape_of(v).size() < 2) {
        throw std::runtime_error("eigvalsh expects a matrix or a stack of matrices (rank ≥ 2)");
    }
    if (shape_of(v).size() > 2) {
        return over_stack<any_tensor>(v, [&](const any_tensor& m) { return eigvalsh(m, upper); });
    }
    if (is_single(v)) {
        return narrowed(eigvalsh(widened(v), upper));
    }
    if (is_complex(v)) {
        return do_eigvalsh<std::complex<double>>(std::get<cplx_tensor>(v), upper);
    }
    return do_eigvalsh<double>(std::get<real_tensor>(v), upper);
}

template <typename T>
any_tensor do_pinv(const numerical::tensor<T>& a, numerical::rank_cutoff cutoff, bool hermitian)
{
    const sb::scoped_gil_release unlocked {releases_gil(a.data().size())};
    return any_tensor {numerical::tensor<T> {numerical::pinv(matrix_from(a), cutoff, hermitian)}};
}

// pinv(a, rtol, hermitian) -> Tensor, as numpy.linalg.pinv.
any_tensor pinv(const any_tensor& v, std::optional<double> rtol, bool hermitian)
{
    if (shape_of(v).size() < 2) {
        throw std::runtime_error("pinv expects a matrix or a stack of matrices (rank ≥ 2)");
    }
    if (shape_of(v).size() > 2) {
        return over_stack<any_tensor>(v, [&](const any_tensor& m) { return pinv(m, rtol, hermitian); });
    }
    if (is_single(v)) {
        return narrowed(pinv(widened(v), single_rtol(v, rtol, std::nullopt), hermitian));
    }
    const numerical::rank_cutoff cutoff {cutoff_named(rtol, std::nullopt)};
    if (is_complex(v)) {
        return do_pinv<std::complex<double>>(std::get<cplx_tensor>(v), cutoff, hermitian);
    }
    return do_pinv<double>(std::get<real_tensor>(v), cutoff, hermitian);
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
    if (shape_of(v).size() < 2) {
        throw std::runtime_error("inverse expects a matrix or a stack of matrices (rank ≥ 2)");
    }
    if (shape_of(v).size() > 2) {
        return over_stack<any_tensor>(v, [&](const any_tensor& m) { return inverse(m); });
    }
    if (is_single(v)) {
        return narrowed(inverse(widened(v)));
    }
    if (is_complex(v)) {
        return do_inverse<std::complex<double>>(std::get<cplx_tensor>(v));
    }
    return do_inverse<double>(std::get<real_tensor>(v));
}

// The least-squares solution with what numpy.linalg.lstsq returns beside it: x (rank-1 for a rank-1 b), the residual
// sums of squares (empty unless a is tall and of full column rank), the rank and the singular values.
using lstsq_result = std::tuple<any_tensor, any_tensor, std::size_t, any_tensor>;

template <typename T>
lstsq_result do_lstsq(const numerical::tensor<T>& a, const numerical::tensor<T>& b, numerical::rank_cutoff cutoff)
{
    const sb::scoped_gil_release unlocked {releases_gil(a.data().size())};
    const bool                   column {b.shape().size() == 1};
    const numerical::matrix<T>   rhs {column ? numerical::matrix<T> {b.shape()[0], 1, b.data()} : matrix_from(b)};
    numerical::least_squares_solution<T> found {numerical::least_squares(matrix_from(a), rhs, cutoff)};
    const auto as_tensor_of = [](const numerical::vector<double>& values) {
        return any_tensor {real_tensor {{values.size()}, numerical::buffer<double>(values.begin(), values.end())}};
    };
    numerical::tensor<T> x {std::move(found.solution)};
    if (column) {
        x = numerical::tensor<T> {{x.shape()[0]}, x.data()};
    }
    return {any_tensor {std::move(x)}, as_tensor_of(found.residuals), found.rank, as_tensor_of(found.singular_values)};
}

// lstsq(a, b, rtol) -> (x, residuals, rank, s), as numpy.linalg.lstsq (SVD; a rank-2; b rank-1, or rank-2 with one
// right-hand side per column).
lstsq_result lstsq(const any_tensor& a, const any_tensor& b, std::optional<double> rtol)
{
    if (shape_of(a).size() != 2 || (shape_of(b).size() != 1 && shape_of(b).size() != 2)) {
        throw std::runtime_error("lstsq(a, b) needs a rank-2 a and a rank-1 or rank-2 b");
    }
    if (a.index() != b.index()) {
        throw std::runtime_error("lstsq operands must share one dtype");
    }
    if (is_single(a)) {
        return narrowed(lstsq(widened(a), widened(b), single_rtol(a, rtol, std::nullopt)));
    }
    const numerical::rank_cutoff cutoff {cutoff_named(rtol, std::nullopt)};
    if (is_complex(a)) {
        return do_lstsq<std::complex<double>>(std::get<cplx_tensor>(a), std::get<cplx_tensor>(b), cutoff);
    }
    return do_lstsq<double>(std::get<real_tensor>(a), std::get<real_tensor>(b), cutoff);
}

// The Cholesky factor, `a` the copy of A the reduction reads (lower_read): L, or U = Lᴴ for NumPy's UPLO="U".
template <typename T>
any_tensor cholesky_copied(numerical::matrix<T> a, bool upper)
{
    numerical::matrix<T> lower {numerical::cholesky_decomposition<T> {std::move(a)}.lower()};
    return any_tensor {numerical::tensor<T> {upper ? numerical::detail::adjoint(lower) : std::move(lower)}};
}

template <typename T>
any_tensor do_cholesky(const numerical::tensor<T>& a, bool upper)
{
    const sb::scoped_gil_release unlocked {releases_gil(a.data().size())};
    return cholesky_copied(lower_read(a, upper), upper);
}

// cholesky(a, upper) -> Tensor: the lower factor L of A = L Lᴴ, or with upper the factor U of A = Uᴴ U, from A's upper
// triangle, as numpy.linalg.cholesky(a, upper=True).
any_tensor cholesky(const any_tensor& v, bool upper)
{
    if (shape_of(v).size() < 2) {
        throw std::runtime_error("cholesky expects a matrix or a stack of matrices (rank ≥ 2)");
    }
    if (shape_of(v).size() > 2) {
        return over_stack<any_tensor>(v, [&](const any_tensor& m) { return cholesky(m, upper); });
    }
    if (is_single(v)) {
        return narrowed(cholesky(widened(v), upper));
    }
    if (is_complex(v)) {
        return do_cholesky<std::complex<double>>(std::get<cplx_tensor>(v), upper);
    }
    return do_cholesky<double>(std::get<real_tensor>(v), upper);
}

// An eigen_method, svd_method or qr_mode by its enumerator's name.
template <typename Method>
Method method_named(std::string_view name, std::initializer_list<std::pair<std::string_view, Method>> methods)
{
    for (const auto& [known, method] : methods) {
        if (name == known) {
            return method;
        }
    }
    throw std::invalid_argument("unknown method '" + std::string {name} + "'");
}

template <typename T>
std::tuple<any_tensor, any_tensor> do_qr(const numerical::tensor<T>& a, numerical::qr_mode mode)
{
    const sb::scoped_gil_release         unlocked {releases_gil(a.data().size())};
    numerical::qr_decomposition<T> factored {matrix_from(a), mode};
    return {any_tensor {numerical::tensor<T> {std::move(factored).q()}}, any_tensor {numerical::tensor<T> {std::move(factored).r()}}};
}

// qr(a, mode) -> (Q, R), mode "reduced" or "complete", through the tuple-caster.
std::tuple<any_tensor, any_tensor> qr(const any_tensor& v, const std::string& name)
{
    using numerical::qr_mode;
    const qr_mode mode {method_named<qr_mode>(name, {{"reduced", qr_mode::reduced}, {"complete", qr_mode::complete}})};
    if (shape_of(v).size() < 2) {
        throw std::runtime_error("qr expects a matrix or a stack of matrices (rank ≥ 2)");
    }
    if (shape_of(v).size() > 2) {
        return over_stack<std::tuple<any_tensor, any_tensor>>(v, [&](const any_tensor& m) { return qr(m, name); });
    }
    if (is_single(v)) {
        return narrowed(qr(widened(v), name));
    }
    if (is_complex(v)) {
        return do_qr<std::complex<double>>(std::get<cplx_tensor>(v), mode);
    }
    return do_qr<double>(std::get<real_tensor>(v), mode);
}


// qr_r(a) -> R, k×n, qr's R to the bit with Q never formed (numpy.linalg.qr's mode "r").
any_tensor qr_r(const any_tensor& v)
{
    if (shape_of(v).size() < 2) {
        throw std::runtime_error("qr expects a matrix or a stack of matrices (rank ≥ 2)");
    }
    if (shape_of(v).size() > 2) {
        return over_stack<any_tensor>(v, [](const any_tensor& m) { return qr_r(m); });
    }
    if (is_single(v)) {
        return narrowed(qr_r(widened(v)));
    }
    if (is_complex(v)) {
        return any_tensor {numerical::tensor<std::complex<double>> {numerical::qr_r(matrix_from(std::get<cplx_tensor>(v)))}};
    }
    return any_tensor {numerical::tensor<double> {numerical::qr_r(matrix_from(std::get<real_tensor>(v)))}};
}

// eigh(a, method, upper) -> (eigenvalues, eigenvectors) for a real symmetric or a complex Hermitian matrix: the values
// are real; its lower triangle is read, or its upper one.
std::tuple<any_tensor, any_tensor> eigh(const any_tensor& v, const std::string& name, bool upper)
{
    using numerical::eigen_method;
    const eigen_method method {method_named<eigen_method>(name, {{"divide_and_conquer", eigen_method::divide_and_conquer},
                                                                 {"tridiagonal_ql", eigen_method::tridiagonal_ql},
                                                                 {"jacobi", eigen_method::jacobi}})};
    if (shape_of(v).size() < 2) {
        throw std::runtime_error("eigh expects a matrix or a stack of matrices (rank ≥ 2)");
    }
    if (shape_of(v).size() > 2) {
        return over_stack<std::tuple<any_tensor, any_tensor>>(v, [&](const any_tensor& m) { return eigh(m, name, upper); });
    }
    if (is_single(v)) {
        return narrowed(eigh(widened(v), name, upper));
    }
    if (is_complex(v)) {
        const cplx_tensor&               a = std::get<cplx_tensor>(v);
        const sb::scoped_gil_release     unlocked {releases_gil(a.data().size())};
        numerical::hermitian_eigen   decomposition {lower_read(a, upper), method};
        return {any_tensor {real_tensor {decomposition.values()}},
                any_tensor {cplx_tensor {std::move(decomposition).vectors()}}};
    }
    const real_tensor&               a = std::get<real_tensor>(v);
    const sb::scoped_gil_release     unlocked {releases_gil(a.data().size())};
    numerical::symmetric_eigen   decomposition {lower_read(a, upper), method};
    return {any_tensor {real_tensor {decomposition.values()}},
            any_tensor {real_tensor {std::move(decomposition).vectors()}}};
}

template <typename T>
std::tuple<any_tensor, any_tensor, any_tensor> do_svd(const numerical::tensor<T>& a, numerical::svd_method method, bool full)
{
    const sb::scoped_gil_release          unlocked {releases_gil(a.data().size())};
    numerical::svd_decomposition<T>       decomposition {matrix_from(a), method, full};
    return {any_tensor {numerical::tensor<T> {std::move(decomposition).u()}},
            any_tensor {real_tensor {decomposition.singular_values()}},
            any_tensor {numerical::tensor<T> {decomposition.vh()}}};
}

// svd(a, method, full) -> (U, singular_values, Vh) for an m×n matrix, as numpy.linalg.svd(a, full_matrices=full).
std::tuple<any_tensor, any_tensor, any_tensor> svd(const any_tensor& v, const std::string& name, bool full)
{
    using numerical::svd_method;
    const svd_method method {method_named<svd_method>(name, {{"divide_and_conquer", svd_method::divide_and_conquer},
                                                             {"bidiagonal_qr", svd_method::bidiagonal_qr},
                                                             {"jacobi", svd_method::jacobi}})};
    if (shape_of(v).size() < 2) {
        throw std::runtime_error("svd expects a matrix or a stack of matrices (rank ≥ 2)");
    }
    if (shape_of(v).size() > 2) {
        return over_stack<std::tuple<any_tensor, any_tensor, any_tensor>>(v, [&](const any_tensor& m) { return svd(m, name, full); });
    }
    if (is_single(v)) {
        return narrowed(svd(widened(v), name, full));
    }
    if (is_complex(v)) {
        return do_svd<std::complex<double>>(std::get<cplx_tensor>(v), method, full);
    }
    return do_svd<double>(std::get<real_tensor>(v), method, full);
}

// _singular_values(a) -> Tensor: the singular values alone, descending, as numpy.linalg.svd(a, compute_uv=False).
any_tensor singular_values(const any_tensor& v)
{
    if (shape_of(v).size() < 2) {
        throw std::runtime_error("svd expects a matrix or a stack of matrices (rank ≥ 2)");
    }
    if (shape_of(v).size() > 2) {
        return over_stack<any_tensor>(v, [&](const any_tensor& m) { return singular_values(m); });
    }
    if (is_single(v)) {
        return narrowed(singular_values(widened(v)));
    }
    if (is_complex(v)) {
        const cplx_tensor&           a = std::get<cplx_tensor>(v);
        const sb::scoped_gil_release unlocked {releases_gil(a.data().size())};
        return any_tensor {real_tensor {numerical::singular_values(matrix_from(a))}};
    }
    const real_tensor&           a = std::get<real_tensor>(v);
    const sb::scoped_gil_release unlocked {releases_gil(a.data().size())};
    return any_tensor {real_tensor {numerical::singular_values(matrix_from(a))}};
}

// The `ord` norm of the magnitudes `values` (all finite or not), as numpy.linalg.vector_norm: 2 and every p > 0 at
// the scale of the largest (a plain sum of squares overflows past 1e154, NumPy's to infinity), p < 0 at the smallest's;
// inf the largest, -inf the smallest, 0 how many are not zero (a NaN counted: it is not zero). A NaN makes every
// other order NaN.
double norm_of(const std::vector<double>& values, double ord)
{
    if (values.empty()) {
        return ord == -std::numeric_limits<double>::infinity() ? std::numeric_limits<double>::infinity() : 0.0;
    }
    if (ord == 0.0) { // how many are not zero, a NaN among them (it is not zero), as NumPy's count
        return static_cast<double>(std::count_if(values.begin(), values.end(), [](double value) { return value != 0.0; }));
    }
    double largest {0.0};
    double smallest {std::numeric_limits<double>::infinity()};
    for (const double value : values) {
        if (std::isnan(value)) {
            return value;
        }
        largest  = std::max(largest, value);
        smallest = std::min(smallest, value);
    }
    if (std::isinf(ord)) {
        return ord > 0 ? largest : smallest;
    }
    const double scale {ord > 0 ? largest : smallest};
    if (scale == 0.0 || std::isinf(scale)) {
        return scale; // a zero (or infinite) largest for p > 0, a zero (or infinite) smallest for p < 0
    }
    double sum {0.0};
    for (const double value : values) {
        const double ratio {value / scale};
        sum += ord == 2.0 ? ratio * ratio : (ord == 1.0 ? ratio : std::pow(ratio, ord));
    }
    return scale * (ord == 2.0 ? std::sqrt(sum) : (ord == 1.0 ? sum : std::pow(sum, 1.0 / ord)));
}

// The parts whose squares plain_two_norm sums as one block: fixed, so the sum does not depend on the threads; as many
// as a chunk of numerical::detail::chunked_dot, whose weight among the threads they take.
constexpr std::size_t norm_block {numerical::detail::dot_chunk};

// `sum + value²`: on x86-64 the square a statement of its own, so never fused (-ffp-contract=on contracts within an
// expression), as avx2_block_squares rounds it whatever the build's target; elsewhere one expression, which arm64 fuses.
inline void add_square(double& sum, double value)
{
#ifdef NUMERICAL_X86_COMPILED
    const double square {value * value};
    sum += square;
#else
    sum += value * value;
#endif
}

// The sum of the squares of `n` parts from `parts`, in double: each one's square added to the sum of its index modulo
// 8 among the whole groups of 8, those eight summed as a tree, then the tail's squares in order.
template <typename R>
double block_squares(const R* parts, std::size_t n)
{
    std::array<double, 8> partial {};
    std::size_t           i {0};
    for (; i + partial.size() <= n; i += partial.size()) {
        for (std::size_t j = 0; j < partial.size(); ++j) {
            add_square(partial[j], static_cast<double>(parts[i + j]));
        }
    }
    double sum {((partial[0] + partial[1]) + (partial[2] + partial[3])) + ((partial[4] + partial[5]) + (partial[6] + partial[7]))};
    for (; i < n; ++i) {
        add_square(sum, static_cast<double>(parts[i]));
    }
    return sum;
}

#ifdef NUMERICAL_X86_COMPILED
// block_squares on the AVX2 units (the AVX-512 path's too): the same eight partial sums, two registers of four, each
// square its own product then added (the baseline that block_squares is built for fuses nothing), the same tree and
// tail, so its bits, at the vectors' width; a float part widened to double first, as there.
template <typename R>
NUMERICAL_TARGET("avx2") double avx2_block_squares(const R* parts, std::size_t n)
{
    __m256d     low {_mm256_setzero_pd()};
    __m256d     high {_mm256_setzero_pd()};
    std::size_t i {0};
    for (; i + 8 <= n; i += 8) {
        __m256d first;
        __m256d second;
        if constexpr (std::is_same_v<R, float>) {
            first  = _mm256_cvtps_pd(_mm_loadu_ps(parts + i));
            second = _mm256_cvtps_pd(_mm_loadu_ps(parts + i + 4));
        }
        else {
            first  = _mm256_loadu_pd(parts + i);
            second = _mm256_loadu_pd(parts + i + 4);
        }
        low  = _mm256_add_pd(low, _mm256_mul_pd(first, first));
        high = _mm256_add_pd(high, _mm256_mul_pd(second, second));
    }
    alignas(32) std::array<double, 8> partial {};
    _mm256_store_pd(partial.data(), low);
    _mm256_store_pd(partial.data() + 4, high);
    double sum {((partial[0] + partial[1]) + (partial[2] + partial[3])) + ((partial[4] + partial[5]) + (partial[6] + partial[7]))};
    for (; i < n; ++i) {
        add_square(sum, static_cast<double>(parts[i]));
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
    return sum;
}
#endif

// block_squares on the vector units the processor has: AVX2's on x86-64 where it runs, the same bits.
template <typename R>
double fast_block_squares(const R* parts, std::size_t n)
{
#ifdef NUMERICAL_X86_COMPILED
    const numerical::gemm_path path {numerical::detail::gemm_path_current()};
    if (path == numerical::gemm_path::avx2 || path == numerical::gemm_path::avx512) {
        return avx2_block_squares(parts, n);
    }
#endif
    return block_squares(parts, n);
}

// The 2-norm from a sum of squares where it neither overflowed nor lost its digits below 2^-900 (a NaN fails the
// first test); no answer elsewhere.
std::optional<double> root_of_squares(double sum)
{
    if (!(sum >= 0x1p-900) || std::isinf(sum)) {
        return std::nullopt;
    }
    return std::sqrt(sum);
}

// The 2-norm of `count` elements from `x` as the square root of their squares' sum, in double, in one pass: blocks of
// norm_block parts in parallel, each by block_squares, their sums added in the blocks' order. Where that sum lies
// between 2^-900 and the largest double, no square overflowed and those lost below the smallest double weigh nothing
// beside it. Elsewhere (a NaN, an infinity, a sum past either bound), no answer: norm_of's scaled one decides. A
// complex element's square is its parts' squares, the parts read as the array of two the standard lays out.
template <typename T>
std::optional<double> plain_two_norm(const T* x, std::size_t count)
{
    using R = real_of_t<T>;
    const auto*       parts {reinterpret_cast<const R*>(x)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the layout above
    const std::size_t n {count * (sizeof(T) / sizeof(R))};
    const std::size_t blocks {(n + norm_block - 1) / norm_block};
    if (blocks <= 1) {
        return root_of_squares(fast_block_squares(parts, n));
    }
    std::vector<double> sums(blocks);
    numerical::detail::parallel_for(blocks, numerical::detail::dot_chunk_weight(blocks), [&](std::size_t b) {
        const std::size_t first {b * norm_block};
        sums[b] = fast_block_squares(parts + first, std::min(norm_block, n - first));
    });
    double sum {0.0};
    for (const double block : sums) {
        sum += block;
    }
    return root_of_squares(sum);
}

// plain_two_norm of a `rows` × `cols` matrix whose elements lie column-major (`x` its first column, `rows` long), its
// elements taken row by row as its C-ordered copy lies, so its bits: eight rows at a time copied row-major into a band
// (each column's eight elements together where they lie), a band starting at a multiple of eight parts, so a block's
// boundary (a multiple of norm_block) between two groups of 8; each square added to the sum of its index modulo 8
// among its block's whole groups of 8, a full block's sum taken at its end, and the last block's tail after its
// groups' sum.
template <typename T>
std::optional<double> plain_two_norm_columns(const T* x, std::size_t rows, std::size_t cols)
{
    using R = real_of_t<T>;
    constexpr std::size_t per {sizeof(T) / sizeof(R)};
    constexpr std::size_t band_rows {8};
    const auto*           parts {reinterpret_cast<const R*>(x)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): plain_two_norm's layout
    std::array<double, 8> partial {};
    std::vector<R>        band(band_rows * cols * per);
    double                sum {0.0};
    std::size_t           in_block {0}; // the parts of the current block already summed
    for (std::size_t r0 {0}; r0 < rows; r0 += band_rows) {
        const std::size_t height {std::min(band_rows, rows - r0)};
        for (std::size_t c {0}; c < cols; ++c) {
            const R* const column {parts + (((c * rows) + r0) * per)};
            for (std::size_t i {0}; i < height; ++i) {
                for (std::size_t q {0}; q < per; ++q) {
                    band[(((i * cols) + c) * per) + q] = column[(i * per) + q];
                }
            }
        }
        const std::size_t length {height * cols * per};
        std::size_t       k {0};
        while (k + partial.size() <= length) {
            const std::size_t take {std::min(norm_block - in_block, ((length - k) / partial.size()) * partial.size())};
            for (const std::size_t end {k + take}; k < end; k += partial.size()) {
                for (std::size_t j = 0; j < partial.size(); ++j) {
                    add_square(partial[j], static_cast<double>(band[k + j]));
                }
            }
            in_block += take;
            if (in_block == norm_block) { // a full block: its sum, and the next block's from zero
                sum      += ((partial[0] + partial[1]) + (partial[2] + partial[3])) + ((partial[4] + partial[5]) + (partial[6] + partial[7]));
                partial   = {};
                in_block  = 0;
            }
        }
        if (r0 + height == rows && (in_block > 0 || k < length)) { // the last block: its groups' sum, then its tail
            double last {((partial[0] + partial[1]) + (partial[2] + partial[3])) + ((partial[4] + partial[5]) + (partial[6] + partial[7]))};
            for (; k < length; ++k) {
                add_square(last, static_cast<double>(band[k]));
            }
            sum += last;
        }
    }
    return root_of_squares(sum);
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
    const any_tensor*       h = as_tensor(obj);
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
            std::visit([&](const auto& t) {
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
        const auto magnitudes {std::visit([](const auto& t) {
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

// The singular value decomposition of the real symmetric or complex Hermitian matrix `v` (its lower triangle read)
// through its eigendecomposition, as numpy.linalg.svd(a, hermitian=True): the magnitudes of the eigenvalues
// descending (ties in the reverse of their ascending order), U their vectors, Vh = (U · sign(λ))ᴴ.
std::tuple<any_tensor, any_tensor, any_tensor> svd_hermitian(const any_tensor& v)
{
    const auto [values, vectors] {eigh(v, "divide_and_conquer", false)};
    return std::visit([&](const auto& vt) -> std::tuple<any_tensor, any_tensor, any_tensor> {
        using tensor_type = std::decay_t<decltype(vt)>;
        using T           = element_of_t<tensor_type>;
        using R           = real_of_t<T>;
        const auto&       w {std::get<numerical::tensor<R>>(values)};
        const std::size_t n {w.size()};
        std::vector<std::size_t> order(n);
        for (std::size_t i = 0; i < n; ++i) {
            order[i] = n - 1 - i;
        }
        std::stable_sort(order.begin(), order.end(), [&](std::size_t x, std::size_t y) {
            return std::abs(w.data()[x]) > std::abs(w.data()[y]);
        });
        numerical::buffer<R> s(n);
        numerical::buffer<T> u(n * n);
        numerical::buffer<T> vh(n * n);
        for (std::size_t c = 0; c < n; ++c) {
            const R value {w.data()[order[c]]};
            s[c] = std::abs(value);
            const R sign {value < R {0} ? R {-1} : R {1}};
            for (std::size_t r = 0; r < n; ++r) {
                const T x {vt.data()[(r * n) + order[c]]};
                u[(r * n) + c]  = x;
                if constexpr (complex_scalar<T>) {
                    vh[(c * n) + r] = std::conj(x) * sign;
                }
                else {
                    vh[(c * n) + r] = x * sign;
                }
            }
        }
        return {tensor_type {{n, n}, std::move(u)}, numerical::tensor<R> {{n}, std::move(s)}, tensor_type {{n, n}, std::move(vh)}};
    }, vectors);
}

// _svd_hermitian(a) -> (U, S, Vh): svd_hermitian of a matrix or of every matrix of a stack.
std::tuple<any_tensor, any_tensor, any_tensor> svd_hermitian_of(const any_tensor& v)
{
    if (shape_of(v).size() < 2 || shape_of(v).back() != shape_of(v)[shape_of(v).size() - 2]) {
        throw std::runtime_error("svd(hermitian=True) expects a square matrix or a stack of them");
    }
    if (shape_of(v).size() > 2) {
        return over_stack<std::tuple<any_tensor, any_tensor, any_tensor>>(v, svd_hermitian_of);
    }
    return svd_hermitian(v);
}

template <typename T>
std::tuple<any_tensor, any_tensor> do_eig(const numerical::tensor<T>& a, bool vectors)
{
    const sb::scoped_gil_release            unlocked {releases_gil(a.data().size())};
    numerical::eigen_decomposition<T>       decomposition {matrix_from(a), vectors};
    return {any_tensor {cplx_tensor {decomposition.values()}},
            any_tensor {vectors ? cplx_tensor {std::move(decomposition).vectors()} : cplx_tensor {{0}, {}}}};
}

// The eigenvalues (and, with `vectors`, the eigenvectors) of the square matrix, or of every matrix of the stack, `v`,
// complex, in double precision.
std::tuple<any_tensor, any_tensor> eig_complex(const any_tensor& v, bool vectors)
{
    if (shape_of(v).size() > 2) {
        return over_stack<std::tuple<any_tensor, any_tensor>>(v, [&](const any_tensor& m) { return eig_complex(m, vectors); });
    }
    if (is_complex(v)) {
        return do_eig<std::complex<double>>(std::get<cplx_tensor>(v), vectors);
    }
    return do_eig<double>(std::get<real_tensor>(v), vectors);
}

// eig_of(a, vectors): numpy.linalg.eig's answers (eigvals' alone without `vectors`), complex as NumPy 2.5's whatever
// the eigenvalues, in the input's precision.
std::tuple<any_tensor, any_tensor> eig_of(const any_tensor& v, bool vectors)
{
    if (shape_of(v).size() < 2 || shape_of(v).back() != shape_of(v)[shape_of(v).size() - 2]) {
        throw std::runtime_error("eig expects a square matrix or a stack of them");
    }
    if (is_single(v)) {
        return narrowed(eig_complex(widened(v), vectors));
    }
    return eig_complex(v, vectors); // moved out: a const result would be copied, its n × n vectors with it
}

// _eig(a) -> (w, v), as numpy.linalg.eig.
std::tuple<any_tensor, any_tensor> eig(const any_tensor& v)
{
    return eig_of(v, true);
}

// _eigvals(a) -> w, as numpy.linalg.eigvals.
any_tensor eigvals(const any_tensor& v)
{
    return std::get<0>(eig_of(v, false));
}

// asarray(obj, dtype=None): an existing Tensor (returned as-is when its dtype already matches, else converted), or a
// copy built from obj's elements (__array_struct__ or __array_interface__, any strides). The dtype is the requested
// one, else the source's own for float32, complex64 and complex128, else float64: a real source becomes complex with a
// zero imaginary part, either precision becomes the other, and a complex source asked as a real dtype is refused (it
// would discard the imaginary part).
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
    // At most two digits of size (16 is the largest supported): a longer one is no type this reads, and would
    // overflow the conversion.
    if (typestr.size() < 3 || typestr.size() > 4 || typestr.find_first_not_of("0123456789", 2) != std::string::npos) {
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
        *out = converted<Out>(read_scalar<Src>(base));
        return;
    }
    const std::size_t        inner  {shape[rank - 1]};
    if (inner == 0) {
        return; // nothing to read, and the address may be null
    }
    const Py_ssize_t         step   {strides[rank - 1]};
    // A matrix whose columns run contiguously (Fortran order, a transpose): copied by tiles, each column of a tile read
    // where it runs and its rows written in the cache, where a row of the output walked a column of the source a
    // stride apart for every element (48 × 275 doubles: 7.1 µs, NumPy's own copy 1).
    constexpr std::size_t tile {32}; // fewer rows than a tile: the rows' loop below reads a few elements a column
    if (rank == 2 && shape[0] >= tile && strides[0] == static_cast<Py_ssize_t>(sizeof(Src)) && step != strides[0]) {
        const std::size_t rows {shape[0]};
        // Bands of a tile's rows on the threads, each element written once: the copy, and the first touch of the
        // output's pages, shared.
        numerical::detail::parallel_for((rows + tile - 1) / tile, numerical::detail::memory_element_weight * tile * inner, [&](std::size_t band) {
            const std::size_t r0 {band * tile};
            for (std::size_t c0 = 0; c0 < inner; c0 += tile) {
                for (std::size_t c = c0; c < std::min(c0 + tile, inner); ++c) {
                    const unsigned char* const column {base + (static_cast<Py_ssize_t>(c) * step)};
                    for (std::size_t r = r0; r < std::min(r0 + tile, rows); ++r) {
                        if constexpr (std::is_same_v<Src, bool>) {
                            out[(r * inner) + c] = static_cast<Out>(column[r] != 0 ? 1.0 : 0.0);
                        }
                        else {
                            out[(r * inner) + c] = converted<Out>(read_scalar<Src>(column + (r * sizeof(Src))));
                        }
                    }
                }
            }
        });
        return;
    }
    std::size_t rows {1};
    for (std::size_t axis = 0; axis + 1 < rank; ++axis) {
        rows *= shape[axis];
    }
    if (rows == 0) {
        return; // an empty extent before the last: nothing to read, and no group to run
    }
    // Groups of rows of about 2^14 elements on the threads, each group's first row's index from its number: each
    // element written once, the copy and the first touch of the output's pages shared (a copy from one thread left
    // them to page faults on it alone). A complex output, its pages touched by the threads' zeros already, in one
    // group: its copy shared took longer (M1 Pro, 500 × 1000 complex128: 470 → 580 µs).
    constexpr std::size_t group_elements {std::size_t {1} << 14U};
    const std::size_t     group {complex_scalar<Out> ? rows : std::max<std::size_t>(1, group_elements / inner)};
    numerical::detail::parallel_for((rows + group - 1) / group, numerical::detail::memory_element_weight * group * inner, [&](std::size_t g) {
        const std::size_t        first {g * group};
        const std::size_t        last {std::min(rows, first + group)};
        std::vector<std::size_t> index(rank - 1, 0);
        for (std::size_t axis = rank - 1, rest = first; axis-- > 0;) {
            index[axis]  = rest % shape[axis];
            rest        /= shape[axis];
        }
        Out* to {out + (first * inner)};
        for (std::size_t row = first; row < last; ++row) {
            Py_ssize_t offset {0};
            for (std::size_t axis = 0; axis + 1 < rank; ++axis) {
                offset += static_cast<Py_ssize_t>(index[axis]) * strides[axis];
            }
            const unsigned char* at {base + offset};
            if (step == static_cast<Py_ssize_t>(sizeof(Src)) && std::is_same_v<Src, Out>) {
                std::memcpy(to, at, inner * sizeof(Src)); // a contiguous run of the output's own type
            }
            else {
                for (std::size_t i = 0; i < inner; ++i) {
                    if constexpr (std::is_same_v<Src, bool>) {
                        to[i] = static_cast<Out>(at[static_cast<Py_ssize_t>(i) * step] != 0 ? 1.0 : 0.0);
                    }
                    else {
                        to[i] = converted<Out>(read_scalar<Src>(at + (static_cast<Py_ssize_t>(i) * step)));
                    }
                }
            }
            to += inner;
            for (std::size_t axis = rank - 1; axis-- > 0;) {
                if (++index[axis] < shape[axis]) {
                    break;
                }
                index[axis] = 0;
            }
        }
    });
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
            if constexpr (complex_scalar<Out>) {
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

// The tensor of the `total` elements of type `element` at `base` (byte `strides` per axis, C order), in the dtype
// of variant index `out`: the conversion picked once, the copy without the GIL.
template <typename Out>
any_tensor elements_as(const unsigned char*            base,
                       const std::vector<std::size_t>& shape,
                       const std::vector<Py_ssize_t>&  strides,
                       std::size_t                     total,
                       const element_type&             element)
{
    // Each element written below: a real one left unset until then; a complex one's own constructor zeroes it, on
    // one thread, where filling with zeros shares the work (and the pages' first touch) among them.
    numerical::buffer<Out> values {complex_scalar<Out> ? numerical::buffer<Out>(total, Out {}) : numerical::buffer<Out>(total)};
    if (!element.swapped) { // the common case: it reads only the buffer
        bool gathered {false};
        {
            const sb::scoped_gil_release unlocked {releases_gil(total)};
            gathered = gather(base, shape, strides, element, values.data());
        }
        if (!gathered) { // a complex source into a real tensor: asarray refuses it before, so never its unset elements
            throw std::invalid_argument("asarray: a complex array cannot become a real tensor");
        }
        return any_tensor {numerical::tensor<Out> {shape, std::move(values)}};
    }
    // A byte-swapped element: each read through read_element, in C order.
    std::vector<std::size_t> index(shape.size(), 0);
    for (std::size_t n = 0; n < total; ++n) {
        Py_ssize_t offset = 0;
        for (std::size_t axis = 0; axis < shape.size(); ++axis) {
            offset += static_cast<Py_ssize_t>(index[axis]) * strides[axis];
        }
        const std::complex<double> value {read_element(base + offset, element)};
        if constexpr (complex_scalar<Out>) {
            values[n] = static_cast<Out>(value);
        }
        else {
            values[n] = static_cast<Out>(value.real());
        }
        for (std::size_t axis = shape.size(); axis-- > 0;) {
            if (++index[axis] < shape[axis]) {
                break;
            }
            index[axis] = 0;
        }
    }
    return any_tensor {numerical::tensor<Out> {shape, std::move(values)}};
}

PyObject* tensor_from_elements(const unsigned char*            base,
                               const std::vector<std::size_t>& shape,
                               const std::vector<Py_ssize_t>&  strides,
                               std::size_t                     total,
                               const element_type&             element,
                               std::size_t                     out)
{
    try {
        switch (out) {
            case 0: return wrap(elements_as<double>(base, shape, strides, total, element));
            case 1: return wrap(elements_as<std::complex<double>>(base, shape, strides, total, element));
            case 2: return wrap(elements_as<float>(base, shape, strides, total, element));
            default: return wrap(elements_as<std::complex<float>>(base, shape, strides, total, element));
        }
    }
    catch (...) {
        return set_cpp_error();
    }
}

// NumPy's PyArrayInterface (numpy/ndarraytypes.h), which __array_struct__ holds in a capsule: an array's
// elements described without the dict __array_interface__ builds on every access (most of a small asarray).
struct array_struct
{
    int          two;      // 2: the version of the struct
    int          nd;       // the rank
    char         typekind; // b, i, u, f or c, as a typestr's kind
    int          itemsize; // bytes per element
    int          flags;    // NPY_ARRAY_* bits
    Py_intptr_t* shape;    // nd extents
    Py_intptr_t* strides;  // nd byte strides
    void*        data;     // the first element
    PyObject*    descr;    // unused here
};

// NPY_ARRAY_NOTSWAPPED: the elements are in this machine's byte order.
constexpr int array_not_swapped {0x200};

// The element count of `shape`, or false with an error set when it overflows a complex128 buffer.
bool element_count(const std::vector<std::size_t>& shape, std::size_t& total)
{
    total = 1;
    for (const std::size_t extent : shape) {
        if (extent != 0 && total > SIZE_MAX / sizeof(std::complex<double>) / extent) {
            PyErr_SetString(sciforge_module_error(), "asarray: the shape's element count overflows");
            return false;
        }
        total *= extent;
    }
    return true;
}

// asarray's output dtype (a variant index) for a source of `element`: the requested one (`wanted`), else the
// source's own when it is one of the four (float32, complex64, complex128), else float64; false with an error set
// for a complex source asked as a real dtype.
bool output_dtype(const element_type& element, std::optional<std::size_t> wanted, std::size_t& out)
{
    const std::size_t own {element.kind == 'c' ? (element.size == 8 ? std::size_t {3} : std::size_t {1})
                           : element.kind == 'f' && element.size == 4 ? std::size_t {2}
                                                                      : std::size_t {0}};
    out = wanted.value_or(own);
    if (element.kind == 'c' && (out == 0 || out == 2)) {
        PyErr_SetString(sciforge_module_error(),
                        "asarray: refusing to cast a complex array to a real dtype (discards the imaginary part)");
        return false;
    }
    return true;
}

// obj's elements through its __array_struct__ capsule (NumPy arrays and scalars), into `*result`; false when obj
// has none (or not version 2), with no error set, so the caller reads __array_interface__ instead.
bool asarray_from_struct(PyObject* obj, std::optional<std::size_t> wanted, PyObject** result)
{
    PyObject* capsule = PyObject_GetAttrString(obj, "__array_struct__"); // new ref
    if (capsule == nullptr) {
        PyErr_Clear();
        return false;
    }
    const auto* view {static_cast<const array_struct*>(PyCapsule_GetPointer(capsule, nullptr))};
    if (view == nullptr || view->two != 2) {
        PyErr_Clear();
        Py_DECREF(capsule);
        return false;
    }
    const char  order {view->itemsize == 1                          ? '|'
                       : (view->flags & array_not_swapped) != 0      ? '='
                       : byte_order() == '<'                         ? '>'
                                                                     : '<'};
    const std::optional<element_type> element {parse_typestr(std::string {order, view->typekind} + std::to_string(view->itemsize))};
    std::size_t                       out {0};
    if (!element) {
        Py_DECREF(capsule);
        PyErr_SetString(sciforge_module_error(),
                        "asarray: unsupported typestr (bool, integers, float32/64 and complex64/128 are read)");
        *result = nullptr;
        return true;
    }
    std::vector<std::size_t> shape;
    std::vector<Py_ssize_t>  strides;
    for (int axis = 0; axis < view->nd; ++axis) {
        shape.push_back(static_cast<std::size_t>(view->shape[axis]));
        strides.push_back(static_cast<Py_ssize_t>(view->strides[axis]));
    }
    std::size_t total {0};
    *result = output_dtype(*element, wanted, out) && element_count(shape, total)
                  ? tensor_from_elements(static_cast<const unsigned char*>(view->data), shape, strides, total, *element, out)
                  : nullptr;
    Py_DECREF(capsule);
    return true;
}

// NPY_ARRAY_C_CONTIGUOUS and NPY_ARRAY_ALIGNED: the elements row-major with no gaps, each at an address of its type's
// alignment.
constexpr int array_c_contiguous {0x1};
// NPY_ARRAY_F_CONTIGUOUS: the elements column-major with no gaps (a transposed C array).
constexpr int array_f_contiguous {0x2};
constexpr int array_aligned {0x100};

// An operand a product reads where it lies: a Tensor's elements, or a NumPy array's (through its __array_struct__) when
// they are row-major, contiguous, aligned, in this machine's byte order and of one of the four dtypes. `owner` (a new
// reference: the Tensor, or the capsule, which holds the array) keeps them alive.
struct borrowed_operand
{
    PyObject*                owner {nullptr};
    std::size_t              dtype {0}; // the any_tensor index of the elements' type
    std::vector<std::size_t> shape;
    const void*              data {nullptr};
    bool                     transposed {false}; // a matrix whose elements run column-major (borrow_operand's columns)
    std::size_t              ld {0};             // a matrix's elements between its rows (column-major: its columns)
    std::vector<std::byte>   gathered;           // a small strided array's elements copied in C order, `data` then

    borrowed_operand() = default;
    borrowed_operand(const borrowed_operand&) = delete;
    borrowed_operand& operator=(const borrowed_operand&) = delete;
    ~borrowed_operand() { Py_XDECREF(owner); }
};

// The most elements of a strided NumPy array borrow_operand copies itself: past them the copy outweighs the converting
// path's calls (a 4 × 31 strided matrix times a strided vector, M1 Pro: 5.1 µs, 3.8 of them asarray and the Python
// around it, where a contiguous pair took 0.7).
constexpr std::size_t gathered_most {std::size_t {1} << 16};

// The elements of `view` (aligned, in this machine's byte order, of a dtype index `dtype`) gathered in C order into
// `out.gathered`, `out.data` pointing there; false, nothing copied, past gathered_most elements.
bool gather_operand(const array_struct& view, int dtype, borrowed_operand& out)
{
    std::vector<std::size_t> shape;
    std::vector<Py_ssize_t>  strides;
    std::size_t              count {1};
    for (int axis = 0; axis < view.nd; ++axis) {
        shape.push_back(static_cast<std::size_t>(view.shape[axis]));
        strides.push_back(static_cast<Py_ssize_t>(view.strides[axis]));
        count *= shape.back();
    }
    if (count > gathered_most) {
        return false;
    }
    out.gathered.resize(count * static_cast<std::size_t>(view.itemsize));
    const auto* const base {static_cast<const unsigned char*>(view.data)};
    const auto gather_as = [&]<typename T>(std::type_identity<T>) {
        gather_typed<T>(base, shape, strides, reinterpret_cast<T*>(out.gathered.data())); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): bytes from the allocator, aligned for any scalar
    };
    switch (dtype) {
    case 0: gather_as(std::type_identity<double> {}); break;
    case 1: gather_as(std::type_identity<std::complex<double>> {}); break;
    case 2: gather_as(std::type_identity<float> {}); break;
    default: gather_as(std::type_identity<std::complex<float>> {}); break;
    }
    out.data = out.gathered.data();
    return true;
}

// Whether `obj` is read where it lies, into `out`; false (no error set) when it is not a Tensor or such an array. With
// `columns`, a matrix whose elements run column-major (Fortran order, a transpose) is read too, `out.transposed` set:
// for a caller whose answer does not depend on the order the elements lie in. With `rows_apart`, a matrix of rows of
// two or more elements together but the rows farther apart (every other row of a larger one) is read too, `out.ld`
// their distance. A small strided array (up to gathered_most elements, aligned, in this machine's byte order) is
// copied in C order instead, as the converting path would copy it, without the calls around it. `out.ld` is a
// matrix's distance between its rows (its columns, column-major).
bool borrow_operand(PyObject* obj, borrowed_operand& out, bool columns = false, bool rows_apart = false)
{
    if (PyObject_TypeCheck(obj, reinterpret_cast<PyTypeObject*>(sb::class_type<any_tensor>())) != 0) {
        const any_tensor* v {reinterpret_cast<sb::wrapper<any_tensor>*>(obj)->held};
        if (v == nullptr) {
            return false;
        }
        out.owner = Py_NewRef(obj);
        out.dtype = v->index();
        out.shape = shape_of(*v);
        out.data  = data_of(*v);
        out.ld    = out.shape.empty() ? 0 : out.shape.back();
        return true;
    }
    PyObject* capsule = PyObject_GetAttrString(obj, "__array_struct__"); // new ref
    if (capsule == nullptr) {
        PyErr_Clear();
        return false;
    }
    const auto* view {static_cast<const array_struct*>(PyCapsule_GetPointer(capsule, nullptr))};
    constexpr int needed {array_c_contiguous | array_aligned | array_not_swapped};
    constexpr int needed_columns {array_f_contiguous | array_aligned | array_not_swapped};
    const bool    rows_usable {view != nullptr && view->two == 2 && view->data != nullptr && (view->flags & needed) == needed};
    const bool    columns_usable {columns && !rows_usable && view != nullptr && view->two == 2 && view->data != nullptr && view->nd == 2
                                  && (view->flags & needed_columns) == needed_columns};
    constexpr int needed_gathered {array_aligned | array_not_swapped};
    const bool    plain {view != nullptr && view->two == 2 && view->data != nullptr && (view->flags & needed_gathered) == needed_gathered};
    const bool    apart_usable {rows_apart && !rows_usable && !columns_usable && plain && view->nd == 2 && view->shape[1] > 1
                                && view->strides[1] == view->itemsize && view->strides[0] % view->itemsize == 0
                                && view->strides[0] / view->itemsize >= view->shape[1]};
    const bool    gatherable {!rows_usable && !columns_usable && !apart_usable && plain};
    const bool    usable {rows_usable || columns_usable || apart_usable || gatherable};
    if (view == nullptr) {
        PyErr_Clear();
    }
    const char kind {usable ? view->typekind : '\0'};
    const int  size {usable ? view->itemsize : 0};
    const int  dtype {kind == 'f' && size == 8 ? 0 : kind == 'c' && size == 16 ? 1 : kind == 'f' && size == 4 ? 2 : kind == 'c' && size == 8 ? 3 : -1};
    if (dtype < 0 || (gatherable && !gather_operand(*view, dtype, out))) {
        Py_DECREF(capsule);
        return false;
    }
    out.owner      = capsule;
    out.dtype      = static_cast<std::size_t>(dtype);
    out.data       = gatherable ? out.data : view->data;
    out.transposed = columns_usable;
    for (int axis = 0; axis < view->nd; ++axis) {
        out.shape.push_back(static_cast<std::size_t>(view->shape[axis]));
    }
    out.ld = view->nd != 2 ? 0
             : apart_usable ? static_cast<std::size_t>(view->strides[0] / view->itemsize)
             : columns_usable ? out.shape[0]
                              : out.shape[1];
    return true;
}

// _borrowed_einsum(spec, x) -> Tensor or NotImplemented: einsum of one Tensor or contiguous NumPy array read where it
// lies, for a spec that sums a run of its axes into the others in order ("ij->i", "ij->", "ij->j", "ijk->ik":
// numerical::detail::sum_over_run, the bits of einsum's own path); NotImplemented for any other spec or operand, which
// einsum then reads from a copy as before. A NumPy array was copied into a Tensor first: 344 × 961 doubles summed by
// rows, 16 µs of the 217.
PyObject* module_borrowed_einsum(PyObject* /*self*/, PyObject* args)
{
    const char* spec_text = nullptr;
    PyObject*   obj       = nullptr;
    if (PyArg_ParseTuple(args, "sO", &spec_text, &obj) == 0) {
        return nullptr;
    }
    std::string spec;
    for (const char* c = spec_text; *c != '\0'; ++c) {
        if (*c != ' ') {
            spec.push_back(*c);
        }
    }
    const std::size_t arrow {spec.find("->")};
    if (arrow == std::string::npos) {
        return Py_NewRef(Py_NotImplemented);
    }
    const std::string in {spec.substr(0, arrow)};
    const std::string out {spec.substr(arrow + 2)};
    const auto letters_once = [](const std::string& labels) {
        for (std::size_t k = 0; k < labels.size(); ++k) {
            if (std::isalpha(static_cast<unsigned char>(labels[k])) == 0 || labels.find(labels[k], k + 1) != std::string::npos) {
                return false;
            }
        }
        return true;
    };
    // A matrix summed over one of its axes may lie column-major: it is then its transpose row-major, read with its
    // labels and extents reversed, each output element the same pairwise sum of the same elements (a column summed
    // down a block by rows is the pairwise sum of that column alone). A whole sum would take them in another order.
    // A permutation of the axes (each label kept) likewise: a copy in the output's order, a column-major one's labels
    // reversed first.
    const bool       one_axis {in.size() == 2 && out.size() == 1};
    const bool       permutes {out.size() == in.size()};
    borrowed_operand x;
    if (!letters_once(in) || !letters_once(out) || out.find_first_not_of(in) != std::string::npos
        || !borrow_operand(obj, x, one_axis || permutes)
        || x.shape.size() != in.size()) {
        return Py_NewRef(Py_NotImplemented);
    }
    std::string labels {in};
    if (x.transposed) {
        std::reverse(labels.begin(), labels.end());
        std::reverse(x.shape.begin(), x.shape.end());
    }
    try {
        std::optional<any_tensor> result;
        {
            std::size_t count {1};
            for (const std::size_t extent : x.shape) {
                count *= extent;
            }
            const sb::scoped_gil_release unlocked {releases_gil(count)};
            const auto sum = [&]<typename T>(std::type_identity<T>) {
                const auto* const data {static_cast<const T*>(x.data)};
                if (permutes) { // the elements in the output's order: the einsum path's permutation, or its copy
                    std::map<char, std::size_t> sizes;
                    std::vector<std::size_t>    shape;
                    for (std::size_t axis {0}; axis < labels.size(); ++axis) {
                        sizes[labels[axis]] = x.shape[axis];
                    }
                    for (const char label : out) {
                        shape.push_back(sizes.at(label));
                    }
                    numerical::buffer<T> elements {labels == out ? numerical::buffer<T>(data, data + count)
                                                                         : numerical::detail::permuted(data, count, labels, out, sizes)};
                    result.emplace(numerical::tensor<T> {std::move(shape), std::move(elements)});
                }
                else if (std::optional<numerical::tensor<T>> sums {numerical::detail::sum_over_run(labels, out, x.shape, data)}) {
                    result.emplace(*std::move(sums));
                }
            };
            switch (x.dtype) {
            case 0: sum(std::type_identity<double> {}); break;
            case 1: sum(std::type_identity<std::complex<double>> {}); break;
            case 2: sum(std::type_identity<float> {}); break;
            default: sum(std::type_identity<std::complex<float>> {}); break;
            }
        }
        return result ? wrap(std::move(*result)) : Py_NewRef(Py_NotImplemented);
    }
    catch (...) {
        return set_cpp_error();
    }
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

// _borrowed_einsum_pair(spec, a, b) -> Tensor or NotImplemented: einsum of two Tensors or C-ordered NumPy arrays of one
// dtype read where they lie, for an explicit spec whose operands carry each
// label once, none of them its own alone (numerical::detail::einsum_pair, the path einsum takes for such a pair, so its
// bits); NotImplemented for anything else, which einsum then reads from copies (and refuses there, as before). Two
// NumPy arrays were copied into Tensors first: ij,i->ij on 112 × 204, 36 µs where the product takes 14.
PyObject* module_borrowed_einsum_pair(PyObject* /*self*/, PyObject* args)
{
    const char* spec_text = nullptr;
    PyObject*   a_obj     = nullptr;
    PyObject*   b_obj     = nullptr;
    if (PyArg_ParseTuple(args, "sOO", &spec_text, &a_obj, &b_obj) == 0) {
        return nullptr;
    }
    std::string spec;
    for (const char* c = spec_text; *c != '\0'; ++c) {
        if (*c != ' ') {
            spec.push_back(*c);
        }
    }
    const std::size_t arrow {spec.find("->")};
    const std::size_t comma {spec.find(',')};
    if (arrow == std::string::npos || comma == std::string::npos || comma > arrow || spec.find(',', comma + 1) != std::string::npos) {
        return Py_NewRef(Py_NotImplemented);
    }
    const std::string s0 {spec.substr(0, comma)};
    const std::string s1 {spec.substr(comma + 1, arrow - comma - 1)};
    const std::string out {spec.substr(arrow + 2)};
    const auto letters_once = [](const std::string& labels) {
        for (std::size_t k = 0; k < labels.size(); ++k) {
            if (std::isalpha(static_cast<unsigned char>(labels[k])) == 0 || labels.find(labels[k], k + 1) != std::string::npos) {
                return false;
            }
        }
        return true;
    };
    const auto elsewhere = [&out](const std::string& labels, const std::string& other) {
        return std::all_of(labels.begin(), labels.end(), [&](char label) { return other.find(label) != std::string::npos || out.find(label) != std::string::npos; });
    };
    if (!letters_once(s0) || !letters_once(s1) || !letters_once(out) || !elsewhere(s0, s1) || !elsewhere(s1, s0)
        || out.find_first_not_of(s0 + s1) != std::string::npos) {
        return Py_NewRef(Py_NotImplemented);
    }
    borrowed_operand x;
    borrowed_operand y;
    // A column-major matrix stays with the copy: read with its labels reversed, the pair would take another route
    // (ji,j->i is no canonical product), and other bits.
    if (!borrow_operand(a_obj, x, false) || !borrow_operand(b_obj, y, false) || x.dtype != y.dtype || x.shape.size() != s0.size()
        || y.shape.size() != s1.size()) {
        return Py_NewRef(Py_NotImplemented);
    }
    std::map<char, std::size_t> sizes;
    for (auto [labels, operand] : {std::pair<const std::string*, const borrowed_operand*> {&s0, &x}, std::pair<const std::string*, const borrowed_operand*> {&s1, &y}}) {
        for (std::size_t axis {0}; axis < labels->size(); ++axis) {
            const auto [at, fresh] {sizes.emplace((*labels)[axis], operand->shape[axis])};
            if (!fresh && at->second != operand->shape[axis]) {
                return Py_NewRef(Py_NotImplemented); // einsum's own path refuses it
            }
        }
    }
    std::string summed;
    for (const auto& [label, extent] : sizes) {
        if (out.find(label) == std::string::npos) {
            summed.push_back(label);
        }
    }
    const auto count_of = [](const borrowed_operand& operand) {
        std::size_t count {1};
        for (const std::size_t extent : operand.shape) {
            count *= extent;
        }
        return count;
    };
    try {
        std::optional<any_tensor> result;
        {
            const sb::scoped_gil_release unlocked {releases_gil(count_of(x) + count_of(y))};
            const auto pair = [&]<typename T>(std::type_identity<T>) {
                result.emplace(numerical::detail::einsum_pair<T>(s0, {static_cast<const T*>(x.data), count_of(x)}, s1,
                                                                 {static_cast<const T*>(y.data), count_of(y)}, out, summed, sizes));
            };
            switch (x.dtype) {
            case 0: pair(std::type_identity<double> {}); break;
            case 1: pair(std::type_identity<std::complex<double>> {}); break;
            case 2: pair(std::type_identity<float> {}); break;
            default: pair(std::type_identity<std::complex<float>> {}); break;
            }
        }
        return wrap(std::move(*result));
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

// A vector, or a matrix of one row, times a column-major matrix of two columns or more, both read where they lie: the
// vector times the matrix as canonical_product sums it, a column of the matrix at a time (vector_times_transposed), so
// the bits of the product of its C-ordered copy, which was copied first. None for any other pair.
std::optional<PyObject*> borrowed_row_times_transposed(const borrowed_operand& left, const borrowed_operand& right)
{
    const bool row {left.shape.size() == 1 || (left.shape.size() == 2 && left.shape[0] == 1)};
    if (!row || left.transposed || !right.transposed || right.shape.size() != 2 || left.shape.back() != right.shape[0]) {
        return std::nullopt;
    }
    const std::size_t inner {right.shape[0]};
    const std::size_t cols {right.shape[1]};
    if (inner == 0 || cols < 2) {
        return std::nullopt;
    }
    std::vector<std::size_t> out_shape {cols};
    if (left.shape.size() == 2) {
        out_shape.insert(out_shape.begin(), 1);
    }
    try {
        std::optional<any_tensor> result;
        {
            const sb::scoped_gil_release unlocked {releases_gil(inner * (cols + 1))};
            const auto product = [&]<typename T>(std::type_identity<T>) {
                result.emplace(numerical::tensor<T> {std::move(out_shape),
                                                     numerical::detail::vector_times_transposed(static_cast<const T*>(left.data),
                                                                                                static_cast<const T*>(right.data),
                                                                                                inner, inner, cols)});
            };
            switch (left.dtype) {
            case 0: product(std::type_identity<double> {}); break;
            case 1: product(std::type_identity<std::complex<double>> {}); break;
            case 2: product(std::type_identity<float> {}); break;
            default: product(std::type_identity<std::complex<float>> {}); break;
            }
        }
        return wrap(std::move(*result));
    }
    catch (...) {
        return set_cpp_error();
    }
}

// Whether the product of `left` and `right` is a Gram matrix A·Aᵀ or Aᵀ·A (`a @ a.T`, `a.T @ a`): one operand the other's
// transpose, the same elements where they lie, and enough work for the packed kernel, as syrk_all takes it.
bool gram_product(const borrowed_operand& left, const borrowed_operand& right, std::size_t n, std::size_t k)
{
    return left.data == right.data && left.ld == right.ld && left.transposed != right.transposed && k >= numerical::detail::gemm_min_depth
           && n * n >= ((numerical::detail::gemm_packed_threshold + k) - 1) / k;
}

// C = A·Aᵀ (`transposed` false: `a` the row-major n × k A, `ld` apart) or Aᵀ·A (`transposed`: `a` A's column-major n × k
// transpose) into the zeroed n × n `c`: the lower triangle by the matrix product's tiles (gemm_triangle, each element its
// sum in the product), then mirrored, element (i, j) the same products in the same order as (j, i). Half the
// multiply-adds of the whole product (a @ a.T of 1000 × 1000, M1 Pro: 4.33 → 2.89 ms; 300 × 2000: 1.08 → 0.79).
void gram_triangle_mirrored(const double* a, std::size_t ld, bool transposed, std::size_t n, std::size_t k, double* c)
{
    namespace nd = numerical::detail;
    if (transposed) {
        nd::gemm_triangle<nd::a_transposed, nd::b_rows>(n, k, a, ld, a, ld, c, n, true);
    }
    else {
        nd::gemm_triangle<nd::a_rows, nd::b_transposed>(n, k, a, ld, a, ld, c, n, true);
    }
    constexpr std::size_t tile {32};
    const std::size_t     tiles {(n + tile - 1) / tile};
    nd::parallel_for(tiles, tile * n, [&](std::size_t bi) { // block row bi's upper blocks from the lower ones
        const std::size_t i0 {bi * tile};
        for (std::size_t j0 {i0}; j0 < n; j0 += tile) {
            for (std::size_t i {i0}; i < std::min(i0 + tile, n); ++i) {
                for (std::size_t j {std::max(j0, i + 1)}; j < std::min(j0 + tile, n); ++j) {
                    c[(i * n) + j] = c[(j * n) + i];
                }
            }
        }
    });
}

// A column-major matrix (a transpose) times a vector of its dtype, both read where they lie (columns_times_vector):
// each element the bits of its C-ordered copy's product, which was copied first. None for any other pair, or where
// columns_times_vector declines.
std::optional<PyObject*> borrowed_transposed_times_vector(const borrowed_operand& left, const borrowed_operand& right)
{
    if (!left.transposed || right.transposed || left.dtype != right.dtype || left.shape.size() != 2 || right.shape.size() != 1
        || left.shape[1] != right.shape[0] || left.shape[0] == 0 || left.shape[1] == 0) {
        return std::nullopt;
    }
    const std::size_t rows {left.shape[0]};
    const std::size_t cols {left.shape[1]};
    const auto product = [&]<typename T>(std::type_identity<T>) -> std::optional<PyObject*> {
        numerical::buffer<T> y(rows);
        bool                 ran {false};
        {
            const sb::scoped_gil_release unlocked {releases_gil(rows * cols)};
            ran = numerical::detail::columns_times_vector(static_cast<const T*>(left.data), left.ld, rows, cols,
                                                          static_cast<const T*>(right.data), y.data());
        }
        if (!ran) {
            return std::nullopt;
        }
        return wrap(any_tensor {numerical::tensor<T> {{rows}, std::move(y)}});
    };
    try {
        switch (left.dtype) {
            case 0: return product(std::type_identity<double> {});
            case 1: return product(std::type_identity<std::complex<double>> {});
            case 2: return product(std::type_identity<float> {});
            default: return product(std::type_identity<std::complex<float>> {});
        }
    }
    catch (...) {
        return set_cpp_error();
    }
}

// The product of two matrices read where they lie, one or both column-major, in any of the four dtypes: packed (or,
// through gemm_split, copied into their parts) from there by gemm's layouts (gemm_at, gemm_bt, gemm_atbt), its tiles
// gemm's and so its bits, where the column-major one was copied first (NumPy passes BLAS the transpose; a 300 × 300
// product of a transposed operand copied 0.7 MB). A vector or a single row times such a matrix is
// borrowed_row_times_transposed's; every other product with such an operand (a single column, a matrix times a vector,
// whose kernels sum in their own order) is left to the converting path: none.
std::optional<PyObject*> borrowed_transposed_product(const borrowed_operand& left, const borrowed_operand& right)
{
    if (const std::optional<PyObject*> row {borrowed_row_times_transposed(left, right)}) {
        return row;
    }
    if (const std::optional<PyObject*> column {borrowed_transposed_times_vector(left, right)}) {
        return column;
    }
    if (left.shape.size() != 2 || right.shape.size() != 2 || left.shape[1] != right.shape[0]) {
        return std::nullopt;
    }
    const std::size_t m {left.shape[0]};
    const std::size_t k {left.shape[1]};
    const std::size_t n {right.shape[1]};
    if (m < 2 || n < 2 || k == 0) {
        return std::nullopt;
    }
    try {
        std::optional<any_tensor> result;
        {
            const sb::scoped_gil_release unlocked {releases_gil((m * k) + (k * n))};
            const auto product = [&]<typename T>(std::type_identity<T>) {
                numerical::buffer<T> c(m * n);
                numerical::detail::parallel_zero(c.data(), c.size()); // gemm adds to C
                const auto* const a {static_cast<const T*>(left.data)};
                const auto* const b {static_cast<const T*>(right.data)};
                if constexpr (std::is_same_v<T, double>) {
                    if (gram_product(left, right, m, k)) {
                        gram_triangle_mirrored(a, left.ld, left.transposed, m, k, c.data());
                        result.emplace(numerical::tensor<T> {{m, n}, std::move(c)});
                        return;
                    }
                }
                if (left.transposed && right.transposed) {
                    numerical::detail::gemm_atbt(m, n, k, a, left.ld, b, right.ld, c.data(), n);
                }
                else if (left.transposed) {
                    numerical::detail::gemm_at(m, n, k, a, left.ld, b, right.ld, c.data(), n);
                }
                else {
                    numerical::detail::gemm_bt(m, n, k, a, left.ld, b, right.ld, c.data(), n);
                }
                result.emplace(numerical::tensor<T> {{m, n}, std::move(c)});
            };
            switch (left.dtype) {
            case 0: product(std::type_identity<double> {}); break;
            case 1: product(std::type_identity<std::complex<double>> {}); break;
            case 2: product(std::type_identity<float> {}); break;
            default: product(std::type_identity<std::complex<float>> {}); break;
            }
        }
        return wrap(std::move(*result));
    }
    catch (...) {
        return set_cpp_error();
    }
}

std::optional<PyObject*> borrowed_product(PyObject* a, PyObject* b)
{
    borrowed_operand left;
    borrowed_operand right;
    if (!borrow_operand(a, left, true, true) || !borrow_operand(b, right, true, true) || left.dtype != right.dtype) {
        return std::nullopt;
    }
    if (left.transposed || right.transposed) {
        return borrowed_transposed_product(left, right);
    }
    const std::size_t ranks[2] {left.shape.size(), right.shape.size()};
    if (ranks[0] < 1 || ranks[0] > 2 || ranks[1] < 1 || ranks[1] > 2 || left.shape.back() != right.shape.front()) {
        return std::nullopt; // the converting path computes it, or raises its own error
    }
    // As Tensor._matmul's einsum ("ij,jk->ik", "ij,j->i", "i,ij->j", "i,i->") routes a plain product.
    const std::size_t        rows {ranks[0] == 2 ? left.shape[0] : 1};
    const std::size_t        inner {left.shape.back()};
    const std::size_t        cols {ranks[1] == 2 ? right.shape[1] : 1};
    std::vector<std::size_t> out_shape;
    if (ranks[0] == 2) {
        out_shape.push_back(rows);
    }
    if (ranks[1] == 2) {
        out_shape.push_back(cols);
    }
    try {
        std::optional<any_tensor> result;
        {
            const sb::scoped_gil_release unlocked {releases_gil((rows * inner) + (inner * cols))};
            const auto product = [&]<typename T>(std::type_identity<T>) {
                // a vector's elements together, ld then unused: a row's (left), or a column's of one element each (right)
                result.emplace(numerical::detail::canonical_product<T>(static_cast<const T*>(left.data), ranks[0] == 2 ? left.ld : inner,
                                                                       static_cast<const T*>(right.data), ranks[1] == 2 ? right.ld : cols,
                                                                       rows, inner, cols, std::move(out_shape)));
            };
            switch (left.dtype) {
            case 0: product(std::type_identity<double> {}); break;
            case 1: product(std::type_identity<std::complex<double>> {}); break;
            case 2: product(std::type_identity<float> {}); break;
            default: product(std::type_identity<std::complex<float>> {}); break;
            }
        }
        if (ranks[0] == 1 && ranks[1] == 1) { // two vectors give a number, as NumPy's
            return std::visit([](const auto& t) { return from_scalar(t.data()[0]); }, *result);
        }
        return wrap(std::move(*result));
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _borrowed_product(a, b) -> Tensor, number or NotImplemented: numerical.matmul's product of operands read where they
// lie (borrowed_product), NotImplemented when they cannot be, the caller then converting them.
PyObject* module_borrowed_product(PyObject* /*self*/, PyObject* args)
{
    PyObject* a = nullptr;
    PyObject* b = nullptr;
    if (PyArg_ParseTuple(args, "OO", &a, &b) == 0) {
        return nullptr;
    }
    if (const std::optional<PyObject*> product {borrowed_product(a, b)}) {
        return *product;
    }
    Py_RETURN_NOTIMPLEMENTED;
}

// The elements of a borrowed operand of `T` (borrow_operand's dtype index 0 or 1) as an owned tensor: asarray's copy.
template <typename T>
numerical::tensor<T> owned_copy(const borrowed_operand& operand)
{
    std::size_t count {1};
    for (const std::size_t extent : operand.shape) {
        count *= extent;
    }
    const T* const       first {static_cast<const T*>(operand.data)};
    numerical::buffer<T> values(first, first + count);
    return numerical::tensor<T> {operand.shape, std::move(values)};
}

// _borrowed_linalg(name, a[, b]) -> the answer, or NotImplemented: cholesky (b its upper flag), solve, determinant, slogdet or inverse of a square
// float64 or complex128 NumPy array read where it lies (borrow_operand), its one copy the one the factorisation takes
// in place, where asarray's Tensor was a copy before it; b, if any, of the same dtype and rank 1 or 2 (a NumPy array
// copied, or a Tensor; its rows are the solve's to check). NotImplemented for anything else (a Tensor, a stack, single
// precision, a shape the converting path rejects), the caller then converting: each answer is the converting
// path's, from the same values, and so is each error.
PyObject* module_borrowed_linalg(PyObject* /*self*/, PyObject* args)
{
    const char* name = nullptr;
    PyObject*   a    = nullptr;
    PyObject*   b    = nullptr;
    if (PyArg_ParseTuple(args, "sO|O", &name, &a, &b) == 0) {
        return nullptr;
    }
    const std::string operation {name};
    if (operation == "cholesky") { // the third argument is the upper flag, not a right-hand side
        const int upper {b == nullptr ? 0 : PyObject_IsTrue(b)};
        if (upper < 0) {
            return nullptr;
        }
        borrowed_operand matrix;
        const bool       tensor {PyObject_TypeCheck(a, reinterpret_cast<PyTypeObject*>(sb::class_type<any_tensor>())) != 0};
        if (tensor || !borrow_operand(a, matrix) || matrix.dtype > 1 || matrix.shape.size() != 2 || matrix.shape[0] != matrix.shape[1]
            || matrix.shape[0] == 0) {
            Py_RETURN_NOTIMPLEMENTED;
        }
        const std::size_t order {matrix.shape[0]};
        const auto factor = [&]<typename T>(std::type_identity<T>) -> PyObject* {
            std::optional<any_tensor> l;
            {
                const sb::scoped_gil_release unlocked {releases_gil(order * order)};
                l.emplace(cholesky_copied(lower_read(matrix_of_elements(static_cast<const T*>(matrix.data), order, order), upper != 0), upper != 0));
            }
            return wrap(std::move(*l));
        };
        try {
            return matrix.dtype == 0 ? factor(std::type_identity<double> {}) : factor(std::type_identity<std::complex<double>> {});
        }
        catch (...) {
            return set_cpp_error();
        }
    }
    borrowed_operand  left;
    const bool        tensor {PyObject_TypeCheck(a, reinterpret_cast<PyTypeObject*>(sb::class_type<any_tensor>())) != 0};
    if (tensor || !borrow_operand(a, left) || left.dtype > 1 || left.shape.size() != 2 || left.shape[0] != left.shape[1]
        || left.shape[0] == 0 || (operation == "solve") != (b != nullptr)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    const std::size_t n {left.shape[0]};
    borrowed_operand  right;
    if (b != nullptr && (!borrow_operand(b, right) || right.dtype != left.dtype || right.shape.empty() || right.shape.size() > 2)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    const auto run = [&]<typename T>(std::type_identity<T>) -> PyObject* {
        std::optional<numerical::tensor<T>> rhs;
        if (b != nullptr) {
            const bool rhs_tensor {PyObject_TypeCheck(b, reinterpret_cast<PyTypeObject*>(sb::class_type<any_tensor>())) != 0};
            if (rhs_tensor) {
                rhs.emplace(std::get<numerical::tensor<T>>(*reinterpret_cast<sb::wrapper<any_tensor>*>(b)->held));
            }
            else {
                rhs.emplace(owned_copy<T>(right));
            }
        }
        if (operation == "solve") {
            std::optional<any_tensor> x;
            {
                const sb::scoped_gil_release unlocked {releases_gil(n * n)};
                x.emplace(solve_copied(matrix_of_elements(static_cast<const T*>(left.data), n, n), *rhs));
            }
            return wrap(std::move(*x));
        }
        if (operation == "inverse") {
            std::optional<numerical::tensor<T>> inverse;
            {
                const sb::scoped_gil_release unlocked {releases_gil(n * n)};
                inverse.emplace(numerical::inverse(matrix_of_elements(static_cast<const T*>(left.data), n, n)));
            }
            return wrap(any_tensor {std::move(*inverse)});
        }
        if (operation == "determinant") {
            std::optional<T> value;
            {
                const sb::scoped_gil_release unlocked {releases_gil(n * n)};
                value.emplace(numerical::determinant(matrix_of_elements(static_cast<const T*>(left.data), n, n)));
            }
            return from_scalar(*value);
        }
        if (operation == "slogdet") {
            std::optional<std::pair<T, double>> value;
            {
                const sb::scoped_gil_release unlocked {releases_gil(n * n)};
                value.emplace(numerical::lu_decomposition<T> {matrix_of_elements(static_cast<const T*>(left.data), n, n)}.log_determinant());
            }
            PyObject* sign {from_scalar(value->first)}; // new ref
            if (sign == nullptr) {
                return nullptr;
            }
            return Py_BuildValue("(Nd)", sign, value->second);
        }
        Py_RETURN_NOTIMPLEMENTED;
    };
    try {
        return left.dtype == 0 ? run(std::type_identity<double> {}) : run(std::type_identity<std::complex<double>> {});
    }
    catch (...) {
        return set_cpp_error();
    }
}

// A Python float, int (bool included) or complex as a rank-0 tensor of the requested dtype, into `*result`;
// false for anything else, with no error set.
bool asarray_from_number(PyObject* obj, std::optional<std::size_t> wanted, PyObject** result)
{
    const bool complex_in {PyComplex_CheckExact(obj) != 0};
    if (!complex_in && PyFloat_CheckExact(obj) == 0 && PyLong_Check(obj) == 0) {
        return false;
    }
    const std::complex<double> value {complex_in ? std::complex<double> {PyComplex_RealAsDouble(obj), PyComplex_ImagAsDouble(obj)}
                                                 : std::complex<double> {PyFloat_AsDouble(obj), 0.0}};
    if (PyErr_Occurred() != nullptr) { // an int too large for a double
        *result = nullptr;
        return true;
    }
    const element_type element {complex_in ? 'c' : 'f', complex_in ? std::size_t {16} : std::size_t {8}, false};
    std::size_t        out {0};
    if (!output_dtype(element, wanted, out)) {
        *result = nullptr;
        return true;
    }
    try {
        switch (out) {
            case 0: *result = wrap(any_tensor {real_tensor {{}, numerical::buffer<double> {value.real()}}}); break;
            case 1: *result = wrap(any_tensor {cplx_tensor {{}, numerical::buffer<std::complex<double>> {value}}}); break;
            case 2: *result = wrap(any_tensor {single_tensor {{}, numerical::buffer<float> {static_cast<float>(value.real())}}}); break;
            default: *result = wrap(any_tensor {csingle_tensor {{}, numerical::buffer<std::complex<float>> {static_cast<std::complex<float>>(value)}}}); break;
        }
    }
    catch (...) {
        *result = set_cpp_error();
    }
    return true;
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
    std::optional<std::size_t> wanted;
    if (dtype_str != nullptr) {
        wanted = dtype_index(dtype_str);
        if (!wanted) {
            PyErr_SetString(sciforge_module_error(),
                            "asarray: dtype must be 'float64', 'complex128', 'float32' or 'complex64'");
            return nullptr;
        }
    }

    if (PyObject_TypeCheck(obj, reinterpret_cast<PyTypeObject*>(sb::class_type<any_tensor>())) != 0) {
        const any_tensor* const h = as_tensor(obj);
        if (h == nullptr) {
            return nullptr;
        }
        if (!wanted || h->index() == *wanted) {
            return Py_NewRef(obj); // already the requested dtype (or none requested)
        }
        if (is_complex(*h) && (*wanted == 0 || *wanted == 2)) { // complex source, a real dtype requested
            PyErr_SetString(sciforge_module_error(),
                            "asarray: refusing to cast a complex tensor to a real dtype (discards the imaginary part)");
            return nullptr;
        }
        try { // another precision, or real to complex with a zero imaginary part
            return wrap(std::visit([&](const auto& t) -> any_tensor {
                                       using T = element_of_t<std::decay_t<decltype(t)>>;
                                       switch (*wanted) {
                                           case 0: if constexpr (!complex_scalar<T>) { return any_tensor {converted<double>(t)}; } break;
                                           case 1: return any_tensor {converted<std::complex<double>>(t)};
                                           case 2: if constexpr (!complex_scalar<T>) { return any_tensor {converted<float>(t)}; } break;
                                           default: return any_tensor {converted<std::complex<float>>(t)};
                                       }
                                       return t; // unreachable: a complex source asked as a real dtype is refused above
                                   }, *h));
        }
        catch (...) {
            return set_cpp_error();
        }
    }

    PyObject* fast {nullptr};
    if (asarray_from_number(obj, wanted, &fast) || asarray_from_struct(obj, wanted, &fast)) {
        return fast;
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
    std::size_t out {0};
    if (!output_dtype(*element, wanted, out)) {
        Py_DECREF(ai);
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

    PyObject* result {tensor_from_elements(static_cast<const unsigned char*>(address), shape, strides, total, *element, out)};
    Py_DECREF(ai);
    return result;
}


// set_num_threads(n): numerical::set_max_threads; a negative count is refused by the size_t conversion.
void set_num_threads(std::size_t threads)
{
    numerical::set_max_threads(threads);
}

std::size_t get_num_threads()
{
    return numerical::max_threads();
}

// The GEMM path as its code (numerical::gemm_path's value); a code past the last is the last.
int limit_gemm_path_code(int code)
{
    const int last {static_cast<int>(numerical::gemm_path::amx)};
    const int most {code < 0 ? 0 : (code > last ? last : code)};
    return static_cast<int>(numerical::limit_gemm_path(static_cast<numerical::gemm_path>(most)));
}

int gemm_path_code()
{
    return static_cast<int>(numerical::gemm_path_in_use());
}

// Whether the AMX's self-test has run in this process (always false where the AMX path is not compiled).
bool amx_probed()
{
#ifdef NUMERICAL_AMX_COMPILED
    return numerical::detail::amx_probed().load();
#else
    return false;
#endif
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
    .raw("_matmul", tensor_matmul, METH_O,
         "_matmul(other) -> Tensor or number\nself @ other (Tensor.__matmul__).")
    .raw("tolist", tensor_tolist, METH_NOARGS,
         "tolist() -> list\nThe elements as nested Python lists (a bare scalar when rank-0).")
    .def_prop_ro<&get_shape>("shape", "The size of each axis, as a tuple.")
    .def_prop_ro<&get_ndim>("ndim", "The number of axes.")
    .def_prop_ro<&get_size>("size", "The total number of elements.")
    .def_prop_ro<&get_dtype>("dtype", "'float64' or 'complex128'.")
    .def_prop_ro<&get_array_interface>("__array_interface__",
                                       "NumPy array-interface (v3) describing the shared, read-only buffer.");
    // The 7 linear-algebra functions whose args/return are caster-able -> m.def.
    m.def<&set_num_threads>("_set_num_threads",
                            "_set_num_threads(n)\nCap the threads a kernel uses at n (see numerical.set_num_threads).");
    m.def<&set_numpy_loops>("_set_numpy_loops",
                            "_set_numpy_loops(products, scalar_products, quotients)\nWhich of NumPy's complex loops fuse (see numerical._numpy_loops).");
    m.def<&get_num_threads>("get_num_threads", "get_num_threads() -> int\nThe threads a kernel may use.");
    m.def<&limit_gemm_path_code>("_limit_gemm_path", "_limit_gemm_path(code) -> int\nThe path code now in use (see numerical.limit_gemm_path).");
    m.def<&gemm_path_code>("_gemm_path", "_gemm_path() -> int\nThe path code float64 matrix products take.");
    m.def<&amx_probed>("_amx_probed", "_amx_probed() -> bool\nWhether the AMX's self-test has run in this process.");
    m.def<&solve>("solve", "solve(a, b) -> Tensor\nSolve the linear system a @ x = b (a rank-2; b rank-1, or rank-2 with one right-hand side per column).");
    m.def<&inverse>("inverse", "inverse(a) -> Tensor\nThe inverse of a square matrix.");
    m.def<&lstsq>("lstsq", "lstsq(a, b, rtol) -> (x, residuals, rank, s)\nThe minimum-norm least-squares solution of a @ x ≈ b with its residual sums of squares, the rank and the singular values of a, as numpy.linalg.lstsq (SVD; a rank-2; b rank-1, or rank-2 with one right-hand side per column).");
    m.def<&cholesky>("cholesky", "cholesky(a, upper) -> Tensor\nThe lower factor L of A = L Lᴴ (Hermitian positive-definite), or with upper the factor U of A = Uᴴ U from its upper triangle.");
    m.def<&qr>("qr", "qr(a, mode) -> (Q, R)\nThe QR factorisation of an m×n matrix, as numpy.linalg.qr: reduced, Q is m×k, R k×n, k = min(m, n); complete, Q is m×m, R m×n.");
    m.def<&qr_r>("qr_r", "qr_r(a) -> R\nqr's R alone, k×n, k = min(m, n), to the bit, Q never formed (numpy.linalg.qr's mode=\"r\").");
    m.def<&eigh>("eigh", "eigh(a, method, upper) -> (values, vectors)\nEigenpairs of a real symmetric or complex Hermitian matrix (real values, ascending); its lower triangle is read, or with upper its upper one, as numpy.linalg.eigh's UPLO.");
    m.def<&matmul_nd>("_matmul_nd", "_matmul_nd(a, b) -> Tensor\nThe matrix product of two stacks of one dtype, their leading axes broadcast, as numpy.matmul.");
    m.def<&eig>("_eig", "_eig(a) -> (w, v)\nThe eigenvalues and right eigenvectors of a general square matrix (or stack), as numpy.linalg.eig: complex, as NumPy 2.5's.");
    m.def<&eigvals>("_eigvals", "_eigvals(a) -> w\nThe eigenvalues alone of a general square matrix (or stack), as numpy.linalg.eigvals: complex.");
    m.def<&fft_axis>("_fft", "_fft(a, n, axis, norm, inverse) -> Tensor\nnumpy.fft.fft (or ifft) of n points along axis (from 0), norm 0 backward, 1 ortho, 2 forward (see numerical.fft).");
    m.raw("_fftshift", module_fftshift, METH_VARARGS,
          "_fftshift(a, axes, inverse) -> Tensor\nnumpy.fft.fftshift (or ifftshift) along each of axes (from 0), in one copy (see numerical.fft).");
    m.def<&rfft_axis>("_rfft", "_rfft(a, n, axis, norm, hermitian) -> Tensor\nnumpy.fft.rfft (or ihfft) of a real tensor along axis (from 0) (see numerical.fft).");
    m.def<&irfft_axis>("_irfft", "_irfft(a, n, axis, norm, hermitian) -> Tensor\nnumpy.fft.irfft (or hfft) of a complex tensor along axis (from 0) (see numerical.fft).");
    m.def<&eigvalsh>("eigvalsh", "eigvalsh(a, upper) -> Tensor\nThe eigenvalues alone of a real symmetric or complex Hermitian matrix, ascending, as numpy.linalg.eigvalsh (its lower triangle is read, or with upper its upper one).");
    m.def<&matrix_ranks>("_matrix_ranks", "_matrix_ranks(a, rtol, tol, hermitian) -> Tensor\nThe rank of every matrix of a stack, as float64 numbers along its leading axes.");
    m.def<&matrix_rank>("matrix_rank", "matrix_rank(a, rtol, tol, hermitian) -> int\nThe numerical rank of a matrix, as numpy.linalg.matrix_rank: its singular values above tol, else above rtol times the largest (None, None: NumPy's default).");
    m.def<&pinv>("pinv", "pinv(a, rtol, hermitian) -> Tensor\nThe Moore–Penrose pseudo-inverse of an m×n matrix, as numpy.linalg.pinv (rtol None: NumPy's default).");
    m.def<&svd>("svd", "svd(a, method, full) -> (U, S, Vh)\nThe singular value decomposition of an m×n matrix, as numpy.linalg.svd(a, full_matrices=full).");
    m.def<&svd_hermitian_of>("_svd_hermitian", "_svd_hermitian(a) -> (U, S, Vh)\nThe SVD of a real symmetric or complex Hermitian matrix (or stack) from its eigendecomposition, as numpy.linalg.svd(a, hermitian=True).");
    m.def<&singular_values>("_singular_values", "_singular_values(a) -> Tensor\nThe singular values alone, descending, as numpy.linalg.svd(a, compute_uv=False).");
    // The 3 that stay manual: determinant (dtype-dependent scalar), einsum (variadic),
    // asarray (returns the same object when the dtype matches — Py_NewRef identity).
    m.raw("determinant", module_determinant, METH_VARARGS,
          "determinant(a) -> number\nThe determinant of a square matrix (float or complex).");
    m.raw("_take", module_take, METH_VARARGS,
          "_take(t, starts, steps, counts, keep) -> Tensor or number\nThe strided copy behind Tensor.__getitem__.");
    m.raw("_power_shortcut", module_power_shortcut, METH_VARARGS,
          "_power_shortcut(kind, a) -> Tensor\nA complex tensor squared (\"2\"), inverted (\"r\") or rooted (\"h\") as NumPy's ** 2, ** -1, ** 0.5.");
    m.raw("_binary", module_binary, METH_VARARGS,
          "_binary(op, a, b) -> Tensor\nElement-wise a op b (op one of + - * /), broadcast; one dtype.");
    m.raw("_stack_last", module_stack_last, METH_VARARGS,
          "_stack_last(parts) -> Tensor\nTensors of one shape and dtype side by side along a new last axis (a copy).");
    m.raw("_negate", module_negate, METH_VARARGS, "_negate(a) -> Tensor\nElement-wise negation.");
    m.raw("_equal", module_equal, METH_VARARGS,
          "_equal(a, b) -> bool\nWhether two tensors of one dtype and size hold equal elements (a NaN equals nothing).");
    m.raw("_unary", module_unary, METH_VARARGS,
          "_unary(name, a) -> Tensor\nexp, log, sqrt, sin, cos, tan, tanh or abs of every element.");
    m.raw("_borrowed_einsum_pair", module_borrowed_einsum_pair, METH_VARARGS,
          "_borrowed_einsum_pair(spec, a, b) -> Tensor or NotImplemented\nTwo operands of one dtype, each label once and none its own, read where they lie.");
    m.raw("_borrowed_fft", module_borrowed_fft, METH_VARARGS,
          "_borrowed_fft(kind, a, n, axis, norm, inverse) -> Tensor or NotImplemented\nfft (kind 0), rfft (1) or irfft (2) of an operand read where it lies.");
    m.raw("_borrowed_einsum", module_borrowed_einsum, METH_VARARGS,
          "_borrowed_einsum(spec, x) -> Tensor or NotImplemented\nOne operand summed over a run of its axes, read where it lies.");
    m.raw("_borrowed_unary", module_borrowed_unary, METH_VARARGS,
          "_borrowed_unary(name, x) -> Tensor or NotImplemented\nAn element-wise function of every element of a NumPy array read where it lies.");
    m.raw("_borrowed_reduce", module_borrowed_reduce, METH_VARARGS,
          "_borrowed_reduce(op, x) -> number or NotImplemented\nsum, mean, max or min of every element of a NumPy array read where it lies.");
    m.raw("_borrowed_norm", module_borrowed_norm, METH_O,
          "_borrowed_norm(x) -> float or NotImplemented\nThe 2-norm of every element of a Tensor or a NumPy array read where it lies.");
    m.raw("_vector_norm", module_vector_norm, METH_VARARGS,
          "_vector_norm(x, ord, axes, keepdims) -> Tensor or number\nThe ord norm over the given axes, as numpy.linalg.vector_norm, without overflow.");
    m.raw("_reduce", module_reduce, METH_VARARGS,
          "_reduce(op, a, axis) -> number or Tensor\nsum, mean, max or min of every element (axis -1) or along an axis.");
    m.raw("_reshape", module_reshape, METH_VARARGS, "_reshape(a, shape) -> Tensor\nThe elements under another shape.");
    m.raw("_full", module_full, METH_VARARGS, "_full(shape, value, dtype) -> Tensor\nA tensor of one value.");
    m.raw("_eye", module_eye, METH_VARARGS, "_eye(n, dtype) -> Tensor\nThe n×n identity.");
    m.raw("_arange", module_arange, METH_VARARGS, "_arange(start, stop, step) -> Tensor\nAs numpy.arange (float64).");
    m.raw("_linspace", module_linspace, METH_VARARGS,
          "_linspace(start, stop, num) -> Tensor\nAs numpy.linspace with the end point (float64).");
    m.raw("slogdet", module_slogdet, METH_VARARGS,
          "slogdet(a) -> (sign, logabsdet)\nThe sign and the natural log of |det(a)|, as numpy.linalg.slogdet; it\n"
          "does not underflow or overflow where determinant(a) does.");
    m.raw("_borrowed_linalg", module_borrowed_linalg, METH_VARARGS,
          "_borrowed_linalg(name, a, b=None) -> the answer or NotImplemented\ncholesky, solve, determinant, slogdet or "
          "inverse of a NumPy array read where it lies.");
    m.raw("_borrowed_product", module_borrowed_product, METH_VARARGS,
          "_borrowed_product(a, b) -> Tensor, number or NotImplemented\nThe product of two operands read where they lie.");
    m.raw("_set_matmul_fallback", module_set_matmul_fallback, METH_O,
          "_set_matmul_fallback(f)\nThe Python product Tensor._matmul hands every case it does not take.");
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
