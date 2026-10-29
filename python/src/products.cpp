// NumericAL — Python extension module: the products, einsum, indexing, reshaping and creation (their helpers in
// binding.hpp).

#include "binding.hpp"

namespace numerical_binding {

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
        // inner dimensions that differ go to the fallback, which says so with both shapes
        if (a->index() == b->index() && ranks[0] >= 1 && ranks[0] <= 2 && ranks[1] >= 1 && ranks[1] <= 2
            && shape_of(*a).back() == shape_of(*b).front()) {
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
        || !borrow_operand(obj, x, one_axis || permutes, permutes && in.size() == 2)
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
                    const bool           apart {x.shape.size() == 2 && x.ld != x.shape[1]}; // a matrix of rows ld apart
                    numerical::buffer<T> elements;
                    if (labels != out) {
                        elements = numerical::detail::permuted(data, count, labels, out, sizes, apart ? x.ld : 0);
                    }
                    else if (apart) { // its rows gathered
                        elements.reserve(count);
                        for (std::size_t r {0}; r < x.shape[0]; ++r) {
                            elements.insert(elements.end(), data + (r * x.ld), data + (r * x.ld) + x.shape[1]);
                        }
                    }
                    else {
                        elements.assign(data, data + count);
                    }
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

} // namespace numerical_binding
