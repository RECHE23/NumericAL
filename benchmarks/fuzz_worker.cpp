// One library's side of `make bench-fuzz EIGEN=<include dir>`, compiled once for each: -DBENCH_EIGEN for Eigen,
// -DBENCH_NUMERICAL for NumericAL's C++ API. Both builds share everything below but run_case, so the two are timed
// by the same loop, on the same operands, and answer in the same files.
//
//   fuzz_worker <dir> [--results]
//
// <dir>/manifest.json lists the cases: {"rounds", "target", "numerical_threads", "cases": [{"index", "op",
// "dtype", "operands": [{"file", "shape"}]}]}, each operand a raw little-endian file of its elements in row-major
// order. Each library receives its operands in its own type, built before the timing (an Eigen::Matrix, column-major;
// a numerical::matrix or numerical::vector, row-major): the layouts in which bench_fuzz.py hands NumPy arrays over
// are not a C++ caller's. What a call copies (a decomposition's working copy, a single-precision matrix widened
// where NumericAL decomposes in double precision, as its binding does) is timed. The standard output is
// {"<index>": seconds per call}, the fastest of `rounds` batches each at least `target` seconds long, after a first
// untimed call. <dir>/<library>-info.json records the library's version, threads and SIMD path. With --results,
// each case's answers are written beside the operands as <library>-<index>-out-<k>.bin, described in
// <dir>/<library>-results.json.
//
// bench_fuzz.py builds both with the same compiler: Eigen with -O3 -DNDEBUG, without -march=native (its SIMD set is
// fixed at compile time, and the comparison takes the ISA baseline the NumericAL wheels are built for, theirs chosen
// at run time) and without OpenMP unless asked (one thread, as Eigen ships); NumericAL with the wheels' flags.
#if defined(BENCH_EIGEN) == defined(BENCH_NUMERICAL)
#error "define one of BENCH_EIGEN and BENCH_NUMERICAL"
#endif

#ifdef BENCH_EIGEN
#include <Eigen/Dense>
#else
#include <numerical/numerical.hpp>
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>


static_assert(std::endian::native == std::endian::little, "the operand files are little-endian");

namespace {

  //! \brief A JSON value, as far as the manifest needs one: numbers, strings, booleans, arrays and objects.
  struct Value {
    enum class Kind { null, boolean, number, string, array, object };
    Kind                                       kind   {Kind::null};
    double                                     number {0.0};
    std::string                                text;
    std::vector<Value>                         items;
    std::vector<std::pair<std::string, Value>> fields;

    const Value& operator[](std::string_view key) const
    {
      for (const auto& [name, value] : fields) {
        if (name == key) {
          return value;
        }
      }
      throw std::runtime_error("manifest: no field " + std::string {key});
    }

    long integer() const
    {
      return static_cast<long>(number);
    }
  };

  //! \brief A recursive-descent reader of the manifest's JSON (no escapes beyond \" and \\ are written into it).
  class Reader {
  public:

    explicit Reader(std::string_view source) : source_ {source}
    {}

    Value parse()
    {
      skip();
      if (at_ >= source_.size()) {
        throw std::runtime_error("manifest: unexpected end");
      }
      const char next {source_[at_]};
      Value      value;
      if (next == '{') {
        value.kind = Value::Kind::object;
        ++at_;
        while (!close('}')) {
          std::string name {string()};
          skip();
          expect(':');
          value.fields.emplace_back(std::move(name), parse());
          separator();
        }
      }
      else if (next == '[') {
        value.kind = Value::Kind::array;
        ++at_;
        while (!close(']')) {
          value.items.push_back(parse());
          separator();
        }
      }
      else if (next == '"') {
        value.kind = Value::Kind::string;
        value.text = string();
      }
      else if (source_.substr(at_, 4) == "true" || source_.substr(at_, 5) == "false") {
        value.kind   = Value::Kind::boolean;
        value.number = next == 't' ? 1.0 : 0.0;
        at_         += next == 't' ? 4 : 5;
      }
      else if (source_.substr(at_, 4) == "null") {
        at_ += 4;
      }
      else {
        std::size_t used {0};
        value.kind   = Value::Kind::number;
        value.number = std::stod(std::string {source_.substr(at_, 32)}, &used);
        at_         += used;
      }
      return value;
    }

  private:

    void skip()
    {
      while (at_ < source_.size() && (source_[at_] == ' ' || source_[at_] == '\n' || source_[at_] == '\t' ||
                                      source_[at_] == '\r')) {
        ++at_;
      }
    }

    void expect(char wanted)
    {
      if (at_ >= source_.size() || source_[at_] != wanted) {
        throw std::runtime_error(std::string {"manifest: expected "} + wanted);
      }
      ++at_;
    }

    bool close(char closing)
    {
      skip();
      if (at_ < source_.size() && source_[at_] == closing) {
        ++at_;
        return true;
      }
      return false;
    }

    void separator()
    {
      skip();
      if (at_ < source_.size() && source_[at_] == ',') {
        ++at_;
      }
    }

    std::string string()
    {
      skip();
      expect('"');
      std::string text;
      while (at_ < source_.size() && source_[at_] != '"') {
        if (source_[at_] == '\\') {
          ++at_;
        }
        if (at_ < source_.size()) {
          text += source_[at_++];
        }
      }
      expect('"');
      return text;
    }

    std::string_view source_;
    std::size_t      at_ {0};
  };

  std::string slurp(const std::string& path)
  {
    std::ifstream in {path, std::ios::binary};
    if (!in) {
      throw std::runtime_error("cannot read " + path);
    }
    return {std::istreambuf_iterator<char> {in}, std::istreambuf_iterator<char> {}};
  }

  //! \brief Keeps \p value alive past the optimiser.
  template <typename T>
  void keep(const T& value)
  {
    asm volatile ("" : : "r" (&value) : "memory");
  }

  template <typename Run>
  double batch_time(const Run& run,
                    long       number)
  {
    const auto start {std::chrono::steady_clock::now()};
    for (long call {0}; call < number; ++call) {
      run();
    }
    const std::chrono::duration<double> spent {std::chrono::steady_clock::now() - start};
    return spent.count() / static_cast<double>(number);
  }

  //! \brief The fastest per-call time over \p rounds batches, each at least \p target seconds long: bench_fuzz.py's
  //! fastest(), so the sides are timed alike.
  template <typename Run>
  double fastest(const Run& run,
                 long       rounds,
                 double     target)
  {
    run();
    long number {1};
    while (number < (1L << 20) && batch_time(run, number) * static_cast<double>(number) < target) {
      number *= 2;
    }
    double best {std::numeric_limits<double>::infinity()};
    for (long round {0}; round < rounds; ++round) {
      best = std::min(best, batch_time(run, number));
    }
    return best;
  }

  //! \brief The fastest time of \p make, as fastest(), each call's answer moved into \p answer: allocated and freed
  //! at each call, as a NumPy function's is.
  template <typename Answer, typename Make>
  double timed(Answer&     answer,
               const Make& make,
               long        rounds,
               double      target)
  {
    return fastest([&] {
                     Answer made = make();
                     keep(made);
                     answer = std::move(made);
                   }, rounds, target);
  }

  template <typename T>
  constexpr const char* dtype_name()
  {
    if constexpr (std::is_same_v<T, double>) {
      return "float64";
    }
    else if constexpr (std::is_same_v<T, float>) {
      return "float32";
    }
    else if constexpr (std::is_same_v<T, std::complex<double>>) {
      return "complex128";
    }
    else {
      static_assert(std::is_same_v<T, std::complex<float>>);
      return "complex64";
    }
  }

  //! \brief An answer as bench_fuzz.py reads it back: its elements in row-major order, its NumPy shape and dtype.
  struct Output {
    std::vector<char> bytes;
    std::vector<long> shape;
    std::string       dtype;
  };

  template <typename Scalar>
  Output output(const std::vector<Scalar>& flat,
                std::vector<long>          shape)
  {
    Output out {std::vector<char>(flat.size() * sizeof(Scalar)), std::move(shape), dtype_name<Scalar>()};
    std::copy_n(reinterpret_cast<const char*>(flat.data()), out.bytes.size(), out.bytes.data());
    return out;
  }

  template <typename T>
  Output scalar_output(T value)
  {
    return output(std::vector<T> {value}, {});
  }

  //! \brief An operand as bench_fuzz.py wrote it: its elements in row-major order and its NumPy shape.
  template <typename T>
  struct Operand {
    std::vector<T>    elements;
    std::vector<long> shape;

    [[nodiscard]] std::size_t rows() const
    {
      return static_cast<std::size_t>(shape[0]);
    }

    [[nodiscard]] std::size_t cols() const
    {
      return shape.size() > 1 ? static_cast<std::size_t>(shape[1]) : 1;
    }
  };

  template <typename T>
  Operand<T> load(const std::string& path,
                  std::vector<long>  shape)
  {
    const std::string bytes   {slurp(path)};
    Operand<T>        operand {std::vector<T>(bytes.size() / sizeof(T)), std::move(shape)};
    std::copy_n(bytes.data(), operand.elements.size() * sizeof(T), reinterpret_cast<char*>(operand.elements.data()));
    return operand;
  }

  template <typename T>
  struct real_of {
    using type = T;
  };

  template <typename R>
  struct real_of<std::complex<R>> {
    using type = R;
  };

  template <typename T>
  using real_t = typename real_of<T>::type;

#ifdef BENCH_EIGEN

  constexpr const char* library {"eigen"};

  template <typename T>
  using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  template <typename T>
  using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  template <typename T>
  Matrix<T> matrix_of(const Operand<T>& operand)
  {
    using RowMajor = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    return Eigen::Map<const RowMajor> {operand.elements.data(), static_cast<Eigen::Index>(operand.rows()),
                                       static_cast<Eigen::Index>(operand.cols())};
  }

  template <typename T>
  Vector<T> vector_of(const Operand<T>& operand)
  {
    return Eigen::Map<const Vector<T>> {operand.elements.data(), static_cast<Eigen::Index>(operand.elements.size())};
  }

  template <typename Derived>
  std::vector<typename Derived::Scalar> row_major(const Eigen::MatrixBase<Derived>& answer)
  {
    std::vector<typename Derived::Scalar> flat;
    flat.reserve(static_cast<std::size_t>(answer.size()));
    for (Eigen::Index i {0}; i < answer.rows(); ++i) {
      for (Eigen::Index j {0}; j < answer.cols(); ++j) {
        flat.push_back(answer(i, j));
      }
    }
    return flat;
  }

  template <typename Derived>
  Output matrix_output(const Eigen::MatrixBase<Derived>& answer)
  {
    return output(row_major(answer), {static_cast<long>(answer.rows()), static_cast<long>(answer.cols())});
  }

  template <typename Derived>
  Output vector_output(const Eigen::MatrixBase<Derived>& answer)
  {
    return output(row_major(answer), {static_cast<long>(answer.size())});
  }

  //! \brief The sign and the natural logarithm of the modulus of the determinant LU factors, as numpy.linalg.slogdet:
  //! a sign of 0 and a logarithm of -inf for a singular matrix.
  template <typename Lu>
  std::pair<typename Lu::Scalar, typename Lu::RealScalar> sign_log(const Lu& lu)
  {
    using T    = typename Lu::Scalar;
    using Real = typename Lu::RealScalar;
    T    sign {static_cast<Real>(lu.permutationP().determinant())};
    Real log  {0};
    for (Eigen::Index i {0}; i < lu.matrixLU().rows(); ++i) {
      const T    pivot   {lu.matrixLU()(i, i)};
      const Real modulus {std::abs(pivot)};
      if (modulus == Real {0}) {
        return {T {0}, -std::numeric_limits<Real>::infinity()};
      }
      sign *= pivot / modulus;
      log  += std::log(modulus);
    }
    return {sign, log};
  }

  void configure(const Value& /*manifest*/)
  {
  }

  void describe(std::ostream& info)
  {
    bool openmp {false};
#ifdef _OPENMP
    openmp = true;
#endif
    info << "{\"version\": \"" << EIGEN_WORLD_VERSION << '.' << EIGEN_MAJOR_VERSION << '.' << EIGEN_MINOR_VERSION
         << "\", \"threads\": " << Eigen::nbThreads() << ", \"openmp\": " << (openmp ? "true" : "false")
         << ", \"simd\": \"" << Eigen::SimdInstructionSetsInUse() << "\"}\n";
  }

  //! \brief Times one case and fills \p answers with the last timed call's answers.
  template <typename T>
  double run_case(const std::string&             op,
                  const std::vector<Operand<T>>& operands,
                  long                           rounds,
                  double                         target,
                  std::vector<Output>&           answers)
  {
    using Real    = real_t<T>;
    using Complex = std::complex<Real>;
    const Matrix<T> a       {matrix_of(operands[0])};
    double          seconds {0.0};
    if (op == "matmul") {
      const Matrix<T> b {matrix_of(operands[1])};
      Matrix<T>       c;
      seconds = timed(c, [&]() -> Matrix<T> { return a * b; }, rounds, target);
      answers.push_back(matrix_output(c));
    }
    else if (op == "matvec") {
      const Vector<T> x {vector_of(operands[1])};
      Vector<T>       y;
      seconds = timed(y, [&]() -> Vector<T> { return a * x; }, rounds, target);
      answers.push_back(vector_output(y));
    }
    else if (op == "solve" && operands[1].shape.size() == 1) {
      const Vector<T> b {vector_of(operands[1])};
      Vector<T>       x;
      seconds = timed(x, [&]() -> Vector<T> { return Eigen::PartialPivLU<Matrix<T>> {a}.solve(b); }, rounds, target);
      answers.push_back(vector_output(x));
    }
    else if (op == "solve") {
      const Matrix<T> b {matrix_of(operands[1])};
      Matrix<T>       x;
      seconds = timed(x, [&]() -> Matrix<T> { return Eigen::PartialPivLU<Matrix<T>> {a}.solve(b); }, rounds, target);
      answers.push_back(matrix_output(x));
    }
    else if (op == "det") {
      T det {0};
      seconds = timed(det, [&] { return Eigen::PartialPivLU<Matrix<T>> {a}.determinant(); }, rounds, target);
      answers.push_back(scalar_output(det));
    }
    else if (op == "slogdet") {
      std::pair<T, Real> result;
      seconds = timed(result, [&] { return sign_log(Eigen::PartialPivLU<Matrix<T>> {a}); }, rounds, target);
      answers.push_back(scalar_output(result.first));
      answers.push_back(scalar_output(result.second));
    }
    else if (op == "inv") {
      Matrix<T> inverse;
      seconds = timed(inverse, [&]() -> Matrix<T> { return Eigen::PartialPivLU<Matrix<T>> {a}.inverse(); }, rounds,
                      target);
      answers.push_back(matrix_output(inverse));
    }
    else if (op == "cholesky") {
      Matrix<T> lower;
      seconds = timed(lower, [&]() -> Matrix<T> { return Eigen::LLT<Matrix<T>> {a}.matrixL(); }, rounds, target);
      answers.push_back(matrix_output(lower));
    }
    else if (op == "qr") {
      const Eigen::Index k {std::min(a.rows(), a.cols())};
      Matrix<T>          q;
      Matrix<T>          r;
      seconds = fastest([&] {
                          const Eigen::HouseholderQR<Matrix<T>> qr {a};
                          Matrix<T> thin_q = qr.householderQ() * Matrix<T>::Identity(a.rows(), k);
                          Matrix<T> upper  = qr.matrixQR().topRows(k).template triangularView<Eigen::Upper>();
                          keep(thin_q);
                          keep(upper);
                          q = std::move(thin_q);
                          r = std::move(upper);
                        }, rounds, target);
      answers.push_back(matrix_output(q));
      answers.push_back(matrix_output(r));
    }
    else if (op == "eigh" || op == "eigvalsh") {
      // A solver constructed at each call, as the other answers are: compute() would reuse the last call's storage.
      const int options {op == "eigh" ? Eigen::ComputeEigenvectors : Eigen::EigenvaluesOnly};

      std::optional<Eigen::SelfAdjointEigenSolver<Matrix<T>>> solver;
      seconds = fastest([&] { keep(solver.emplace(a, options)); }, rounds, target);
      answers.push_back(vector_output(solver->eigenvalues()));
      if (op == "eigh") {
        answers.push_back(matrix_output(solver->eigenvectors()));
      }
    }
    else if (op == "svd" || op == "svdvals") {
      const unsigned int thin    {static_cast<unsigned int>(Eigen::ComputeThinU | Eigen::ComputeThinV)};
      const unsigned int options {op == "svd" ? thin : 0U};

      std::optional<Eigen::BDCSVD<Matrix<T>>> svd;
      seconds = fastest([&] { keep(svd.emplace(a, options)); }, rounds, target);
      answers.push_back(vector_output(svd->singularValues()));
      if (op == "svd") {
        answers.insert(answers.begin(), matrix_output(svd->matrixU()));
        answers.push_back(matrix_output(svd->matrixV().adjoint()));
      }
    }
    else if (op == "lstsq") {
      const Matrix<T> b {matrix_of(operands[1])};
      Matrix<T>       x;
      seconds = timed(x, [&]() -> Matrix<T> {
                        return Eigen::BDCSVD<Matrix<T>> {a, Eigen::ComputeThinU | Eigen::ComputeThinV}.solve(b);
                      }, rounds, target);
      answers.push_back(matrix_output(x));
    }
    else if (op == "eig") {
      Vector<Complex> values;
      seconds = timed(values, [&]() -> Vector<Complex> {
                        if constexpr (Eigen::NumTraits<T>::IsComplex) {
                          return Eigen::ComplexEigenSolver<Matrix<T>> {a, false}.eigenvalues();
                        }
                        else {
                          return Eigen::EigenSolver<Matrix<T>> {a, false}.eigenvalues();
                        }
                      }, rounds, target);
      answers.push_back(vector_output(values));
    }
    else {
      throw std::runtime_error("no Eigen equivalent for " + op);
    }
    return seconds;
  }

#else // BENCH_NUMERICAL

  constexpr const char* library {"numerical"};

  //! \brief The precision NumericAL's decompositions compute in: a single-precision operand is widened to it and the
  //! answers rounded back, as its binding does, inside the timing.
  template <typename T>
  using wide_t = std::conditional_t<std::is_same_v<real_t<T>, T>, double, std::complex<double>>;

  template <typename T>
  numerical::matrix<T> matrix_of(const Operand<T>& operand)
  {
    return numerical::matrix<T> {operand.rows(), operand.cols(),
                                 numerical::buffer<T>(operand.elements.begin(), operand.elements.end())};
  }

  template <typename T>
  numerical::vector<T> vector_of(const Operand<T>& operand)
  {
    return numerical::vector<T> {numerical::buffer<T>(operand.elements.begin(), operand.elements.end())};
  }

  //! \brief The elements of \p x (a numerical::matrix or numerical::vector) converted to \p To, in one pass.
  template <typename To, typename X>
  numerical::buffer<To> converted_elements(const X& x)
  {
    numerical::buffer<To> out(static_cast<std::size_t>(std::distance(x.begin(), x.end())));
    std::transform(x.begin(), x.end(), out.begin(), [](const auto& element) { return static_cast<To>(element); });
    return out;
  }

  template <typename To, typename From>
  numerical::matrix<To> converted(const numerical::matrix<From>& m)
  {
    return numerical::matrix<To> {m.rows(), m.cols(), converted_elements<To>(m)};
  }

  template <typename To, typename From>
  numerical::vector<To> converted(const numerical::vector<From>& v)
  {
    return numerical::vector<To> {converted_elements<To>(v)};
  }

  //! \brief \p x in the precision NumericAL decomposes in: itself (no copy) where it is there already.
  template <typename X>
  decltype(auto) widen(const X& x)
  {
    using T = std::decay_t<decltype(*x.begin())>;
    if constexpr (std::is_same_v<wide_t<T>, T>) {
      return (x);
    }
    else {
      return converted<wide_t<T>>(x);
    }
  }

  //! \brief \p x rounded to the element type \p To: itself where it is of that type.
  template <typename To, typename X>
  auto narrowed(X x)
  {
    using From = std::decay_t<decltype(*x.begin())>;
    if constexpr (std::is_same_v<From, To>) {
      return x;
    }
    else {
      return converted<To>(x);
    }
  }

  template <typename S>
  Output matrix_output(const numerical::matrix<S>& answer)
  {
    return output(std::vector<S>(answer.begin(), answer.end()),
                  {static_cast<long>(answer.rows()), static_cast<long>(answer.cols())});
  }

  template <typename S>
  Output vector_output(const numerical::vector<S>& answer)
  {
    return output(std::vector<S>(answer.begin(), answer.end()), {static_cast<long>(answer.size())});
  }

  //! \brief The threads NumericAL may use: the manifest's numerical_threads, or the hardware's where it is 0.
  void configure(const Value& manifest)
  {
    numerical::set_max_threads(static_cast<std::size_t>(manifest["numerical_threads"].integer()));
  }

  void describe(std::ostream& info)
  {
    static constexpr std::array<const char*, 5> paths {"portable", "neon", "avx2", "avx512", "amx"};
    info << "{\"threads\": " << numerical::max_threads() << ", \"gemm_path\": \""
         << paths.at(static_cast<std::size_t>(numerical::detail::gemm_path_current())) << "\"}\n";
  }

  //! \brief Times one case and fills \p answers with the last timed call's answers.
  template <typename T>
  double run_case(const std::string&             op,
                  const std::vector<Operand<T>>& operands,
                  long                           rounds,
                  double                         target,
                  std::vector<Output>&           answers)
  {
    using Real    = real_t<T>;
    using Complex = std::complex<Real>;
    using Matrix  = numerical::matrix<T>;
    using Vector  = numerical::vector<T>;
    const Matrix a       {matrix_of(operands[0])};
    double       seconds {0.0};
    if (op == "matmul") {
      const Matrix b {matrix_of(operands[1])};
      Matrix       c;
      seconds = timed(c, [&] { return a * b; }, rounds, target);
      answers.push_back(matrix_output(c));
    }
    else if (op == "matvec") {
      const Vector x {vector_of(operands[1])};
      Vector       y;
      seconds = timed(y, [&] { return a * x; }, rounds, target);
      answers.push_back(vector_output(y));
    }
    else if (op == "solve" && operands[1].shape.size() == 1) {
      const Vector b {vector_of(operands[1])};
      Vector       x;
      seconds = timed(x, [&] { return narrowed<T>(numerical::solve(widen(a), widen(b))); }, rounds, target);
      answers.push_back(vector_output(x));
    }
    else if (op == "solve") {
      const Matrix b {matrix_of(operands[1])};
      Matrix       x;
      seconds = timed(x, [&] { return narrowed<T>(numerical::solve(widen(a), widen(b))); }, rounds, target);
      answers.push_back(matrix_output(x));
    }
    else if (op == "det") {
      T det {0};
      seconds = timed(det, [&] { return static_cast<T>(numerical::determinant(widen(a))); }, rounds, target);
      answers.push_back(scalar_output(det));
    }
    else if (op == "slogdet") {
      std::pair<T, Real> result;
      seconds = timed(result, [&] {
                        const auto [sign, log] {numerical::lu_decomposition<wide_t<T>> {widen(a)}.log_determinant()};
                        return std::pair<T, Real> {static_cast<T>(sign), static_cast<Real>(log)};
                      }, rounds, target);
      answers.push_back(scalar_output(result.first));
      answers.push_back(scalar_output(result.second));
    }
    else if (op == "inv") {
      Matrix inverse;
      seconds = timed(inverse, [&] { return narrowed<T>(numerical::inverse(widen(a))); }, rounds, target);
      answers.push_back(matrix_output(inverse));
    }
    else if (op == "cholesky") {
      Matrix lower;
      seconds = timed(lower, [&] { return narrowed<T>(numerical::cholesky(widen(a)).lower()); }, rounds, target);
      answers.push_back(matrix_output(lower));
    }
    else if (op == "qr") {
      Matrix q;
      Matrix r;
      seconds = fastest([&] {
                          numerical::qr_decomposition<wide_t<T>> factored {widen(a), numerical::qr_mode::reduced};
                          Matrix thin_q {narrowed<T>(std::move(factored).q())};
                          Matrix upper  {narrowed<T>(std::move(factored).r())};
                          keep(thin_q);
                          keep(upper);
                          q = std::move(thin_q);
                          r = std::move(upper);
                        }, rounds, target);
      answers.push_back(matrix_output(q));
      answers.push_back(matrix_output(r));
    }
    else if (op == "eigh") {
      using Solver = std::conditional_t<std::is_same_v<Real, T>, numerical::symmetric_eigen,
                                        numerical::hermitian_eigen>;
      numerical::vector<Real> values;
      Matrix                  vectors;
      seconds = fastest([&] {
                          Solver                  solver {widen(a)};
                          numerical::vector<Real> made_values {narrowed<Real>(solver.values())};
                          Matrix                  made_vectors {narrowed<T>(std::move(solver).vectors())};
                          keep(made_values);
                          keep(made_vectors);
                          values  = std::move(made_values);
                          vectors = std::move(made_vectors);
                        }, rounds, target);
      answers.push_back(vector_output(values));
      answers.push_back(matrix_output(vectors));
    }
    else if (op == "eigvalsh") {
      numerical::vector<Real> values;
      seconds = timed(values, [&] { return narrowed<Real>(numerical::symmetric_eigenvalues(widen(a))); }, rounds,
                      target);
      answers.push_back(vector_output(values));
    }
    else if (op == "svd") {
      Matrix                  u;
      numerical::vector<Real> s;
      Matrix                  vh;
      seconds = fastest([&] {
                          numerical::svd_decomposition<wide_t<T>> svd {widen(a)}; // divide and conquer, thin
                          numerical::vector<Real> made_s  {narrowed<Real>(svd.singular_values())};
                          Matrix                  made_vh {narrowed<T>(svd.vh())};
                          Matrix                  made_u  {narrowed<T>(std::move(svd).u())};
                          keep(made_u);
                          keep(made_s);
                          keep(made_vh);
                          u  = std::move(made_u);
                          s  = std::move(made_s);
                          vh = std::move(made_vh);
                        }, rounds, target);
      answers.push_back(matrix_output(u));
      answers.push_back(vector_output(s));
      answers.push_back(matrix_output(vh));
    }
    else if (op == "svdvals") {
      numerical::vector<Real> values;
      seconds = timed(values, [&] { return narrowed<Real>(numerical::singular_values(widen(a))); }, rounds, target);
      answers.push_back(vector_output(values));
    }
    else if (op == "lstsq") {
      // NumPy's default cutoff at the operands' own precision: float32's epsilon for a single-precision matrix.
      const double epsilon {static_cast<double>(std::numeric_limits<Real>::epsilon())};
      const double largest {static_cast<double>(std::max(a.rows(), a.cols()))};
      const auto   cutoff  {std::is_same_v<Real, double> ? numerical::rank_cutoff {}
                                                        : numerical::rank_cutoff::relative(epsilon * largest)};
      const Matrix b       {matrix_of(operands[1])};
      Matrix       x;
      seconds = timed(x, [&] { return narrowed<T>(numerical::lstsq(widen(a), widen(b), cutoff)); }, rounds, target);
      answers.push_back(matrix_output(x));
    }
    else if (op == "eig") {
      numerical::vector<Complex> values;
      seconds = timed(values, [&] { return narrowed<Complex>(numerical::eigenvalues(widen(a))); }, rounds, target);
      answers.push_back(vector_output(values));
    }
    else {
      throw std::runtime_error("no NumericAL C++ function for " + op);
    }
    return seconds;
  }

#endif

  template <typename T>
  double run_typed(const Value&         item,
                   const std::string&   dir,
                   long                 rounds,
                   double               target,
                   std::vector<Output>& answers)
  {
    std::vector<Operand<T>> operands;
    for (const Value& operand : item["operands"].items) {
      std::vector<long> shape;
      for (const Value& extent : operand["shape"].items) {
        shape.push_back(extent.integer());
      }
      operands.push_back(load<T>(dir + "/" + operand["file"].text, std::move(shape)));
    }
    return run_case<T>(item["op"].text, operands, rounds, target, answers);
  }

  std::string shape_json(const std::vector<long>& shape)
  {
    std::string text {"["};
    for (std::size_t i {0}; i < shape.size(); ++i) {
      text += (i == 0 ? "" : ", ") + std::to_string(shape[i]);
    }
    return text + "]";
  }
} // namespace

int main(int    argc,
         char** argv)
{
  if (argc < 2) {
    std::fprintf(stderr, "usage: fuzz_worker <dir> [--results]\n");
    return 2;
  }
  const std::string dir     {argv[1]};
  const bool        results {argc > 2 && std::string_view {argv[2]} == "--results"};
  const std::string prefix  {dir + "/" + library};
  try {
    const Value  manifest {Reader {slurp(dir + "/manifest.json")}.parse()};
    const long   rounds   {manifest["rounds"].integer()};
    const double target   {manifest["target"].number};
    configure(manifest);
    {
      std::ofstream info {prefix + "-info.json"};
      describe(info);
    }
    std::ostringstream times;
    std::ostringstream described;
    times.precision(17);
    bool first {true};
    for (const Value& item : manifest["cases"].items) {
      const std::string   index   {std::to_string(item["index"].integer())};
      const std::string&  dtype   {item["dtype"].text};
      std::vector<Output> answers;
      double              seconds {0.0};
      if (dtype == "float64") {
        seconds = run_typed<double>(item, dir, rounds, target, answers);
      }
      else if (dtype == "complex128") {
        seconds = run_typed<std::complex<double>>(item, dir, rounds, target, answers);
      }
      else if (dtype == "float32") {
        seconds = run_typed<float>(item, dir, rounds, target, answers);
      }
      else if (dtype == "complex64") {
        seconds = run_typed<std::complex<float>>(item, dir, rounds, target, answers);
      }
      else {
        throw std::runtime_error("unknown dtype " + dtype);
      }
      times << (first ? "" : ", ") << '"' << index << "\": " << seconds;
      if (results) {
        described << (first ? "" : ", ") << '"' << index << "\": [";
        for (std::size_t k {0}; k < answers.size(); ++k) {
          const std::string file {std::string {library} + "-" + index + "-out-" + std::to_string(k) + ".bin"};
          std::ofstream     out  {dir + "/" + file, std::ios::binary};
          out.write(answers[k].bytes.data(), static_cast<std::streamsize>(answers[k].bytes.size()));
          described << (k == 0 ? "" : ", ") << "{\"file\": \"" << file << "\", \"dtype\": \"" << answers[k].dtype
                    << "\", \"shape\": " << shape_json(answers[k].shape) << '}';
        }
        described << ']';
      }
      first = false;
    }
    if (results) {
      std::ofstream {prefix + "-results.json"} << '{' << described.str() << "}\n";
    }
    std::printf("{%s}\n", times.str().c_str());
  }
  catch (const std::exception& caught) {
    std::fprintf(stderr, "fuzz_worker (%s): %s\n", library, caught.what());
    return 1;
  }
  return 0;
}
