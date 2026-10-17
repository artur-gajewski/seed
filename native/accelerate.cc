#include <node_api.h>
#include <Accelerate/Accelerate.h>
#include <climits>
#include <cmath>
#include <cstdint>

// Synchronous BLAS calls borrow Node's Float64Array storage; no copies or retained pointers.
static napi_value Matmul(napi_env env, napi_callback_info info) {
  size_t argc = 9;
  napi_value args[9];
  if (napi_get_cb_info(env, info, &argc, args, nullptr, nullptr) != napi_ok || argc != 9) {
    napi_throw_type_error(env, nullptr, "matmul requires 9 arguments");
    return nullptr;
  }
  double* buffers[3];
  size_t lengths[3];
  for (int i = 0; i < 3; i++) {
    bool typed = false;
    napi_typedarray_type type;
    void* data;
    napi_value arraybuffer;
    size_t offset;
    if (napi_is_typedarray(env, args[i], &typed) != napi_ok || !typed ||
        napi_get_typedarray_info(env, args[i], &type, &lengths[i], &data,
                                &arraybuffer, &offset) != napi_ok || type != napi_float64_array) {
      napi_throw_type_error(env, nullptr, "matmul requires Float64Array buffers");
      return nullptr;
    }
    buffers[i] = static_cast<double*>(data);
  }
  int dimensions[3];
  for (int i = 0; i < 3; i++) {
    double value;
    if (napi_get_value_double(env, args[i + 3], &value) != napi_ok ||
        !std::isfinite(value) || value < 1 || value > INT_MAX || std::floor(value) != value) {
      napi_throw_range_error(env, nullptr, "matmul dimensions must be positive BLAS integers");
      return nullptr;
    }
    dimensions[i] = static_cast<int>(value);
  }
  bool transposeA, transposeB;
  double beta;
  if (napi_get_value_bool(env, args[6], &transposeA) != napi_ok ||
      napi_get_value_bool(env, args[7], &transposeB) != napi_ok ||
      napi_get_value_double(env, args[8], &beta) != napi_ok || !std::isfinite(beta)) {
    napi_throw_type_error(env, nullptr, "Invalid matmul transpose or accumulation arguments");
    return nullptr;
  }
  const int m = dimensions[0], n = dimensions[1], k = dimensions[2];
  if (lengths[0] < size_t(m) * k || lengths[1] < size_t(k) * n || lengths[2] < size_t(m) * n) {
    napi_throw_range_error(env, nullptr, "matmul buffer is smaller than its dimensions");
    return nullptr;
  }
  // BLAS forbids output overlapping either input, including overlapping typed-array views.
  for (int i = 0; i < 2; i++) {
    const uintptr_t start = reinterpret_cast<uintptr_t>(buffers[i]);
    const uintptr_t end = start + lengths[i] * sizeof(double);
    const uintptr_t output = reinterpret_cast<uintptr_t>(buffers[2]);
    if (output < end && start < output + lengths[2] * sizeof(double)) {
      napi_throw_range_error(env, nullptr, "matmul output must not overlap an input");
      return nullptr;
    }
  }
  cblas_dgemm(CblasRowMajor, transposeA ? CblasTrans : CblasNoTrans,
              transposeB ? CblasTrans : CblasNoTrans, m, n, k, 1.0,
              buffers[0], transposeA ? m : k, buffers[1], transposeB ? k : n,
              beta, buffers[2], n);
  napi_value result;
  napi_get_undefined(env, &result);
  return result;
}

static napi_value Init(napi_env env, napi_value exports) {
  napi_value function;
  napi_create_function(env, "matmul", NAPI_AUTO_LENGTH, Matmul, nullptr, &function);
  napi_set_named_property(env, exports, "matmul", function);
  return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)
