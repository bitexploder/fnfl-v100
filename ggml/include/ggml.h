#pragma once

//
// GGML Tensor Library
//
// This documentation is still a work in progress.
// If you wish some specific topics to be covered, feel free to drop a comment:
//
//   https://github.com/ggml-org/whisper.cpp/issues/40
//
// ## Overview
//
// This library implements:
//
//  - a set of tensor operations
//  - automatic differentiation
//  - basic optimization algorithms
//
// The aim of this library is to provide a minimalistic approach for various machine learning tasks. This includes,
// but is not limited to, the following:
//
//  - linear regression
//  - support vector machines
//  - neural networks
//
// The library allows the user to define a certain function using the available tensor operations. This function
// definition is represented internally via a computation graph. Each tensor operation in the function definition
// corresponds to a node in the graph. Having the computation graph defined, the user can choose to compute the
// function's value and/or its gradient with respect to the input variables. Optionally, the function can be optimized
// using one of the available optimization algorithms.
//
// For example, here we define the function: f(x) = a*x^2 + b
//
//   {
//       struct ggml_init_params params = {
//           .mem_size   = 16*1024*1024,
//           .mem_buffer = NULL,
//       };
//
//       // memory allocation happens here
//       struct ggml_context * ctx = ggml_init(params);
//
//       struct ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
//
//       ggml_set_param(ctx, x); // x is an input variable
//
//       struct ggml_tensor * a  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
//       struct ggml_tensor * b  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
//       struct ggml_tensor * x2 = ggml_mul(ctx, x, x);
//       struct ggml_tensor * f  = ggml_add(ctx, ggml_mul(ctx, a, x2), b);
//
//       ...
//   }
//
// Notice that the function definition above does not involve any actual computation. The computation is performed only
// when the user explicitly requests it. For example, to compute the function's value at x = 2.0:
//
//   {
//       ...
//
//       struct ggml_cgraph * gf = ggml_new_graph(ctx);
//       ggml_build_forward_expand(gf, f);
//
//       // set the input variable and parameter values
//       ggml_set_f32(x, 2.0f);
//       ggml_set_f32(a, 3.0f);
//       ggml_set_f32(b, 4.0f);
//
//       ggml_graph_compute_with_ctx(ctx, &gf, n_threads);
//
//       printf("f = %f\n", ggml_get_f32_1d(f, 0));
//
//       ...
//   }
//
// The actual computation is performed in the ggml_graph_compute() function.
//
// The ggml_new_tensor_...() functions create new tensors. They are allocated in the memory buffer provided to the
// ggml_init() function. You have to be careful not to exceed the memory buffer size. Therefore, you have to know
// in advance how much memory you need for your computation. Alternatively, you can allocate a large enough memory
// and after defining the computation graph, call the ggml_used_mem() function to find out how much memory was
// actually needed.
//
// The ggml_set_param() function marks a tensor as an input variable. This is used by the automatic
// differentiation and optimization algorithms.
//
// The described approach allows to define the function graph once and then compute its forward or backward graphs
// multiple times. All computations will use the same memory buffer allocated in the ggml_init() function. This way
// the user can avoid the memory allocation overhead at runtime.
//
// The library supports multi-dimensional tensors - up to 4 dimensions. The FP16 and FP32 data types are first class
// citizens, but in theory the library can be extended to support FP8 and integer data types.
//
// Each tensor operation produces a new tensor. Initially the library was envisioned to support only the use of unary
// and binary operations. Most of the available operations fall into one of these two categories. With time, it became
// clear that the library needs to support more complex operations. The way to support these operations is not clear
// yet, but a few examples are demonstrated in the following operations:
//
//   - ggml_permute()
//   - ggml_conv_1d_1s()
//   - ggml_conv_1d_2s()
//
// For each tensor operator, the library implements a forward and backward computation function. The forward function
// computes the output tensor value given the input tensor values. The backward function computes the adjoint of the
// input tensors given the adjoint of the output tensor. For a detailed explanation of what this means, take a
// calculus class, or watch the following video:
//
//   What is Automatic Differentiation?
//   https://www.youtube.com/watch?v=wG_nF1awSSY
//
//
// ## Tensor data (struct ggml_tensor)
//
// The tensors are stored in memory via the ggml_tensor struct. The structure provides information about the size of
// the tensor, the data type, and the memory buffer where the tensor data is stored. Additionally, it contains
// pointers to the "source" tensors - i.e. the tensors that were used to compute the current tensor. For example:
//
//   {
//       struct ggml_tensor * c = ggml_add(ctx, a, b);
//
//       assert(c->src[0] == a);
//       assert(c->src[1] == b);
//   }
//
// The multi-dimensional tensors are stored in row-major order. The ggml_tensor struct contains fields for the
// number of elements in each dimension ("ne") as well as the number of bytes ("nb", a.k.a. stride). This allows
// to store tensors that are not contiguous in memory, which is useful for operations such as transposition and
// permutation. All tensor operations have to take the stride into account and not assume that the tensor is
// contiguous in memory.
//
// The data of the tensor is accessed via the "data" pointer. For example:
//
//   {
//       const int nx = 2;
//       const int ny = 3;
//
//       struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nx, ny);
//
//       for (int y = 0; y < ny; y++) {
//           for (int x = 0; x < nx; x++) {
//               *(float *) ((char *) a->data + y*a->nb[1] + x*a->nb[0]) = x + y;
//           }
//       }
//
//       ...
//   }
//
// Alternatively, there are helper functions, such as ggml_get_f32_1d() and ggml_set_f32_1d() that can be used.
//
// ## The matrix multiplication operator (ggml_mul_mat)
//
// TODO
//
//
// ## Multi-threading
//
// TODO
//
//
// ## Overview of ggml.c
//
// TODO
//
//
// ## SIMD optimizations
//
// TODO
//
//
// ## Debugging ggml
//
// TODO
//
//

#ifdef GGML_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef GGML_BUILD
#            define GGML_API __declspec(dllexport) extern
#        else
#            define GGML_API __declspec(dllimport) extern
#        endif
#    else
#        define GGML_API __attribute__ ((visibility ("default"))) extern
#    endif
#else
#    define GGML_API extern
#endif

// TODO: support for clang
#ifdef __GNUC__
#    define GGML_DEPRECATED(func, hint) func __attribute__((deprecated(hint)))
#elif defined(_MSC_VER)
#    define GGML_DEPRECATED(func, hint) __declspec(deprecated(hint)) func
#else
#    define GGML_DEPRECATED(func, hint) func
#endif

#ifndef __GNUC__
#    define GGML_ATTRIBUTE_FORMAT(...)
#elif defined(__MINGW32__) && !defined(__clang__)
#    define GGML_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#    define GGML_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif

#if defined(_WIN32) && !defined(_WIN32_WINNT)
#    define _WIN32_WINNT 0x0A00
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define GGML_FILE_MAGIC   0x67676d6c // "ggml"
#define GGML_FILE_VERSION 2

#define GGML_QNT_VERSION        2    // bump this on quantization format changes
#define GGML_QNT_VERSION_FACTOR 1000 // do not change this

#define GGML_MAX_DIMS           4
#define GGML_MAX_PARAMS         2048
#define GGML_MAX_SRC            10
#define GGML_MAX_N_THREADS      512
#define GGML_MAX_OP_PARAMS      64

#ifndef GGML_MAX_NAME
#   define GGML_MAX_NAME        64
#endif

#define GGML_DEFAULT_N_THREADS  4
#define GGML_DEFAULT_GRAPH_SIZE 2048

#if UINTPTR_MAX == 0xFFFFFFFF
    #define GGML_MEM_ALIGN 4
#elif defined(__EMSCRIPTEN__)
// emscripten uses max_align_t == 8, so we need GGML_MEM_ALIGN == 8 for 64-bit wasm.
// (for 32-bit wasm, the first conditional is true and GGML_MEM_ALIGN stays 4.)
// ref: https://github.com/ggml-org/llama.cpp/pull/18628
    #define GGML_MEM_ALIGN 8
#else
    #define GGML_MEM_ALIGN 16
#endif

#define GGML_EXIT_SUCCESS 0
#define GGML_EXIT_ABORTED 1

// TODO: convert to enum https://github.com/ggml-org/llama.cpp/pull/16187#discussion_r2388538726
#define GGML_ROPE_TYPE_NORMAL 0
#define GGML_ROPE_TYPE_NEOX   2
#define GGML_ROPE_TYPE_MROPE  8
#define GGML_ROPE_TYPE_VISION 24
#define GGML_ROPE_TYPE_IMROPE 40 // binary: 101000

#define GGML_MROPE_SECTIONS   4

#define GGML_UNUSED(x) (void)(x)
#ifdef __CUDACC__
template<typename... Args>
__host__ __device__ constexpr inline void ggml_unused_vars_impl(Args&&...) noexcept {}
#define GGML_UNUSED_VARS(...) ggml_unused_vars_impl(__VA_ARGS__)
#else
#define GGML_UNUSED_VARS(...) do { (void)sizeof((__VA_ARGS__, 0)); } while(0)
#endif // __CUDACC__

#define GGML_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))

#ifndef NDEBUG
#   define GGML_UNREACHABLE() do { fprintf(stderr, "statement should be unreachable\n"); abort(); } while(0)
#elif defined(__GNUC__)
#   define GGML_UNREACHABLE() __builtin_unreachable()
#elif defined(_MSC_VER)
#   define GGML_UNREACHABLE() __assume(0)
#else
#   define GGML_UNREACHABLE() ((void) 0)
#endif

#ifdef __cplusplus
#   define GGML_NORETURN [[noreturn]]
#elif defined(_MSC_VER)
#   define GGML_NORETURN __declspec(noreturn)
#else
#   define GGML_NORETURN _Noreturn
#endif

#define GGML_ABORT(...) ggml_abort(__FILE__, __LINE__, __VA_ARGS__)
#define GGML_ASSERT(x) if (!(x)) GGML_ABORT("GGML_ASSERT(%s) failed", #x)

// used to copy the number of elements and stride in bytes of tensors into local variables.
// main purpose is to reduce code duplication and improve readability.
//
// example:
//
//    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne);
//    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb);
//
#define GGML_TENSOR_LOCALS_1(type, prefix, pointer, array) \
    const type prefix##0 = (pointer) ? (pointer)->array[0] : 0; \
    GGML_UNUSED(prefix##0);
#define GGML_TENSOR_LOCALS_2(type, prefix, pointer, array) \
    GGML_TENSOR_LOCALS_1    (type, prefix, pointer, array) \
    const type prefix##1 = (pointer) ? (pointer)->array[1] : 0; \
    GGML_UNUSED(prefix##1);
#define GGML_TENSOR_LOCALS_3(type, prefix, pointer, array) \
    GGML_TENSOR_LOCALS_2    (type, prefix, pointer, array) \
    const type prefix##2 = (pointer) ? (pointer)->array[2] : 0; \
    GGML_UNUSED(prefix##2);
#define GGML_TENSOR_LOCALS(type, prefix, pointer, array) \
    GGML_TENSOR_LOCALS_3  (type, prefix, pointer, array) \
    const type prefix##3 = (pointer) ? (pointer)->array[3] : 0; \
    GGML_UNUSED(prefix##3);

#define GGML_TENSOR_UNARY_OP_LOCALS \
    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb) \
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne) \
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

#define GGML_TENSOR_BINARY_OP_LOCALS \
    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb) \
    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb) \
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne) \
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

#define GGML_TENSOR_TERNARY_OP_LOCALS \
    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb) \
    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb) \
    GGML_TENSOR_LOCALS(int64_t, ne2, src2, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb2, src2, nb) \
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne) \
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

#define GGML_TENSOR_BINARY_OP_LOCALS01 \
    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb) \
    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

#ifdef  __cplusplus
extern "C" {
#endif

    // Function type used in fatal error callbacks
    typedef void (*ggml_abort_callback_t)(const char * error_message);

    // Set the abort callback (passing null will restore original abort functionality: printing a message to stdout)
    // Returns the old callback for chaining
    GGML_API ggml_abort_callback_t ggml_set_abort_callback(ggml_abort_callback_t callback);

    GGML_NORETURN GGML_ATTRIBUTE_FORMAT(3, 4)
    GGML_API void ggml_abort(const char * file, int line, const char * fmt, ...);

    enum ggml_status {
        GGML_STATUS_ALLOC_FAILED = -2,
        GGML_STATUS_FAILED = -1,
        GGML_STATUS_SUCCESS = 0,
        GGML_STATUS_ABORTED = 1,
    };

    // get ggml_status name string
    GGML_API const char * ggml_status_to_string(enum ggml_status status);

    // ieee 754-2008 half-precision float16
    // todo: make this not an integral type
    typedef uint16_t ggml_fp16_t;
    GGML_API float       ggml_fp16_to_fp32(ggml_fp16_t);
    GGML_API ggml_fp16_t ggml_fp32_to_fp16(float);
    GGML_API void        ggml_fp16_to_fp32_row(const ggml_fp16_t *, float *, int64_t);
    GGML_API void        ggml_fp32_to_fp16_row(const float *, ggml_fp16_t *, int64_t);

    // google brain half-precision bfloat16
    typedef struct { uint16_t bits; } ggml_bf16_t;
    GGML_API ggml_bf16_t ggml_fp32_to_bf16(float);
    GGML_API float       ggml_bf16_to_fp32(ggml_bf16_t);  // consider just doing << 16
    GGML_API void        ggml_bf16_to_fp32_row(const ggml_bf16_t *, float *, int64_t);
    GGML_API void        ggml_fp32_to_bf16_row_ref(const float *, ggml_bf16_t *, int64_t);
    GGML_API void        ggml_fp32_to_bf16_row(const float *, ggml_bf16_t *, int64_t);

    struct ggml_object;
    struct ggml_context;
    struct ggml_cgraph;

    // NOTE: always add types at the end of the enum to keep backward compatibility
    enum ggml_type {
        GGML_TYPE_F32     = 0,
        GGML_TYPE_F16     = 1,
        GGML_TYPE_Q4_0    = 2,
        GGML_TYPE_Q4_1    = 3,
        // GGML_TYPE_Q4_2 = 4, support has been removed
        // GGML_TYPE_Q4_3 = 5, support has been removed
        GGML_TYPE_Q5_0    = 6,
        GGML_TYPE_Q5_1    = 7,
        GGML_TYPE_Q8_0    = 8,
        GGML_TYPE_Q8_1    = 9,
        GGML_TYPE_Q2_K    = 10,
        GGML_TYPE_Q3_K    = 11,
        GGML_TYPE_Q4_K    = 12,
        GGML_TYPE_Q5_K    = 13,
        GGML_TYPE_Q6_K    = 14,
        GGML_TYPE_Q8_K    = 15,
        GGML_TYPE_IQ2_XXS = 16,
        GGML_TYPE_IQ2_XS  = 17,
        GGML_TYPE_IQ3_XXS = 18,
        GGML_TYPE_IQ1_S   = 19,
        GGML_TYPE_IQ4_NL  = 20,
        GGML_TYPE_IQ3_S   = 21,
        GGML_TYPE_IQ2_S   = 22,
        GGML_TYPE_IQ4_XS  = 23,
        GGML_TYPE_I8      = 24,
        GGML_TYPE_I16     = 25,
        GGML_TYPE_I32     = 26,
        GGML_TYPE_I64     = 27,
        GGML_TYPE_F64     = 28,
        GGML_TYPE_IQ1_M   = 29,
        GGML_TYPE_BF16    = 30,
        // GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
        // GGML_TYPE_Q4_0_4_8 = 32,
        // GGML_TYPE_Q4_0_8_8 = 33,
        GGML_TYPE_TQ1_0   = 34,
        GGML_TYPE_TQ2_0   = 35,
        // GGML_TYPE_IQ4_NL_4_4 = 36,
        // GGML_TYPE_IQ4_NL_4_8 = 37,
        // GGML_TYPE_IQ4_NL_8_8 = 38,
        GGML_TYPE_MXFP4   = 39, // MXFP4 (1 block)
        GGML_TYPE_NVFP4   = 40, // NVFP4 (4 blocks, E4M3 scale)
        GGML_TYPE_Q1_0    = 41,
        GGML_TYPE_Q2_0    = 42,
        // IQ1_S's 8-element ternary codebook with a narrower index; see ggml-common.h.
        // 42..63 stay reserved for upstream (42 is already Q2_0), so fork-local types start
        // at 64 and a stock build rejects these files instead of misreading them.
        GGML_TYPE_IQ1_XS   = 64, // 1.4375 bpw, 1024-entry grid
        GGML_TYPE_IQ1_XXS  = 65, // 1.3125 bpw,  512-entry grid
        GGML_TYPE_IQ1_XXXS = 66, // 1.1875 bpw,  256-entry grid
        GGML_TYPE_COUNT   = 67,
    };

    // [TAG_GGML_PREC]
    // this enum is used to declare the allowed numerical precision/data-types types that can be used during the compute of an op
    // the declared types can be:
    //  - result accumulation type
    //  - source tensor data representation type
    //  - etc.
    // the precision parameters are stored as ggml_tensor.op_params to the respective ops
    enum ggml_prec {
        GGML_PREC_UNDEFINED = 0,
        GGML_PREC_DEFAULT   = 0,  // note: deprecated, use GGML_PREC_UNDEFINED
        // stricter than F32: true IEEE F32 arithmetic, no TF32 / tensor-core / reassociating
        // shortcuts (cuBLAS pedantic mode). Ranks below GGML_PREC_F32 so that
        // "at least F32" tests (prec != UNDEFINED && prec <= GGML_PREC_F32) accept it.
        GGML_PREC_F32_PEDANTIC = 5,
        GGML_PREC_F32       = 10,
        GGML_PREC_BF16      = 15,
        GGML_PREC_F16       = 20,
        GGML_PREC_Q8        = 30,
        GGML_PREC_Q4        = 40,
    };

    // op hint
    enum ggml_op_hint {
        GGML_HINT_NONE             = 0,
        GGML_HINT_SRC0_IS_HADAMARD = 1,
        GGML_HINT_HC_PROJ          = 2, // BF16 gated-residual projection; backends may use a dedicated matrix-vector kernel
    };

    // model file types
    enum ggml_ftype {
        GGML_FTYPE_UNKNOWN        = -1,
        GGML_FTYPE_ALL_F32        = 0,
        GGML_FTYPE_MOSTLY_F16     = 1,  // except 1d tensors
        GGML_FTYPE_MOSTLY_Q4_0    = 2,  // except 1d tensors
        GGML_FTYPE_MOSTLY_Q4_1    = 3,  // except 1d tensors
        GGML_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4, // tok_embeddings.weight and output.weight are F16
        GGML_FTYPE_MOSTLY_Q8_0    = 7,  // except 1d tensors
        GGML_FTYPE_MOSTLY_Q5_0    = 8,  // except 1d tensors
        GGML_FTYPE_MOSTLY_Q5_1    = 9,  // except 1d tensors
        GGML_FTYPE_MOSTLY_Q2_K    = 10, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q3_K    = 11, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q4_K    = 12, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q5_K    = 13, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q6_K    = 14, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ2_XXS = 15, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ2_XS  = 16, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ3_XXS = 17, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ1_S   = 18, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ4_NL  = 19, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ3_S   = 20, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ2_S   = 21, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ4_XS  = 22, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ1_M   = 23, // except 1d tensors
        GGML_FTYPE_MOSTLY_BF16    = 24, // except 1d tensors
        GGML_FTYPE_MOSTLY_MXFP4   = 25, // except 1d tensors
        GGML_FTYPE_MOSTLY_NVFP4   = 26, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q1_0    = 27, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q2_0    = 28, // except 1d tensors
        // 28..39 are reserved for upstream ggml, which is already using 28 for Q2_0.
        GGML_FTYPE_MOSTLY_IQ1_XS   = 40, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ1_XXS  = 41, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ1_XXXS = 42, // except 1d tensors
    };

    // available tensor operations:
    enum ggml_op {
        GGML_OP_NONE = 0,

        GGML_OP_DUP,
        GGML_OP_ADD,
        GGML_OP_ADD_ID,
        GGML_OP_ADD1,
        GGML_OP_ACC,
        GGML_OP_SUB,
        GGML_OP_MUL,
        GGML_OP_DIV,
        GGML_OP_SQR,
        GGML_OP_SQRT,
        GGML_OP_LOG,
        GGML_OP_SIN,
        GGML_OP_COS,
        GGML_OP_SUM,
        GGML_OP_SUM_ROWS,
        GGML_OP_CUMSUM,
        GGML_OP_MEAN,
        GGML_OP_ARGMAX,
        GGML_OP_COUNT_EQUAL,
        GGML_OP_REPEAT,
        GGML_OP_REPEAT_BACK,
        GGML_OP_CONCAT,
        GGML_OP_SILU_BACK,
        GGML_OP_NORM, // normalize
        GGML_OP_RMS_NORM,
        GGML_OP_RMS_NORM_BACK,
        GGML_OP_GROUP_NORM,
        GGML_OP_L2_NORM,

        GGML_OP_MUL_MAT,
        GGML_OP_MUL_MAT_ID,
        GGML_OP_OUT_PROD,

        GGML_OP_SCALE,
        GGML_OP_SET,
        GGML_OP_CPY,
        GGML_OP_CONT,
        GGML_OP_RESHAPE,
        GGML_OP_VIEW,
        GGML_OP_PERMUTE,
        GGML_OP_TRANSPOSE,
        GGML_OP_GET_ROWS,
        GGML_OP_GET_ROWS_BACK,
        GGML_OP_SET_ROWS,
        GGML_OP_DIAG,
        GGML_OP_DIAG_MASK_INF,
        GGML_OP_DIAG_MASK_ZERO,
        GGML_OP_SOFT_MAX,
        GGML_OP_SOFT_MAX_BACK,
        GGML_OP_ROPE,
        GGML_OP_ROPE_BACK,
        GGML_OP_CLAMP,
        GGML_OP_CONV_TRANSPOSE_1D,
        GGML_OP_IM2COL,
        GGML_OP_IM2COL_BACK,
        GGML_OP_IM2COL_3D,
        GGML_OP_COL2IM_1D,
        GGML_OP_CONV_2D,
        GGML_OP_CONV_3D,
        GGML_OP_CONV_2D_DW,
        GGML_OP_CONV_TRANSPOSE_2D,
        GGML_OP_POOL_1D,
        GGML_OP_POOL_2D,
        GGML_OP_POOL_2D_BACK,
        GGML_OP_UPSCALE,
        GGML_OP_PAD,
        GGML_OP_PAD_REFLECT_1D,
        GGML_OP_ROLL,
        GGML_OP_ARANGE,
        GGML_OP_TIMESTEP_EMBEDDING,
        GGML_OP_ARGSORT,
        GGML_OP_TOP_K,
        GGML_OP_LEAKY_RELU,
        GGML_OP_TRI,
        GGML_OP_FILL,

        GGML_OP_FLASH_ATTN_EXT,
        GGML_OP_FLASH_ATTN_EXT_BANDED,
        GGML_OP_FLASH_ATTN_BACK,
        GGML_OP_SSM_CONV,
        GGML_OP_SSM_SCAN,
        GGML_OP_WIN_PART,
        GGML_OP_WIN_UNPART,
        GGML_OP_GET_REL_POS,
        GGML_OP_ADD_REL_POS,
        GGML_OP_RWKV_WKV6,
        GGML_OP_GATED_LINEAR_ATTN,
        GGML_OP_RWKV_WKV7,
        GGML_OP_SOLVE_TRI,
        GGML_OP_GATED_DELTA_NET,
        GGML_OP_LIGHTNING_INDEXER,
        GGML_OP_DSV4_HC_COMB,
        GGML_OP_DSV4_HC_PRE,
        GGML_OP_DSV4_HC_POST,
        GGML_OP_QSA_SELECT,
        GGML_OP_QSA_VIS_ROWS,
        GGML_OP_QSA_UNION,

        GGML_OP_UNARY,

        GGML_OP_MAP_CUSTOM1,
        GGML_OP_MAP_CUSTOM2,
        GGML_OP_MAP_CUSTOM3,

        GGML_OP_CUSTOM,

        GGML_OP_CROSS_ENTROPY_LOSS,
        GGML_OP_CROSS_ENTROPY_LOSS_BACK,
        GGML_OP_OPT_STEP_ADAMW,
        GGML_OP_OPT_STEP_SGD,

        GGML_OP_GLU,

        GGML_OP_DRAFT_PICK,

        GGML_OP_COUNT,
    };

    enum ggml_unary_op {
        GGML_UNARY_OP_ABS,
        GGML_UNARY_OP_SGN,
        GGML_UNARY_OP_NEG,
        GGML_UNARY_OP_STEP,
        GGML_UNARY_OP_TANH,
        GGML_UNARY_OP_ELU,
        GGML_UNARY_OP_RELU,
        GGML_UNARY_OP_SIGMOID,
        GGML_UNARY_OP_GELU,
        GGML_UNARY_OP_GELU_QUICK,
        GGML_UNARY_OP_SILU,
        GGML_UNARY_OP_HARDSWISH,
        GGML_UNARY_OP_HARDSIGMOID,
        GGML_UNARY_OP_EXP,
        GGML_UNARY_OP_EXPM1,
        GGML_UNARY_OP_SOFTPLUS,
        GGML_UNARY_OP_GELU_ERF,
        GGML_UNARY_OP_XIELU,
        GGML_UNARY_OP_FLOOR,
        GGML_UNARY_OP_CEIL,
        GGML_UNARY_OP_ROUND,
        GGML_UNARY_OP_TRUNC,

        GGML_UNARY_OP_COUNT,
    };

    enum ggml_glu_op {
        GGML_GLU_OP_REGLU,
        GGML_GLU_OP_GEGLU,
        GGML_GLU_OP_SWIGLU,
        GGML_GLU_OP_SWIGLU_OAI,
        GGML_GLU_OP_GEGLU_ERF,
        GGML_GLU_OP_GEGLU_QUICK,
        GGML_GLU_OP_SWIGLU_CLAMP,

        GGML_GLU_OP_COUNT,
    };

    enum ggml_object_type {
        GGML_OBJECT_TYPE_TENSOR,
        GGML_OBJECT_TYPE_GRAPH,
        GGML_OBJECT_TYPE_WORK_BUFFER
    };

    enum ggml_log_level {
        GGML_LOG_LEVEL_NONE  = 0,
        GGML_LOG_LEVEL_DEBUG = 1,
        GGML_LOG_LEVEL_INFO  = 2,
        GGML_LOG_LEVEL_WARN  = 3,
        GGML_LOG_LEVEL_ERROR = 4,
        GGML_LOG_LEVEL_CONT  = 5, // continue previous log
    };

    // this tensor...
    enum ggml_tensor_flag {
        GGML_TENSOR_FLAG_INPUT   =  1, // ...is an input for the GGML compute graph
        GGML_TENSOR_FLAG_OUTPUT  =  2, // ...is an output for the GGML compute graph
        GGML_TENSOR_FLAG_PARAM   =  4, // ...contains trainable parameters
        GGML_TENSOR_FLAG_LOSS    =  8, // ...defines loss for numerical optimization (multiple loss tensors add up)
        GGML_TENSOR_FLAG_COMPUTE = 16, // ...must be computed
        GGML_TENSOR_FLAG_BACKEND_LAYOUT = 32, // ...data is held in a backend-private layout (ticket 0078): only that backend's matching op may read it
    };

    enum ggml_tri_type {
        GGML_TRI_TYPE_UPPER_DIAG = 0,
        GGML_TRI_TYPE_UPPER      = 1,
        GGML_TRI_TYPE_LOWER_DIAG = 2,
        GGML_TRI_TYPE_LOWER      = 3
    };

    struct ggml_init_params {
        // memory pool
        size_t mem_size;   // bytes
        void * mem_buffer; // if NULL, memory will be allocated internally
        bool   no_alloc;   // don't allocate memory for the tensor data
    };

    // n-dimensional tensor
    struct ggml_tensor {
        enum ggml_type type;

        struct ggml_backend_buffer * buffer;

        int64_t ne[GGML_MAX_DIMS]; // number of elements
        size_t  nb[GGML_MAX_DIMS]; // stride in bytes:
                                   // nb[0] = ggml_type_size(type)
                                   // nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
                                   // nb[i] = nb[i-1] * ne[i-1]

        // compute data
        enum ggml_op op;

        // op params - allocated as int32_t for alignment
        int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];

        int32_t flags;

        struct ggml_tensor * src[GGML_MAX_SRC];

        // source tensor and offset for views
        struct ggml_tensor * view_src;
        size_t               view_offs;

        void * data;

        char name[GGML_MAX_NAME];

        void * extra; // extra things e.g. for ggml-cuda.cu

        char padding[8];
    };

    static const size_t GGML_TENSOR_SIZE = sizeof(struct ggml_tensor);

    // Abort callback
    // If not NULL, called before ggml computation
    // If it returns true, the computation is aborted
    typedef bool (*ggml_abort_callback)(void * data);

    //
    // GUID
    //

    // GUID types
    typedef uint8_t ggml_guid[16];
    typedef ggml_guid * ggml_guid_t;

    GGML_API bool ggml_guid_matches(ggml_guid_t guid_a, ggml_guid_t guid_b);

    // misc

    GGML_API const char * ggml_version(void);
    GGML_API const char * ggml_commit(void);

    GGML_API void    ggml_time_init(void); // call this once at the beginning of the program
    GGML_API int64_t ggml_time_ms(void);
    GGML_API int64_t ggml_time_us(void);
    GGML_API int64_t ggml_cycles(void);
    GGML_API int64_t ggml_cycles_per_ms(void);

    // accepts a UTF-8 path, even on Windows
    GGML_API FILE *  ggml_fopen(const char * fname, const char * mode);

    GGML_API void    ggml_print_object (const struct ggml_object * obj);
    GGML_API void    ggml_print_objects(const struct ggml_context * ctx);

    GGML_API int64_t ggml_nelements (const struct ggml_tensor * tensor);
    GGML_API int64_t ggml_nrows     (const struct ggml_tensor * tensor);
    GGML_API size_t  ggml_nbytes    (const struct ggml_tensor * tensor);
    GGML_API size_t  ggml_nbytes_pad(const struct ggml_tensor * tensor); // same as ggml_nbytes() but padded to GGML_MEM_ALIGN

    GGML_API int64_t ggml_blck_size(enum ggml_type type);
    GGML_API size_t  ggml_type_size(enum ggml_type type);             // size in bytes for all elements in a block
    GGML_API size_t  ggml_row_size (enum ggml_type type, int64_t ne); // size in bytes for all elements in a row

    GGML_DEPRECATED(
    GGML_API double ggml_type_sizef(enum ggml_type type), // ggml_type_size()/ggml_blck_size() as float
    "use ggml_row_size() instead");

    GGML_API const char * ggml_type_name(enum ggml_type type);
    GGML_API const char * ggml_op_name  (enum ggml_op   op);
    GGML_API const char * ggml_op_symbol(enum ggml_op   op);

    GGML_API const char * ggml_unary_op_name(enum ggml_unary_op op);
    GGML_API const char * ggml_glu_op_name(enum ggml_glu_op op);
    GGML_API const char * ggml_op_desc(const struct ggml_tensor * t); // unary or op name

    GGML_API size_t  ggml_element_size(const struct ggml_tensor * tensor);

    GGML_API bool    ggml_is_quantized(enum ggml_type type);

    // TODO: temporary until model loading of ggml examples is refactored
    GGML_API enum ggml_type ggml_ftype_to_ggml_type(enum ggml_ftype ftype);

    GGML_API bool ggml_is_transposed(const struct ggml_tensor * tensor);
    GGML_API bool ggml_is_permuted  (const struct ggml_tensor * tensor);
    GGML_API bool ggml_is_empty     (const struct ggml_tensor * tensor);
    GGML_API bool ggml_is_view      (const struct ggml_tensor * tensor);
    GGML_API bool ggml_is_scalar    (const struct ggml_tensor * tensor);
    GGML_API bool ggml_is_vector    (const struct ggml_tensor * tensor);
    GGML_API bool ggml_is_matrix    (const struct ggml_tensor * tensor);
    GGML_API bool ggml_is_3d        (const struct ggml_tensor * tensor);
    GGML_API int  ggml_n_dims       (const struct ggml_tensor * tensor); // returns 1 for scalars

    // returns whether the tensor elements can be iterated over with a flattened index (no gaps, no permutation)
    GGML_API bool ggml_is_contiguous  (const struct ggml_tensor * tensor);
    GGML_API bool ggml_is_contiguous_0(const struct ggml_tensor * tensor); // same as ggml_is_contiguous()
    GGML_API bool ggml_is_contiguous_1(const struct ggml_tensor * tensor); // contiguous for dims >= 1
    GGML_API bool ggml_is_contiguous_2(const struct ggml_tensor * tensor); // contiguous for dims >= 2

    GGML_API bool ggml_is_contiguous_to_1(const struct ggml_tensor * tensor); // contiguous for dims < 1
    GGML_API bool ggml_is_contiguous_to_2(const struct ggml_tensor * tensor); // contiguous for dims < 2
    GGML_API bool ggml_is_contiguous_to_3(const struct ggml_tensor * tensor); // contiguous for dims < 3

    // returns whether the tensor elements are allocated as one contiguous block of memory (no gaps, but permutation ok)
    GGML_API bool ggml_is_contiguously_allocated(const struct ggml_tensor * tensor);

    // true for tensor that is stored in memory as CxWxHxN and has been permuted to WxHxCxN
    GGML_API bool ggml_is_contiguous_channels(const struct ggml_tensor * tensor);

    // true if the elements in dimension 0 are contiguous, or there is just 1 block of elements
    GGML_API bool ggml_is_contiguous_rows(const struct ggml_tensor * tensor);

    GGML_API bool ggml_are_same_shape (const struct ggml_tensor * t0, const struct ggml_tensor * t1);
    GGML_API bool ggml_are_same_stride(const struct ggml_tensor * t0, const struct ggml_tensor * t1);

    GGML_API bool ggml_can_repeat(const struct ggml_tensor * t0, const struct ggml_tensor * t1);

    // use this to compute the memory overhead of a tensor
    GGML_API size_t ggml_tensor_overhead(void);

    GGML_API bool ggml_validate_row_data(enum ggml_type type, const void * data, size_t nbytes);

    // main

    GGML_API struct ggml_context * ggml_init (struct ggml_init_params params);
    GGML_API void                  ggml_reset(struct ggml_context * ctx);
    GGML_API void                  ggml_free (struct ggml_context * ctx);

    GGML_API size_t  ggml_used_mem(const struct ggml_context * ctx);

    GGML_API bool    ggml_get_no_alloc(struct ggml_context * ctx);
    GGML_API void    ggml_set_no_alloc(struct ggml_context * ctx, bool no_alloc);

    GGML_API void *  ggml_get_mem_buffer     (const struct ggml_context * ctx);
    GGML_API size_t  ggml_get_mem_size       (const struct ggml_context * ctx);
    GGML_API size_t  ggml_get_max_tensor_size(const struct ggml_context * ctx);

    GGML_API struct ggml_tensor * ggml_new_tensor(
            struct ggml_context * ctx,
            enum   ggml_type type,
            int    n_dims,
            const int64_t *ne);

    GGML_API struct ggml_tensor * ggml_new_tensor_1d(
            struct ggml_context * ctx,
            enum   ggml_type type,
            int64_t ne0);

    GGML_API struct ggml_tensor * ggml_new_tensor_2d(
            struct ggml_context * ctx,
            enum   ggml_type type,
            int64_t ne0,
            int64_t ne1);

    GGML_API struct ggml_tensor * ggml_new_tensor_3d(
            struct ggml_context * ctx,
            enum   ggml_type type,
            int64_t ne0,
            int64_t ne1,
            int64_t ne2);

    GGML_API struct ggml_tensor * ggml_new_tensor_4d(
            struct ggml_context * ctx,
            enum   ggml_type type,
            int64_t ne0,
            int64_t ne1,
            int64_t ne2,
            int64_t ne3);

    GGML_API void * ggml_new_buffer(struct ggml_context * ctx, size_t nbytes);

    GGML_API struct ggml_tensor * ggml_dup_tensor (struct ggml_context * ctx, const struct ggml_tensor * src);
    GGML_API struct ggml_tensor * ggml_view_tensor(struct ggml_context * ctx, struct ggml_tensor * src);

    // Context tensor enumeration and lookup
    GGML_API struct ggml_tensor * ggml_get_first_tensor(const struct ggml_context * ctx);
    GGML_API struct ggml_tensor * ggml_get_next_tensor (const struct ggml_context * ctx, struct ggml_tensor * tensor);
    GGML_API struct ggml_tensor * ggml_get_tensor(struct ggml_context * ctx, const char * name);

    // Converts a flat index into coordinates
    GGML_API void ggml_unravel_index(const struct ggml_tensor * tensor, int64_t i, int64_t * i0, int64_t * i1, int64_t * i2, int64_t * i3);

    GGML_API enum ggml_unary_op ggml_get_unary_op(const struct ggml_tensor * tensor);
    GGML_API enum ggml_glu_op ggml_get_glu_op(const struct ggml_tensor * tensor);

    GGML_API void *  ggml_get_data    (const struct ggml_tensor * tensor);
    GGML_API float * ggml_get_data_f32(const struct ggml_tensor * tensor);

    GGML_API const char *         ggml_get_name   (const struct ggml_tensor * tensor);
    GGML_API struct ggml_tensor * ggml_set_name   (      struct ggml_tensor * tensor, const char * name);
    GGML_ATTRIBUTE_FORMAT(2, 3)
    GGML_API struct ggml_tensor * ggml_format_name(      struct ggml_tensor * tensor, const char * fmt, ...);

    //
    // op parameter types (ggml-cuda reads them from tensor op_params)
    //

    // compute correction dims for YaRN RoPE scaling
    GGML_API void ggml_rope_yarn_corr_dims(
        int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]);

    enum ggml_op_pool {
        GGML_OP_POOL_MAX,
        GGML_OP_POOL_AVG,
        GGML_OP_POOL_COUNT,
    };

    enum ggml_scale_mode {
        GGML_SCALE_MODE_NEAREST  = 0,
        GGML_SCALE_MODE_BILINEAR = 1,
        GGML_SCALE_MODE_BICUBIC  = 2,

        GGML_SCALE_MODE_COUNT
    };

    enum ggml_scale_flag {
        GGML_SCALE_FLAG_ALIGN_CORNERS = (1 << 8),
        GGML_SCALE_FLAG_ANTIALIAS     = (1 << 9),
    };

    // sort rows
    enum ggml_sort_order {
        GGML_SORT_ORDER_ASC,
        GGML_SORT_ORDER_DESC,
    };

    typedef void (*ggml_custom1_op_t)(struct ggml_tensor * dst , const struct ggml_tensor * a, int ith, int nth, void * userdata);
    typedef void (*ggml_custom2_op_t)(struct ggml_tensor * dst , const struct ggml_tensor * a, const struct ggml_tensor * b, int ith, int nth, void * userdata);
    typedef void (*ggml_custom3_op_t)(struct ggml_tensor * dst , const struct ggml_tensor * a, const struct ggml_tensor * b, const struct ggml_tensor * c, int ith, int nth, void * userdata);

#define GGML_N_TASKS_MAX (-1)
    // n_tasks == GGML_N_TASKS_MAX means to use max number of tasks

    typedef void (*ggml_custom_op_t)(struct ggml_tensor * dst , int ith, int nth, void * userdata);

    // TODO these functions were sandwiched in the old optimization interface, is there a better place for them?
    typedef void (*ggml_log_callback)(enum ggml_log_level level, const char * text, void * user_data);

    // Set callback for all future logging events.
    // If this is not called, or NULL is supplied, everything is output on stderr.
    GGML_API void ggml_log_get(ggml_log_callback * log_callback, void ** user_data);
    GGML_API void ggml_log_set(ggml_log_callback   log_callback, void *  user_data);

    //
    // quantization
    //

    // - ggml_quantize_init can be called multiple times with the same type
    //   it will only initialize the quantization tables for the first call or after ggml_quantize_free
    //   automatically called by ggml_quantize_chunk for convenience
    //
    // - ggml_quantize_free will free any memory allocated by ggml_quantize_init
    //   call this at the end of the program to avoid memory leaks
    //
    // note: these are thread-safe
    //
    GGML_API void ggml_quantize_init(enum ggml_type type);
    GGML_API void ggml_quantize_free(void);

    // some quantization type cannot be used without an importance matrix
    GGML_API bool ggml_quantize_requires_imatrix(enum ggml_type type);

    // calls ggml_quantize_init internally (i.e. can allocate memory)
    GGML_API size_t ggml_quantize_chunk(
            enum ggml_type   type,
               const float * src,
                      void * dst,
                   int64_t   start,
                   int64_t   nrows,
                   int64_t   n_per_row,
               const float * imatrix);

#ifdef __cplusplus
    // restrict not standard in C++
#    if defined(__GNUC__)
#        define GGML_RESTRICT __restrict__
#    elif defined(__clang__)
#        define GGML_RESTRICT __restrict
#    elif defined(_MSC_VER)
#        define GGML_RESTRICT __restrict
#    else
#        define GGML_RESTRICT
#    endif
#else
#    if defined (_MSC_VER) && (__STDC_VERSION__ < 201112L)
#        define GGML_RESTRICT __restrict
#    else
#        define GGML_RESTRICT restrict
#    endif
#endif
    typedef void (*ggml_to_float_t)  (const void  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
    typedef void (*ggml_from_float_t)(const float * GGML_RESTRICT x, void  * GGML_RESTRICT y, int64_t k);

    struct ggml_type_traits {
        const char             * type_name;
        int64_t                  blck_size;
        int64_t                  blck_size_interleave; // interleave elements in blocks
        size_t                   type_size;
        bool                     is_quantized;
        ggml_to_float_t          to_float;
        ggml_from_float_t        from_float_ref;
    };

    GGML_API const struct ggml_type_traits * ggml_get_type_traits(enum ggml_type type);

    // ggml threadpool
    // TODO: currently, only a few functions are in the base ggml API, while the rest are in the CPU backend
    // the goal should be to create an API that other backends can use move everything to the ggml base

    // scheduling priorities
    enum ggml_sched_priority {
        GGML_SCHED_PRIO_LOW = -1,
        GGML_SCHED_PRIO_NORMAL,
        GGML_SCHED_PRIO_MEDIUM,
        GGML_SCHED_PRIO_HIGH,
        GGML_SCHED_PRIO_REALTIME
    };

    // threadpool params
    // Use ggml_threadpool_params_default() or ggml_threadpool_params_init() to populate the defaults
    struct ggml_threadpool_params {
        bool                cpumask[GGML_MAX_N_THREADS]; // mask of cpu cores (all-zeros means use default affinity settings)
        int                 n_threads;                   // number of threads
        enum ggml_sched_priority prio;                   // thread priority
        uint32_t            poll;                        // polling level (0 - no polling, 100 - aggressive polling)
        bool                strict_cpu;                  // strict cpu placement
        bool                paused;                      // start in paused state
    };

    struct ggml_threadpool;     // forward declaration, see ggml.c

    typedef struct ggml_threadpool * ggml_threadpool_t;

    GGML_API struct ggml_threadpool_params ggml_threadpool_params_default(int n_threads);
    GGML_API void                          ggml_threadpool_params_init   (struct ggml_threadpool_params * p, int n_threads);
    GGML_API bool                          ggml_threadpool_params_match  (const struct ggml_threadpool_params * p0, const struct ggml_threadpool_params * p1);

#ifdef  __cplusplus
}
#endif
