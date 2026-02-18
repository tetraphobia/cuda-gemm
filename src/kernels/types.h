#ifndef KERNEL_ARGS_H
#define KERNEL_ARGS_H
#include <cuda_runtime_api.h>

#define KERNEL_ARGS_DEFAULT                                                    \
  {                                                                            \
      .stream = NULL,                                                          \
  };

typedef struct {
  cudaStream_t stream;
} kernel_args_t;
#endif
