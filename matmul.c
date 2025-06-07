#include <stdio.h>
#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

#define MAX_DIM 8
#define NUM_THREADS 8

typedef struct {
  float* data;
  int ndim;
  int shape[MAX_DIM]; 
  int strides[MAX_DIM]; 
  int size;
} NDArray;

typedef struct {
  NDArray *A, *B, *C;
  int m, k, n;
  int start_i, end_i;
} ThreadArgs;

typedef struct {
  NDArray *A, *B, *C;
  int m, k, n;
  int bm, bn, bk;
  int start_i, end_i;
} ThreadArgs2;

void init_ndarray(NDArray* arr, int ndim, int* shape) {
  if (ndim < 1 || ndim > MAX_DIM) {
    arr->ndim = 0;
    arr->size = 0;
    arr->data = NULL;
    return;
  }

  arr->ndim = ndim;
  arr->size = 1;
  for (int i = 0; i < ndim; i++) {
    if (shape[i] <= 0) {
      arr->ndim = 0;
      arr->size = 0;
      arr->data = NULL;
      return;
    }
    arr->shape[i] = shape[i];
    arr->size *= shape[i];
  }
  
  arr->data = (float*)_mm_malloc(arr->size * sizeof(float), 32);
  if (arr->data == NULL) {
    arr->ndim = 0;
    arr->size = 0;
    return;
  }

  arr->strides[ndim-1] = 1;
  for (int i = ndim - 2; i >= 0; i--) {
    arr->strides[i] = arr->strides[i + 1] * arr->shape[i + 1];
  }
}

/////////////////////////
/// Naive 
/////////////////////////

void matmul_naive(NDArray* A, NDArray* B, NDArray* C) {
  int m = A->shape[0];
  int k = A->shape[1];
  int n = B->shape[1];

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      C->data[i*C->strides[0] + j*C->strides[1]] = 0.0f;  
      for (int k_idx = 0; k_idx < k; k_idx++) {
        C->data[i*C->strides[0] + j*C->strides[1]] += A->data[i*A->strides[0] + k_idx*A->strides[1]] * B->data[k_idx*B->strides[0] + j*B->strides[1]];
      }
    }
  }
}

/////////////////////////
/// AVX2 SIMD 
/////////////////////////

void matmul_avx2(NDArray* A, NDArray* B, NDArray* C) {
  int m = A->shape[0];
  int k = A->shape[1];
  int n = B->shape[1];

  for(int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      __m256 sum = _mm256_setzero_ps();
      float* a_ptr = A->data + i * A->strides[0];
      float* b_ptr = B->data + j * B->strides[1];
      int k_idx;
      for (k_idx = 0; k_idx < k - 7; k_idx+=8) {
        __m256 a = _mm256_load_ps(a_ptr + k_idx * A->strides[1]);

        float temp[8];
        for (int t = 0; t < 8; t++) {
          temp[t] = b_ptr[(k_idx + t) * B->strides[0]];
        }
        __m256 b = _mm256_loadu_ps(temp);

        sum = _mm256_fmadd_ps(a, b, sum);
      }

      float temp_sum[8];
      _mm256_storeu_ps(temp_sum, sum);
      float result = 0.0f;
      for (int t = 0; t < 8; t++) {
        result += temp_sum[t];
      }

      for (; k_idx < m; k_idx++) {
        result += a_ptr[k_idx * A->strides[1]] * b_ptr[k_idx * B->strides[0]];
      }

      C->data[i * C->strides[0] + j * C->strides[1]] = result;
    }
  }
}

/////////////////////////
/// Cache Blocked 
/////////////////////////


void matmul_blocked(NDArray* A, NDArray* B, NDArray* C, int bm, int bn, int bk) {
  int m = A->shape[0];
  int k = A->shape[1];
  int n = B->shape[1];

  for (int i = 0; i < m; i += bm) {
    for(int j = 0; j < n; j += bn) {
      for (int k_idx = 0; k_idx < k; k_idx += bk) {
        for(int ii = i; ii < i + bm && ii < m; ii++) {
          for (int jj = j; jj < j + bn && jj < n; jj++) {
            if (k_idx == 0) {
              C->data[ii * C->strides[0] + jj * C->strides[1]] = 0.0f;
            }

            for (int kk = k_idx; kk < k_idx + bk && kk < k; kk++) {
              C->data[ii * C->strides[0] + jj * C->strides[1]] += A->data[ii * A->strides[0] + kk * A->strides[1]] * B->data[kk * B->strides[0] + jj * A->strides[1]]; 
            }
          }
        }
      }
    }
  }
}

//////////////////////////////
///  AVX2 SIMD + Cache Blocked 
//////////////////////////////

void matmul_simd_blocked(NDArray* A, NDArray* B, NDArray* C, int bm, int bn, int bk) {
  int m = A->shape[0];
  int k = A->shape[1];
  int n = B->shape[1];

  for (int i = 0; i < m; i += bm) {
    for(int j = 0; j < n; j += bn) {
      for (int k_idx = 0; k_idx < k; k_idx += bk) {
        for(int ii = i; ii < i + bm && ii < m; ii++) {
          int jj;
          for (jj = j; jj < j + bn && jj + 7 < n; jj += 8) {
            __m256 c_vec;
            if (k_idx == 0) {
              c_vec = _mm256_setzero_ps();
            } else {
              c_vec = _mm256_loadu_ps(&C->data[ii * C->strides[0] + jj * C->strides[1]]);
            }

            for (int kk = k_idx; kk < k_idx + bk && kk < k; kk++) {
              __m256 a_vec = _mm256_set1_ps(A->data[ii*k + kk*C->strides[1]]);
              __m256 b_vec = _mm256_loadu_ps(&B->data[kk*C->strides[0] + jj*C->strides[1]]);
              c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
            }
            _mm256_storeu_ps(&C->data[ii * C->strides[0] + jj * C->strides[1]], c_vec);
          }

          for (; jj < j + bn && jj < n; jj++) {
            if (k_idx == 0) {
              C->data[ii * C->strides[0] + jj * C->strides[1]] = 0.0f;
            }

            for (int kk = k_idx; kk < k_idx + bk && kk < k; kk++) {
              C->data[ii * C->strides[0] + jj * C->strides[1]] += A->data[ii * A->strides[0] + kk * A->strides[1]] * B->data[kk * B->strides[0] + jj * A->strides[1]]; 
            }
          }
        }
      }
    }
  }
}

//////////////////////////////
/// Multi-threaded
//////////////////////////////

void *compute_rows(void* arg) {
  ThreadArgs* targs = (ThreadArgs*)arg;
  NDArray *A = targs->A, *B = targs->B, *C = targs->C;
  int n = targs->n, k = targs->k;
  int start_i = targs->start_i, end_i = targs->end_i;

  for (int i = start_i; i < end_i; i++) {
    for (int j = 0; j < n; j++) {
      C->data[i*C->strides[0] + j*C->strides[1]] = 0;
      for (int k_idx = 0; k_idx < k; k_idx++) {
        C->data[i*C->strides[0] + j*C->strides[1]] += A->data[i*A->strides[0] + k_idx*A->strides[1]] * B->data[k_idx*B->strides[0] + j*B->strides[1]];
      }
    }
  }
  return NULL;
}

void matmul_threaded(NDArray* A, NDArray* B, NDArray* C) {
  int m = A->shape[0];
  int k = A->shape[1];
  int n = B->shape[1];

  pthread_t threads[NUM_THREADS];
  ThreadArgs args[NUM_THREADS];
  int rows_per_thread = m / NUM_THREADS;
  int extra_rows = m % NUM_THREADS;

  int current_row = 0;

  for (int t = 0; t < NUM_THREADS; t++) {
    args[t].start_i = current_row;
    args[t].end_i = current_row + rows_per_thread + (t < extra_rows ? 1 : 0);
    args[t].A = A;
    args[t].B = B;
    args[t].C = C;
    args[t].m = m;
    args[t].n = n;
    args[t].k = k;

    pthread_create(&threads[t], NULL, compute_rows, &args[t]);
    current_row = args[t].end_i;
  }

  for (int t = 0; t < NUM_THREADS; t++) {
    pthread_join(threads[t], NULL);
  }
}

////////////////////////////////////
/// Multi-threaded + SIMD + Blocked
////////////////////////////////////

void *compute_rows2(void* arg) {
  ThreadArgs2* targs = (ThreadArgs2*)arg;
  NDArray *A = targs->A, *B = targs->B, *C = targs->C;
  int m = targs->m, n = targs->n, k = targs->k;
  int bm = targs->bm, bn = targs->bn, bk = targs->bk;
  int start_i = targs->start_i, end_i = targs->end_i;
  

  for (int i = start_i; i < end_i; i += bm) {
    for(int j = 0; j < n; j += bn) {
      for (int k_idx = 0; k_idx < k; k_idx += bk) {
        for(int ii = i; ii < i + bm && ii < end_i && ii < m; ii++) {
          int jj;
          for (jj = j; jj < j + bn && jj + 7 < n; jj += 8) {
            __m256 c_vec;
            if (k_idx == 0) {
              c_vec = _mm256_setzero_ps();
            } else {
              c_vec = _mm256_loadu_ps(&C->data[ii * C->strides[0] + jj * C->strides[1]]);
            }

            for (int kk = k_idx; kk < k_idx + bk && kk < k; kk++) {
              __m256 a_vec = _mm256_set1_ps(A->data[ii*k + kk*C->strides[1]]);
              __m256 b_vec = _mm256_loadu_ps(&B->data[kk*C->strides[0] + jj*C->strides[1]]);
              c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
            }
            _mm256_storeu_ps(&C->data[ii * C->strides[0] + jj * C->strides[1]], c_vec);
          }

          for (; jj < j + bn && jj < n; jj++) {
            if (k_idx == 0) {
              C->data[ii * C->strides[0] + jj * C->strides[1]] = 0.0f;
            }

            for (int kk = k_idx; kk < k_idx + bk && kk < k; kk++) {
              C->data[ii * C->strides[0] + jj * C->strides[1]] += A->data[ii * A->strides[0] + kk * A->strides[1]] * B->data[kk * B->strides[0] + jj * A->strides[1]]; 
            }
          }
        }
      }
    }
  }
  return NULL;
}

void matmul_threaded_blocked_avx2(NDArray* A, NDArray* B, NDArray* C, int bm, int bn, int bk) {
  int m = A->shape[0];
  int k = A->shape[1];
  int n = B->shape[1];

  pthread_t threads[NUM_THREADS];
  ThreadArgs2 args[NUM_THREADS];
  int rows_per_thread = m / NUM_THREADS;
  int extra_rows = m % NUM_THREADS;

  int current_row = 0;

  for (int t = 0; t < NUM_THREADS; t++) {
    args[t].start_i = current_row;
    args[t].end_i = current_row + rows_per_thread + (t < extra_rows ? 1 : 0);
    args[t].A = A;
    args[t].B = B;
    args[t].C = C;
    args[t].m = m;
    args[t].n = n;
    args[t].k = k;
    args[t].bm = bm;
    args[t].bn = bn;
    args[t].bk = bk;

    pthread_create(&threads[t], NULL, compute_rows2, &args[t]);
    current_row = args[t].end_i;
  }

  for (int t = 0; t < NUM_THREADS; t++) {
    pthread_join(threads[t], NULL);
  }
}

int main() {
  struct timespec start, end;
  double time_spent;

  // defining arrays
  NDArray A, B, C;
  int shape_A[] = {2000, 3000};
  int shape_B[] = {3000, 1200};
  int shape_C[] = {2000, 1200};

  // initializing arrays
  init_ndarray(&A, 2, shape_A);
  init_ndarray(&B, 2, shape_B);
  init_ndarray(&C, 2, shape_C);

  for (int i = 0; i < A.size; i++) A.data[i] = i + 1.0;
  for (int i = 0; i < B.size; i++) B.data[i] = 1.0;

  // calling naive matmul
  clock_gettime(CLOCK_MONOTONIC, &start);
  matmul_naive(&A, &B, &C);
  clock_gettime(CLOCK_MONOTONIC, &end);
  time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
  printf("Naive time: %lf\n", time_spent);
  // for (int i = 0; i < C.shape[0]; i++) {
  //   for (int j = 0; j < C.shape[1]; j++) {
  //     printf("%.2f ",C.data[i*C.strides[0] + j*C.strides[1]]);
  //   }
  //   printf("\n");
  // }
  // printf("\n");

  // calling avx2 matmul
  clock_gettime(CLOCK_MONOTONIC, &start);
  matmul_avx2(&A, &B, &C);
  clock_gettime(CLOCK_MONOTONIC, &end);
  time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
  printf("SIMD time: %lf\n", time_spent);
  // for (int i = 0; i < C.shape[0]; i++) {
  //   for (int j = 0; j < C.shape[1]; j++) {
  //     printf("%.2f ",C.data[i*C.strides[0] + j*C.strides[1]]);
  //   }
  //   printf("\n");
  // }
  // printf("\n");

  // calling blocked matmul
  clock_gettime(CLOCK_MONOTONIC, &start);
  matmul_blocked(&A, &B, &C, 18, 18, 18);
  clock_gettime(CLOCK_MONOTONIC, &end);
  time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
  printf("Cache blocked time: %lf\n", time_spent);
  // for (int i = 0; i < C.shape[0]; i++) {
  //   for (int j = 0; j < C.shape[1]; j++) {
  //     printf("%.2f ",C.data[i*C.strides[0] + j*C.strides[1]]);
  //   }
  //   printf("\n");
  // }
  // printf("\n");

  // calling blocked + simd matmul
  clock_gettime(CLOCK_MONOTONIC, &start);
  matmul_simd_blocked(&A, &B, &C, 18, 18, 18);
  clock_gettime(CLOCK_MONOTONIC, &end);
  time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
  printf("Cache blocked + SIMD time: %lf\n", time_spent);
  // for (int i = 0; i < C.shape[0]; i++) {
  //   for (int j = 0; j < C.shape[1]; j++) {
  //     printf("%.2f ",C.data[i*C.strides[0] + j*C.strides[1]]);
  //   }
  //   printf("\n");
  // }
  // printf("\n");

  // calling threaded matmul
  clock_gettime(CLOCK_MONOTONIC, &start);
  matmul_threaded(&A, &B, &C);
  clock_gettime(CLOCK_MONOTONIC, &end);
  time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
  printf("Multi-threaded time: %lf\n", time_spent);
  // for (int i = 0; i < C.shape[0]; i++) {
  //   for (int j = 0; j < C.shape[1]; j++) {
  //     printf("%.2f ",C.data[i*C.strides[0] + j*C.strides[1]]);
  //   }
  //   printf("\n");
  // }
  // printf("\n");

  // calling threaded + simd + blocked matmul
  clock_gettime(CLOCK_MONOTONIC, &start);
  matmul_threaded_blocked_avx2(&A, &B, &C, 18, 18, 18);
  clock_gettime(CLOCK_MONOTONIC, &end);
  time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
  printf("Multi-threaded + SIMD + Blocked time: %lf\n", time_spent);
  // for (int i = 0; i < C.shape[0]; i++) {
  //   for (int j = 0; j < C.shape[1]; j++) {
  //     printf("%.2f ",C.data[i*C.strides[0] + j*C.strides[1]]);
  //   }
  //   printf("\n");
  // }
  // printf("\n");

  // freeing memory
  _mm_free(A.data);
  _mm_free(B.data);
  _mm_free(C.data);
  return 0;
}
