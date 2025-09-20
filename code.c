/*
  Skew this! Loop skewing and other transformations

  Instructions: find all instances of "STUDENT_TODO" and replace with code
                that makes the test corresponding to that function pass. Note
                that while there is a particular loop optimization being
                emphasized in a given problem, multiple optimizations may
                be performed simualtaneously.

                At the command prompt in the directory containing this code
                run 'make'

  Submission: For this assignment you will upload three artifacts to canvas.
              1. [figures.pdf] For each of the tasks you will draw a diagram for
  the original and transformed code.
              2. [results.txt] containing the test output of your code.
              3. [code.c] Your modified version of this code.


  - richard.m.veras@ou.edu
*/

/*

  Loop Transformations:
  0. Unswitching
  1. Peeling (Splitting)
  2. Index Set splitting
  3. Fusion
  4. Fission
  5. Loop Interchange
  6. Strip Mining
  7. Skewing

  Resources:
  Lect 14 (OUCS_4473_5473_lect14_week10_day02.pptx)
  Lect 21 (OUCS_4473_5473_lect21_week15_day02 - transpose hints, AVX SIMD.pptx)

https://sites.cs.ucsb.edu/~tyang/class/240a13w/slides/LectureParallelization2s.pdf
https://www.inf.ed.ac.uk/teaching/courses/copt/lecture-9.pdf
https://www.cri.ensmp.fr/~tadonki/PaperForWeb/tadonki_loop.pdf

*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*
  Helper functions
*/

float max_pair_wise_diff(int m, float *a, float *b) {
  float max_diff = 0.0;

  for (int i = 0; i < m; ++i) {
    float sum = fabs(a[i] + b[i]);
    float diff = fabs(a[i] - b[i]);

    float res = 0.0f;

    if (sum == 0.0f)
      res = diff;
    else
      res = 2 * diff / sum;

    if (res > max_diff)
      max_diff = res;
  }

  return max_diff;
}

void print_8xfloat_mem(char *name, float *src) {
  const int vlen = 8;

  printf("%s = [ ", name);
  for (int vid = 0; vid < vlen; ++vid) {
    if (src[vid] < 0.0f)
      printf(" x, ", src[vid]);
    else
      printf("%2.f, ", src[vid]);
  }
  printf("]\n");
}

void print_float_mem(char *name, int vlen, float *src) {

  printf("%s = [ ", name);
  for (int vid = 0; vid < vlen; ++vid) {
    if (src[vid] < 0.0f)
      printf(" x, ", src[vid]);
    else
      printf("%2.f, ", src[vid]);
  }
  printf("]\n");
}

void print_float_mem_as_vects(char *name, int size, int vlen, float *src) {

  for (int i = 0; i < size; i += vlen) {
    printf("%s[%2i:%2i] = [ ", name, i, i + vlen);
    for (int vid = 0; vid < vlen; ++vid) {
      if (src[vid + i] < 0.0f)
        printf(" x, ", src[vid + i]);
      else
        printf("%2.f, ", src[vid + i]);
    }
    printf("]\n");
  }
  printf("\n");
}

// Loop Unswitching
/*
  0 <= size
  size%2 == 0
  src and dst do not alias (overlap)
*/
void reference_I_tensor_DFT2(int size, float *src, float *dst) {

  for (int i = 0; i < size; ++i) {
    if (i % 2 == 0)
      dst[i] = src[i] - src[i + 1];
    else
      dst[i] = src[i - 1] + src[i];
  }
}

void student_I_tensor_DFT2_unswitch(int size, float *src, float *dst) {
  // reduce branch operations by only switching once for each case (even or odd
  // iterations)
  for (int ii = 0; ii < 2; ++ii)
    if (ii % 2 == 0) { // only runs when ii is 0, so only runs once
      for (int io = 0; io < size;
           io += 2) { // makes i iterate over 0, 2, 4, 6,...
        int i = io + ii;
        dst[i] = src[i] - src[i + 1];
      }
    } else { // only runs when ii is 1, so only runs once
      for (int io = 0; io < size;
           io += 2) { // makes i iterate over 1, 3, 5, 7, ...
        int i = io + ii;
        dst[i] = src[i - 1] + src[i];
      }
    }
}

void test_I_tensor_DFT2_unswitch() {
  const int size = 8;
  float a[] = {7, 6, 5, 4, 3, 2, 1, 0};
  float bt[] = {-1, -1, -1, -1, -1, -1, -1, -1};
  float br[] = {-1, -1, -1, -1, -1, -1, -1, -1};

  reference_I_tensor_DFT2(size, a, br);
  student_I_tensor_DFT2_unswitch(size, a, bt);

  float res = max_pair_wise_diff(8, bt, br);

  printf("test_I_tensor_DFT2_unswitch: ");
  if (res > 1e-6) {
    printf("FAIL\n");

    print_8xfloat_mem(" a", a);
    print_8xfloat_mem("bt", bt);
    print_8xfloat_mem("br", br);

    printf("\n");
  } else {
    printf("PASS\n");
  }
}

// PROBLEM: Index Set Splitting
/*
  0 <= size
  0 <= shift < size
  src and dst do not alias (overlap)
 */
void reference_rotate_no_mod(int size, int shift, float *src, float *dst) {

  for (int i = 0; i < size; ++i) {
    if (i + shift >= size)
      dst[i] = src[i + shift - size];
    else
      dst[i] = src[i + shift];
  }
}

void student_rotate_no_mod_index_set_splitting(int size, int shift, float *src,
                                               float *dst) {
  int index_start = 0;
  int index_split = size - shift;
  int index_end = size;

  for (int i = index_start; i < index_split; ++i) {
    dst[i] = src[i + shift];
  }

  for (int i = index_split; i < index_end; ++i) {
    dst[i] = src[i + shift - size];
  }
}

void test_rotate_no_mod_index_set_splitting() {
  const int size = 8;
  const int shift = 1;
  float a[] = {0, 1, 2, 3, 4, 5, 6, 7};
  float bt[] = {-1, -1, -1, -1, -1, -1, -1, -1};
  float br[] = {-1, -1, -1, -1, -1, -1, -1, -1};

  reference_rotate_no_mod(size, shift, a, br);
  student_rotate_no_mod_index_set_splitting(size, shift, a, bt);

  float res = max_pair_wise_diff(8, bt, br);

  printf("test_rotate_no_mod_index_set_splitting: ");
  if (res > 1e-6) {
    printf("FAIL\n");

    print_8xfloat_mem(" a", a);
    print_8xfloat_mem("bt", bt);
    print_8xfloat_mem("br", br);

    printf("\n");
  } else {
    printf("PASS\n");
  }
}

// Loop Peeling
// reference alpha times X plus Y (AXPY) software pipelined
void reference_axpy(int size, float *src, float *dst) {
  float alpha = 2.0f;
  float y = 1.0f;
  for (int i = 0; i < size; ++i) {
    dst[i] = alpha * src[i] + y;
  }
}

void reference_axpy_sftwr_pipeln(int size, float *src, float *dst) {
  float alpha = 2.0f;
  float y = 1.0f;

  float reg_0;
  for (int i = 0; i < size - 1; ++i) { 
    if (i == 0)
      reg_0 = alpha * src[0];

    float reg_1 = reg_0;
    reg_0 = alpha * src[i + 1];
    dst[i] = reg_1 + y;

    if (i == size - 2)
      dst[size - 1] = reg_0 + y;
  }
}

void student_axpy_sftwr_pipeln_peel(int size, float *src, float *dst) {
  float alpha = 2.0f;
  float y = 1.0f;

  float reg_0 = alpha * src[0];
  for (int i = 0; i < size - 1; ++i) {
    float reg_1 = reg_0;
    reg_0 = alpha * src[i + 1];
    dst[i] = reg_1 + y;
  }
  dst[size - 1] = reg_0 + y;
}

void test_axpy_sftwr_pipeln_peel() {
  const int size = 8;
  float a[] = {0, 1, 2, 3, 4, 5, 6, 7};
  float bt[] = {-1, -1, -1, -1, -1, -1, -1, -1};
  float br[] = {-1, -1, -1, -1, -1, -1, -1, -1};

  reference_axpy_sftwr_pipeln(size, a, br);
  student_axpy_sftwr_pipeln_peel(size, a, bt);

  float res = max_pair_wise_diff(8, bt, br);

  printf("test_axpy_sftwr_pipelin_peel: ");
  if (res > 1e-6) {
    printf("FAIL\n");

    print_8xfloat_mem(" a", a);
    print_8xfloat_mem("bt", bt);
    print_8xfloat_mem("br", br);

    printf("\n");
  } else {
    printf("PASS\n");
  }
}

// Loop Fusion
void reference_apply_weight_then_activate(int size, float *src, float *dst) {
  float x[size];

  float alpha = 2.0f;
  float y = -5.0f;

  // apply weight
  for (int i = 0; i < size; ++i) {
    x[i] = alpha * src[i] + y;
  }

  // activate with relu!
  for (int i = 0; i < size; ++i) {
    if (x[i] < 0)
      dst[i] = 0.0f;
    else
      dst[i] = x[i];
  }
}

void student_apply_weight_then_activate_fusion(int size, float *src,
                                               float *dst) {
  float x[size];

  float alpha = 2.0f;
  float y = -5.0f;

  // the two loops iterate over the same index, so we fuse them
  for (int i = 0; i < size; ++i) {
    x[i] = alpha * src[i] + y;
    dst[i] = (x[i] < 0) ? 0.0f : x[i];
  }
}

void test_apply_weight_then_activate_fusion() {
  const int size = 8;
  float a[] = {0, 1, 2, 3, 4, 5, 6, 7};
  float bt[] = {-1, -1, -1, -1, -1, -1, -1, -1};
  float br[] = {-1, -1, -1, -1, -1, -1, -1, -1};

  reference_apply_weight_then_activate(size, a, br);
  student_apply_weight_then_activate_fusion(size, a, bt);

  float res = max_pair_wise_diff(8, bt, br);

  printf("test_apply_weight_then_activate_fusion: ");
  if (res > 1e-6) {
    printf("FAIL\n");

    print_8xfloat_mem(" a", a);
    print_8xfloat_mem("bt", bt);
    print_8xfloat_mem("br", br);

    printf("\n");
  } else {
    printf("PASS\n");
  }
}

// Loop Fission:
void reference_deinterleave(int size, float *src, float *dst) {
  float x[size];

  for (int i = 0; i < size / 2; ++i) {
    dst[0 * size / 2 + i] = src[2 * i + 0];
    dst[1 * size / 2 + i] = src[2 * i + 1];
  }
}

void student_deinterleave_fission(int size, float *src, float *dst) {
  float x[size];

  for (int i = 0; i < size / 2; ++i) {
    dst[i] = src[2 * i]; // interleave even indices
  }
  for (int i = 0; i < size / 2; ++i) {
    dst[size / 2 + i] = src[2 * i + 1]; // interleave odd indices
  }
}

void test_deinterleave_fission() {
  const int size = 8;
  float a[] = {0, 1, 2, 3, 4, 5, 6, 7};
  float bt[] = {-1, -1, -1, -1, -1, -1, -1, -1};
  float br[] = {-1, -1, -1, -1, -1, -1, -1, -1};

  reference_deinterleave(size, a, br);
  student_deinterleave_fission(size, a, bt);

  float res = max_pair_wise_diff(8, bt, br);

  printf("test_deinterleave_fission: ");
  if (res > 1e-6) {
    printf("FAIL\n");

    print_8xfloat_mem(" a", a);
    print_8xfloat_mem("bt", bt);
    print_8xfloat_mem("br", br);

    printf("\n");
  } else {
    printf("PASS\n");
  }
}

// Loop Interchange
void reference_matvec_4x4_colmaj_16xfloat(float *src, float *x, float *dst) {
  const int m = 4; // number of rows of the matrix and the vector
  const int n = 4; // number of columns
  const int rs_s = m;
  const int cs_s = 1; // vector only has 1 column

  for (int i = 0; i < m; ++i) {
    dst[i] = 0.0f;
    for (int j = 0; j < n; ++j)
      dst[i] += src[i * cs_s + j * rs_s] * x[j]; // 
  }
}

void student_matvec_4x4_colmaj_8xfloat_interchange(float *src, float *x,
                                                   float *dst) {
  const int m = 4;
  const int n = 4;
  const int rs_s = m;
  const int cs_s = 1;

  for (int i = 0; i < m; ++i)
    dst[i] = 0.0f;

  // outer loop over columns, inner loop over rows
  for (int j = 0; j < n; ++j) {
    const float xj = x[j];
    for (int i = 0; i < m; ++i) {
      dst[i] += src[i * cs_s + j * rs_s] * xj;
    }
  }
}

void test_matvec_4x4_colmaj_8xfloat() {
  float a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  float x[] = {1, 2, 3, 4};
  float bt[4] = {-1, -1, -1, -1};
  float br[4] = {-1, -1, -1, -1};

  reference_matvec_4x4_colmaj_16xfloat(a, x, br);
  student_matvec_4x4_colmaj_8xfloat_interchange(a, x, bt);

  float res = max_pair_wise_diff(4, bt, br);

  printf("test_matvec_4x4_colmaj_8xfloat: ");
  if (res > 1e-6) {
    printf("FAIL\n");

    print_float_mem(" a", 16, a);
    print_float_mem("bt", 4, bt);
    print_float_mem("br", 4, br);

    printf("\n");
  } else {
    printf("PASS\n");
  }
}

// Loop Stripmining
void reference_axpy_stripmine(int size, float *src, float *dst) {
  float alpha = 2.0f;
  float y = 1.0f;
  for (int i = 0; i < size; ++i) {
    dst[i] = alpha * src[i] + y;
  }
}

void student_axpy_stripmine(int size, float *src, float *dst) {
  float alpha = 2.0f;
  float y = 1.0f;

  int block = 3;
  int split = size - (size % block); // we'll use this to break the loop after the blocks have been processed

  // Note: small check to make sure first loop is executed.
  if (split <= block)
    return;

  for (int io = 0; io < split; io += block) {
    for (int ii = 0; ii < block; ++ii) {
      dst[ii + io] = alpha * src[ii + io] + y;
    }
  }
  for (int i = split; i < size; ++i)
    dst[i] = alpha * src[i] + y; // finish the remaining iterations
}

void test_axpy_stripmine() {
  const int size = 8;
  float a[] = {0, 1, 2, 3, 4, 5, 6, 7};
  float bt[] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
  float br[] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

  reference_axpy_stripmine(size, a, br);
  student_axpy_stripmine(size, a, bt);

  float res = max_pair_wise_diff(12, bt, br);

  printf("test_axpy_stripmine: ");
  if (res > 1e-6) {
    printf("FAIL\n");

    print_float_mem(" a", 8, a);
    print_float_mem("bt", 12, bt);
    print_float_mem("br", 12, br);

    printf("\n");
  } else {
    printf("PASS\n");
  }
}

// Loop Skewing - Sort of
void reference_blur_skew(int size, float *src, float *dst) {
  float w[3] = {2, 1, 1};
  // float w[3] = {1,0,0}; //hint
  // float w[3] = {0,1,0}; //hint
  // float w[3] = {0,0,1}; //hint

  for (int i = 0; i < size; ++i) {
    dst[i] = 0.0f;

    for (int p = 0; p < 3; ++p) {
      dst[i] += w[p] * src[(i + p) % size];
    }
  }
}

void student_blur_skew(int size, float *src, float *dst) {
  float w[3] = {2, 1, 1};
  // float w[3] = {1,0,0}; //hint
  // float w[3] = {0,1,0}; //hint
  // float w[3] = {0,0,1}; //hint

  for (int i = 0; i < size; ++i)
    dst[i] = 0.0f;

  for (int i = 0; i < size; ++i) {
    for (int p = 0; p < 3; ++p) {
      int index = (i - p + size) % size;  //  since we add size, it causes wrap around despite negative i-p
      dst[index] += w[p] * src[i];
    }
  }
}

void test_blur_skew() {
  const int size = 8;
  float a[] = {0, 1, 2, 3, 4, 5, 6, 7};
  float bt[] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
  float br[] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

  reference_blur_skew(size, a, br);
  student_blur_skew(size, a, bt);

  float res = max_pair_wise_diff(12, bt, br);

  printf("test_blur_skew: ");
  if (res > 1e-6) {
    printf("FAIL\n");

    print_float_mem(" a", 8, a);
    print_float_mem("bt", 12, bt);
    print_float_mem("br", 12, br);

    printf("\n");
  } else {
    printf("PASS\n");
  }
}

int main(int argc, char *argv[]) {

  printf("00: Unswitching: ");
  test_I_tensor_DFT2_unswitch();
  printf("01: Loop Peeling: ");
  test_axpy_sftwr_pipeln_peel();
  printf("02: Index Set Splitting: ");
  test_rotate_no_mod_index_set_splitting();
  printf("03: Loop Fusion: ");
  test_apply_weight_then_activate_fusion();
  printf("04: Loop Fission: ");
  test_deinterleave_fission();
  printf("05: Loop Interchange: ");
  test_matvec_4x4_colmaj_8xfloat();
  printf("06: Strip Mining: ");
  test_axpy_stripmine();
  printf("07: Skewing: ");
  test_blur_skew();

  return 0;
}
