#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>

// gcc md5simdtest3.c -o md5simdtest3_SSE2.bin -msse2 -lm -fstrict-aliasing -Wstrict-aliasing=2 -O3 -std=c11 -Wall 
// gcc md5simdtest3.c -o md5simdtest3_AVX.bin -mavx -lm -fstrict-aliasing -Wstrict-aliasing=2 -O3 -std=c11 -Wall 
// gcc md5simdtest3.c -o md5simdtest3_AVX2.bin -mavx2 -lm -fstrict-aliasing -Wstrict-aliasing=2 -O3 -std=c11 -Wall 

#ifndef alignof
  #define alignof _Alignof
#endif

#ifndef __AVX2__
  #define X_SSE2_ONLY 1 
#endif

#ifdef X_SSE2_ONLY
  #define SIMD_SUPPORT SIMD_SUPPORT_FOR_SSE2
  #define X_mm256i X_mm256i_union_t
  #define X_mm256_set1_epi32(a) Y_mm256_set1_epi32(a) 
  #define X_mm256_setr_epi32(a0, a1, a2, a3, a4, a5, a6, a7) Y_mm256_setr_epi32(a0, a1, a2, a3, a4, a5, a6, a7) 
  #define X_mm256_add_epi32(a, b) Y_mm256_add_epi32(a, b)
  #define X_mm256_sub_epi32(a, b) Y_mm256_sub_epi32(a, b)
  #define X_mm256_srli_epi32(a, b) Y_mm256_srli_epi32(a, b)
  #define X_mm256_slli_epi32(a, b) Y_mm256_slli_epi32(a, b)
  #define X_mm256_and_si256(a, b) Y_mm256_and_si256(a, b)
  #define X_mm256_xor_si256(a, b) Y_mm256_xor_si256(a, b)
  #define X_mm256_or_si256(a, b) Y_mm256_or_si256(a, b)
  #define X_mm256_andnot_si256(a, b) Y_mm256_andnot_si256(a, b)
  #define X_mm256_loadu_si256(a) Y_mm256_loadu_si256(a)
  #define X_mm256_storeu_si256(a, b) Y_mm256_storeu_si256(a, b)
#else
  #define SIMD_SUPPORT SIMD_SUPPORT_FOR_AVX2
  #define X_mm256i __m256i
  #define X_mm256_set1_epi32(a)  _mm256_set1_epi32(a)
  #define X_mm256_setr_epi32(a0, a1, a2, a3, a4, a5, a6, a7)  _mm256_setr_epi32(a0, a1, a2, a3, a4, a5, a6, a7)
  #define X_mm256_add_epi32(a, b) _mm256_add_epi32(a, b)
  #define X_mm256_sub_epi32(a, b) _mm256_sub_epi32(a, b)
  #define X_mm256_srli_epi32(a, b) _mm256_srli_epi32(a, b)
  #define X_mm256_slli_epi32(a, b) _mm256_slli_epi32(a, b)
  #define X_mm256_and_si256(a, b) _mm256_and_si256(a, b)
  #define X_mm256_xor_si256(a, b) _mm256_xor_si256(a, b)
  #define X_mm256_or_si256(a, b) _mm256_or_si256(a, b)
  #define X_mm256_andnot_si256(a, b) _mm256_andnot_si256(a, b)
  #define X_mm256_loadu_si256(a) _mm256_loadu_si256(a)
  #define X_mm256_storeu_si256(a, b) _mm256_storeu_si256(a, b)
#endif

#ifdef __linux__
  #define IS_LINUX 'Y'
#else
  #define IS_LINUX 'N'
#endif

#define F(x, y, z) X_mm256_xor_si256(z, X_mm256_and_si256(X_mm256_xor_si256(z, y), x))
#define G(x, y, z) X_mm256_xor_si256(y, X_mm256_and_si256(X_mm256_xor_si256(x, y), z))
#define H(x, y, z) X_mm256_xor_si256(x, X_mm256_xor_si256(y, z))
#define I(x, y, z) X_mm256_xor_si256(y, X_mm256_or_si256(x, X_mm256_andnot_si256(z, X_mm256i_allones)))

#define STEP(f, az, bz, cz, dz, x, t, s, u) \
  az = X_mm256_add_epi32(X_mm256_add_epi32(X_mm256_add_epi32(f(bz, cz, dz), x), t), az); \
  az = X_mm256_add_epi32(X_mm256_or_si256(X_mm256_slli_epi32(az, s), X_mm256_srli_epi32(az, u)), bz);
#define GET(i) (msg[(i)])
  
#ifdef X_SSE2_ONLY
typedef union {
  __m128i v128[2];
  long longelement[4];
  int intelement[8];
  short shortelement[16];
  char charelement[32];
} X_mm256i_union_t;

X_mm256i Y_mm256_set1_epi32(int a0) {
  //SSE2
  X_mm256i ret;
  ret.v128[0] = _mm_set1_epi32(a0);
  ret.v128[1] = _mm_set1_epi32(a0);
  return ret;
}
X_mm256i Y_mm256_setr_epi32(int a0, int a1, int a2, int a3, int a4, int a5, int a6, int a7) {
  //SSE2
  X_mm256i ret;
  ret.v128[0] = _mm_setr_epi32(a0, a1, a2, a3);
  ret.v128[1] = _mm_setr_epi32(a4, a5, a6, a7);
  return ret;
}
X_mm256i Y_mm256_add_epi32(X_mm256i a0, X_mm256i a1) {
  //SSE2
  X_mm256i ret;
  ret.v128[0] = _mm_add_epi32(a0.v128[0], a1.v128[0]);
  ret.v128[1] = _mm_add_epi32(a0.v128[1], a1.v128[1]);
  return ret;
}
X_mm256i Y_mm256_sub_epi32(X_mm256i a0, X_mm256i a1) {
  //SSE2
  X_mm256i ret;
  ret.v128[0] = _mm_sub_epi32(a0.v128[0], a1.v128[0]);
  ret.v128[1] = _mm_sub_epi32(a0.v128[1], a1.v128[1]);
  return ret;
}
X_mm256i Y_mm256_srli_epi32(X_mm256i a0, int a1) {
  //SSE2
  X_mm256i ret;
  ret.v128[0] = _mm_srli_epi32(a0.v128[0], a1);
  ret.v128[1] = _mm_srli_epi32(a0.v128[1], a1);
  return ret;
}
X_mm256i Y_mm256_slli_epi32(X_mm256i a0, int a1) {
  //SSE2
  X_mm256i ret;
  ret.v128[0] = _mm_slli_epi32(a0.v128[0], a1);
  ret.v128[1] = _mm_slli_epi32(a0.v128[1], a1);
  return ret;
}
X_mm256i Y_mm256_and_si256(X_mm256i a0, X_mm256i a1) {
  //SSE2
  X_mm256i ret;
  ret.v128[0] = _mm_and_si128(a0.v128[0], a1.v128[0]);
  ret.v128[1] = _mm_and_si128(a0.v128[1], a1.v128[1]);
  return ret;
}
X_mm256i Y_mm256_xor_si256(X_mm256i a0, X_mm256i a1) {
  //SSE2
  X_mm256i ret;
  ret.v128[0] = _mm_xor_si128(a0.v128[0], a1.v128[0]);
  ret.v128[1] = _mm_xor_si128(a0.v128[1], a1.v128[1]);
  return ret;
}
X_mm256i Y_mm256_or_si256(X_mm256i a0, X_mm256i a1) {
  //SSE2
  X_mm256i ret;
  ret.v128[0] = _mm_or_si128(a0.v128[0], a1.v128[0]);
  ret.v128[1] = _mm_or_si128(a0.v128[1], a1.v128[1]);
  return ret;
}
X_mm256i Y_mm256_andnot_si256(X_mm256i a0, X_mm256i a1) {
  //SSE2
  X_mm256i ret; // ret = (~a) & b
  ret.v128[0] = _mm_andnot_si128(a0.v128[0], a1.v128[0]);
  ret.v128[1] = _mm_andnot_si128(a0.v128[1], a1.v128[1]);
  return ret;
}
X_mm256i Y_mm256_loadu_si256(X_mm256i* mem) {
  //SSE2
  X_mm256i ret;
  ret.v128[0] = _mm_loadu_si128(&mem->v128[0]);
  ret.v128[1] = _mm_loadu_si128(&mem->v128[1]);
  return ret;
}
void Y_mm256_storeu_si256(X_mm256i* mem, X_mm256i a0) {
  //SSE2
  _mm_storeu_si128(&mem->v128[0], a0.v128[0]);
  _mm_storeu_si128(&mem->v128[1], a0.v128[1]);
}
#else
typedef union {
  __m256i v256;
  __m128i v128[2];
  long longelement[4];
  int intelement[8];
  short shortelement[16];
  char charelement[32];
} X_mm256i_union_t;
#endif

X_mm256i X_mm256i_allones;

typedef struct {
  X_mm256i a,b,c,d;
} md5_avx2_t;

X_mm256i K0xd76aa478;
X_mm256i K0xe8c7b756;
X_mm256i K0x242070db;
X_mm256i K0xc1bdceee;
X_mm256i K0xf57c0faf;
X_mm256i K0x4787c62a;
X_mm256i K0xa8304613;
X_mm256i K0xfd469501;
X_mm256i K0x698098d8;
X_mm256i K0x8b44f7af;
X_mm256i K0xffff5bb1;
X_mm256i K0x895cd7be;
X_mm256i K0x6b901122;
X_mm256i K0xfd987193;
X_mm256i K0xa679438e;
X_mm256i K0x49b40821;
X_mm256i K0xf61e2562;
X_mm256i K0xc040b340;
X_mm256i K0x265e5a51;
X_mm256i K0xe9b6c7aa;
X_mm256i K0xd62f105d;
X_mm256i K0x02441453;
X_mm256i K0xd8a1e681;
X_mm256i K0xe7d3fbc8;
X_mm256i K0x21e1cde6;
X_mm256i K0xc33707d6;
X_mm256i K0xf4d50d87;
X_mm256i K0x455a14ed;
X_mm256i K0xa9e3e905;
X_mm256i K0xfcefa3f8;
X_mm256i K0x676f02d9;
X_mm256i K0x8d2a4c8a;
X_mm256i K0xfffa3942;
X_mm256i K0x8771f681;
X_mm256i K0x6d9d6122;
X_mm256i K0xfde5380c;
X_mm256i K0xa4beea44;
X_mm256i K0x4bdecfa9;
X_mm256i K0xf6bb4b60;
X_mm256i K0xbebfbc70;
X_mm256i K0x289b7ec6;
X_mm256i K0xeaa127fa;
X_mm256i K0xd4ef3085;
X_mm256i K0x04881d05;
X_mm256i K0xd9d4d039;
X_mm256i K0xe6db99e5;
X_mm256i K0x1fa27cf8;
X_mm256i K0xc4ac5665;
X_mm256i K0xf4292244;
X_mm256i K0x432aff97;
X_mm256i K0xab9423a7;
X_mm256i K0xfc93a039;
X_mm256i K0x655b59c3;
X_mm256i K0x8f0ccc92;
X_mm256i K0xffeff47d;
X_mm256i K0x85845dd1;
X_mm256i K0x6fa87e4f;
X_mm256i K0xfe2ce6e0;
X_mm256i K0xa3014314;
X_mm256i K0x4e0811a1;
X_mm256i K0xf7537e82;
X_mm256i K0xbd3af235;
X_mm256i K0x2ad7d2bb;
X_mm256i K0xeb86d391;
void md5initK() {
K0xd76aa478= X_mm256_set1_epi32(0xd76aa478);
K0xe8c7b756= X_mm256_set1_epi32(0xe8c7b756);
K0x242070db= X_mm256_set1_epi32(0x242070db);
K0xc1bdceee= X_mm256_set1_epi32(0xc1bdceee);
K0xf57c0faf= X_mm256_set1_epi32(0xf57c0faf);
K0x4787c62a= X_mm256_set1_epi32(0x4787c62a);
K0xa8304613= X_mm256_set1_epi32(0xa8304613);
K0xfd469501= X_mm256_set1_epi32(0xfd469501);
K0x698098d8= X_mm256_set1_epi32(0x698098d8);
K0x8b44f7af= X_mm256_set1_epi32(0x8b44f7af);
K0xffff5bb1= X_mm256_set1_epi32(0xffff5bb1);
K0x895cd7be= X_mm256_set1_epi32(0x895cd7be);
K0x6b901122= X_mm256_set1_epi32(0x6b901122);
K0xfd987193= X_mm256_set1_epi32(0xfd987193);
K0xa679438e= X_mm256_set1_epi32(0xa679438e);
K0x49b40821= X_mm256_set1_epi32(0x49b40821);
K0xf61e2562= X_mm256_set1_epi32(0xf61e2562);
K0xc040b340= X_mm256_set1_epi32(0xc040b340);
K0x265e5a51= X_mm256_set1_epi32(0x265e5a51);
K0xe9b6c7aa= X_mm256_set1_epi32(0xe9b6c7aa);
K0xd62f105d= X_mm256_set1_epi32(0xd62f105d);
K0x02441453= X_mm256_set1_epi32(0x02441453);
K0xd8a1e681= X_mm256_set1_epi32(0xd8a1e681);
K0xe7d3fbc8= X_mm256_set1_epi32(0xe7d3fbc8);
K0x21e1cde6= X_mm256_set1_epi32(0x21e1cde6);
K0xc33707d6= X_mm256_set1_epi32(0xc33707d6);
K0xf4d50d87= X_mm256_set1_epi32(0xf4d50d87);
K0x455a14ed= X_mm256_set1_epi32(0x455a14ed);
K0xa9e3e905= X_mm256_set1_epi32(0xa9e3e905);
K0xfcefa3f8= X_mm256_set1_epi32(0xfcefa3f8);
K0x676f02d9= X_mm256_set1_epi32(0x676f02d9);
K0x8d2a4c8a= X_mm256_set1_epi32(0x8d2a4c8a);
K0xfffa3942= X_mm256_set1_epi32(0xfffa3942);
K0x8771f681= X_mm256_set1_epi32(0x8771f681);
K0x6d9d6122= X_mm256_set1_epi32(0x6d9d6122);
K0xfde5380c= X_mm256_set1_epi32(0xfde5380c);
K0xa4beea44= X_mm256_set1_epi32(0xa4beea44);
K0x4bdecfa9= X_mm256_set1_epi32(0x4bdecfa9);
K0xf6bb4b60= X_mm256_set1_epi32(0xf6bb4b60);
K0xbebfbc70= X_mm256_set1_epi32(0xbebfbc70);
K0x289b7ec6= X_mm256_set1_epi32(0x289b7ec6);
K0xeaa127fa= X_mm256_set1_epi32(0xeaa127fa);
K0xd4ef3085= X_mm256_set1_epi32(0xd4ef3085);
K0x04881d05= X_mm256_set1_epi32(0x04881d05);
K0xd9d4d039= X_mm256_set1_epi32(0xd9d4d039);
K0xe6db99e5= X_mm256_set1_epi32(0xe6db99e5);
K0x1fa27cf8= X_mm256_set1_epi32(0x1fa27cf8);
K0xc4ac5665= X_mm256_set1_epi32(0xc4ac5665);
K0xf4292244= X_mm256_set1_epi32(0xf4292244);
K0x432aff97= X_mm256_set1_epi32(0x432aff97);
K0xab9423a7= X_mm256_set1_epi32(0xab9423a7);
K0xfc93a039= X_mm256_set1_epi32(0xfc93a039);
K0x655b59c3= X_mm256_set1_epi32(0x655b59c3);
K0x8f0ccc92= X_mm256_set1_epi32(0x8f0ccc92);
K0xffeff47d= X_mm256_set1_epi32(0xffeff47d);
K0x85845dd1= X_mm256_set1_epi32(0x85845dd1);
K0x6fa87e4f= X_mm256_set1_epi32(0x6fa87e4f);
K0xfe2ce6e0= X_mm256_set1_epi32(0xfe2ce6e0);
K0xa3014314= X_mm256_set1_epi32(0xa3014314);
K0x4e0811a1= X_mm256_set1_epi32(0x4e0811a1);
K0xf7537e82= X_mm256_set1_epi32(0xf7537e82);
K0xbd3af235= X_mm256_set1_epi32(0xbd3af235);
K0x2ad7d2bb= X_mm256_set1_epi32(0x2ad7d2bb);
K0xeb86d391= X_mm256_set1_epi32(0xeb86d391);
}
void md5init(md5_avx2_t*input) {
    input->a = X_mm256_set1_epi32(0x67452301);
    input->b = X_mm256_set1_epi32(0xefcdab89);
    input->c = X_mm256_set1_epi32(0x98badcfe);
    input->d = X_mm256_set1_epi32(0x10325476);
}
void md5round(md5_avx2_t*input, X_mm256i *msg) {
  X_mm256i a = input->a;
  X_mm256i b = input->b;
  X_mm256i c = input->c;
  X_mm256i d = input->d;
  STEP(F, a, b, c, d, GET(0), K0xd76aa478, 7, 25)
  STEP(F, d, a, b, c, GET(1), K0xe8c7b756, 12, 20)
  STEP(F, c, d, a, b, GET(2), K0x242070db, 17, 15)
  STEP(F, b, c, d, a, GET(3), K0xc1bdceee, 22, 10)
  STEP(F, a, b, c, d, GET(4), K0xf57c0faf, 7, 25)
  STEP(F, d, a, b, c, GET(5), K0x4787c62a, 12, 20)
  STEP(F, c, d, a, b, GET(6), K0xa8304613, 17, 15)
  STEP(F, b, c, d, a, GET(7), K0xfd469501, 22, 10)
  STEP(F, a, b, c, d, GET(8), K0x698098d8, 7, 25)
  STEP(F, d, a, b, c, GET(9), K0x8b44f7af, 12, 20)
  STEP(F, c, d, a, b, GET(10), K0xffff5bb1, 17, 15)
  STEP(F, b, c, d, a, GET(11), K0x895cd7be, 22, 10)
  STEP(F, a, b, c, d, GET(12), K0x6b901122, 7, 25)
  STEP(F, d, a, b, c, GET(13), K0xfd987193, 12, 20)
  STEP(F, c, d, a, b, GET(14), K0xa679438e, 17, 15)
  STEP(F, b, c, d, a, GET(15), K0x49b40821, 22, 10)

  STEP(G, a, b, c, d, GET(1), K0xf61e2562, 5, 27)
  STEP(G, d, a, b, c, GET(6), K0xc040b340, 9, 23)
  STEP(G, c, d, a, b, GET(11), K0x265e5a51, 14, 18)
  STEP(G, b, c, d, a, GET(0), K0xe9b6c7aa, 20, 12)
  STEP(G, a, b, c, d, GET(5), K0xd62f105d, 5, 27)
  STEP(G, d, a, b, c, GET(10), K0x02441453, 9, 23)
  STEP(G, c, d, a, b, GET(15), K0xd8a1e681, 14, 18)
  STEP(G, b, c, d, a, GET(4), K0xe7d3fbc8, 20, 12)
  STEP(G, a, b, c, d, GET(9), K0x21e1cde6, 5, 27)
  STEP(G, d, a, b, c, GET(14), K0xc33707d6, 9, 23)
  STEP(G, c, d, a, b, GET(3), K0xf4d50d87, 14, 18)
  STEP(G, b, c, d, a, GET(8), K0x455a14ed, 20, 12)
  STEP(G, a, b, c, d, GET(13), K0xa9e3e905, 5, 27)
  STEP(G, d, a, b, c, GET(2), K0xfcefa3f8, 9, 23)
  STEP(G, c, d, a, b, GET(7), K0x676f02d9, 14, 18)
  STEP(G, b, c, d, a, GET(12), K0x8d2a4c8a, 20, 12)

  STEP(H, a, b, c, d, GET(5), K0xfffa3942, 4, 28)
  STEP(H, d, a, b, c, GET(8), K0x8771f681, 11, 21)
  STEP(H, c, d, a, b, GET(11), K0x6d9d6122, 16, 16)
  STEP(H, b, c, d, a, GET(14), K0xfde5380c, 23, 9)
  STEP(H, a, b, c, d, GET(1), K0xa4beea44, 4, 28)
  STEP(H, d, a, b, c, GET(4), K0x4bdecfa9, 11, 21)
  STEP(H, c, d, a, b, GET(7), K0xf6bb4b60, 16, 16)
  STEP(H, b, c, d, a, GET(10), K0xbebfbc70, 23, 9)
  STEP(H, a, b, c, d, GET(13), K0x289b7ec6, 4, 28)
  STEP(H, d, a, b, c, GET(0), K0xeaa127fa, 11, 21)
  STEP(H, c, d, a, b, GET(3), K0xd4ef3085, 16, 16)
  STEP(H, b, c, d, a, GET(6), K0x04881d05, 23, 9)
  STEP(H, a, b, c, d, GET(9), K0xd9d4d039, 4, 28)
  STEP(H, d, a, b, c, GET(12), K0xe6db99e5, 11, 21)
  STEP(H, c, d, a, b, GET(15), K0x1fa27cf8, 16, 16)
  STEP(H, b, c, d, a, GET(2), K0xc4ac5665, 23, 9)

  STEP(I, a, b, c, d, GET(0), K0xf4292244, 6, 26)
  STEP(I, d, a, b, c, GET(7), K0x432aff97, 10, 22)
  STEP(I, c, d, a, b, GET(14), K0xab9423a7, 15, 17)
  STEP(I, b, c, d, a, GET(5), K0xfc93a039, 21, 11)
  STEP(I, a, b, c, d, GET(12), K0x655b59c3, 6, 26)
  STEP(I, d, a, b, c, GET(3), K0x8f0ccc92, 10, 22)
  STEP(I, c, d, a, b, GET(10), K0xffeff47d, 15, 17)
  STEP(I, b, c, d, a, GET(1), K0x85845dd1, 21, 11)
  STEP(I, a, b, c, d, GET(8), K0x6fa87e4f, 6, 26)
  STEP(I, d, a, b, c, GET(15), K0xfe2ce6e0, 10, 22)
  STEP(I, c, d, a, b, GET(6), K0xa3014314, 15, 17)
  STEP(I, b, c, d, a, GET(13), K0x4e0811a1, 21, 11)
  STEP(I, a, b, c, d, GET(4), K0xf7537e82, 6, 26)
  STEP(I, d, a, b, c, GET(11), K0xbd3af235, 10, 22)
  STEP(I, c, d, a, b, GET(2), K0x2ad7d2bb, 15, 17)
  STEP(I, b, c, d, a, GET(9), K0xeb86d391, 21, 11)
  
  input->a = X_mm256_add_epi32(a, input->a);
  input->b = X_mm256_add_epi32(b, input->b);
  input->c = X_mm256_add_epi32(c, input->c);
  input->d = X_mm256_add_epi32(d, input->d);
}

void md5roundx2(md5_avx2_t*input1, md5_avx2_t*input2, X_mm256i *msg) {
  //md5round(input1, msg);
  //md5round(input2, msg + 16);
  //return;
  // Do two lots at once to mitigate instruction dependencies. 50% faster on i7-6700 compiled with AVX2.
  
  X_mm256i a1 = input1->a;
  X_mm256i b1 = input1->b;
  X_mm256i c1 = input1->c;
  X_mm256i d1 = input1->d;
  
  X_mm256i a2 = input2->a;
  X_mm256i b2 = input2->b;
  X_mm256i c2 = input2->c;
  X_mm256i d2 = input2->d;
  
  STEP(F, a1, b1, c1, d1, GET(0), K0xd76aa478, 7, 25)
  STEP(F, a2, b2, c2, d2, GET(0 + 16), K0xd76aa478, 7, 25)
  STEP(F, d1, a1, b1, c1, GET(1), K0xe8c7b756, 12, 20)
  STEP(F, d2, a2, b2, c2, GET(1 + 16), K0xe8c7b756, 12, 20)
  STEP(F, c1, d1, a1, b1, GET(2), K0x242070db, 17, 15)
  STEP(F, c2, d2, a2, b2, GET(2 + 16), K0x242070db, 17, 15)
  STEP(F, b1, c1, d1, a1, GET(3), K0xc1bdceee, 22, 10)
  STEP(F, b2, c2, d2, a2, GET(3 + 16), K0xc1bdceee, 22, 10)
  STEP(F, a1, b1, c1, d1, GET(4), K0xf57c0faf, 7, 25)
  STEP(F, a2, b2, c2, d2, GET(4 + 16), K0xf57c0faf, 7, 25)
  STEP(F, d1, a1, b1, c1, GET(5), K0x4787c62a, 12, 20)
  STEP(F, d2, a2, b2, c2, GET(5 + 16), K0x4787c62a, 12, 20)
  STEP(F, c1, d1, a1, b1, GET(6), K0xa8304613, 17, 15)
  STEP(F, c2, d2, a2, b2, GET(6 + 16), K0xa8304613, 17, 15)
  STEP(F, b1, c1, d1, a1, GET(7), K0xfd469501, 22, 10)
  STEP(F, b2, c2, d2, a2, GET(7 + 16), K0xfd469501, 22, 10)
  STEP(F, a1, b1, c1, d1, GET(8), K0x698098d8, 7, 25)
  STEP(F, a2, b2, c2, d2, GET(8 + 16), K0x698098d8, 7, 25)
  STEP(F, d1, a1, b1, c1, GET(9), K0x8b44f7af, 12, 20)
  STEP(F, d2, a2, b2, c2, GET(9 + 16), K0x8b44f7af, 12, 20)
  STEP(F, c1, d1, a1, b1, GET(10), K0xffff5bb1, 17, 15)
  STEP(F, c2, d2, a2, b2, GET(10 + 16), K0xffff5bb1, 17, 15)
  STEP(F, b1, c1, d1, a1, GET(11), K0x895cd7be, 22, 10)
  STEP(F, b2, c2, d2, a2, GET(11 + 16), K0x895cd7be, 22, 10)
  STEP(F, a1, b1, c1, d1, GET(12), K0x6b901122, 7, 25)
  STEP(F, a2, b2, c2, d2, GET(12 + 16), K0x6b901122, 7, 25)
  STEP(F, d1, a1, b1, c1, GET(13), K0xfd987193, 12, 20)
  STEP(F, d2, a2, b2, c2, GET(13 + 16), K0xfd987193, 12, 20)
  STEP(F, c1, d1, a1, b1, GET(14), K0xa679438e, 17, 15)
  STEP(F, c2, d2, a2, b2, GET(14 + 16), K0xa679438e, 17, 15)
  STEP(F, b1, c1, d1, a1, GET(15), K0x49b40821, 22, 10)
  STEP(F, b2, c2, d2, a2, GET(15 + 16), K0x49b40821, 22, 10)

  STEP(G, a1, b1, c1, d1, GET(1), K0xf61e2562, 5, 27)
  STEP(G, a2, b2, c2, d2, GET(1 + 16), K0xf61e2562, 5, 27)
  STEP(G, d1, a1, b1, c1, GET(6), K0xc040b340, 9, 23)
  STEP(G, d2, a2, b2, c2, GET(6 + 16), K0xc040b340, 9, 23)
  STEP(G, c1, d1, a1, b1, GET(11), K0x265e5a51, 14, 18)
  STEP(G, c2, d2, a2, b2, GET(11 + 16), K0x265e5a51, 14, 18)
  STEP(G, b1, c1, d1, a1, GET(0), K0xe9b6c7aa, 20, 12)
  STEP(G, b2, c2, d2, a2, GET(0 + 16), K0xe9b6c7aa, 20, 12)
  STEP(G, a1, b1, c1, d1, GET(5), K0xd62f105d, 5, 27)
  STEP(G, a2, b2, c2, d2, GET(5 + 16), K0xd62f105d, 5, 27)
  STEP(G, d1, a1, b1, c1, GET(10), K0x02441453, 9, 23)
  STEP(G, d2, a2, b2, c2, GET(10 + 16), K0x02441453, 9, 23)
  STEP(G, c1, d1, a1, b1, GET(15), K0xd8a1e681, 14, 18)
  STEP(G, c2, d2, a2, b2, GET(15 + 16), K0xd8a1e681, 14, 18)
  STEP(G, b1, c1, d1, a1, GET(4), K0xe7d3fbc8, 20, 12)
  STEP(G, b2, c2, d2, a2, GET(4 + 16), K0xe7d3fbc8, 20, 12)
  STEP(G, a1, b1, c1, d1, GET(9), K0x21e1cde6, 5, 27)
  STEP(G, a2, b2, c2, d2, GET(9 + 16), K0x21e1cde6, 5, 27)
  STEP(G, d1, a1, b1, c1, GET(14), K0xc33707d6, 9, 23)
  STEP(G, d2, a2, b2, c2, GET(14 + 16), K0xc33707d6, 9, 23)
  STEP(G, c1, d1, a1, b1, GET(3), K0xf4d50d87, 14, 18)
  STEP(G, c2, d2, a2, b2, GET(3 + 16), K0xf4d50d87, 14, 18)
  STEP(G, b1, c1, d1, a1, GET(8), K0x455a14ed, 20, 12)
  STEP(G, b2, c2, d2, a2, GET(8 + 16), K0x455a14ed, 20, 12)
  STEP(G, a1, b1, c1, d1, GET(13), K0xa9e3e905, 5, 27)
  STEP(G, a2, b2, c2, d2, GET(13 + 16), K0xa9e3e905, 5, 27)
  STEP(G, d1, a1, b1, c1, GET(2), K0xfcefa3f8, 9, 23)
  STEP(G, d2, a2, b2, c2, GET(2 + 16), K0xfcefa3f8, 9, 23)
  STEP(G, c1, d1, a1, b1, GET(7), K0x676f02d9, 14, 18)
  STEP(G, c2, d2, a2, b2, GET(7 + 16), K0x676f02d9, 14, 18)
  STEP(G, b1, c1, d1, a1, GET(12), K0x8d2a4c8a, 20, 12)
  STEP(G, b2, c2, d2, a2, GET(12 + 16), K0x8d2a4c8a, 20, 12)

  STEP(H, a1, b1, c1, d1, GET(5), K0xfffa3942, 4, 28)
  STEP(H, a2, b2, c2, d2, GET(5 + 16), K0xfffa3942, 4, 28)
  STEP(H, d1, a1, b1, c1, GET(8), K0x8771f681, 11, 21)
  STEP(H, d2, a2, b2, c2, GET(8 + 16), K0x8771f681, 11, 21)
  STEP(H, c1, d1, a1, b1, GET(11), K0x6d9d6122, 16, 16)
  STEP(H, c2, d2, a2, b2, GET(11 + 16), K0x6d9d6122, 16, 16)
  STEP(H, b1, c1, d1, a1, GET(14), K0xfde5380c, 23, 9)
  STEP(H, b2, c2, d2, a2, GET(14 + 16), K0xfde5380c, 23, 9)
  STEP(H, a1, b1, c1, d1, GET(1), K0xa4beea44, 4, 28)
  STEP(H, a2, b2, c2, d2, GET(1 + 16), K0xa4beea44, 4, 28)
  STEP(H, d1, a1, b1, c1, GET(4), K0x4bdecfa9, 11, 21)
  STEP(H, d2, a2, b2, c2, GET(4 + 16), K0x4bdecfa9, 11, 21)
  STEP(H, c1, d1, a1, b1, GET(7), K0xf6bb4b60, 16, 16)
  STEP(H, c2, d2, a2, b2, GET(7 + 16), K0xf6bb4b60, 16, 16)
  STEP(H, b1, c1, d1, a1, GET(10), K0xbebfbc70, 23, 9)
  STEP(H, b2, c2, d2, a2, GET(10 + 16), K0xbebfbc70, 23, 9)
  STEP(H, a1, b1, c1, d1, GET(13), K0x289b7ec6, 4, 28)
  STEP(H, a2, b2, c2, d2, GET(13 + 16), K0x289b7ec6, 4, 28)
  STEP(H, d1, a1, b1, c1, GET(0), K0xeaa127fa, 11, 21)
  STEP(H, d2, a2, b2, c2, GET(0 + 16), K0xeaa127fa, 11, 21)
  STEP(H, c1, d1, a1, b1, GET(3), K0xd4ef3085, 16, 16)
  STEP(H, c2, d2, a2, b2, GET(3 + 16), K0xd4ef3085, 16, 16)
  STEP(H, b1, c1, d1, a1, GET(6), K0x04881d05, 23, 9)
  STEP(H, b2, c2, d2, a2, GET(6 + 16), K0x04881d05, 23, 9)
  STEP(H, a1, b1, c1, d1, GET(9), K0xd9d4d039, 4, 28)
  STEP(H, a2, b2, c2, d2, GET(9 + 16), K0xd9d4d039, 4, 28)
  STEP(H, d1, a1, b1, c1, GET(12), K0xe6db99e5, 11, 21)
  STEP(H, d2, a2, b2, c2, GET(12 + 16), K0xe6db99e5, 11, 21)
  STEP(H, c1, d1, a1, b1, GET(15), K0x1fa27cf8, 16, 16)
  STEP(H, c2, d2, a2, b2, GET(15 + 16), K0x1fa27cf8, 16, 16)
  STEP(H, b1, c1, d1, a1, GET(2), K0xc4ac5665, 23, 9)
  STEP(H, b2, c2, d2, a2, GET(2 + 16), K0xc4ac5665, 23, 9)

  STEP(I, a1, b1, c1, d1, GET(0), K0xf4292244, 6, 26)
  STEP(I, a2, b2, c2, d2, GET(0 + 16), K0xf4292244, 6, 26)
  STEP(I, d1, a1, b1, c1, GET(7), K0x432aff97, 10, 22)
  STEP(I, d2, a2, b2, c2, GET(7 + 16), K0x432aff97, 10, 22)
  STEP(I, c1, d1, a1, b1, GET(14), K0xab9423a7, 15, 17)
  STEP(I, c2, d2, a2, b2, GET(14 + 16), K0xab9423a7, 15, 17)
  STEP(I, b1, c1, d1, a1, GET(5), K0xfc93a039, 21, 11)
  STEP(I, b2, c2, d2, a2, GET(5 + 16), K0xfc93a039, 21, 11)
  STEP(I, a1, b1, c1, d1, GET(12), K0x655b59c3, 6, 26)
  STEP(I, a2, b2, c2, d2, GET(12 + 16), K0x655b59c3, 6, 26)
  STEP(I, d1, a1, b1, c1, GET(3), K0x8f0ccc92, 10, 22)
  STEP(I, d2, a2, b2, c2, GET(3 + 16), K0x8f0ccc92, 10, 22)
  STEP(I, c1, d1, a1, b1, GET(10), K0xffeff47d, 15, 17)
  STEP(I, c2, d2, a2, b2, GET(10 + 16), K0xffeff47d, 15, 17)
  STEP(I, b1, c1, d1, a1, GET(1), K0x85845dd1, 21, 11)
  STEP(I, b2, c2, d2, a2, GET(1 + 16), K0x85845dd1, 21, 11)
  STEP(I, a1, b1, c1, d1, GET(8), K0x6fa87e4f, 6, 26)
  STEP(I, a2, b2, c2, d2, GET(8 + 16), K0x6fa87e4f, 6, 26)
  STEP(I, d1, a1, b1, c1, GET(15), K0xfe2ce6e0, 10, 22)
  STEP(I, d2, a2, b2, c2, GET(15 + 16), K0xfe2ce6e0, 10, 22)
  STEP(I, c1, d1, a1, b1, GET(6), K0xa3014314, 15, 17)
  STEP(I, c2, d2, a2, b2, GET(6 + 16), K0xa3014314, 15, 17)
  STEP(I, b1, c1, d1, a1, GET(13), K0x4e0811a1, 21, 11)
  STEP(I, b2, c2, d2, a2, GET(13 + 16), K0x4e0811a1, 21, 11)
  STEP(I, a1, b1, c1, d1, GET(4), K0xf7537e82, 6, 26)
  STEP(I, a2, b2, c2, d2, GET(4 + 16), K0xf7537e82, 6, 26)
  STEP(I, d1, a1, b1, c1, GET(11), K0xbd3af235, 10, 22)
  STEP(I, d2, a2, b2, c2, GET(11 + 16), K0xbd3af235, 10, 22)
  STEP(I, c1, d1, a1, b1, GET(2), K0x2ad7d2bb, 15, 17)
  STEP(I, c2, d2, a2, b2, GET(2 + 16), K0x2ad7d2bb, 15, 17)
  STEP(I, b1, c1, d1, a1, GET(9), K0xeb86d391, 21, 11)
  STEP(I, b2, c2, d2, a2, GET(9 + 16), K0xeb86d391, 21, 11)
  
  input1->a = X_mm256_add_epi32(a1, input1->a);
  input1->b = X_mm256_add_epi32(b1, input1->b);
  input1->c = X_mm256_add_epi32(c1, input1->c);
  input1->d = X_mm256_add_epi32(d1, input1->d);
  
  input2->a = X_mm256_add_epi32(a2, input2->a);
  input2->b = X_mm256_add_epi32(b2, input2->b);
  input2->c = X_mm256_add_epi32(c2, input2->c);
  input2->d = X_mm256_add_epi32(d2, input2->d);
  
}

unsigned int Endian32(unsigned int x)
{
	return ((x >> 24) + ((x & 0xff0000) >> 8) + ((x & 0xff00) << 8) + ((x & 0xff) << 24));
}
char* md5str(unsigned int a, unsigned int b, unsigned int c, unsigned int d) {
  char *md5out = (char*)calloc(1,33);
  sprintf(md5out, "%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x", 
(a & 0xff000000) >> 24, (a & 0x00ff0000) >> 16, (a & 0x0000ff00) >> 8, (a & 0x000000ff),
(b & 0xff000000) >> 24, (b & 0x00ff0000) >> 16, (b & 0x0000ff00) >> 8, (b & 0x000000ff),
(c & 0xff000000) >> 24, (c & 0x00ff0000) >> 16, (c & 0x0000ff00) >> 8, (c & 0x000000ff),
(d & 0xff000000) >> 24, (d & 0x00ff0000) >> 16, (d & 0x0000ff00) >> 8, (d & 0x000000ff)
);
  return md5out;
}

int main(int argc, char*argv[])
{
    char userstring[16] = {0};      
    long i;
    if (argc <= 1) {
      printf("Usage: %s somestring\nThis program demonstrates the capability of SIMD. Enter a string of up to 15 characters and it will find strings to append to it which md5 hash with 8 hex leading zeroes, using only a single thread.\n", argv[0]);
      exit(0);
    }
    #ifdef __AVX__
      #ifdef __AVX2__
        printf("Using AVX2.\n");
      #else
        printf("Using AVX only.\n");
      #endif
    #else
      printf("Using SSE* only.\n");
    #endif
    long argv1len = strlen(argv[1]);
    if (argv1len > 15) argv1len = 15;
    memcpy(userstring, argv[1], argv1len);
    for (i=argv1len; i<15; i++) {
      userstring[i] = '0';
    }
    int msg0, msg1, msg2, msg3;
    msg0 = userstring[0] + (userstring[1] << 8) + (userstring[2] << 16) + (userstring[3] << 24);
    msg1 = userstring[4] + (userstring[5] << 8) + (userstring[6] << 16) + (userstring[7] << 24);
    msg2 = userstring[8] + (userstring[9] << 8) + (userstring[10] << 16) + (userstring[11] << 24);
    msg3 = userstring[12] + (userstring[13] << 8) + (userstring[14] << 16);
    X_mm256i_allones = X_mm256_set1_epi32(-1);
    md5initK();
    md5_avx2_t myvars[2];
    X_mm256i *msg = aligned_alloc(alignof(X_mm256i), 32*sizeof(X_mm256i));
    msg[0] = X_mm256_set1_epi32(msg0);
    msg[1] = X_mm256_set1_epi32(msg1);
    msg[2] = X_mm256_set1_epi32(msg2);
    msg[3] = X_mm256_setr_epi32(0x30000000 + msg3, 0x31000000 + msg3, 0x32000000 + msg3, 0x33000000 + msg3, 0x34000000 + msg3, 0x35000000 + msg3, 0x36000000 + msg3, 0x37000000 + msg3);
    msg[4] = X_mm256_set1_epi32(0x30303030);
    msg[5] = X_mm256_set1_epi32(0x30303030);
    msg[6] = X_mm256_set1_epi32(0x30303030);
    msg[7] = X_mm256_set1_epi32(0x30303030);
    msg[8] = X_mm256_set1_epi32(0x80);
    msg[9] = X_mm256_set1_epi32(0);
    msg[10] = X_mm256_set1_epi32(0);
    msg[11] = X_mm256_set1_epi32(0);
    msg[12] = X_mm256_set1_epi32(0);
    msg[13] = X_mm256_set1_epi32(0);
    msg[14] = X_mm256_set1_epi32(0x00000100);
    msg[15] = X_mm256_set1_epi32(0);
    
    msg[16] = X_mm256_set1_epi32(msg0);
    msg[17] = X_mm256_set1_epi32(msg1);
    msg[18] = X_mm256_set1_epi32(msg2);
    msg[19] = X_mm256_setr_epi32(0x38000000 + msg3, 0x39000000 + msg3, 0x41000000 + msg3, 0x42000000 + msg3, 0x43000000 + msg3, 0x44000000 + msg3, 0x45000000 + msg3, 0x46000000 + msg3);
    msg[20] = X_mm256_set1_epi32(0x30303030);
    msg[21] = X_mm256_set1_epi32(0x30303030);
    msg[22] = X_mm256_set1_epi32(0x30303030);
    msg[23] = X_mm256_set1_epi32(0x30303030);
    msg[24] = X_mm256_set1_epi32(0x80);
    msg[25] = X_mm256_set1_epi32(0);
    msg[26] = X_mm256_set1_epi32(0);
    msg[27] = X_mm256_set1_epi32(0);
    msg[28] = X_mm256_set1_epi32(0);
    msg[29] = X_mm256_set1_epi32(0);
    msg[30] = X_mm256_set1_epi32(0x00000100);
    msg[31] = X_mm256_set1_epi32(0);

    
    X_mm256i_union_t res_union; 
    int a,b,c,d;   
    long j, starttime, endtime, duration, iterations;
    unsigned int temp;
    unsigned int msg6, msg7;
    char* md5str_ptr;
    starttime = time(0);
    iterations = 0x25000000;
    for (i=0; i<iterations; i++) {
      temp = i;
      msg6 = 0x41414141 + (temp & 0xf) + (((temp & 0xf0) << 4)) + (((temp & 0xf00) << 8)) + (((temp & 0xf000) << 12));
      msg7 = 0x41414141 + ((temp & 0xf0000) >> 16) + (((temp & 0xf00000) >> 12)) + (((temp & 0xf000000) >> 8)) + (((temp & 0xf0000000) >> 4));
      msg[6] = X_mm256_set1_epi32(msg6);
      msg[7] = X_mm256_set1_epi32(msg7);
      msg[22] = X_mm256_set1_epi32(msg6);
      msg[23] = X_mm256_set1_epi32(msg7);
      md5init(&myvars[0]);
      md5init(&myvars[1]);
      char hashstream[] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'};
      md5roundx2(&myvars[0], &myvars[1], msg);   
      for (int k=0; k < 2; k++) { 
        memcpy(&res_union, &myvars[k].a, sizeof(myvars[k].a));      
        for (j=0; j<8; j++) {
          if ((res_union.intelement[j] & 0xffffffff) == 0) {
            printf("%s%c00000000%c%c%c%c%c%c%c%c\n", userstring, hashstream[j+8*k], msg6 & 0xff, (msg6 >> 8) & 0xff, (msg6 >> 16) & 0xff, (msg6 >> 24), msg7 & 0xff, (msg7 >> 8) & 0xff, (msg7 >> 16) & 0xff, (msg7 >> 24));
            memcpy(&res_union, &myvars[k].a, sizeof(myvars[k].a));
            a = res_union.intelement[j];
            memcpy(&res_union, &myvars[k].b, sizeof(myvars[k].a));
            b = res_union.intelement[j];
            memcpy(&res_union, &myvars[k].c, sizeof(myvars[k].a));
            c = res_union.intelement[j];
            memcpy(&res_union, &myvars[k].d, sizeof(myvars[k].a));
            d = res_union.intelement[j];
            md5str_ptr = md5str(Endian32(a),Endian32(b),Endian32(c),Endian32(d));
            printf("%s\n", md5str_ptr);
            free(md5str_ptr);
          } 
        }  
      }      
    }
    endtime = time(0);
    duration = endtime - starttime;
    printf("Calculated %li md5 hash blocks in %li seconds.\n", 16*iterations, duration);
    if (duration > 0) printf("md5 Hash Rate approx. = %f MH/s\n", 16.0f*iterations/(1000000*duration));
    free(msg);
}
