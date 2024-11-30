# MD5-SIMD-Test
This program demonstrates x86 vector instruction performance using SSE2 or later and AVX*. The user provides a runtime argument as a short string (<= 15 chars), and the program calculates approximately 10 billion md5 hash rounds, 16 at a time, to find input strings which produce md5 hashes beginning with at least 8 hex zeroes. It runs on a single thread which can be verified using system monitoring tools. It is meant to allow performance comparisons to be made between AVX2 and SSE2. The program is optimised for AVX2 which is why 16 md5 hashes are computed at once.

E.g. On a i7-6700 @3.4GHz

./md5simdtest3_AVX2.bin simonwozere!

Using AVX2.

simonwozere!000B00000000KFMLIIAA

00000000976f57802ab10c6cebdd101a ...etc

Calculated 9932111872 md5 hash blocks in 99 seconds.

md5 Hash Rate approx. = 100.324364 MH/s
