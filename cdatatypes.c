/*
  For programs that need to know the exact sizes of certain c data types, include this with a call at the start of main to
  
      assert_type_sizes(ASSERT_TYPE_SIZE_LL_64 | ASSERT_TYPE_SIZE_VOID_8);

  choosing of course, appropriate flags for the argument.
*/

#define ASSERT_TYPE_SIZE_LL_128 4
#define ASSERT_TYPE_SIZE_LL_80 2
#define ASSERT_TYPE_SIZE_LL_64 1
#define ASSERT_TYPE_SIZE_LD_128 32
#define ASSERT_TYPE_SIZE_LD_80 16
#define ASSERT_TYPE_SIZE_LD_64 8
#define ASSERT_TYPE_SIZE_VOID_8 64
#define ASSERT_TYPE_SIZE_VOID_32 128
#define ASSERT_TYPE_SIZE_LD_80bit_128 256

void assert_type_sizes(int variation) {
  unsigned int flags;
  long double testlongdouble = 1;
  char testlongdoublechars[16];
  flags = 0;
  if (sizeof(char) != 1) {
    fprintf(stderr, "Program requires 8 bit char data type. %li bits given.\n", 8*sizeof(char));
    flags |= 1; 
  }
  if (sizeof(short) != 2) {
    fprintf(stderr, "Program requires 16 bit short data type. %li bits given.\n", 8*sizeof(short));
    flags |= 2; 
  }
  if (sizeof(int) != 4) {
    fprintf(stderr, "Program requires 32 bit int data type. %li bits given.\n", 8*sizeof(int));
    flags |= 4; 
  }
  if (sizeof(long) != 8) {
    fprintf(stderr, "Program requires 64 bit long data type. %li bits given.\n", 8*sizeof(long));
    flags |= 8; 
  }
  if ((variation & ASSERT_TYPE_SIZE_LL_64) != 0) {
    if (sizeof(long long) != 8) {
      fprintf(stderr, "Program requires 64 bit long long data type. %li bits given.\n", 8*sizeof(long long));
      flags |= 16; 
    }
  } 
  if ((variation & ASSERT_TYPE_SIZE_LL_80) != 0) {
    if (sizeof(long long) != 10) {
      fprintf(stderr, "Program requires 80 bit long long data type. %li bits given.\n", 8*sizeof(long long));
      flags |= 16; 
    }
  }
  if ((variation & ASSERT_TYPE_SIZE_LL_128) != 0) {
    if (sizeof(long long) != 16) {
      fprintf(stderr, "Program requires 128 bit long long data type %li bits given.\n", 8*sizeof(long long));
      flags |= 16; 
    }
  }
  if ((variation & ASSERT_TYPE_SIZE_LD_64) != 0) {
    if (sizeof(long double) != 8) {
      fprintf(stderr, "Program requires 64 bit long double data type. %li bits given.\n", 8*sizeof(long double));
      flags |= 16; 
    }
  } 
  if ((variation & ASSERT_TYPE_SIZE_LD_80) != 0) {
    if (sizeof(long double) != 10) {
      fprintf(stderr, "Program requires 80 bit long double data type. %li bits given.\n", 8*sizeof(long double));
      flags |= 16; 
    }
  }
  if ((variation & (ASSERT_TYPE_SIZE_LD_128 | ASSERT_TYPE_SIZE_LD_80bit_128)) != 0) {
    if (sizeof(long double) != 16) {
      fprintf(stderr, "Program requires 128 bit long double data type. %li bits given.\n", 8*sizeof(long double));
      flags |= 16; 
    } else {
      if ((variation & ASSERT_TYPE_SIZE_LD_80bit_128) != 0) {
        memcpy(testlongdoublechars, &testlongdouble, 16);
        if (testlongdoublechars[9] != 0x3f) {
          fprintf(stderr, "Program requires 128 bit long double data type of 80 bit format.\n");
          flags |= 256; 
        }
      }
    }
  }
  if (sizeof(float) != 4) {
    fprintf(stderr, "Program requires 32 bit float data type. %li bits given.\n", 8*sizeof(float));
    flags |= 32; 
  }
  if (sizeof(double) != 8) {
    fprintf(stderr, "Program requires 64 bit double data type. %li bits given.\n", 8*sizeof(double));
    flags |= 64; 
  }
  if ((variation & ASSERT_TYPE_SIZE_VOID_8) != 0) {
    if (sizeof(void) != 1) {
      fprintf(stderr, "Program requires 8 bit void data type. %li bits given.\n", 8*sizeof(void));
      flags |= 128; 
    }
  }
  if ((variation & ASSERT_TYPE_SIZE_VOID_32) != 0) {
    if (sizeof(void) != 4) {
      fprintf(stderr, "Program requires 32 bit void data type. %li bits given.\n", 8*sizeof(void));
      flags |= 128; 
    }
  }
  if (flags != 0) exit(1);
};

