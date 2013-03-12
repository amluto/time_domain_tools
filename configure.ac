#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.59])
AC_INIT(td_capture, 0.1, luto@mit.edu)
AM_INIT_AUTOMAKE([subdir-objects])
AC_CONFIG_MACRO_DIR(m4)
AC_CONFIG_SRCDIR([src/td_lib.cc])
AM_CONFIG_HEADER([config.h])

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC

# Checks for libraries.
AC_CHECK_LIB([gcrypt], [gcry_create_nonce])
AC_CHECK_LIB([rt], [clock_gettime])

# Checks for header files.
AC_CHECK_HEADERS([arpa/inet.h fcntl.h inttypes.h stdint.h string.h sys/socket.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_INLINE

# Checks for library functions.
AC_FUNC_MMAP
AC_FUNC_STRERROR_R
AC_CHECK_FUNCS([clock_gettime ftruncate munmap socket strtoull recvmmsg])

AC_CHECK_DECL(be64toh,[AC_DEFINE([HAVE_BE64TOH],[1],[endian conversion available])],,[[
#include <endian.h>
]])

AC_CHECK_DECL(fallocate,[AC_DEFINE([HAVE_FALLOCATE],[1],[fallocate conversion available])],,[[
#define _GNU_SOURCE
#include <fcntl.h>
]])

AC_CHECK_TYPE(struct mmsghdr,[AC_DEFINE([HAVE_MMSGHDR],[1],[struct mmsghdr available])],,[[
#define _GNU_SOURCE
#include <sys/socket.h>
]])

# Checks for compiler support.
AX_CXXFLAGS_GCC_OPTION(-msse4)

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
