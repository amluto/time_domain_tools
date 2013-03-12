// -*- mode: c++; tab-width: 4; c-file-style: "bsd"; c-basic-offset: 4 -*-

#pragma once

#include <endian.h>
#include "config.h"

#ifndef HAVE_BE64TOH

#include <byteswap.h>

#if __BYTE_ORDER == __LITTLE_ENDIAN
# define be64toh bswap_64
# define htobe64 bswap_64
# define be16toh bswap_16
# define htobe16 bswap_16
# define le16toh(x) (x)
#else
# error If you want big-endian support, try a newer glibc.
#endif

#endif
