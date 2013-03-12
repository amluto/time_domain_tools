// -*- mode: c++; tab-width: 4; c-file-style: "bsd"; c-basic-offset: 4 -*-
/*
 * Time domain tools for CASPER
 * Copyright (C) 2011 Massachusetts Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * version 2 as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
 * USA.
 */

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
