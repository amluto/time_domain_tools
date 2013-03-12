// -*- mode: c++; tab-width: 4; c-file-style: "linux"; c-basic-offset: 4 -*-
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
#include <stdint.h>
#include <stdexcept>
#include <netinet/ip.h>
#include <boost/noncopyable.hpp>

struct CaptureSpec
{
	struct in_addr source_addr;		// If all zeros, Capture() will guess
	unsigned short source_port;		// host order.  0 to guess
	unsigned short dest_port;		// host order

	int channels;
	int bits_per_sample;  // Behavior is unspecified if not a multiple of 8.

	// If true, then you will get shift == 1, endian == HOST.
	bool want_native_format;  
};

struct CaptureTarget
{
	void *buffer;
	size_t len;

	virtual ~CaptureTarget() {}
};

struct CaptureResult
{
	bool ok;
	char error[128];  // NULL terminated.  Valid if !ok.

	int channels;
	int bits_per_sample;  // For now, always a multiple of 8.
	enum Endianness {
		BIG,
		LITTLE,
	} endian;
	int shift;  // Amount to right-shift each sample.

	uint64_t recv_time_ns, postproc_time_ns;
	size_t n_recv_calls;
};

// On failure, target will be scribbled on.
CaptureResult Capture(const CaptureTarget *target,
					  const CaptureSpec &spec,
					  int ntries = 2,
					  bool verbose = false);

class MemoryTarget : public CaptureTarget
{
public:
	MemoryTarget(size_t len);
	~MemoryTarget();

	void StreamToFd(int fd, bool verbose = false);  // Frees memory and streams to fd.

	void DropBeginning(size_t bytes);  // 0..bytes-1 become invalid

private:
	size_t dropped_;

	MemoryTarget(const MemoryTarget &);
	void operator = (const MemoryTarget &);
};

/*
 * This doesn't work well.  Maybe in some future Linux it will
 * and we can enable it.
 */

#if 0
class FileTarget : public CaptureTarget
{
public:
	// WARNING: This will truncate the file.
	// Takes ownership of fd.
	FileTarget(int fd, size_t len);
	~FileTarget();

	// Keeps the output (otherwise it gets truncated)
	// Do not access buffer after calling this.
	void Commit();

	int fd() const { return fd_; }

private:
	int fd_;

	FileTarget(const FileTarget &);
	void operator = (const FileTarget &);
};
#endif
