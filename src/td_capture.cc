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

#define _FILE_OFFSET_BITS 64
#define _BSD_SOURCE 1

#include "td_capture.h"
#include <argp.h>

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <stdint.h>
#include <string.h>
#include <errno.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <fcntl.h>

#include <vector>
#include <utility>
#include <algorithm>

#include <gcrypt.h>
#include <boost/scoped_ptr.hpp>

using namespace std;

static bool verbose = false;
static in_addr localaddr;
static unsigned short dport = 0;
static int bits = 0, channels = 0;
static bool want_native = true;
static size_t num_samples = 0;
static bool write_stdout = false;
static bool sha1 = false;
static bool randomdata = false;
static const char *outfile = 0;
static bool force = false;
static bool dontneed = false;

enum {
	OPT_DPORT = 257,
	OPT_STDOUT,
	OPT_SHA1,
	OPT_RANDOMDATA,
	OPT_DONTNEED,
};

static struct argp_option options[] = {
	{0, 0, 0, 0, "General settings:"},
	{"samples", 'n', "samples", 0, "Number of samples to acquire (required)"},
	{"bits", 'b', "bits", 0, "Bits per sample (required)"},
	{"channels", 'c', "channels", 0, "Number of channels (required)"},
	{"raw", 'r', 0, 0, "Suppress conversion to native format"},
	{"verbose", 'v', 0, 0, "Verbose output (mainly for debugging)"},

	{0, 0, 0, 0, "Capture source selection:"},
//	{"source", 'S', "addr", 0, "Roach IP and (optional) port"},
	{"localaddr", 'A', "ip", 0, "Local IP on which to listen"},
	{"dport", OPT_DPORT, "port", 0, "Port on which to listen"},

	{0, 0, 0, 0, "Capture target settings:"},
	{"out", 'o', "file", 0, "Write to file"},
	{"force", 'f', 0, 0, "Allow overwriting files"},
	{"dontneed", OPT_DONTNEED, 0, 0, "File will not be used soon (conserves memory)"},
	{"stdout", OPT_STDOUT, 0, 0, "Write acquired data to stdout"},

	{0, 0, 0, 0, "Debugging:"},
	{"sha1", OPT_SHA1, 0, 0, "Calculate SHA-1 of acquired data"},
	{"randomdata", OPT_RANDOMDATA, 0, 0, "Generate random data"},

	{}
};

static unsigned long long arg_to_ull(struct argp_state *state, const char *arg)
{
	char *end;
	errno = 0;
	unsigned long long ret = strtoull(arg, &end, 0);
	if (errno || *end)
		argp_error(state, "Number out of range: %s", arg);
	return ret;
}

static error_t parse_opt (int key, char *arg, struct argp_state *state)
{
	unsigned long long tmp;

	switch (key)
	{
	case 'v':
		verbose = true;
		break;

	case 'n':
		num_samples = arg_to_ull(state, arg);
		break;

	case 'b':
		bits = (int)arg_to_ull(state, arg);
		break;

	case 'c':
		channels = (int)arg_to_ull(state, arg);
		break;

	case 'r':
		want_native = false;
		break;

	case OPT_DPORT:
		tmp = arg_to_ull(state, arg);
		if (tmp > 65535)
			argp_error(state, "Bad port number");
		dport = (unsigned short)tmp;
		break;

	case 'S':
		break;  // Not yet implemented.

	case 'A':
		if (!inet_pton(AF_INET, arg, &localaddr))
			argp_error(state, "Failed to parse local address\n");

	case OPT_STDOUT:
		write_stdout = true;
		break;

	case 'o':
		if (outfile)
			argp_error(state, "Can only write to one destination");
		outfile = arg;
		break;

	case 'f':
		force = true;
		break;

	case OPT_DONTNEED:
		dontneed = true;
		break;

	case OPT_SHA1:
		sha1 = true;
		break;

	case OPT_RANDOMDATA:
		randomdata = true;
		break;

	case ARGP_KEY_END:
		if (ntohs(dport) == 0)
			argp_error(state, "Need a dport for now");

		if (num_samples == 0)
			argp_error(state, "Need samples > 0");

		if (bits == 0)
			argp_error(state, "Need bits > 0");

		if (channels == 0)
			argp_error(state, "Need channels > 0");

		if (randomdata && !sha1)
			argp_error(state, "--randomdata without --sha1 is useless");

		if (write_stdout && outfile)
			argp_error(state, "Can only write to one destination");

		break;

	case ARGP_KEY_ARG:
		argp_usage(state);
		return ARGP_ERR_UNKNOWN;

	default:
		return ARGP_ERR_UNKNOWN;
	}

	return 0;
}

static struct argp argp = { options, parse_opt, "",
							"td_capture -- time-domain capture from the ROACH" };

int main(int argc, char *argv[])
{
	localaddr.s_addr = htonl(INADDR_ANY);

	if (argp_parse(&argp, argc, argv, 0, 0, 0) != 0)
		return 1;

	CaptureSpec spec;
	spec.want_native_format = want_native;
	spec.bits_per_sample = bits;
	spec.channels = channels;

	spec.dest_port = dport;

	size_t bytes = num_samples * channels * bits / 8;

	boost::scoped_ptr<MemoryTarget> target;

	try {
		target.reset(new MemoryTarget(bytes));
	} catch (std::runtime_error &e) {
		fprintf(stderr, "Memory allocation failed: %s\n", e.what());
		return 1;
	}

	if (!randomdata) {
		CaptureResult result = Capture(target.get(), spec, 2, verbose);
		if (!result.ok) {
			fprintf(stderr, "Capture failed: %s\n", result.error);
			return 1;
		}

		if (verbose) {
			double recv_secs = 1e-9 * result.recv_time_ns;
			double postproc_secs = 1e-9 * result.postproc_time_ns;
			
			fprintf(stderr, "Capture took after %.2f seconds (rate = %.2f MiB/sec) + %.2f postprocessing\n",
					recv_secs, target->len / (1048576 * recv_secs),
					postproc_secs);
			fprintf(stderr, "Output is %s endian, shifted by %d\n",
					(result.endian == CaptureResult::BIG ? "big" : "little"),
					result.shift);
		}
	} else {
		fprintf(stderr, "As requested, the output will be random garbage\n");
		gcry_create_nonce(target->buffer, target->len);
	}

	if (sha1) {
		unsigned char val[160 / 8];
		gcry_md_hash_buffer(GCRY_MD_SHA1, val, target->buffer, target->len);
		fprintf(stderr, "SHA-1 hash is ");
		for (size_t i = 0; i < sizeof(val); i++)
			fprintf(stderr, "%02x", val[i]);
		fprintf(stderr, "\n");
	}

	if (write_stdout) {
		target->StreamToFd(1, verbose);

		// This is unnecessary, but it makes strace give nicer output when
		// file IO causes lag.
		close(1);
	}

	if (outfile) {
		int fd = open(outfile, O_CREAT | O_RDWR | O_NOATIME | O_TRUNC
					  | (force ? 0 : O_EXCL), 0777);
		if (fd < 0) {
			perror("open");
			return 1;
		}

		target->StreamToFd(fd);

		if (dontneed) {
			if (posix_fadvise(fd, 0, bytes, POSIX_FADV_DONTNEED) != 0)
				perror("POSIX_FADV_DONTNEED");
		}

		close(fd);
	}

	return 0;
}


