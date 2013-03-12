// -*- mode: c++; tab-width: 4; c-file-style: "bsd"; c-basic-offset: 4 -*-

#define _FILE_OFFSET_BITS 64
#define _BSD_SOURCE 1
#define __STDC_FORMAT_MACROS

#include "config.h"
#include "td_capture.h"
#include "myendian.h"

#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <errno.h>

#include <sys/types.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <fcntl.h>
#include <arpa/inet.h>

#include <utility>
#include <algorithm>

#define HIDDEN __attribute__((visibility("hidden")))

// Source format specifications
enum {
	TD_LEGACY			= 0,
	TD_8BIT_1CHAN		= 1,
	TD_8BIT_2CHAN		= 2,
	TD_12BIT_RPAD_4CHAN	= 3,
};

bool swab16_shift(int16_t *buf, size_t bytes, int shift);

struct HIDDEN Socket : boost::noncopyable
{
	int fd;

	Socket() : fd(-1) {}
	~Socket()
	{
		if (fd != -1)
			close(fd);
	}
};

#ifndef HAVE_MMSGHDR
struct mmsghdr
{
    struct msghdr msg_hdr;      /* Actual message header.  */
    unsigned int msg_len;       /* Number of received bytes for the entry.  */
};
#endif

static bool has_recvmmsg = true;

static int safe_recvmmsg(int fd, struct mmsghdr *vmessages,
						 unsigned int vlen, int flags)
{
	int nr_datagrams;

#ifdef HAVE_RECVMMSG
	if (has_recvmmsg) {
		nr_datagrams = recvmmsg(fd, vmessages, vlen, flags, 0);
		if (nr_datagrams != -1 || errno != ENOSYS)
			return nr_datagrams;
	}

	has_recvmmsg = false;
#endif
	
	// Fallback to recvmsg
	vmessages[0].msg_len = recvmsg(fd, &vmessages[0].msg_hdr, flags);
	if (vmessages[0].msg_len >= 0)
		return 1;
	else
		return -1;
}


CaptureResult Capture(const CaptureTarget *target,
					  const CaptureSpec &spec,
					  int ntries,
					  bool verbose)
{
	const char *errtxt;

	CaptureResult result;
	result.ok = false;
	result.error[0] = 0;

	{  // For error handling.  Intentionally not indented.

	if (spec.bits_per_sample % 8 != 0) {
		errtxt = "bits_per_sample must be a multiple of 8";
		goto fail;
	}

	size_t chunk_size = spec.channels * spec.bits_per_sample / 8;

	if (target->len % chunk_size != 0) {
		errtxt = "target length must be a multiple of sample size";
		goto fail;
	}

	struct timespec start_time;
	clock_gettime(CLOCK_MONOTONIC, &start_time);

	/*
	 * Step 1: set up the socket.
	 */
	Socket s;

	s.fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	if (s.fd == -1) {
		errtxt = "socket";
		goto fail_errno;
	}
	
	{
		int one = 1;
		setsockopt(s.fd, SOL_SOCKET, SO_REUSEADDR, &one, (socklen_t)sizeof(one));
	}

	sockaddr_in bind_addr;
	bind_addr.sin_family = AF_INET;
	bind_addr.sin_port = htons(spec.dest_port);
	bind_addr.sin_addr.s_addr = INADDR_ANY;
	if (bind(s.fd, (sockaddr*)&bind_addr,
			 sizeof(bind_addr)) != 0) {
		errtxt = "bind";
		goto fail_errno;
	}

	/*
	 * Step 2: Acquire a single packet to get the packet info.
	 */

	size_t payload_len;
	sockaddr_in source_addr;
	uint8_t source_info;

	{
		if (verbose)
			fprintf(stderr, "Waiting for first packet... ");
		socklen_t addrlen = sizeof(source_addr);
		ssize_t len = recvfrom(s.fd, &source_info, 1, MSG_TRUNC,
							   (sockaddr *)&source_addr, &addrlen);

		if (len < 0) {
			errtxt = "recvfrom";
			goto fail_errno;
		}

		if (verbose) {
			char addrname[128];
			fprintf(stderr, "%d bytes from %s:%d\n", (int)len,
					inet_ntop(AF_INET, &source_addr.sin_addr,
							  addrname, sizeof(addrname)),
					(int)ntohs(source_addr.sin_port));
		}

		if (len <= 8) {
			errtxt = "packet too short";
			goto fail_errno;
		}

		payload_len = len - 8;
	}

	// Validate the source info.
	if (source_info == TD_12BIT_RPAD_4CHAN) {
		if (spec.channels != 4) {
			errtxt = "this source has 4 channels";
			goto fail;
		}

		if (spec.bits_per_sample != 16) {
			errtxt = "this source has 16 bits per sample";
			goto fail;
		}

		result.channels = 4;
		result.bits_per_sample = 16;
		result.shift = 4;
		result.endian = CaptureResult::BIG;  // Will swap later if needed.
	} else {
		if (source_info == TD_LEGACY)
			errtxt = "your source model is too old";
		else
			errtxt = "this source is not supported";
		goto fail;
	}

	/*
	 * Step 3: Set up I/O vectors
	 */

	const int batch_size = 8;
	uint64_t headers[batch_size];
	struct iovec iovecs[batch_size][2];
	struct sockaddr addr[batch_size];
	struct mmsghdr mmsgs[batch_size];

	for (int j = 0; j < batch_size; j++)
	{
		iovecs[j][0].iov_base = &headers[j];
		iovecs[j][0].iov_len = 8;
		iovecs[j][1].iov_base = 0;  /* Will fill in later. */
		iovecs[j][1].iov_len = payload_len;
		mmsgs[j].msg_hdr.msg_iov = iovecs[j];
		mmsgs[j].msg_hdr.msg_iovlen = 2;
		mmsgs[j].msg_hdr.msg_name = &addr[j];
		mmsgs[j].msg_hdr.msg_namelen = sizeof(addr[j]);	
		mmsgs[j].msg_hdr.msg_control = 0;
		mmsgs[j].msg_hdr.msg_controllen = 0;
		mmsgs[j].msg_len = 0;
	}

	/*
	 * Step 4: Connect the socket.
	 */

	if (connect(s.fd, (sockaddr *)&source_addr, sizeof(source_addr)) != 0) {
		errtxt = "failed to connect socket";
		goto fail_errno;
	}

	// Ask for a large buffer.
	{
		int rcvbuf = 16 * 1048576;
		if (setsockopt(s.fd, SOL_SOCKET, SO_RCVBUF,
					   &rcvbuf, sizeof(rcvbuf)) != 0)
			if (verbose)
				perror("SO_RCVBUF");

		if (verbose) {
			socklen_t optlen = sizeof(rcvbuf);
			rcvbuf = 0;
			if (getsockopt(s.fd, SOL_SOCKET, SO_RCVBUF,
						   &rcvbuf, &optlen))
				perror("SO_RCVBUF");
			fprintf(stderr, "Using receive buffer size %d\n", rcvbuf);
		}
	}

	// Flush socket buffer
	if (target->len > payload_len) while(true)
	{
		char buf;
		ssize_t ret = recv(s.fd, &buf, 1, MSG_TRUNC | MSG_DONTWAIT);

		if (ret < 0) {
			if (errno == EAGAIN || errno == EWOULDBLOCK)
				break;

			errtxt = "recv";
			goto fail_errno;
		}
	}

	/*
	 * Step 5: Acquire the data
	 */

	result.n_recv_calls = 0;
	size_t bytes_acquired = 0;
	uint64_t next_offset = 0;
	while(bytes_acquired < target->len)
	{
		/* How many packets are left? */
		size_t packets_left = (target->len - bytes_acquired + payload_len - 1)
			/ payload_len;
		size_t this_batch_size = std::min<size_t>(batch_size, packets_left);

		/* Fix up the receive vectors */
		size_t bytes_left = target->len - bytes_acquired;
		for(size_t j = 0; j < this_batch_size; j++)
		{
			iovecs[j][1].iov_base =
				(char *)target->buffer + bytes_acquired + j * payload_len;

			if (bytes_left < payload_len)
				iovecs[j][1].iov_len = bytes_left;
			bytes_left -= payload_len;
		}
		
		int nr_datagrams = safe_recvmmsg(s.fd, mmsgs, this_batch_size, MSG_TRUNC);
		if (nr_datagrams < 0) {
			perror("recvmmsg");
			errtxt = "recvmmsg failed";
			goto fail_errno;
		}
		result.n_recv_calls++;

		/* Inspect the data we just received. */

		for (int j = 0; j < nr_datagrams; j++)
		{
			if (mmsgs[j].msg_len != 8 + payload_len) {
				errtxt = "got packet with bad length";
				goto fail;
			}

			uint64_t header = be64toh(headers[j]);
			if ((header >> 56) != source_info) {
				errtxt = "source info changed";
				goto fail;
			}

			if (bytes_acquired == 0)
				next_offset = header & ((1ULL<<56) - 1);

			uint64_t offset = header & ((1ULL<<56) - 1);
			if (offset != next_offset) {
				if (verbose) {
					fprintf(stderr, "Lost %"PRIi64" bytes after acquiring "
							"%"PRIu64" bytes (pos %"PRIu64")\n",
							(int64_t)(offset - next_offset) * 8,
							bytes_acquired, next_offset);
					fprintf(stderr, "Prev offset was 0x%"PRIX64"; got 0x%"PRIX64"\n",
							next_offset - payload_len / 8, offset);
				}

				// TODO: Implement ntries.
				errtxt = "too much packet loss";
				goto fail;
			}

			next_offset += payload_len / 8;
			next_offset &= ((1ULL<<56) - 1);
			bytes_acquired += payload_len;
		}
	}

	struct timespec end_recv_time;
	clock_gettime(CLOCK_MONOTONIC, &end_recv_time);

	/*
	 * Step 6: Fix and validate data format
	 */
	if (spec.bits_per_sample == 16) {
		// We're lazy.  If no conversion requested, don't validate either.
		if (spec.want_native_format) {
			if (result.endian == CaptureResult::BIG
				&& __BYTE_ORDER == __LITTLE_ENDIAN) {
				// This is an optimized special case.
				if (!swab16_shift((int16_t*)target->buffer, target->len,
								 result.shift)) {
					errtxt = "received data has bad padding";
					goto fail;
				}
			} else {
				uint16_t badmask = (1 << result.shift) - 1;

				int16_t *words = (int16_t*)target->buffer;
				for (size_t i = 0; i < target->len / 2; i++) {
					int16_t sample = words[i];

					if (result.endian == CaptureResult::BIG)
						sample = be16toh(sample);
					else
						sample = le16toh(sample);

					if (sample & badmask) {
						errtxt = "received data has bad padding";
						goto fail;
					}

					words[i] = sample >> result.shift;
				}
			}

			result.shift = 0;
			result.endian = (__BYTE_ORDER == __LITTLE_ENDIAN
							 ? CaptureResult::LITTLE
							 : CaptureResult::BIG);
		}
	}

	struct timespec end_postproc_time;
	clock_gettime(CLOCK_MONOTONIC, &end_postproc_time);

	result.recv_time_ns = (end_recv_time.tv_nsec - start_time.tv_nsec)
		+ 1000000000 * (end_recv_time.tv_sec - start_time.tv_sec);
	result.postproc_time_ns = (end_postproc_time.tv_nsec - end_recv_time.tv_nsec)
		+ 1000000000 * (end_postproc_time.tv_sec - end_recv_time.tv_sec);

	result.ok = true;
	return result;

	}  // end of error handling region.

fail_errno:
	{
		char errno_buf[256] = "unknown error";
		// Bloody GNU extensions
		char *errno_text = strerror_r(errno, errno_buf, sizeof(errno_buf));
		snprintf(result.error, sizeof(result.error), "%s: %s",
				 errtxt, errno_text);
		result.error[sizeof(result.error) - 1] = 0;
		return result;
	}

fail:
	strncpy(result.error, errtxt, sizeof(result.error));
	result.error[sizeof(result.error) - 1] = 0;
	return result;
}

static size_t source_blocksizes[] = { 1, 1, 2, 8 };

size_t source_block_size(uint8_t source_info)
{
	if (source_info >= sizeof(source_blocksizes) / sizeof(source_blocksizes[0]))
		return 0;

	return source_blocksizes[source_info];
}

MemoryTarget::MemoryTarget(size_t len) : dropped_(0)
{
	this->len = len;
	buffer = mmap(0, len, PROT_READ | PROT_WRITE,
				  MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
	if (buffer == MAP_FAILED)
		throw std::runtime_error("failed to allocate memory");

	// Make sure that every page can be written.
	for (char *p = (char *)buffer; p < (char *)buffer + len; p += 4096)
		*p = 0;
}

MemoryTarget::~MemoryTarget()
{
	if (dropped_ < len)
		munmap((char *)buffer + dropped_, len - dropped_);
}

void MemoryTarget::DropBeginning(size_t bytes)
{
	if (bytes > len)
		bytes = len;
	bytes -= (bytes % sysconf(_SC_PAGESIZE));
	if (bytes <= dropped_)
		return;

	munmap((char *)buffer + dropped_, bytes - dropped_);
	dropped_ = bytes;
}

static void FallbackStream(MemoryTarget &target, int fd, size_t start, bool verbose)
{
	const size_t chunk = 1048576;
	for (size_t pos = start; pos < target.len; ) {
		size_t to_write = std::min<size_t>(chunk, target.len - pos);
		ssize_t written = write(fd, (const char *)target.buffer + pos, to_write);
		if (written < 0) {
			if (verbose)
				perror("write");
			throw std::runtime_error("failed to write data");
		}
		
		pos += written;
		target.DropBeginning(pos - (pos % chunk));
	}
}

static bool writeall(int fd, const void *buf, size_t len)
{
	while(len > 0) {
		int ret = write(fd, buf, len);
		if (len == 0)
			errno = EAGAIN;
		if (len <= 0)
			return false;

		len -= ret;
		buf = ((const char *)buf) + ret;
	}

	return true;
}

void MemoryTarget::StreamToFd(int fd, bool verbose)
{
#ifdef HAVE_FALLOCATE
	// Try to preallocate space.
	off_t offset = lseek(fd, 0, SEEK_CUR);
	if (offset >= 0)
		(void)fallocate(fd, 0, offset, offset + len);
#endif

	// We'll optimistically try splicing.
	int pipefd[2];
	if (pipe(pipefd) != 0) {
		if (verbose)
			perror("pipe");
		FallbackStream(*this, fd, 0, verbose);
		return;
	}

	const size_t chunk = 1048576;
	for (size_t pos = 0; pos < len; ) {
		// Splice some data into a pipe.
		size_t to_write = std::min<size_t>(chunk, len - pos);

		// The kernel is silly.
		if (to_write > 65536)
			to_write -= to_write % 65536;

		iovec iov;
		iov.iov_base = (char *)buffer + pos;
		iov.iov_len = to_write;
		ssize_t written = vmsplice(pipefd[1], &iov, 1, SPLICE_F_GIFT);

		if (written < 0) {
			if (verbose) {
				// Avoid confusion
				if (errno != EINVAL) {
					perror("vmsplice");
					fprintf(stderr, "Falling back to standard write "
							"(spliced %"PRIu64" bytes)\n",
							(uint64_t)pos);
				}
			}

			close(pipefd[0]);
			close(pipefd[1]);
			FallbackStream(*this, fd, pos, verbose);
			return;
		}

		// Release the pages.
		pos += written;
		DropBeginning(pos - (pos % chunk));

		// Splice from the pipe into the output.
		size_t data_left = (size_t)written;
		while(data_left) {
			ssize_t spliced = splice(pipefd[0], 0, fd, 0, data_left,
									 SPLICE_F_MOVE | (pos < len ? SPLICE_F_MORE : 0));
			if (spliced < 0) {
				if (verbose) {
					if (errno != EINVAL)  // Avoid confusion
						perror("splice");
					fprintf(stderr, "Failed to splice; will fall back to write\n");
				}

				// Recover the data!
				close(pipefd[1]);  // So read will eventually return 0.
				while(true) {
					char buf[65536];
					ssize_t bytes = read(pipefd[0], buf, sizeof(buf));
					if (bytes == 0)
						break;

					if (bytes < 0) {
						if (verbose)
							perror("read from pipe");
						throw std::runtime_error("unable to recover from splice failure");
					}

					if (!writeall(fd, buf, (size_t)bytes)) {
						if (verbose)
							perror("write");
						throw std::runtime_error("splice recovery failed to write");
					}
				}

				close(pipefd[1]);
				FallbackStream(*this, fd, pos, verbose);
				return;
			}

			data_left -= spliced;
		}
	}

	close(pipefd[0]);
	close(pipefd[1]);
}

#if 0

FileTarget::FileTarget(int fd, size_t len) : fd_(fd)
{
	if (ftruncate(fd, len) != 0)
		throw std::runtime_error("failed to set file target size");

#ifdef HAVE_FALLOCATE
	fallocate(fd, 0, 0, len);  // Failure doesn't really matter
#endif

	this->len = len;
	buffer = mmap(0, len, PROT_READ | PROT_WRITE,
				  MAP_SHARED | MAP_POPULATE, fd, 0);
	if (buffer == MAP_FAILED)
		throw std::runtime_error("failed to map target file");

	// Make sure that every page can be written.
	for (char *p = (char *)buffer; p < (char *)buffer + len; p += 4096)
		*p = 0;
}

FileTarget::~FileTarget()
{
	if (buffer != 0) {
		// Data was not committed.  Trash it.
		// There's no way to munmap without an implied msync, so try it this way.
		int ret = ftruncate(fd_, 0);
		munmap(buffer, len);

		if (ret != 0)
			(void)ftruncate(fd_, 0);
	}

	close(fd_);
}

void FileTarget::Commit()
{
	munmap(buffer, len);  // Implies msync, so the kernel will start writing
	buffer = 0;
}

#endif
