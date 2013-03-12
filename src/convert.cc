// -*- mode: c++; tab-width: 4; c-file-style: "linux"; c-basic-offset: 4 -*-

#include <stdint.h>
#include <sys/types.h>

#define HIDDEN __attribute__((visibility("hidden")))

#ifdef __SSE4_1__

#include <emmintrin.h>
#include <smmintrin.h>
#include "myendian.h"

// 0: unknown
// 1: present
// 2: not present
static int sse41_state = 0;

static bool has_sse41()
{
	// This function is threadsafe.

	if (sse41_state != 0)
		return sse41_state == 1;

	int eax = 1, ecx;
	asm volatile ("cpuid" : "+a" (eax), "=c" (ecx) : : "ebx", "edx");
	bool ok = ecx & (1 << 19);

	sse41_state = ok ? 1 : 2;
	return ok;
}

static inline __m128i swab16_m128(__m128i x)
{
	return _mm_or_si128(_mm_slli_epi16(x, 8), _mm_srli_epi16(x, 8));
}

#define unlikely(x) __builtin_expect((x), 0)

// one block is 32 words or 64 bytes
#define SWAB16_WORDS_PER_BLOCK 32
static bool swab16_shift_sse4(int16_t *buf, size_t blocks, int shift)
{
	int16_t wordmask = ((int16_t)1 << shift) - 1;
	__m128i mask = _mm_set1_epi16(htobe16(wordmask));

	__m128i *data = (__m128i *)buf;

	for (size_t i = 0; i < blocks; i++, data += 4) {
		__m128i d0 = _mm_stream_load_si128(&data[0]);
		__m128i d1 = _mm_stream_load_si128(&data[1]);
		__m128i d2 = _mm_stream_load_si128(&data[2]);
		__m128i d3 = _mm_stream_load_si128(&data[3]);

		bool bad;

		bad = !_mm_testz_si128(d0, mask);
		d0 = _mm_srai_epi16(swab16_m128(d0), shift);
		if (unlikely(bad))
			return false;

		bad = !_mm_testz_si128(d1, mask);
		d1 = _mm_srai_epi16(swab16_m128(d1), shift);
		if (unlikely(bad))
			return false;

		bad = !_mm_testz_si128(d2, mask);
		d2 = _mm_srai_epi16(swab16_m128(d2), shift);
		if (unlikely(bad))
			return false;

		bad = !_mm_testz_si128(d3, mask);
		d3 = _mm_srai_epi16(swab16_m128(d3), shift);
		if (unlikely(bad))
			return false;

		_mm_stream_si128(&data[0], d0);
		_mm_stream_si128(&data[1], d1);
		_mm_stream_si128(&data[2], d2);
		_mm_stream_si128(&data[3], d3);
	}

	return true;
}

#endif

HIDDEN bool swab16_shift(int16_t *buf, size_t bytes, int shift)
{
	size_t words = bytes / 2;

#ifdef __SSE4_1__
	if (has_sse41()) {
		size_t blocks = words / SWAB16_WORDS_PER_BLOCK;

		if (!swab16_shift_sse4(buf, words / SWAB16_WORDS_PER_BLOCK, shift))
			return false;

		buf += blocks * SWAB16_WORDS_PER_BLOCK;
		words -= blocks * SWAB16_WORDS_PER_BLOCK;
	}
#endif

	for (size_t i = 0; i < words; i++) {
		buf[i] = (int16_t)((uint16_t)buf[i] << 8 | (uint16_t)buf[i] >> 8);
		if (buf[i] & ((1<<shift) - 1))
			return false;
		buf[i] >>= shift;
	}

	return true;
}
