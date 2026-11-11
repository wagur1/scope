/* x264 ROI encoder: raw planar I420 plus a text delta-QP map, not PNG/NPZ.
 * Build: gcc -std=gnu99 -O2 -o x264_roi_encode x264_roi_encode.c -lx264 -lm
 * GNU99 exposes MinGW's native file-descriptor stdio declarations.
 * Requires x264 development headers and libraries.
 * Usage: x264_roi_encode <input.yuv> <w> <h> <fps> <crf> <map.txt|->
 *                        <delta> <output.h264> [nframes]
 * Map: exactly nframes lines, each with ceil(w/16)*ceil(h/16) finite floats
 * in raster order. '-' disables the map. Delta is integer metadata only:
 * offsets already include it. Omitting nframes infers it from the exact raw
 * size; explicit nframes must match that size, not select a prefix.
 * Verified policy: CRF, AQ mode 1, strength 1, mbtree off. quant_offsets
 * flows through AQ; strength 0 silently disables the intervention.
 */
#ifdef _WIN32
#define __USE_MINGW_ANSI_STDIO 1 /* C99 format specifiers with MinGW's CRT. */
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0600 /* Expose 64-bit stdio on MinGW (Vista or later). */
#endif
#else
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64
#endif
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h> /* x264.h requires the integer types to be defined first. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <sys/types.h>
#include <unistd.h>
#endif
#include <x264.h>

static int parse_int(const char *text, int *value) {
    char *end;
    long v;
    errno = 0;
    v = strtol(text, &end, 10);
    if (!*text || isspace((unsigned char)*text) || *end || errno == ERANGE ||
        v < INT_MIN || v > INT_MAX)
        return 0;
    *value = (int)v;
    return 1;
}

static int parse_fps(const char *text, uint32_t *num, uint32_t *den) {
    char *end;
    double fps, scale, rounded;
    uint32_t a, b;
    errno = 0;
    fps = strtod(text, &end);
    if (!*text || isspace((unsigned char)*text) || *end || errno == ERANGE ||
        !isfinite(fps) || fps <= 0 || fps > UINT32_MAX)
        return 0;
    /* Microsecond precision for usual rates; fit x264's uint32 rational.
     * Small rates use the full denominator range rather than round to zero. */
    scale = fps < 1 ? UINT32_MAX : fmin(1000000.0, floor(UINT32_MAX / fps));
    rounded = floor(fps * scale + 0.5);
    if (rounded < 1 || rounded > UINT32_MAX)
        return 0; /* Not representable by x264's rational FPS fields. */
    *num = (uint32_t)rounded;
    *den = (uint32_t)scale;
    a = *num; b = *den;
    while (b) { uint32_t r = a % b; a = b; b = r; }
    *num /= a; *den /= a;
    return 1;
}

static int raw_size(FILE *file, uint64_t *size) {
#ifdef _WIN32
    int64_t length;
    if (_fseeki64(file, 0, SEEK_END) != 0) return 0;
    length = _ftelli64(file);
    if (length < 0 || _fseeki64(file, 0, SEEK_SET) != 0) return 0;
#else
    off_t length;
    if (fseeko(file, 0, SEEK_END) != 0) return 0;
    length = ftello(file);
    if (length < 0 || fseeko(file, 0, SEEK_SET) != 0) return 0;
#endif
    *size = (uint64_t)length;
    return 1;
}

/* Read one physical line without letting a short row borrow the next row.
 * A bounded token avoids allocating a whole map or arbitrarily long lines. */
static int map_row(FILE *file, float *offsets, size_t expected) {
    size_t count = 0;
    int c = fgetc(file);
    while (c != EOF && c != '\n') {
        char token[128], *end;
        size_t length = 0;
        float value;
        if (isspace((unsigned char)c)) { c = fgetc(file); continue; }
        if (count == expected) return 0;
        do {
            if (length == sizeof(token) - 1) return 0;
            token[length++] = (char)c;
            c = fgetc(file);
        } while (c != EOF && !isspace((unsigned char)c));
        token[length] = '\0';
        errno = 0;
        value = strtof(token, &end);
        if (end != token + length || errno == ERANGE || !isfinite(value))
            return 0;
        offsets[count++] = value;
    }
    return !ferror(file) && count == expected;
}

static int at_end(FILE *file) {
    return fgetc(file) == EOF && !ferror(file);
}

/* Compare opened files, not path strings. Windows CRT stat has no reliable
 * inode; use the volume/file ID instead. Return -1 if identity is unknown. */
static int same_file(FILE *a, FILE *b) {
#ifdef _WIN32
    BY_HANDLE_FILE_INFORMATION x, y;
    HANDLE ha = (HANDLE)_get_osfhandle(_fileno(a));
    HANDLE hb = (HANDLE)_get_osfhandle(_fileno(b));
    if (ha == INVALID_HANDLE_VALUE || hb == INVALID_HANDLE_VALUE ||
        !GetFileInformationByHandle(ha, &x) || !GetFileInformationByHandle(hb, &y))
        return -1;
    return x.dwVolumeSerialNumber == y.dwVolumeSerialNumber &&
           x.nFileIndexHigh == y.nFileIndexHigh && x.nFileIndexLow == y.nFileIndexLow;
#else
    struct stat x, y;
    if (fstat(fileno(a), &x) != 0 || fstat(fileno(b), &y) != 0) return -1;
    return x.st_dev == y.st_dev && x.st_ino == y.st_ino;
#endif
}

/* Output is opened in append mode so identity can be checked without damage.
 * Truncate that same open handle (no pathname reopen/alias-check race).
 * Non-regular POSIX outputs such as /dev/full still get checked writes. */
static int truncate_output(FILE *file) {
#ifdef _WIN32
    HANDLE handle = (HANDLE)_get_osfhandle(_fileno(file));
    LARGE_INTEGER zero;
    zero.QuadPart = 0;
    return handle != INVALID_HANDLE_VALUE &&
           SetFilePointerEx(handle, zero, NULL, FILE_BEGIN) && SetEndOfFile(handle);
#else
    struct stat info;
    if (fstat(fileno(file), &info) != 0) return 0;
    return !S_ISREG(info.st_mode) || ftruncate(fileno(file), 0) == 0;
#endif
}

static int write_nals(FILE *out, x264_nal_t *nals, int count, uint64_t *total) {
    if (count < 0 || (count && !nals)) return 0;
    for (int i = 0; i < count; ++i) {
        if (nals[i].i_payload <= 0 || !nals[i].p_payload ||
            fwrite(nals[i].p_payload, 1, (size_t)nals[i].i_payload, out) !=
                (size_t)nals[i].i_payload)
            return 0;
        *total += (uint64_t)nals[i].i_payload;
    }
    return !ferror(out);
}

int main(int argc, char **argv) {
    int w, h, crf, delta, n = 0, encoded = 0, status = 1;
    uint32_t fps_num, fps_den;
    uint64_t size, total = 0;
    size_t pixels, frame_size, mb_count;
    FILE *yuv = NULL, *map = NULL, *out = NULL;
    uint8_t *frame = NULL;
    float *offsets = NULL;
    x264_t *enc = NULL;
    x264_param_t p;
    x264_picture_t pic, pic_out;
    x264_nal_t *nals = NULL;
    int nnals = 0, bytes;
    char crf_text[16];
    const char *error = NULL;

    if ((argc != 9 && argc != 10) ||
        !parse_int(argv[2], &w) || !parse_int(argv[3], &h) ||
        w <= 0 || h <= 0 || w % 2 || h % 2 ||
        !parse_fps(argv[4], &fps_num, &fps_den) ||
        !parse_int(argv[5], &crf) || crf < 0 || crf > 51 ||
        !parse_int(argv[7], &delta) ||
        (argc == 10 && (!parse_int(argv[9], &n) || n < 2))) {
        fprintf(stderr, "args: <input.yuv> <positive even w> <positive even h> "
                        "<finite positive fps> <crf 0..51> <map.txt|-> "
                        "<integer delta> <output.h264> [nframes >=2]\n");
        return 2;
    }
    (void)delta; /* Applied by the map producer, never applied twice here. */
    if ((size_t)w > SIZE_MAX / (size_t)h) {
        error = "I420 dimensions overflow"; goto cleanup;
    }
    pixels = (size_t)w * (size_t)h;
    if (pixels > SIZE_MAX - pixels / 2) {
        error = "I420 frame size overflow"; goto cleanup;
    }
    frame_size = pixels + pixels / 2;
    mb_count = ((size_t)w / 16 + (w % 16 != 0)) *
               ((size_t)h / 16 + (h % 16 != 0));
    if (mb_count > SIZE_MAX / sizeof(float)) {
        error = "ROI map size overflow"; goto cleanup;
    }
    if (!strcmp(argv[1], argv[8]) ||
        (strcmp(argv[6], "-") && !strcmp(argv[6], argv[8]))) {
        error = "output must not overwrite an input"; goto cleanup;
    }
    yuv = fopen(argv[1], "rb");
    if (!yuv || !raw_size(yuv, &size)) {
        error = "cannot open or size raw I420 input"; goto cleanup;
    }
    if (size % frame_size || size / frame_size < 2 ||
        size / frame_size > INT_MAX ||
        (argc == 10 && size / frame_size != (uint64_t)n)) {
        error = "raw I420 size must match exactly n >=2 complete frames";
        goto cleanup;
    }
    n = (int)(size / frame_size);
    frame = malloc(frame_size);
    if (!frame) { error = "cannot allocate I420 frame"; goto cleanup; }
    if (strcmp(argv[6], "-")) {
        offsets = malloc(mb_count * sizeof(*offsets));
        map = fopen(argv[6], "rb");
        if (!offsets || !map) { error = "cannot allocate/open ROI map"; goto cleanup; }
        /* Validate before opening the output, then stream the map again. */
        for (int t = 0; t < n; ++t) {
            if (!map_row(map, offsets, mb_count)) {
                fprintf(stderr, "invalid ROI frame line %d (expected %zu finite offsets)\n",
                        t + 1, mb_count);
                goto cleanup;
            }
        }
        if (!at_end(map) || fseek(map, 0, SEEK_SET) != 0) {
            error = "extra ROI frame lines/data or map read/rewind failure"; goto cleanup;
        }
    }

    x264_param_default(&p);
    p.i_width = w; p.i_height = h; p.i_csp = X264_CSP_I420;
    p.i_threads = 1; p.i_log_level = X264_LOG_ERROR;
    p.i_keyint_max = 250; p.i_bframe = 0;
    p.b_vfr_input = 0;
    p.i_fps_num = fps_num; p.i_fps_den = fps_den;
    p.i_timebase_num = fps_den; p.i_timebase_den = fps_num;
    snprintf(crf_text, sizeof(crf_text), "%d", crf);
    if (x264_param_parse(&p, "crf", crf_text) < 0 ||
        x264_param_parse(&p, "aq-mode", "1") < 0 ||
        x264_param_parse(&p, "aq-strength", "1.0") < 0 ||
        x264_param_parse(&p, "mbtree", "0") < 0) {
        error = "encoder rejected verified CRF/AQ/mbtree policy"; goto cleanup;
    }
    enc = x264_encoder_open(&p);
    if (!enc) { error = "encoder_open failed"; goto cleanup; }
    out = fopen(argv[8], "ab"); /* Never truncate before checking opened-file identity. */
    if (!out) { error = "cannot open output bitstream"; goto cleanup; }
    if (same_file(out, yuv) != 0 || (map && same_file(out, map) != 0)) {
        error = "output aliases an input or file identity check failed"; goto cleanup;
    }
    if (!truncate_output(out)) {
        error = "cannot truncate output bitstream"; goto cleanup;
    }
    bytes = x264_encoder_headers(enc, &nals, &nnals);
    if (bytes <= 0 || nnals <= 0 || !write_nals(out, nals, nnals, &total)) {
        error = "encoder headers or bitstream write failed"; goto cleanup;
    }
    x264_picture_init(&pic);
    pic.img.i_csp = X264_CSP_I420; pic.img.i_plane = 3;
    pic.img.i_stride[0] = w; pic.img.i_stride[1] = w / 2; pic.img.i_stride[2] = w / 2;
    pic.img.plane[0] = frame;
    pic.img.plane[1] = frame + pixels;
    pic.img.plane[2] = frame + pixels + pixels / 4;
    pic.prop.quant_offsets = offsets;
    for (int t = 0; t < n; ++t) {
        if (fread(frame, 1, frame_size, yuv) != frame_size ||
            (map && !map_row(map, offsets, mb_count))) {
            error = "raw/map read failed or input changed during encode"; goto cleanup;
        }
        pic.i_pts = t;
        bytes = x264_encoder_encode(enc, &nals, &nnals, &pic, &pic_out);
        if (bytes < 0 || !write_nals(out, nals, nnals, &total)) {
            error = "frame encode or bitstream write failed"; goto cleanup;
        }
        if (bytes > 0) ++encoded;
    }
    if (!at_end(yuv) || (map && !at_end(map))) {
        error = "raw/map has extra data or read error after nframes"; goto cleanup;
    }
    for (int flush = 0; x264_encoder_delayed_frames(enc) > 0; ++flush) {
        if (flush >= n) { error = "encoder flush did not finish"; goto cleanup; }
        bytes = x264_encoder_encode(enc, &nals, &nnals, NULL, &pic_out);
        if (bytes <= 0 || !write_nals(out, nals, nnals, &total)) {
            error = "encoder flush or bitstream write failed"; goto cleanup;
        }
        ++encoded;
    }
    if (encoded != n) { error = "encoder produced an incomplete sequence"; goto cleanup; }
    if (fflush(out) != 0) { error = "bitstream flush failed"; goto cleanup; }
    status = 0;

cleanup:
    if (out && fclose(out) != 0) { error = "bitstream close failed"; status = 1; }
    if (map && fclose(map) != 0) { error = "ROI map close failed"; status = 1; }
    if (yuv && fclose(yuv) != 0) { error = "raw input close failed"; status = 1; }
    if (enc) x264_encoder_close(enc);
    free(offsets);
    free(frame);
    if (error) fprintf(stderr, "%s\n", error);
    if (!status)
        fprintf(stderr, "encoded %d frames, %llu bytes\n", n, (unsigned long long)total);
    return status;
}
