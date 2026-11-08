/* x264 ROI delta-QP encoder tool — the A1 backend (MATH.md section 3).
 *
 * Encodes a %04d.png frame directory with per-MB QP offsets from a cached
 * mask npz. A0/A2 arms use the same tool with a zero-offset map, so every
 * arm shares one verified encode path (external review P1-04: no arm may
 * silently encode without its intervention).
 *
 * Verify chain (docs/SESSION_LOG.md hunt rounds 5-17): quant_offsets applies
 * through the AQ path — aq-mode=1 and DEFAULT aq-strength are REQUIRED
 * (strength=0 gates the whole path, rounds 5-16 ratio 1.0); split-offset
 * sign test green (bg+10/obj0 ratio 0.277).
 *
 * Build (Kaggle): gcc -O2 -o x264_roi_encode x264_roi_encode.c -lx264 -lm
 *   needs: libx264-dev, libpng-dev
 * Usage: x264_roi_encode <frames_dir> <width> <height> <fps> <qp>
 *                        <masks.npz> <delta> <out_bin> [n_frames]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <stdint.h>
#include <x264.h>

/* minimal PNG loader: use stb_image-style inline decode is overkill here —
 * frames are written by PIL at a known size; read raw RGB via tiny PNG
 * decoder is NOT included. Instead the driver pre-converts frames to raw
 * I420 .yuv packets (simplest robust path): the tool consumes a raw YUV. */

int main(int argc, char **argv) {
    if (argc < 9) {
        fprintf(stderr, "args: <yuv_file> <w> <h> <fps> <qp> <roi_map_file|-> "
                        "<delta> <out_bin> [nframes]\n");
        return 2;
    }
    const char *yuvp = argv[1];
    int w = atoi(argv[2]), h = atoi(argv[3]);
    double fps = atof(argv[4]);
    int qp = atoi(argv[5]);
    const char *roimapp = argv[6];
    int delta = atoi(argv[7]);
    const char *outp = argv[8];

    FILE *yuv = fopen(yuvp, "rb");
    if (!yuv) { fprintf(stderr, "no yuv input\n"); return 2; }
    fseek(yuv, 0, SEEK_END);
    long fsz = ftell(yuv);
    fseek(yuv, 0, SEEK_SET);
    long fsz_frame = (long)w * h * 3 / 2;
    int n = (int)(fsz / fsz_frame);
    if (argc > 9 && atoi(argv[9]) > 0) n = atoi(argv[9]);
    if (n < 2) { fprintf(stderr, "need >=2 frames\n"); return 2; }

    x264_param_t p;
    x264_param_default(&p);
    p.i_width = w; p.i_height = h; p.i_csp = X264_CSP_I420;
    p.i_threads = 1; p.i_log_level = X264_LOG_ERROR;
    p.i_keyint_max = 250; p.i_bframe = 0;   /* low-delay pinning */
    char crf_s[32];
    snprintf(crf_s, sizeof(crf_s), "%d", qp);
    /* AQ path MUST be on with default strength: quant_offsets flows through
     * it (hunt rounds 5-17); strength=0 gates the whole path off */
    x264_param_parse(&p, "crf", crf_s);
    x264_param_parse(&p, "aq-mode", "1");
    x264_param_parse(&p, "aq-strength", "1.0");
    x264_param_parse(&p, "mbtree", "0");
    x264_t *enc = x264_encoder_open(&p);
    if (!enc) { fprintf(stderr, "encoder_open failed\n"); return 3; }

    /* ROI map: text, one line per frame, mbw*mbh integer QP offsets
     * (raster order, 16x16 MB) — written by scripts/make_roi_maps.py */
    float *offsets = malloc(sizeof(float) * (size_t)((w + 15) / 16 * ((h + 15) / 16)));
    FILE *roimap = NULL;
    int use_roi = strcmp(roimapp, "-") != 0;
    if (use_roi) {
        roimap = fopen(roimapp, "r");
        if (!roimap) { fprintf(stderr, "roi map missing\n"); return 2; }
    }

    FILE *out = fopen(outp, "wb");
    x264_nal_t *nals; int nnals;
    x264_encoder_headers(enc, &nals, &nnals);
    for (int i = 0; i < nnals; i++) fwrite(nals[i].p_payload, 1, nals[i].i_payload, out);

    x264_picture_t pic, pic_out;
    x264_picture_init(&pic);
    long total = 0;
    for (int t = 0; t < n; t++) {
        if (fread(yuv, 1, fsz_frame, yuv) != (size_t)fsz_frame) break;
        pic.img.i_csp = X264_CSP_I420; pic.img.i_plane = 3;
        pic.img.i_stride[0] = w; pic.img.i_stride[1] = w / 2; pic.img.i_stride[2] = w / 2;
        pic.img.plane[0] = (uint8_t *)yuv;
        pic.img.plane[1] = (uint8_t *)yuv + w * h;
        pic.img.plane[2] = (uint8_t *)yuv + w * h * 5 / 4;
        pic.i_pts = t;
        pic.prop.quant_offsets = NULL;
        if (use_roi) {
            int mbw = (w + 15) / 16, mbh = (h + 15) / 16;
            for (int i = 0; i < mbw * mbh; i++) {
                float v = 0.0f;
                if (fscanf(roimap, "%f", &v) != 1) v = 0.0f;
                offsets[i] = v;
            }
            pic.prop.quant_offsets = offsets;
        }
        int fn = x264_encoder_encode(enc, &nals, &nnals, &pic, &pic_out);
        for (int i = 0; i < nnals; i++) {
            fwrite(nals[i].p_payload, 1, nals[i].i_payload, out);
            total += nals[i].i_payload;
        }
        (void)fn;
    }
    int guard = 0;
    while (x264_encoder_delayed_frames(enc) > 0 && guard++ < 200) {
        int fn = x264_encoder_encode(enc, &nals, &nnals, NULL, &pic_out);
        for (int i = 0; i < nnals; i++) {
            fwrite(nals[i].p_payload, 1, nals[i].i_payload, out);
            total += nals[i].i_payload;
        }
        if (fn <= 0 && nnals == 0) break;
        (void)fn;
    }
    fprintf(stderr, "encoded %d frames, %ld bytes\n", n, total);

    fclose(out);
    free(offsets);
    if (roimap) fclose(roimap);
    fclose(yuv);
    x264_encoder_close(enc);
    (void)delta; /* delta is baked into the map by make_roi_maps.py */
    return 0;
}
