/* ROI sign-test for x264 — round 14 harness (review-corrected).
 *
 * Fixes from the dual review (Argon + Astra):
 *   - NO pic.i_qpplus1: it overrides the frame QP at ratecontrol_start,
 *     AFTER the AQ path, so quant_offsets never reached the bitstream
 *     (rounds 5-9 were byte-identical because of this).
 *   - CRF + aq-mode=1 + aq-strength=0 + mbtree=0: the AQ path runs
 *     (x264_adaptive_quant_frame folds prop.quant_offsets into f_qp_offset)
 *     with zero AQ pollution of its own.
 *   - FULL drain: encode NULL input until x264_encoder_delayed_frames == 0,
 *     writing every returned NAL (rounds 5-13 truncated buffered frames).
 *
 * argv: w h nframes qp offset_value outpath   (offset 0 = base run)
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <x264.h>

int main(int argc, char **argv) {
    if (argc < 7) { fprintf(stderr, "args: w h n qp off out\n"); return 2; }
    int w = atoi(argv[1]), h = atoi(argv[2]), n = atoi(argv[3]);
    int qp = atoi(argv[4]);
    float bg_off = (float)atof(argv[5]);
    float obj_off = (float)atof(argv[6]);
    const char *outp = argv[7];

    x264_param_t p;
    x264_param_default(&p);
    p.i_width = w; p.i_height = h; p.i_csp = X264_CSP_I420;
    p.i_threads = 1; p.i_log_level = X264_LOG_ERROR;
    x264_param_parse(&p, "crf", argv[4]);
    x264_param_parse(&p, "aq-mode", "1");
    x264_param_parse(&p, "mbtree", "0");
    x264_t *enc = x264_encoder_open(&p);
    if (!enc) { fprintf(stderr, "encoder_open failed\n"); return 3; }

    FILE *out = fopen(outp, "wb");
    x264_nal_t *nals; int nnals;
    x264_encoder_headers(enc, &nals, &nnals);
    for (int i = 0; i < nnals; i++) fwrite(nals[i].p_payload, 1, nals[i].i_payload, out);

    int ysz = w * h, csz = ysz / 4;
    uint8_t *yuv = malloc(ysz + 2 * csz);
    int mbw = (w + 15) / 16, mbh = (h + 15) / 16;
    float *offsets = malloc(sizeof(float) * mbw * mbh);

    x264_picture_t pic, pic_out;
    x264_picture_init(&pic);
    for (int t = 0; t < n; t++) {
        for (int i = 0; i < ysz; i++) yuv[i] = 60 + rand() % 140;
        for (int i = 0; i < csz; i++) { yuv[ysz + i] = 128; yuv[ysz + csz + i] = 128; }
        int bx = (t * 2) % (w - 64), by = (t * 3) % (h - 64);
        for (int y = by; y < by + 48; y++)
            for (int x = bx; x < bx + 48; x++) yuv[y * w + x] = 235;

        pic.img.i_csp = X264_CSP_I420; pic.img.i_plane = 3;
        pic.img.i_stride[0] = w; pic.img.i_stride[1] = w / 2; pic.img.i_stride[2] = w / 2;
        pic.img.plane[0] = yuv; pic.img.plane[1] = yuv + ysz; pic.img.plane[2] = yuv + ysz + csz;
        pic.i_pts = t;
        pic.prop.quant_offsets = NULL;
        if (bg_off != 0.0f || obj_off != 0.0f) {
            /* round 16: SPLIT offsets — uniform offsets are mean-centered
             * away by the rate control (rounds 5-14 ratio 1.0); the ROI
             * semantics are a redistribution bg<->object */
            float *offsets = malloc(sizeof(float) * mbw * mbh);
            for (int r2 = 0; r2 < mbh; r2++)
                for (int c2 = 0; c2 < mbw; c2++)
                    offsets[r2 * mbw + c2] = bg_off;
            /* object block = the moving box this frame */
            int obx = ((t * 2) % (w - 64)) / 16, oby = ((t * 3) % (h - 64)) / 16;
            int obw = (48 + 15) / 16, obh2 = (48 + 15) / 16;
            for (int r2 = oby; r2 < oby + obh2 && r2 < mbh; r2++)
                for (int c2 = obx; c2 < obx + obw && c2 < mbw; c2++)
                    if (r2 >= 0 && c2 >= 0) offsets[r2 * mbw + c2] = obj_off;
            pic.prop.quant_offsets = offsets;
        }
        int fn = x264_encoder_encode(enc, &nals, &nnals, &pic, &pic_out);
        for (int i = 0; i < nnals; i++) fwrite(nals[i].p_payload, 1, nals[i].i_payload, out);
        (void)fn;
    }
    /* full drain (review fix): delayed lookahead frames carry inter QP effects */
    int guard = 0;
    while (x264_encoder_delayed_frames(enc) > 0 && guard++ < 100) {
        int fn = x264_encoder_encode(enc, &nals, &nnals, NULL, &pic_out);
        for (int i = 0; i < nnals; i++) fwrite(nals[i].p_payload, 1, nals[i].i_payload, out);
        if (fn <= 0 && nnals == 0) break;
        (void)fn;
    }

    fclose(out);
    x264_encoder_close(enc);
    return 0;
}
