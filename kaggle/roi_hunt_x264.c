/* ROI sign-test for x264: per-MB quant_offsets via x264_image_prop_t.
 * Encodes the SAME deterministic clip twice; the caller compares bitstream
 * sizes. argv: w h nframes qp use_offsets(0/1) outpath
 * Offsets (when enabled) = +10 QP on EVERY MB -> bitrate MUST drop sharply.
 * This pins both field availability and sign semantics empirically.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <x264.h>

int main(int argc, char **argv) {
    if (argc < 7) { fprintf(stderr, "args: w h n qp use_off out\n"); return 2; }
    int w = atoi(argv[1]), h = atoi(argv[2]), n = atoi(argv[3]);
    int qp = atoi(argv[4]), use_off = atoi(argv[5]);
    const char *outp = argv[6];

    x264_param_t p;
    x264_param_default(&p);
    p.i_width = w; p.i_height = h; p.i_csp = X264_CSP_I420;
    p.rc.i_rc_method = X264_RC_CQP; p.rc.i_qp_constant = qp;
    p.i_threads = 1; p.i_log_level = X264_LOG_ERROR;
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
        if (use_off) {
            for (int i = 0; i < mbw * mbh; i++) offsets[i] = 10.0f;
            pic.prop.quant_offsets = offsets;
        }
        int frame_nals = x264_encoder_encode(enc, &nals, &nnals, &pic, &pic_out);
        for (int i = 0; i < nnals; i++) fwrite(nals[i].p_payload, 1, nals[i].i_payload, out);
        (void)frame_nals;
    }
    int frame_nals = x264_encoder_encode(enc, &nals, &nnals, NULL, &pic_out); /* flush */
    for (int i = 0; i < nnals; i++) fwrite(nals[i].p_payload, 1, nals[i].i_payload, out);
    (void)frame_nals;
    if (pic.prop.quant_offsets) pic.prop.quant_offsets = NULL;

    fclose(out);
    x264_encoder_close(enc);
    return 0;
}
