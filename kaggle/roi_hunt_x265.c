/* ROI sign-test for x265 — round 14 harness (review-corrected).
 *
 * Fixes from the dual review (Argon + Astra):
 *   - NO x265_picture_free on a stack picture (invalid free; rounds 6-12
 *     heap corruption came from the harness, not the library).
 *   - CRF + aqMode=1 + aqStrength=0 + cuTree=0 + qgSize=16: CQP forces
 *     aqMode=0 which disables cu_qp_delta entirely; CRF+AQ keeps the path
 *     quantOffsets flows through, with aqStrength=0 removing AQ's own offset.
 *   - forceqp locks the slice base QP (rateControlStart bypasses CRF drift).
 *   - quantOffsets sized by 16x16 blocks (qgSize=16), fresh per frame.
 *   - FULL drain: encode NULL input until no NALs are returned.
 *
 * argv: w h nframes qp offset_value outpath   (offset 0 = base run)
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <x265.h>

int main(int argc, char **argv) {
    if (argc < 7) { fprintf(stderr, "args: w h n qp off out\n"); return 2; }
    int w = atoi(argv[1]), h = atoi(argv[2]), n = atoi(argv[3]);
    int qp = atoi(argv[4]);
    float offv = (float)atof(argv[5]);
    const char *outp = argv[6];

    x265_param *p = x265_param_alloc();
    x265_param_default(p);
    p->sourceWidth = w; p->sourceHeight = h;
    p->internalCsp = X265_CSP_I420;
    p->frameNumThreads = 1;
    p->bEnableSAO = 0;
    p->rc.rateControlMode = X265_RC_CRF;
    p->rc.rfConstant = (double)qp;
    p->rc.aqMode = X265_AQ_VARIANCE;
    p->rc.aqStrength = 0.0;
    p->rc.cuTree = 0;
    p->rc.qgSize = 16;
    x265_param_parse(p, "fps", "30");
    x265_param_parse(p, "numa-pools", "none");
    x265_encoder *enc = x265_encoder_open(p);
    if (!enc) { fprintf(stderr, "encoder_open failed\n"); return 3; }

    FILE *out = fopen(outp, "wb");
    x265_nal *nals; uint32_t nnals;

    int ysz = w * h, csz = ysz / 4;
    uint8_t *yuv = malloc(ysz + 2 * csz);
    int blk = ((w + 15) / 16) * ((h + 15) / 16);

    x265_picture pic;
    x265_picture_init(p, &pic);
    for (int t = 0; t < n; t++) {
        for (int i = 0; i < ysz; i++) yuv[i] = 60 + rand() % 140;
        for (int i = 0; i < csz; i++) { yuv[ysz + i] = 128; yuv[ysz + csz + i] = 128; }
        int bx = (t * 2) % (w - 64), by = (t * 3) % (h - 64);
        for (int y = by; y < by + 48; y++)
            for (int x = bx; x < bx + 48; x++) yuv[y * w + x] = 235;

        pic.planes[0] = yuv; pic.planes[1] = yuv + ysz; pic.planes[2] = yuv + ysz + csz;
        pic.stride[0] = w; pic.stride[1] = w / 2; pic.stride[2] = w / 2;
        pic.pts = t;
        pic.forceqp = qp + 1;
        pic.quantOffsets = NULL;
        if (offv != 0.0f) {
            float *offsets = malloc(sizeof(float) * blk);
            for (int i = 0; i < blk; i++) offsets[i] = offv;
            pic.quantOffsets = offsets;
        }
        int fn = x265_encoder_encode(enc, &nals, &nnals, &pic, NULL);
        for (uint32_t i = 0; i < nnals; i++) fwrite(nals[i].payload, 1, nals[i].sizeBytes, out);
        (void)fn;
        pic.quantOffsets = NULL;
    }
    /* full drain (review fix): flush until the encoder returns no NALs */
    int guard = 0;
    nnals = 1;
    while (nnals > 0 && guard++ < 20) {
        int fn = x265_encoder_encode(enc, &nals, &nnals, NULL, NULL);
        for (uint32_t i = 0; i < nnals; i++) fwrite(nals[i].payload, 1, nals[i].sizeBytes, out);
        if (fn <= 0) break;
        (void)fn;
    }

    fclose(out);
    x265_encoder_close(enc);
    x265_param_free(p);
    return 0;
}
