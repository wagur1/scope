/* ROI sign-test for x265: per-CU quantOffsets via x265_picture.
 * Identical protocol to roi_hunt_x264.c: same deterministic clip, offsets
 * +10 QP on every CU when enabled -> bitrate MUST drop sharply.
 * Only compiled if x265.h declares quantOffsets (checked by run_roi_hunt.sh).
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <x265.h>

int main(int argc, char **argv) {
    if (argc < 7) { fprintf(stderr, "args: w h n qp use_off out\n"); return 2; }
    int w = atoi(argv[1]), h = atoi(argv[2]), n = atoi(argv[3]);
    int qp = atoi(argv[4]);
    float offv = (float)atof(argv[5]);
    const char *outp = argv[6];

    x265_param *p = x265_param_alloc();
    x265_param_default(p);
    p->sourceWidth = w; p->sourceHeight = h;
    p->internalCsp = X265_CSP_I420;
    /* CQP forces aqMode=0 -> cu_qp_delta disabled -> offsets ignored.
     * CRF + aqMode=1 + aqStrength=0 + forceqp locks base QP (Argon rx) */
    p->rc.rateControlMode = X265_RC_CRF;
    p->rc.rfConstant = (double)qp;
    p->rc.aqMode = X265_AQ_VARIANCE;
    p->rc.aqStrength = 0.0;
    p->rc.cuTree = 0;
    p->rc.qgSize = 16;
    p->frameNumThreads = 1; p->bEnableSAO = 0;
    x265_param_parse(p, "fps", "30");   /* field names differ across builds */
    x265_param_parse(p, "numa-pools", "none");  /* NUMA restricted on containers */
    p->logLevel = X265_LOG_ERROR;
    x265_encoder *enc = x265_encoder_open(p);
    if (!enc) { fprintf(stderr, "encoder_open failed\n"); return 3; }

    FILE *out = fopen(outp, "wb");
    x265_nal *nals; uint32_t nnals;

    int ysz = w * h, csz = ysz / 4;
    uint8_t *yuv = malloc(ysz + 2 * csz);
    int cu64_count = ((w + 15) / 16) * ((h + 15) / 16);  /* 16x16 blocks, per Argon rx */

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
        pic.forceqp = qp + 1;   /* lock slice base QP (rateControlStart) */
        pic.quantOffsets = NULL;
        if (offv != 0.0f) {
            float *offsets = malloc(sizeof(float) * cu64_count);
            for (int i = 0; i < cu64_count; i++) offsets[i] = offv;
            pic.quantOffsets = offsets;
        }
        int frame_nals = x265_encoder_encode(enc, &nals, &nnals, &pic, NULL);
        pic.quantOffsets = NULL;
        for (uint32_t i = 0; i < nnals; i++) fwrite(nals[i].payload, 1, nals[i].sizeBytes, out);
        (void)frame_nals;
    }
    nals = NULL; nnals = 0;
    x265_encoder_encode(enc, &nals, &nnals, NULL, NULL); /* flush (pic_in=NULL) */
    x265_picture_free(&pic);

    fclose(out);
    x265_encoder_close(enc);
    x265_param_free(p);
    return 0;
}
