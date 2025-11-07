//#pragma OPENCL EXTENSION cl_intel_printf : enable
//
//short getDisparity(
//        uchar *data1,
//        uchar *data2,
//        size_t x,
//        size_t y,
//        size_t w,
//        size_t h,
//        int minDisparity,
//        int maxDisparity,
//        size_t windowSize0,
//        uchar sz,
//        float* q,
//        int DISPARITY_PRECISION
//    ) {
//    const uchar *src = data1 + y * w * sz + x;
//    const uchar *dest = data2 + y * w * sz + x;
//
//    float disparity;
//    float avgScore;
//    float maxScore;
//    maxDisparity = max(minDisparity, maxDisparity);
//    minDisparity = min(minDisparity, maxDisparity);
//    int disparityRange = maxDisparity - minDisparity;
//
//    const int maxScoreSize = 5;
//    int scoreSize = max(2, min(maxScoreSize, disparityRange / 6));
//    float score[maxScoreSize];
//    float bestScore[maxScoreSize];
//
//    for (int i = 0; i < scoreSize; ++i) {
//        bestScore[i] = 0;
//        score[i] = 0;
//    }
//
//    int bestI = 0;
//    int bestK = 0;
//
//    int wi = 0;
//    int wis[] = {1, 7};
//    int wstep = 1;
//    do {
//        int windowSize = (int)windowSize0 * wis[wi];
//        maxScore = 0;
//        disparity = 0;
//        avgScore = 0;
//
//        int k = 0;
//        float scoreSum = 0;
//
//        for (int i = minDisparity; i < maxDisparity; i += sz) {
//            int score0 = 0;
//
//            int hstep = (int)w * sz;
//            for (int j = 0; j < h * hstep; j+= hstep) {
//                for (int i0 = -windowSize; i0 <= windowSize; i0+=wstep) {
//                    const int d = src[j + i0] - dest[i + j + i0];
//                    score0 += d * d;
//                }
//            }
//
//            float maxPossibleScore = 255.f * 255.f * ((float)windowSize * 2 + 1) * (float)h;
//            float newScore = maxPossibleScore - (float)score0 / (float)windowSize;
//            float prevScore = score[k % scoreSize];
//            score[k % scoreSize] = (float)newScore;
//
//            scoreSum += (float)newScore - prevScore;
//
//            int n = min(k + 1, (int) scoreSize);
//
//            float currentScore = scoreSum / (float)n;
//
//            avgScore += newScore;
//
//            if (maxScore < currentScore && k >= scoreSize) {
//                maxScore = currentScore;
//                for (int j = 0; j < scoreSize; ++j) {
//                    bestScore[j] = score[j];
//                }
//                bestI = i / sz;
//                bestK = k;
//            }
//            k++;
//        }
//
//        avgScore /= (float)k;
//        wi++;
//    } while (wi < sizeof wis / sizeof *wis && avgScore > maxScore * *q);
//
//    if (avgScore > maxScore * *q) {
//        *q = *q + 3e-7;
//        return 0;
//    }
//
//    *q = *q - 1e-7;
//
//    int n = min(bestK + 1, (int) scoreSize);
//    float mass = 0;
//    float sumX = 0;
//
//    for (int j = 1; j <= n; ++j) {
//        float m = (float)bestScore[(bestK + j) % scoreSize];
//        mass += m;
//        sumX += m * (float)j;
//    }
//    disparity = (float)bestI + (sumX / mass) - (float)n;
//
//    return (short)rint(disparity * DISPARITY_PRECISION);
//    return (short)1;
//}

__kernel void DisparityEvaluator(
        __global char* data,
        __global uchar* frame0,
        __global uchar* frame1,
        __global float* roughDisparity,
        __global short* disparity,
        __global float* q,
        const float q0,
        const int windowHeight,
        const int windowSize,
        const int w,
        const int sz,
        const int DISPARITY_PRECISION
) {
    data[0] = 'H';
    data[1] = 'e';
    data[2] = 'l';
    data[3] = 'l';
    data[4] = 'o';
    data[5] = ' ';
    data[6] = 'W';
    data[7] = 'o';
    data[8] = 'r';
    data[9] = 'l';
    data[10] = 'd';
    data[11] = '!';
    data[12] = '\n';
//    printf("Hello");
//    int x = get_global_id(0);
//    int y = get_global_id(1);
//
//    printf("Hello");
//
//    float _q = q0;
//    int wsz = w * sz;
//
//    short d = getDisparity(frame1, frame0, x * sz, y, w, windowHeight, 0, (wsz - windowSize - x), windowSize, sz, &_q, DISPARITY_PRECISION);
//
//    *q += _q - q0;
//
//    *q = 0.4f;
//
//    disparity[0] = (short)100;
}
