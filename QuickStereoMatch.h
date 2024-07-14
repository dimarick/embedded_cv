//
// Created by dima on 30.06.24.
//

#ifndef EMBEDDED_CV_QUICKSTEREOMATCH_H
#define EMBEDDED_CV_QUICKSTEREOMATCH_H

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#ifdef HAVE_OPENCV_HIGHGUI
#include "opencv2/highgui.hpp"
#endif

using namespace cv;

class QuickStereoMatch {
public:
    void computeDisparityMap(const Mat &left, const Mat &right, Mat &lmap, Mat &rmap, int maxDisparity = 150, int windowSize = 5, int borderSize = 10);

    float getCost(int channels, int windowSize, float cost, const uchar *si, const float *sg, const uchar *ti, const float *tg) const;
};


#endif //EMBEDDED_CV_QUICKSTEREOMATCH_H
