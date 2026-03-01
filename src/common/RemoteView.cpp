#include "RemoteView.h"

using namespace ecv;

void RemoteView::showMat(const std::string& viewName, const cv::Mat& mat) {
#ifdef HAVE_OPENCV_HIGHGUI
    cv::imshow(viewName, mat);
#endif
    // TODO send it to ws_ctl
}
