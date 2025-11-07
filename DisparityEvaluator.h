//
// Created by dima on 03.11.25.
//

#ifndef EMBEDDED_CV_DISPARITYEVALUATOR_H
#define EMBEDDED_CV_DISPARITYEVALUATOR_H

#include <vector>
#include <atomic>
#include "opencv2/core.hpp"

//#define CL_HPP_TARGET_OPENCL_VERSION 210

#include <CL/cl.hpp>

namespace ecv {

    class DisparityEvaluator {
    private:
        bool oclInitialized = false;
        cl::Device device;
        cl::Context context;
        cl::Program program;
        cl::Kernel kernel;
        cl::CommandQueue queue;
        std::vector<cl::Event> ioEvents;
        std::atomic<double> q = 0.5;
        int16_t getDisparity(const uint8_t *data1, const uint8_t *data2, size_t x, size_t y, size_t w, size_t h, int minDisparity, int maxDisparity, size_t windowSize, uint8_t sz);
    public:
        static constexpr const int DISPARITY_PRECISION = 16;
        void evaluateDisparity(const std::vector<cv::Mat> &frames, cv::Mat &disparity);
        void evaluateIncrementally(const std::vector<cv::Mat> &frames, const cv::Mat &roughDisparity, cv::Mat &disparity);
        void evaluateIncrementallyOcl(const std::vector<cv::Mat> &frames, const cv::Mat &roughDisparity, cv::Mat &disparity);

        void lazyInitializeOcl();

    };

} // ecv

#endif //EMBEDDED_CV_DISPARITYEVALUATOR_H
