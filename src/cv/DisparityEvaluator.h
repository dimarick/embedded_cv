#ifndef EMBEDDED_CV_DISPARITYEVALUATOR_H
#define EMBEDDED_CV_DISPARITYEVALUATOR_H

#include <utility>
#include <vector>
#include <atomic>
#include "opencv2/core.hpp"

#define CL_HPP_TARGET_OPENCL_VERSION 210
#define CL_HPP_ENABLE_EXCEPTIONS

#include <CL/cl2.hpp>
#include <stereo.hpp>

namespace ecv {

    class DisparityEvaluator {
    private:
        bool oclInitialized = false;
        cv::Ptr<cv::StereoSGBM> sgbm;
        cv::Ptr<cv::StereoBM> bm;
        bool hasContext = false;
        cl::Device device;
        cl::Context context;
        cl::Program program;
        cl::Kernel kernel;
        cl::CommandQueue queue;
        std::vector<cl::Event> ioEvents;
        std::atomic<double> q = 0.5;
        int16_t getDisparity(const uint8_t *data1, const uint8_t *data2, size_t x, size_t y, size_t w, size_t h, int minDisparity, int maxDisparity, size_t windowSize, uint8_t sz);
        const char *openclErrorString(cl_int err);

        void evaluateIncrementallyOcl(const std::vector<cv::UMat> &frames, const cv::Mat &roughDisparity, cv::Mat &disparity, cv::Mat &variance);
    public:
        explicit DisparityEvaluator(cl_context c) {
            context = cl::Context(c);
            std::vector<cl::Device> devices;
            context.getInfo(CL_CONTEXT_DEVICES, &devices);
            device = devices[0];
            hasContext = true;
        }
        explicit DisparityEvaluator() = default;
        static constexpr const int DISPARITY_PRECISION = 16;
        void evaluateDisparity(const std::vector<cv::UMat> &frames, cv::Mat &disparity, cv::Mat &variance);
        void lazyInitializeOcl();
    };

} // ecv

#endif //EMBEDDED_CV_DISPARITYEVALUATOR_H
