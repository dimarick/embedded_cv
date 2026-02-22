#include <imgproc.hpp>
#include "BlurFrameFilter.h"

using namespace ecv;

/**
 * Вычисляет резкость кадра в абстрактных ненормированных единицах
 * @param frame
 * @return
 */
double BlurFrameFilter::getValue(const cv::UMat &frame) {
    cv::UMat thumb, lap;
    int w3 = frame.cols / 5;
    int h3 = frame.rows / 5;
    cv::extractChannel(frame(cv::Rect(frame.cols * 2 / 5, frame.rows * 2 / 5, w3, h3)), thumb, 1); // green
    cv::Laplacian(thumb, lap, CV_32F);
    cv::absdiff(lap, 0, lap);
    return cv::mean(lap)[0];
}

/**
 * Отделяет percentile лучших value. Лучше == больше.
 * Используется адаптивный порог резкости, устанавливающийся автоматически для достижения
 * целевого percentile. При установившемся пороге, пропускает все кадры, которые незначительно ниже порога
 * @param frame
 * @return
 */
bool BlurFrameFilter::streamingPercentile(double value) {
    if (currentThreshold == 0) {
        currentThreshold = value;
    }

    bool solution = value > currentThreshold;
    if (solution) {
        trueValues++;
    } else {
        falseValues++;
    }

    double relDiff = std::abs(value - currentThreshold) / std::max(value, currentThreshold);
    auto step = relDiff * currentThresholdStep;

    if (trueValues / falseValues > percentile) {
        currentThreshold *= 1 + step;
    } else {
        currentThreshold /= 1 + step;
    }

    if ((trueValues + falseValues) > 100) {
        trueValues *= 0.98;
        falseValues *= 0.98;
    }

    bool result = solution || relDiff < 1e-2;
    return result;
}
