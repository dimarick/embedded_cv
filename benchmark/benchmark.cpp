// SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
// Copyright (c) 2026 Dmitrii Kosenok

// This file is part of EmbeddedCV.

// It is dual-licensed under the terms of the GNU General Public License v3
// and a commercial license. You can choose the license that fits your needs.
// For details, see the LICENSE file in the root of the repository.

#include "KitDisparityBenchmark.h"
#include <cv/DisparityEvaluator.h>
#include <numeric>
#include <fstream>
#include <core/ocl.hpp>

int main(int argc, char **argv) {
    // Конфигурация бенчмарка
    KitDisparityBenchmark::Config config;
    config.datasetBasePath = "datasets/kitti12/"; // Укажите ваш путь
    config.benchmarkYear = "2012";
    config.errorThreshold = 3.0f;
    config.verbose = true;

    if (argc == 2) {
        config.debug = true;
        config.id = std::atoi(argv[1]);
    }

    
    // Инициализация бенчмарка
    KitDisparityBenchmark benchmark(config);
    benchmark.initializeDatasets();

    ecv::DisparityEvaluator stereo((cl_context)cv::ocl::Context::getDefault().ptr());

    std::mutex lock;
    
    // Запуск оценки вашего алгоритма
    auto results = benchmark.runBenchmark([&stereo, &lock](const std::vector<cv::UMat>& frames, cv::Mat& disparity, cv::Mat& variance) {
        cv::Mat disparityI16;

        for (auto &frame : frames) {
            if (frame.channels() == 1) {
                cv::cvtColor(frame, frame, cv::COLOR_GRAY2RGB);
//                frame.convertTo(frame, CV_8UC1);
            } else {
//                cv::cvtColor(frame, frame, cv::COLOR_RGB2GRAY);
//                target.convertTo(frame, CV_8UC1);
            }
        }

        lock.lock();
        stereo.evaluateDisparity(frames, disparityI16, variance);
        lock.unlock();

        disparityI16.convertTo(disparity, CV_32F);

        disparity /= 16;
    });

    return 0;
}