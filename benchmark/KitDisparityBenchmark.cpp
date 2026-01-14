// KitDisparityBenchmark.cpp
#include "KitDisparityBenchmark.h"
#include <filesystem>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <numeric>

namespace fs = std::filesystem;

KitDisparityBenchmark::KitDisparityBenchmark(const Config& config) : config_(config) {
}

void KitDisparityBenchmark::initializeDatasets() {
    leftImagePaths_.clear();
    rightImagePaths_.clear();
    gtDisparityPaths_.clear();
    
    std::string leftImageFolder, rightImageFolder, gtFolder;
    
    if (config_.benchmarkYear == "2012") {
        leftImageFolder = "training/colored_0/";
        rightImageFolder = "training/colored_1/";
        gtFolder = "training/disp_noc/"; // Незакрытые области
    } else if (config_.benchmarkYear == "2015") {
        leftImageFolder = "training/image_2/";
        rightImageFolder = "training/image_2/";
        gtFolder = "training/disp_occ_0/"; // Все области (с окклюзиями)
    } else {
        throw std::runtime_error("Unsupported benchmark year. Use '2012' or '2015'.");
    }
    
    std::string base = config_.datasetBasePath;
    std::string leftBase = base + leftImageFolder;
    std::string rightBase = base + rightImageFolder;
    std::string gtBase = base + gtFolder;
    
    // KITTI использует naming pattern: 000000_10.png, 000001_10.png и т.д.
    for (int i = 0; i < (config_.benchmarkYear == "2012" ? 194 : 200); ++i) {
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(6) << i;
        std::string idxStr = ss.str();
        
        std::string leftPath = leftBase + idxStr + "_10.png";
        std::string rightPath = rightBase + idxStr + "_10.png";
        std::string gtPath = gtBase + idxStr + "_10.png";
        
        if (fs::exists(leftPath) && fs::exists(rightPath) && fs::exists(gtPath)) {
            leftImagePaths_.push_back(leftPath);
            rightImagePaths_.push_back(rightPath);
            gtDisparityPaths_.push_back(gtPath);
        } else {
            if (config_.verbose) {
                std::cout << "Warning: Missing files for index " << i << ", skipping." << "leftPath " << leftPath << " rightPath " << rightPath << std::endl;
            }
        }
    }
    
    if (config_.verbose) {
        std::cout << "Initialized KITTI " << config_.benchmarkYear 
                  << " benchmark with " << leftImagePaths_.size() << " image pairs." << std::endl;
    }
}

cv::Mat KitDisparityBenchmark::loadGTDisparity(const std::string& path) {
    // В KITTI ground truth диспарити хранится в 16-битном PNG, где значения равны диспарити * 256
    cv::Mat gt = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (gt.empty()) {
        throw std::runtime_error("Failed to load GT disparity: " + path);
    }
    
    // Конвертация к float и масштабирование
    cv::Mat gtFloat;
    gt.convertTo(gtFloat, CV_32F);
    gtFloat = gtFloat / 256.0f;
    
    // Пиксели без ground truth помечены как 0
    return gtFloat;
}

KitDisparityBenchmark::ImageMetrics KitDisparityBenchmark::computeMetrics(
    const cv::Mat& predDisparity, const cv::Mat& gtDisparity) {
    
    ImageMetrics metrics;
    
    // Маска валидных пикселей (в GT диспарити > 0)
    cv::Mat validMask = gtDisparity > 0.0f;
    
    if (cv::countNonZero(validMask) == 0) {
        return metrics;
    }
    
    // Ошибка для каждого пикселя
    cv::Mat diff;
    cv::absdiff(predDisparity, gtDisparity, diff);
    
    // 1. Для KITTI 2012: отдельно считаем метрики для незакрытых (noc) и всех (all) областей
    if (config_.benchmarkYear == "2012") {
        // В 2012 датасете есть отдельные GT для незакрытых областей
        // Здесь для простоты считаем, что все валидные пиксели - незакрытые
        cv::Mat errorMaskNoc = diff > config_.errorThreshold;
        errorMaskNoc = errorMaskNoc & validMask;
        
        metrics.outNoc = 100.0f * cv::countNonZero(errorMaskNoc) / cv::countNonZero(validMask);
        metrics.avgNoc = cv::mean(diff, validMask)[0];
        
        // Для 'all' метрики в 2012 требуются дополнительные GT с окклюзиями
        // В этом упрощенном примере полагаем outAll = outNoc, avgAll = avgNoc
        metrics.outAll = metrics.outNoc;
        metrics.avgAll = metrics.avgNoc;
        
    } 
    // 2. Для KITTI 2015: используем метрику D1-all
    else if (config_.benchmarkYear == "2015") {
        // D1-all: процент пикселей с ошибкой > max(3px, 0.05 * gtDisparity)
        cv::Mat thresholdMap;
        cv::max(config_.errorThreshold, 0.05f * gtDisparity, thresholdMap);
        
        cv::Mat errorMask = diff > thresholdMap;
        errorMask = errorMask & validMask;
        
        metrics.d1All = 100.0f * cv::countNonZero(errorMask) / cv::countNonZero(validMask);
        metrics.avgAll = cv::mean(diff, validMask)[0];
    }
    
    return metrics;
}

std::pair<KitDisparityBenchmark::ImageMetrics, double> 
KitDisparityBenchmark::evaluateImage(int index, 
                                     std::function<void(const std::vector<cv::UMat>&, cv::Mat&)> evaluator) {
    
    if (index < 0 || index >= leftImagePaths_.size()) {
        throw std::out_of_range("Image index out of range");
    }
    
    // Загрузка стереопары
    cv::Mat left = cv::imread(leftImagePaths_[index], cv::IMREAD_UNCHANGED);
    cv::Mat right = cv::imread(rightImagePaths_[index], cv::IMREAD_UNCHANGED);
    cv::Mat gt = loadGTDisparity(gtDisparityPaths_[index]);
    
    if (left.empty() || right.empty() || gt.empty()) {
        throw std::runtime_error("Failed to load image pair or GT for index " + std::to_string(index));
    }
    
    // Конвертация в UMat для вашего интерфейса
    std::vector<cv::UMat> frames(2);
    left.copyTo(frames[0]);
    right.copyTo(frames[1]);
    
    // Вычисление диспарити и замер времени
    cv::Mat disparity;
    auto start = std::chrono::high_resolution_clock::now();
    
    evaluator(frames, disparity);
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Приведение predicted disparity к float для сравнения
    if (disparity.type() != CV_32F) {
        disparity.convertTo(disparity, CV_32F);
    }

// --- ДИАГНОСТИКА ---
// 1. Проверьте тип данных
    std::cout << "[DEBUG] Disparity type: " << disparity.type()
              << " (CV_32F = " << CV_32F << ")" << std::endl;

// 2. Проверьте диапазон значений
    double minVal, maxVal, maxValGt;
    cv::minMaxLoc(disparity, &minVal, &maxVal);
    std::cout << "[DEBUG] Disparity range: min=" << minVal << ", max=" << maxVal << std::endl;

// 3. Для сравнения - проверьте Ground Truth
    cv::minMaxLoc(gt, &minVal, &maxValGt);
    std::cout << "[DEBUG] GT range: min=" << minVal << ", max=" << maxValGt << std::endl;
    std::cout << "[DEBUG] GT type: " << gt.type() << std::endl;


    double scale = maxValGt / maxVal;

    // Вычисление метрик
    ImageMetrics metrics = computeMetrics(disparity, gt);

    if (config_.debug) {
        cv::Mat disparity8;
        cv::Mat disparityFp;
        cv::Mat gtFp;
        cv::Mat gt8;

        disparity.copyTo(disparityFp);
        if (minVal == 0 || maxVal == 0) {
            cv::minMaxLoc(disparityFp, &minVal, &maxVal);
        }

        disparityFp -= minVal;
        disparityFp *= 255.0 / (maxVal - minVal);

        disparityFp.convertTo(disparity8, CV_8U);
        cv::applyColorMap(disparity8, disparity8, cv::ColormapTypes::COLORMAP_JET);

        gt.copyTo(gtFp);
        if (minVal == 0 || maxVal == 0) {
            cv::minMaxLoc(gtFp, &minVal, &maxVal);
        }

        gtFp -= minVal;
        gtFp *= 255.0 / (maxVal - minVal);

        gtFp.convertTo(gt8, CV_8U);
        cv::applyColorMap(gt8, gt8, cv::ColormapTypes::COLORMAP_JET);

        cv::imshow("GT", gt8);
        cv::imshow("Disparity", disparity8);

        cv::imshow("Left", left);
        cv::imshow("Right", right);

        cv::waitKey(0);
    }

    return {metrics, elapsed};
}

KitDisparityBenchmark::AggregateMetrics 
KitDisparityBenchmark::runBenchmark(std::function<void(const std::vector<cv::UMat>&, cv::Mat&)> evaluator) {
    
    AggregateMetrics aggregate;
    std::vector<double> times;
    
    size_t totalImages = leftImagePaths_.size();
    int processed = 0;
    
    for (int i = 0; i < totalImages; ++i) {
        if (config_.debug && config_.id >= 0 && i != config_.id) {
            continue;
        }
        try {
            auto [metrics, elapsed] = evaluateImage(i, evaluator);
            
            // Агрегация метрик
            aggregate.meanOutNoc += metrics.outNoc;
            aggregate.meanOutAll += metrics.outAll;
            aggregate.meanAvgNoc += metrics.avgNoc;
            aggregate.meanAvgAll += metrics.avgAll;
            aggregate.meanD1All += metrics.d1All;
            aggregate.inferenceTimes.push_back(elapsed);
            
            processed++;
            
            if (config_.verbose && (i % 20 == 0 || i == totalImages - 1)) {
                std::cout << "Processed " << i + 1 << "/" << totalImages 
                          << " images. Current D1-all: " << metrics.d1All << "%" 
                          << ", Time: " << elapsed << "ms" << std::endl;
            }
            
        } catch (const std::exception& e) {
            if (config_.verbose) {
                std::cout << "Error processing image " << i << ": " << e.what() << std::endl;
            }
        }
    }
    
    // Усреднение метрик
    if (processed > 0) {
        float invProcessed = 1.0f / processed;
        aggregate.meanOutNoc *= invProcessed;
        aggregate.meanOutAll *= invProcessed;
        aggregate.meanAvgNoc *= invProcessed;
        aggregate.meanAvgAll *= invProcessed;
        aggregate.meanD1All *= invProcessed;
    }
    
    if (config_.verbose) {
        std::cout << "\n=== Benchmark Results (KITTI " << config_.benchmarkYear << ") ===" << std::endl;
        std::cout << "Processed " << processed << "/" << totalImages << " image pairs" << std::endl;
        
        if (config_.benchmarkYear == "2012") {
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "Out-Noc: " << aggregate.meanOutNoc << "%" << std::endl;
            std::cout << "Out-All: " << aggregate.meanOutAll << "%" << std::endl;
            std::cout << "Avg-Noc: " << aggregate.meanAvgNoc << " px" << std::endl;
            std::cout << "Avg-All: " << aggregate.meanAvgAll << " px" << std::endl;
        } else {
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "D1-all: " << aggregate.meanD1All << "%" << std::endl;
            std::cout << "Avg-All: " << aggregate.meanAvgAll << " px" << std::endl;
        }
        
        if (!aggregate.inferenceTimes.empty()) {
            double avgTime = std::accumulate(aggregate.inferenceTimes.begin(), 
                                           aggregate.inferenceTimes.end(), 0.0) 
                           / aggregate.inferenceTimes.size();
            std::cout << "Avg inference time: " << avgTime << " ms" << std::endl;
            std::cout << "FPS: " << 1000.0 / avgTime << std::endl;
        }
    }
    
    return aggregate;
}