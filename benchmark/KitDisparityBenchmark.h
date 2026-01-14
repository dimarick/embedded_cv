// KitDisparityBenchmark.h
#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <map>

/**
 * Бенчмарк для оценки карт диспарити на датасете KITTI Stereo.
 * Поддерживает наборы 2012 и 2015 годов. Использует тренировочные данные с ground truth.
 */
class KitDisparityBenchmark {
public:
    // Конфигурация бенчмарка
    struct Config {
        std::string datasetBasePath; // Путь к корневой папке KITTI (например, "KITTI/stereo_2015/")
        std::string benchmarkYear;   // "2012" или "2015"
        float errorThreshold = 3.0f; // Порог для пиксельной ошибки (обычно 3 пикселя)
        bool verbose = true;
        bool debug = false;
        int id = -1;
    };

    // Результаты для одного изображения
    struct ImageMetrics {
        float outNoc = 0.0f; // Процент плохих пикселей (незакрытые области)
        float outAll = 0.0f; // Процент плохих пикселей (все области)
        float avgNoc = 0.0f; // Средняя ошибка (незакрытые области)
        float avgAll = 0.0f; // Средняя ошибка (все области)
        float d1All = 0.0f;  // Метрика D1-all (актуально для 2015)
    };

    // Агрегированные результаты по всему датасету
    struct AggregateMetrics {
        float meanOutNoc = 0.0f;
        float meanOutAll = 0.0f;
        float meanAvgNoc = 0.0f;
        float meanAvgAll = 0.0f;
        float meanD1All = 0.0f;
        std::vector<float> inferenceTimes; // Время выполнения для каждого изображения
    };

    explicit KitDisparityBenchmark(const Config& config);
    
    /**
     * Запускает оценку на всех изображениях тренировочного набора.
     * @param evaluator Функция для вычисления диспарити.
     * @return Агрегированные метрики по датасету.
     */
    AggregateMetrics runBenchmark(std::function<void(const std::vector<cv::UMat>&, cv::Mat&)> evaluator);

    /**
     * Оценка на одном конкретном изображении.
     * @param index Индекс изображения в наборе.
     * @param evaluator Функция для вычисления диспарити.
     * @return Метрики для этого изображения и время выполнения.
     */
    std::pair<ImageMetrics, double> evaluateImage(int index, 
                                                  std::function<void(const std::vector<cv::UMat>&, cv::Mat&)> evaluator);

    // Инициализация путей к данным
    void initializeDatasets();
private:
    Config config_;
    std::vector<std::string> leftImagePaths_;
    std::vector<std::string> rightImagePaths_;
    std::vector<std::string> gtDisparityPaths_;
    
    // Загрузка ground truth диспарити KITTI (16-битное PNG)
    cv::Mat loadGTDisparity(const std::string& path);
    
    // Вычисление метрик для одной пары (предсказание и ground truth)
    ImageMetrics computeMetrics(const cv::Mat& predDisparity, const cv::Mat& gtDisparity);

};