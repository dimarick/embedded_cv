#include "CalibrateMapper.h"
#include "StatStreaming.h"
#include <ext/pb_ds/assoc_container.hpp>
#ifdef HAVE_OPENCV_HIGHGUI
#include <highgui.hpp>
#include <iostream>
#include <random>

#endif

namespace ecv {
    CalibrateMapper::CalibrateMapper() {
        setPattern(patternSize, skew);
    }

    void CalibrateMapper::setPattern(size_t size, float _skew) {
        size_t sz = size - size % 2;

        if (this->patternSize != sz || this->skew != _skew || this->checkBoardCornerPattern.empty()) {
            auto pattern = cv::Mat((int)sz * 3, (int)sz * 3, CV_8U);
            createCheckBoardPatterns(pattern);
            auto half = (float) sz * 3 / 2;
            std::vector<cv::Point2f> src = {{half, half},
                                            {half, 0},
                                            {0,    half}};
            std::vector<cv::Point2f> dest = {{half,               half},
                                             {half + half * _skew, half * _skew},
                                             {half * _skew,        half - half * _skew}};
            auto rot = cv::getAffineTransform(src, dest);
            cv::warpAffine(pattern, pattern, rot,
                           pattern.size());

            const cv::Rect2i &crop = cv::Rect((int) sz, (int) sz, (int) sz, (int) sz);
            const cv::Mat &mat = pattern(crop).clone();
            mat.convertTo(this->checkBoardCornerPattern, CV_32F);

            this->patternSize = sz;
            this->skew = _skew;
        }
    }

    double CalibrateMapper::detectFrameImagePointsGrid(const cv::Size &frameSize, const std::vector<Point3> &peaks,
                                                       std::vector<Point3> &imageGrid,
                                                       int *w, int *h, cv::Mat &debugFrame) {
        BaseSquare square;
        size_t size = peaks.size();
        drawPeaks(debugFrame, peaks, size, cv::Scalar(0, 255, 0));
        auto squareRmse = detectBaseSquare(frameSize, peaks , square);

        drawBaseSquare(debugFrame, square, squareRmse < 0.2f ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 0, 255));

        auto result = detectFrameImagePointsGrid(frameSize, peaks, square, imageGrid, w, h);

        if (result < 0.3 && !debugFrame.empty()) {
            drawBaseSquare(debugFrame, square, squareRmse < 0.2f ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 0, 255));
            drawGrid(debugFrame, imageGrid, *w, *h, cv::Scalar(255, 0, 0));
        }

        if (result < 0.3 && squareRmse < 0.4 && result < prevError * 1.1 && *w >= 3 && *h >= 3) {
            prevError = result;
            setPattern(suggestPatternSize(imageGrid, square, *w, *h), (float) suggestSkew(imageGrid, *w, *h));
        } else if (result > 5 * prevError || result > 0.5) {
            std::normal_distribution<float> rngSkew(skew, 0.05);
            std::normal_distribution<float> rngSize((float)patternSize, 3);
            std::mt19937 r {std::random_device{}()};
            setPattern(std::clamp((int)rngSize(r), 24, 256), std::clamp((float)rngSkew(r), -0.5f, 0.5f));
            prevError = result;
        }

        return result;
    }

    void CalibrateMapper::detectPeaks(const cv::UMat &frame, std::vector<Point3> &peaks, size_t *size) {
        cv::UMat gray, grayF;
        cv::cvtColor(frame, gray, cv::ColorConversionCodes::COLOR_BGR2GRAY);
        auto clahe = cv::createCLAHE(1, cv::Size(3,3));
        clahe->apply(gray, gray);
        gray.convertTo(grayF, CV_32F);
        auto matches = cv::UMat(gray.size(), CV_32F);

        cv::matchTemplate(grayF, this->checkBoardCornerPattern, matches, cv::TemplateMatchModes::TM_CCOEFF_NORMED, cv::noArray());
        cv::multiply(matches, matches, matches);

        cv::Mat mat;
        matches.copyTo(mat);

        findPeaks(mat, peaks, size, patternSize);
    }

    /**
     * Алгоритм:
     * 1. Выбрать область в центре, содержащую 10-15 точек. Предполагается что на 4 истинных точки приходится не более 6 шумовых
     * 2. Перебрать все комбинации, (худший случай не более 15^4, ожидаемо в среднем не более 3000)
     *    и найти ту, которая дает наиболее правильный квадрат, наиболее близко к центру.
     *
     * Допущения принятые при разработке:
     * - точки расположены преимущественно квадратно-гнездовым
     * - все клетки шахматной доски в центре поля распознаны без пропусков (нет ложно-отрицательных срабатываний)
     * - среди точек присутствуют лишние, обусловленные шумом точки, их количество в худшем случае не более чем в полтора раза превышает истинные
     * - угол наклона доски вокруг любой оси не более 30 градусов в любую сторону.
     * - Весовой коэффициент (z-координата) истинной точки, как правило, выше веса ложной.
     *
     * Что остается неизвестным:
     * - размер клеток
     * - цвет клеток
     * - наклоны доски по трем осям
     * - четкий порог веса между истинными и ложными (шумовыми) точками. Также не известно можно ли достоверно разграничить таким порогом.
     *
     * @param size размер фрейма
     * @param points список найденных точек, преимущественно расположенных квадратно-гнездовым
     * @param result возвращаемое значение, 4 точки угла опорного квадрата
     * @return нормированная оценка качества полученного квадрата. Квадрат считается идеальным, если его стороны и диагонали / sqrt(2) равны.
     */
    double CalibrateMapper::detectBaseSquare(const cv::Size &frameSize, const std::vector<Point3> &peaks, BaseSquare &square) {
        if (peaks.size() < 4) {
            return 1.0;
        }

        auto center = findPointsMassCenter(peaks);

        square = {Point3(0, 0, -1), Point3(0, 0, -1), Point3(0, 0, -1), Point3(0, 0, -1)};

        std::vector<Point3> centralPoints = peaks;

        auto roiSize = (double)std::min(frameSize.width, frameSize.height) / 2;

        struct Rect {
            double x;
            double y;
            double x2;
            double y2;
        };

        size_t nextSize, currentSize;
        currentSize = peaks.size();

        // выберем 15-20 точек вокруг центра масс всех точек для поиска опорного квадрата
        do {
            auto roi = Rect{center.x - roiSize, center.y - roiSize, center.x + roiSize, center.y + roiSize};
            nextSize = 0;
            for (auto i = 0; i < currentSize; i++) {
                auto p = centralPoints[i];
                if (p.x > roi.x && p.x < roi.x2 && p.y > roi.y && p.y < roi.y2) {
                    centralPoints[nextSize++] = p;
                }
            }
            roiSize *= std::sqrt(15 / (double)nextSize);
            currentSize = nextSize;
        } while (nextSize >= 20);

        auto q = 1. / 0.;
        auto qNorm = 1. / 0.;

        for (auto i = 0; i < currentSize; i++) {
            auto p = centralPoints[i];
            if (p.x <= center.x && p.y <= center.y) {
                BaseSquare quad;
                quad.topLeft = p;
                findSquareByTopLeft(centralPoints, currentSize, quad);
                auto [q2, size] = squareQualityNormRms(quad);

                if ((q2 * size) < q) {
                    q = (q2 * size);
                    qNorm = q2;
                    square = quad;
                }
            }
        }

        return qNorm;
    }

    size_t CalibrateMapper::suggestPatternSize(const std::vector<Point3> &imageGrid, const BaseSquare &square, int w, int h) const {
        if (h < 2 || w < 2) {
            return patternSize;
        }
        auto list = {
                square.topRight.x - square.topLeft.x,
                square.bottomLeft.y - square.topLeft.y,

                (imageGrid[1].x - imageGrid[0].x),
                (imageGrid[w - 1].x - imageGrid[w - 2].x),
                (imageGrid[(h - 1) * w + 1].x - imageGrid[(h - 1) * w].x),
                (imageGrid[(h - 1) * w + w - 1].x - imageGrid[(h - 1) * w + w - 2].x),

                (imageGrid[1 * w].y - imageGrid[0].y),
                (imageGrid[1 * w + w - 1].y - imageGrid[w - 1].y),

                (imageGrid[(h - 1) * w].y - imageGrid[(h - 2) * w].y),
                (imageGrid[(h - 1) * w + w - 1].y - imageGrid[(h - 2) * w + w - 1].y),
        };

        double m = 1000;

        for (auto item : list) {
            if (item > 0) {
                m = std::min(m, item);
            }
        }

        auto dSize = (m * 1.4f);

        auto ema = 2. / (3. + 1.);
        dSize = ema * dSize + (1 - ema) * (double)patternSize;

        auto pSize = (size_t)dSize - (size_t)dSize % 2;
        pSize = std::clamp(pSize, (size_t)12, (size_t)256);

        return pSize;
    }

    double CalibrateMapper::suggestSkew(const std::vector<Point3> &imageGrid, int w, int h) const {
        StatStreaming stat;

        stat.addFirstValue(0);

        int cH = h / 2;

        for (int i = cH * w; i < cH * w + w - 1; i++) {
            auto vector = imageGrid[i] - imageGrid[i + 1];
            auto tan = vector.y / vector .x;
            stat.addValue(tan);
        }

        if (std::abs(stat.mean()) < 1) {
            auto ema = 2. / (5. + 1.);
            return ema * stat.mean() + (1 - ema) * skew;
        }

        return skew;
    }

    double
    CalibrateMapper::detectFrameImagePointsGrid(const cv::Size &frameSize, const std::vector<Point3> &peaks, const BaseSquare &square, std::vector<Point3> &imageGrid, int *w, int *h) {
        auto pointsCount = peaks.size();

        if (pointsCount < 4 * 4) {
            return 1.0;
        }

        double sqSizeW = (
                 distance2(square.topLeft, square.topRight) +
                 distance2(square.bottomLeft, square.bottomRight)
         ) / 2;
        double sqSizeH = (
                 distance2(square.topLeft, square.bottomLeft) +
                 distance2(square.bottomRight, square.topRight)
         ) / 2;
        auto pixelsPerPointW = sqSizeW;
        auto pixelsPerPointH = sqSizeH;
        auto _w = (int)(frameSize.width / pixelsPerPointW);
        auto _h = (int)(frameSize.height / pixelsPerPointH);
        auto ratio = (double)_w / (double)_h;

        for (auto &item : imageGrid) {
            item = Point3(0, 0, -1);
        }

        if (_h <= 3 || _w <= 3) {
            return 2.0;
        }

        _h = std::min((_h - 3) * 2, (int)std::sqrt((double)imageGrid.size() / ratio));
        _w = std::min((_w - 3) * 2, (int)std::sqrt((double)imageGrid.size() * ratio));

        auto cH = _h / 2 - 1;
        auto cW = _w / 2 - 1;
        *w = _w;
        *h = _h;

        auto gridMaxOffset = &imageGrid[imageGrid.size() - 1];

        imageGrid[cH * _w + cW] = square.topLeft;
        imageGrid[cH * _w + cW + 1] = square.topRight;
        imageGrid[(cH + 1) * _w + cW] = square.bottomLeft;
        imageGrid[(cH + 1) * _w + cW + 1] = square.bottomRight;

        // заполним 2 строки сетки влево и вправо от центрального квадрата
        for (auto j = 0; j < 2; j++) {
            fillGridRow(_w, cH, cW, j, peaks, imageGrid);
        }

        for (auto i = 0; i < cH; i++) {
            for (auto s = -1; s <= 1; s += 2) { // двигаемся в обе стороны от центра
                // заполним 2 столбца сетки вверх и вниз от центрального квадрата
                for (auto j = 0; j < 2; j++) {
                    auto current = imageGrid[(cH + s * i) * _w + cW + j];
                    auto prev = imageGrid[(cH + s * (i - 1)) * _w + cW + j];
                    auto next = &imageGrid[(cH + s * (i + 1)) * _w + cW + j];
                    CV_Assert(next < gridMaxOffset);

                    auto tp = distance2(current, prev);
                    auto searchRadius = tp * 0.2f;

                    auto nextApproximated = approximate(current, prev);
                    *next = findNearestPoint(nextApproximated, peaks, searchRadius);
                }
                // заполним остальную сетку аппроксимируя по двум точкам
                for (auto k = 0; k < cW; k++) {
                    for (auto ks = -1; ks <= 1; ks += 2) {
                        auto current = imageGrid[(cH + s * i) * _w + cW + ks * k];
                        auto left = imageGrid[(cH + s * (i + 1)) * _w + cW + ks * k];
                        auto top = imageGrid[(cH + s * i) * _w + cW + ks * (k + 1)];
                        auto next = &imageGrid[(cH + s * (i + 1)) * _w + cW + ks * (k + 1)];
                        CV_Assert(next < gridMaxOffset);
                        auto tp = distance2(current, left);

                        auto searchRadius = tp * 0.2f;
                        auto nextApproximated = approximate2(current, left, top);
                        *next = findNearestPoint(nextApproximated, peaks, searchRadius);
                    }
                }
            }
        }
        imageGrid.resize(*w * *h);

        return cropGrid(imageGrid, w, h, findPointsMassCenter(imageGrid));
    }

    /**
     * @tparam TP
     * @param imageGrid
     * @param objectGrid
     * @param w
     * @param h
     * @return возвращает среднюю дистанцию между точкой объекта и точкой на изображении, скорректированную на площадь сетки на изображении.
     */
    double CalibrateMapper::generateFrameObjectPointsGrid(std::vector<Point3> &objectGrid, int w, int h) {
        CV_Assert(w * h <= objectGrid.size());

        std::vector<Point3> o(w * h);

        double sellSize = 1;

        for (int x = 0; x < w; ++x) {
            for (int y = 0; y < h; ++y) {
                objectGrid[y * w + x] = Point3(x, y, 0) * sellSize;
            }
        }

        return 0.0;
    }

    void CalibrateMapper::drawPeaks(cv::Mat &target, const std::vector<Point3> &peaks, size_t size, const cv::Scalar& color) {
        for (int i = 0; i < size; ++i) {
            auto p = peaks[i];
            if (p.z <= 0) {
                continue;
            }
            if (p.x <= 0) {
                continue;
            }

            if (p.y <= 0) {
                continue;
            }

            char info[256];
            snprintf(info, sizeof info - sizeof("\0"), "%d", i);

            cv::drawMarker(target, Point((int)p.x, (int)p.y), color, cv::MarkerTypes::MARKER_TILTED_CROSS, (int)std::clamp(50. * p.z, 1., 100.), 1);
        }
    }
    void CalibrateMapper::drawBaseSquare(cv::Mat &target, const BaseSquare &square, const cv::Scalar& color) {
        cv::drawMarker(target, Point((int) square.topLeft.x, (int) square.topLeft.y), color, cv::MarkerTypes::MARKER_TRIANGLE_UP, 50, 6);
        cv::line(target, Point((int) square.topLeft.x, (int) square.topLeft.y),
                 Point((int) square.topRight.x, (int) square.topRight.y), color, 3);
        cv::line(target, Point((int) square.topLeft.x, (int) square.topLeft.y),
                 Point((int) square.bottomLeft.x, (int) square.bottomLeft.y), color, 3);
        cv::line(target, Point((int) square.topLeft.x, (int) square.topLeft.y),
                 Point((int) square.bottomRight.x, (int) square.bottomRight.y), color, 3);
        cv::line(target, Point((int) square.topRight.x, (int) square.topRight.y),
                 Point((int) square.bottomLeft.x, (int) square.bottomLeft.y), color, 3);
        cv::line(target, Point((int) square.topRight.x, (int) square.topRight.y),
                 Point((int) square.bottomRight.x, (int) square.bottomRight.y), color, 3);
        cv::line(target, Point((int) square.bottomLeft.x, (int) square.bottomLeft.y),
                 Point((int) square.bottomRight.x, (int) square.bottomRight.y), color, 3);

    }

    void CalibrateMapper::drawGrid(const cv::Mat &target, const std::vector<Point3> &grid, int w, int h, const cv::Scalar& color, int thickness) {
        for (int x = 0; x < w; ++x) {
            for (int y = 0; y < h; ++y) {
                auto p = grid[y * w + x];
                auto right = x + 1 == w ? Point3(0, 0, 0): grid[y * w + x + 1];
                auto bottom = y + 1 == h ? Point3(0, 0, 0) : grid[(y + 1) * w + x];

                const auto viewPoint = Point((int)std::round(p.x), (int)std::round(p.y));
                if (x < w - 1) {
                    cv::line(target, viewPoint, Point((int) std::round(right.x), (int) std::round(right.y)),
                             color, thickness);
                }

                if (y < h - 1) {
                    cv::line(target, viewPoint, Point((int) std::round(bottom.x), (int) std::round(bottom.y)),
                             color, thickness);
                }

                if (p.z < 0) {
                    cv::drawMarker(target, viewPoint, cv::Scalar(0, 0, 255), cv::MARKER_DIAMOND, 20, 5);
                }
            }
        }
    }

    void CalibrateMapper::createCheckBoardPatterns(cv::Mat &t1) {
        CV_Assert(t1.cols == t1.rows);
        t1.setTo(cv::Scalar(0));

        auto size = t1.cols / 2;

        auto t1tl = cv::Mat(t1, cv::Rect(0, 0, size, size));
        auto t1br = cv::Mat(t1, cv::Rect(size, size, size, size));
        t1tl.setTo(cv::Scalar(255));
        t1br.setTo(cv::Scalar(255));
    }

    /**
     * kernel size = 3
     *   1 1 1
     *   1 2 1
     *   1 1 1
     * kernel size = 5
     *   0 0 0 0 0
     *   0 1 1 1 0
     *   0 1 2 1 0
     *   0 1 1 1 0
     *   0 0 0 0 0
     */
    void CalibrateMapper::findPeaks(const cv::Mat &mat, std::vector<Point3> &points, size_t *size, size_t kernel) {
        const auto data = (float *)mat.data;

        const cv::Mat mask = cv::Mat::zeros(mat.size(), CV_8SC1);

        const int kernelRadius = (int)kernel / 2;
        const int granule = (int)kernel / 2;
        const int granule2 = granule / 2;
        const int w = mat.cols;
        const auto step = 3;

        std::mutex pointsSetLock;
        std::multiset<Point3, Point3Compare> pointsSet;
        int noiseTolerance = 2;

#pragma omp parallel for default(none) shared(kernelRadius, mat, kernel, w, mask, granule, granule2, data, noiseTolerance, pointsSet, pointsSetLock, points)
        for (int i = kernelRadius; i < mat.rows - kernel; i += step) {
            for (int j = kernelRadius; j < w - kernel; j += step) {
                if (mask.at<char>(i, j) != 0) {
                    continue;
                }
                auto found = true;
                for (int k = 1; k <= granule2; ++k) {
                    auto diagonals = (data[(i + (k - 1)) * w + (j + (k - 1))] < data[(i + k) * w + (j + k)])
                                     + (data[(i + (k - 1)) * w + (j - (k - 1))] < data[(i + k) * w + (j - k)])
                                     + (data[(i - (k - 1)) * w + (j + (k - 1))] < data[(i - k) * w + (j + k)])
                                     + (data[(i - (k - 1)) * w + (j - (k - 1))] < data[(i - k) * w + (j - k)]);

                    auto sideWidth = (k - 1) * 2;

                    auto top = 0;
                    for (int s = 0; s < sideWidth; ++s) {
                        top += data[(i - (k - 1)) * w + j - (k - 1) + s] < data[(i - k) * w + j - (k - 1) + s];
                    }

                    auto bottom = 0;
                    for (int s = 0; s < sideWidth; ++s) {
                        bottom += data[(i + (k - 1)) * w + j - (k - 1) + s] < data[(i + k) * w + j - (k - 1) + s];
                    }

                    auto left = 0;
                    for (int s = 0; s < sideWidth; ++s) {
                        left += data[(i - (k - 1) + s) * w + j - (k - 1)] < data[(i - (k - 1) + s) * w + j - k];
                    }

                    auto right = 0;
                    for (int s = 0; s < sideWidth; ++s) {
                        right += data[(i - (k - 1) + s) * w + j + (k - 1)] < data[(i - (k - 1) + s) * w + j + k];
                    }

                    auto r = diagonals + top + bottom + left + right;

                    if (r > noiseTolerance) {
                        found = false;
                        break;
                    }
                }

                if (found) {
                    const cv::Rect2i &roi = cv::Rect(j - granule2, i - granule2, granule, granule);
                    auto processed = cv::Mat(mask, roi);
                    processed.setTo(127);
                    auto point = findMassCenter(mat, j, i, kernelRadius / 2);
                    j += granule;

                    pointsSetLock.lock();
                    pointsSet.insert(point);
                    while (pointsSet.size() >= points.size()) {
                        pointsSet.erase(std::prev(pointsSet.end()));
                    }
                    pointsSetLock.unlock();
                }
            }
        }

        double zMin = 1.0 / 0.0;
        double zMax = 0.0;
        for (const auto &p : pointsSet) {
            zMin = std::min(p.z, zMin);
            zMax = std::max(p.z, zMax);
        }

        double range = zMax - zMin;

        StatStreaming stat;

        stat.addFirstValue(0);

        for (const auto &p : pointsSet) {
            const auto &z = (p.z - zMin) / range;

            if (z < 0.001) {
                continue;
            }

            auto i = stat.n();

            if (i > 5) {
                if (stat.sigmaValue(z) > 6) {
                    continue;
                }
            }
            stat.addValue(z);

            points[i].z = z;
            points[i].x = p.x + (double)kernel / 2;
            points[i].y = p.y + (double)kernel / 2;

            if (stat.n() > points.size()) {
                break;
            }
        }

        *size = stat.n();
    }

    typename CalibrateMapper::Point3 CalibrateMapper::findMassCenter(const cv::Mat &mat, int x, int y, int searchRadius) {
        auto data = (float *)mat.data;
        auto w = mat.cols;
        auto mass = 0.;
        auto sumX = 0.;
        auto sumY = 0.;
        auto count = 0.;

        for (int i = -searchRadius; i <= searchRadius; ++i) {
            for (int j = -searchRadius; j <= searchRadius; ++j) {
                int offset = (y + i) * w + x + j;

                if (offset < 0) {
                    continue;
                }

                if ((uchar *)(data + offset) >= mat.dataend) {
                    continue;
                }

                auto pixelValue = data[offset];
                if (pixelValue <= 0) {
                    continue;
                }
                mass += pixelValue;
                sumX += (double)pixelValue * (double)j;
                sumY += (double)pixelValue * (double)i;
                count++;
            }
        }

        auto cX = (double)x + (sumX / mass);
        auto cY = (double)y + (sumY / mass);

        auto height = 0.0;
        if (mass > 0) {
            height = mass / (double)searchRadius / (double)searchRadius;
        }

        return {cX, cY, height};
    }

    typename CalibrateMapper::Point3 CalibrateMapper::findPointsMassCenter(const std::vector<Point3> &points) {
        auto mass = 0.;
        auto sumX = 0.;
        auto sumY = 0.;

        for (const auto &p : points) {
            mass += std::max(0., p.z);
            sumX += std::max(0., p.z) * p.x;
            sumY += std::max(0., p.z) * p.y;
        }

        auto cX = sumX > 0 && mass > 0 ? (sumX / mass) : 0;
        auto cY = sumY > 0 && mass > 0 ? (sumY / mass) : 0;

        return {cX, cY, 0};
    }

// Быстрая проверка с использованием Bounding Box
    bool CalibrateMapper::isInsideQuadSimple(const Point3& p, BaseSquare quad) {
        // Считаем сумму углов - если 360°, то точка внутри
        double angleSum = 0;

        Point3 q[4] = {
                quad.topLeft,
                quad.topRight,
                quad.bottomRight,
                quad.bottomLeft,
        };

        for (int i = 0; i < 4; i++) {
            int j = (i + 1) % 4;

            // Векторы от точки к вершинам
            double dx1 = q[i].x - p.x;
            double dy1 = q[i].y - p.y;
            double dx2 = q[j].x - p.x;
            double dy2 = q[j].y - p.y;

            // Угол между векторами через скалярное произведение
            double dot = dx1 * dx2 + dy1 * dy2;
            double cross = dx1 * dy2 - dy1 * dx2;
            angleSum += fabs(atan2(cross, dot));
        }

        // Если сумма углов ~±2π, точка внутри
        return fabs(angleSum - 2 * 3.14) < 0.1;
    }

    double CalibrateMapper::findSquareBy3Points(const std::vector<Point3> &points, size_t size, BaseSquare &result) {
        auto q = 1. / 0.;
        for (int i = 0; i < size; ++i) {
            auto p = points[i];
            if (p.x > result.topLeft.x && p.y > result.topLeft.y) {
                BaseSquare quad;
                quad.topLeft = result.topLeft;
                quad.topRight = result.topRight;
                quad.bottomLeft = result.bottomLeft;
                quad.bottomRight = p;
                auto [q2, sz] = squareQualityNormRms(quad);

                for (int j = 0; j < size; ++j) {
                    auto p3 = points[j];
                    if (p3.x > result.topLeft.x && p3.y > result.topLeft.y) {
                        if (isInsideQuadSimple(p3, quad)) {
                            auto z = p3.z * 2;
                            if (z > result.topLeft.z || z > result.bottomRight.z) {
                                q2 = 1. / 0.;
                            }
                        }
                    }
                }

                if (q2 * sz < q) {
                    q = q2 * sz;
                    result = quad;
                }
            }
        }

        return q;
    }

    double CalibrateMapper::findSquareByTop(const std::vector<Point3> &points, size_t size, BaseSquare &result) {
        auto q = 1. / 0.;
        for (int i = 0; i < size; ++i) {
            auto p = points[i];
            if (p.y > result.topLeft.y && p.y > result.topRight.y && p.x < result.topRight.x && p.x > result.topLeft.x - (result.topRight.x - result.topLeft.x)) {
                BaseSquare quad;
                quad.topLeft = result.topLeft;
                quad.topRight = result.topRight;
                quad.bottomLeft = p;

                auto q2 = findSquareBy3Points(points, size, quad) / p.z;

                if (q2 < q) {
                    q = q2;
                    result = quad;
                }
            }
        }

        return q;
    }

    double CalibrateMapper::findSquareByTopLeft(const std::vector<Point3> &points, size_t size, BaseSquare &result) {
        auto q = 1. / 0.;
        for (int i = 0; i < size; ++i) {
            auto p = points[i];
            auto dp = p - result.topLeft;
            double d = distance2(p, result.topLeft);
            if (d > ((double)patternSize * 0.4) && d < ((double)patternSize * 4) && dp.x > 0 && std::abs(dp.x / dp.y) > 1.5f) {
                BaseSquare quad;
                quad.topLeft = result.topLeft;
                quad.topRight = p;
                auto q2 = d * findSquareByTop(points, size, quad) / p.z;

                if (q2 < q) {
                    q = q2;
                    result = quad;
                }
            }
        }

        return q;
    }

    std::pair<double, double> CalibrateMapper::squareQualityNormRms(const BaseSquare &square) {
        auto left = distance2(square.topLeft, square.bottomLeft);
        auto right = distance2(square.topRight, square.bottomRight);
        auto top = distance2(square.topLeft, square.topRight);
        auto bottom = distance2(square.bottomLeft, square.bottomRight);
        auto d1 = (double)(distance2(square.topLeft, square.bottomRight) / sqrt(2));
        auto d2 = (double)(distance2(square.bottomLeft, square.topRight) / sqrt(2));
        auto avg = (left + right + top + bottom + d1 + d2) / 6;
        auto min = std::min({left, right, top, bottom, d1, d2});
        auto max = std::max({left, right, top, bottom, d1, d2});
        if (avg == 0) {
            return {(double)1.0, max};
        }
        auto cost = (
                std::pow(avg - left, 2) + std::pow(avg - right, 2) +
                std::pow(avg - top, 2) + std::pow(avg - bottom, 2) +
                std::pow(avg - d1, 2) + std::pow(avg - d2, 2)
        ) / 6;



        return {sqrt(cost) / (min + 1e-16), max};
    }

    typename CalibrateMapper::Point3 CalibrateMapper::approximate(Point3 current, Point3 prev) {
        return {current.x + (current.x - prev.x), current.y + (current.y - prev.y), (current.z + prev.z) / 2};
    }

    /**
     * Находит вероятное расположение следующей (слева-сверху) точки, по текущей, точке слева и точке сверху,
     * такое что. Эта точка есть сумма векторов current + (left - current) + (top - current)
     *
     *
     * @param current
     * @param prev
     * @param prevVertical
     * @return
     */
    typename CalibrateMapper::Point3 CalibrateMapper::approximate2(Point3 current, Point3 left, Point3 top) {
        auto x = top.x + (left.x - current.x);
        auto y = left.y + (top.y - current.y);

        return {x, y, -1};
    }

    /**
     * Ищет прямоугольник стремясь забрать максимум площади за наименьшую стоимость
     * @param grid
     * @param w
     * @param h
     * @param center
     */
    double CalibrateMapper::cropGrid(std::vector<Point3> &grid, int *w, int *h, Point3 center) const {
        auto centerId = findNearestPointId(center, grid, 100);

        if (centerId < 0) {
            return 1.;
        }

        if (*w < 4 || *h < 4) {
            return 1.;
        }

        int top = 1, left = 1, right = 1, bottom = 1;
        int cW = centerId % *w;
        int cH = centerId / *w;

        auto cost = [&grid, cH, cW, w, h](int top, int left, int right, int bottom) {
            return getGridCost(grid,
                               *w,
                               *h,
                               std::clamp(cH - top, 0, *h),
                               std::clamp(cW - left, 0, *w),
                               std::clamp(*h - cH - bottom - 1, 0, *h),
                               std::clamp(*w - cW - right - 1, 0, *w));
        };

        auto threshold = 2.;
        auto Q = 1.;
        StatStreaming stat;

        stat.addFirstDValue(0);

        auto q = cost(top, left, right, bottom);
        while (top + bottom < *h && left + right < *w) {
            int i = 0;

            auto qt = cost(top + 1, left, right, bottom);
            if ((qt - q) / q < threshold) {
                top++;
                i++;
                q = std::max(q, qt);
            }


            auto ql = cost(top, left + 1, right, bottom);

            if ((ql - q) / q < threshold) {
                left++;
                i++;
                q = std::max(q, ql);
            }


            auto qr = cost(top, left, right + 1, bottom);

            if ((qr - q) / q < threshold) {
                right++;
                i++;
                q = std::max(q, qr);
            }


            auto qb = cost(top, left, right, bottom + 1);

            if ((qb - q) / q < threshold) {
                bottom++;
                i++;
                q = std::max(q, qb);
            }

            if (i == 0 || q > 0.1) {
                break;
            }
            Q = q;
        };

        auto w0 = *w;

        top = std::clamp(top, 0, cH);
        bottom = std::clamp(bottom, 0, *h - cH - 1);
        left = std::clamp(left, 0, cW);
        right = std::clamp(right, 0, *w - cW - 1);

        *w = right + left + 1;
        *h = bottom + top + 1;

        top = cH - top;
        left = cW - left;

        for (int y = 0; y < *h; ++y) {
            for (int x = 0; x < *w; ++x) {
                grid[y * *w + x] = grid[(y + top) * w0 + x + left];
            }
        }

        return Q;
    }

    /**
     * Функция стоимости: средняя относительная ошибка экстраполяции следующей точки по 2 соседним, верхняя оценка 2 сигмы (99+%)
     * @param grid
     * @param w
     * @param h
     * @param top
     * @param left
     * @param bottom
     * @param right
     * @return grid cost
     */
    double CalibrateMapper::getGridCost(const std::vector<Point3> &grid, int w, int h, int top, int left, int bottom, int right) {
        StatStreaming err;

        err.addFirstValue(0);
        for (int y = top; y < h - bottom; ++y) {
            for (int x = left; x < w - right; ++x) {
                auto p = grid[y * w + x];
                int sy = y < h - bottom - 1 ? 1 : -1;
                int sx = x < w - right - 1 ? 1 : -1;
                if (p.z < 0) {
                    err.addValue(1.);
                    continue;
                }
                auto prediction = approximate2(p, grid[y * w + x + sx], grid[(y + sy) * w + x]);
                auto real = grid[(y + sy) * w + x + sx];
                double norm = distance2(p, prediction);
                err.addValue(distance2(prediction, real) / norm * 0.5 + std::abs(real.z - p.z) * 0.5);
            }
        }

        return err.mean() + err.stddev() * 3;
    }

    void CalibrateMapper::fillGridRow(size_t w, size_t cH, size_t cW, int j, const std::vector<Point3> &peaks, std::vector<Point3> &grid) {
        for (auto i = 0; i < cW; i++) {
            for (auto s = -1; s <= 1; s += 2) {
                if (cW + s * i >= w || cW + s * i < 0) {
                    continue;
                }
                auto current = grid[(cH + j) * w + cW + s * i];
                auto prev = grid[(cH + j) * w + cW + s * (i - 1)];
                auto next = &grid[(cH + j) * w + cW + s * (i + 1)];
                auto searchRadius = distance2(current, prev) * 0.3f;
                auto nextApproximated = approximate(current, prev);
                *next = findNearestPoint(nextApproximated, peaks, searchRadius);
            }
        }
    }

    CalibrateMapper::Point3 CalibrateMapper::findNearestPoint(const Point3 &point, const std::vector<Point3> &points,
                                                                       double searchRadius) const {
        auto i = findNearestPointId(point, points, searchRadius);

        if (i < 0) {
            return {point.x, point.y, -1};
        }

        return points[i];
    }

    int CalibrateMapper::findNearestPointId(const Point3 &point, const std::vector<Point3> &points, double searchRadius) const {
        auto found = -1;
        auto foundDistance = 1e6;
        for (int i = 0; i < points.size(); i++) {
            auto p = points[i];
            if (std::abs(p.x - point.x) > searchRadius || std::abs(p.y - point.y) > searchRadius || p.z < 0) {
                continue;
            }

            auto d = distance2(p, point);

            if (d < foundDistance) {
                foundDistance = d;
                found = i;
            }
        }

        if (found < 0) {
            return -1;
        }

        return found;
    }

    double CalibrateMapper::distance2(Point3 p1, Point3 p2) {
        return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
    }

    double CalibrateMapper::sign(double val) const {
        return double((double(0) < val) - (val < double(0)));
    }
} // ecv