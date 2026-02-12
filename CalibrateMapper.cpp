#include "CalibrateMapper.h"
#include <ext/pb_ds/assoc_container.hpp>
#ifdef HAVE_OPENCV_HIGHGUI
#include <highgui.hpp>
#endif

namespace ecv {
    template<typename TP> CalibrateMapper<TP>::CalibrateMapper() {
        setPattern(patternSize, skew);
    }

    template<typename TP> void CalibrateMapper<TP>::setPattern(size_t size, float _skew) {
        size_t sz = size - size % 2;

        if (this->patternSize != sz || this->skew != skew || this->checkBoardCornerPattern.empty()) {
            auto pattern = cv::Mat((int)sz * 3, (int)sz * 3, CV_8U);
            createCheckBoardPatterns(pattern);
            auto half = (float) sz * 3 / 2;
            std::vector<cv::Point2f> src = {{half, half},
                                            {half, 0},
                                            {0,    half}};
            std::vector<cv::Point2f> dest = {{half,               half},
                                             {half + half * skew, half * skew},
                                             {half * skew,        half - half * skew}};
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

    template<typename TP>
    TP CalibrateMapper<TP>::detectFrameImagePointsGrid(const cv::UMat &frame, std::vector<Point3> &imageGrid,
                                                       size_t *w, size_t *h, cv::Mat &debugFrame) {
        std::vector<Point3> peaks(imageGrid.size());
        size_t size = peaks.size();
        BaseSquare square;
        detectPeaks(frame, peaks, &size);
        peaks.resize(size);
        drawPeaks(debugFrame, peaks, size, cv::Scalar(0, 255, 0));
        auto squareRmse = detectBaseSquare(frame.size(), peaks , square);

        if (squareRmse > 0.2) {
            *w = 0;
            *h = 0;

            return 1. / 0.;
        }

        bool updatePattern = false;

        if (squareRmse < prevSquareRmse * 1.01) {
            prevPatternSize = patternSize;
            prevSkew = skew;
            prevSquareRmse = squareRmse;
            prevSquare  = square;
            updatePattern = true;
        } else if (squareRmse > 1.4 * prevSquareRmse) {
            setPattern(64, 0);
            prevSquareRmse = squareRmse;
            prevSquare = square;
            skew = prevSkew;
        }

        auto result = detectFrameImagePointsGrid(frame.size(), peaks, square, imageGrid, w, h);

        if (!debugFrame.empty()) {
            drawBaseSquare(debugFrame, square, squareRmse < 0.1f ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 0, 255));
            drawGrid(debugFrame, imageGrid, *w, *h, cv::Scalar(255, 0, 0));
        }

        if (updatePattern && *w > 4 && *h > 4) {
            setPattern(suggestPatternSize(imageGrid, square, *w, *h), suggestSkew(square));
        }

        return result;
    }

    template<typename TP> void CalibrateMapper<TP>::detectPeaks(const cv::UMat &frame, std::vector<Point3> &peaks, size_t *size) {
        cv::UMat gray, grayf;
        cv::cvtColor(frame, gray, cv::ColorConversionCodes::COLOR_BGR2GRAY);
        auto clahe = cv::createCLAHE(1, cv::Size(3,3));
        clahe->apply(gray, gray);
        gray.convertTo(grayf, CV_32F);
        auto matches = cv::UMat(gray.size(), CV_32F);

        cv::matchTemplate(grayf, this->checkBoardCornerPattern, matches, cv::TemplateMatchModes::TM_CCOEFF_NORMED, cv::noArray());
        cv::multiply(matches, matches, matches);

        cv::Mat mat;
        matches.copyTo(mat);

        cv::imshow("m", mat);

        findPeaks(mat, peaks, size, patternSize, 2);
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
    template<typename TP> TP CalibrateMapper<TP>::detectBaseSquare(const cv::Size &frameSize, const std::vector<Point3> &peaks, BaseSquare &square) {
        if (peaks.size() < 4) {
            return 1.0;
        }

        auto center = findPointsMassCenter(peaks);

        square = {Point3(0, 0, -1), Point3(0, 0, -1), Point3(0, 0, -1), Point3(0, 0, -1)};

        std::vector<Point3> centralPoints = peaks;

        auto roiSize = (TP)std::min(frameSize.width, frameSize.height) / 2;

        struct Rect {
            TP x;
            TP y;
            TP x2;
            TP y2;
        };

        size_t nextSize, currentSize;
        currentSize = peaks.size();

        do {
            auto roi = Rect{center.x - roiSize, center.y - roiSize, center.x + roiSize, center.y + roiSize};
            nextSize = 0;
            for (auto i = 0; i < currentSize; i++) {
                auto p = centralPoints[i];
                if (p.x > roi.x && p.x < roi.x2 && p.y > roi.y && p.y < roi.y2) {
                    centralPoints[nextSize++] = p;
                }
            }
            roiSize *= std::sqrt(15 / (TP)nextSize);
            currentSize = nextSize;
        } while (nextSize >= 20);

        auto q = 1.0f / 0.0f;
        auto qNorm = 1.0f / 0.0f;

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

    template<typename TP> size_t CalibrateMapper<TP>::suggestPatternSize(const std::vector<Point3> &imageGrid, const BaseSquare &square, size_t w, size_t h) {
        if (h < 2 || w < 2) {
            return 64;
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

        TP m = 1000;

        for (auto item : list) {
            if (item > 0) {
                m = std::min(m, item);
            }
        }

        auto pSize = (size_t)(m * 0.8f);
        pSize -= pSize % 2;
        pSize = std::clamp(pSize, (size_t)24, (size_t)256);

        return pSize;
    }

    template<typename TP> TP CalibrateMapper<TP>::detectFrameImagePointsGrid(const cv::Size &frameSize, const std::vector<Point3> &peaks, const BaseSquare &square, std::vector<Point3> &imageGrid, size_t *w, size_t *h) {
        auto pointsCount = peaks.size();

        if (pointsCount < 4 * 4) {
            return 1.0 / 0.0;
        }

        auto imageArea = frameSize.width * frameSize.height;
        auto pixelsPerPointW = (
                distance2(square.topLeft, square.topRight) +
                distance2(square.bottomLeft, square.bottomRight)
        ) / 2;
        auto pixelsPerPointH = (
                distance2(square.topLeft, square.bottomLeft) +
                distance2(square.bottomRight, square.topRight)
        ) / 2;
//        auto pixelsPerPoint = std::sqrt(imageArea / pointsCount);
        auto _w = (size_t)(frameSize.width / pixelsPerPointW);
        auto _h = (size_t)(frameSize.height / pixelsPerPointH);
        auto ratio = (TP)_w / (TP)_h;

        for (int i = 0; i < imageGrid.size(); ++i) {
            imageGrid[i] = Point3(0, 0, -1);
        }

        if (_h <= 3 || _w <= 3) {
            return 1.0 / 0.0;
        }

        _h = std::min((size_t)(_h - 3) * 2, (size_t)std::sqrt(imageGrid.size() / ratio));
        _w = std::min((size_t)(_w - 3) * 2, (size_t)std::sqrt(imageGrid.size() * ratio));

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

        auto err = (TP)0.0;

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

                    err += distance2(*next, nextApproximated) / tp;
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

                        err += distance2(*next, nextApproximated) / tp;
                    }
                }
            }
        }

        cropGrid(imageGrid, w, h);

        return err / (TP)(_w * _h);
    }

    /**
     * @tparam TP
     * @param imageGrid
     * @param objectGrid
     * @param w
     * @param h
     * @return возвращает среднюю дистанцию между точкой объекта и точкой на изображении, скорректированную на площадь сетки на изображении.
     */
    template<typename TP> TP CalibrateMapper<TP>::generateFrameObjectPointsGrid(const std::vector<Point3> &imageGrid,
                                                                                std::vector<Point3> &objectGrid, size_t w, size_t h) {
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

    template<typename TP> bool CalibrateMapper<TP>::isGridValid(const cv::Size &frameSize, const std::vector<Point3> &gridPoints, size_t w, size_t h) {
        if (w < 5 || h < 5 || w > frameSize.width || h > frameSize.height || w * h >= gridPoints.size()) {
            return false;
        }

        return true;
    }

    template<typename TP> void CalibrateMapper<TP>::drawPeaks(cv::Mat &target, const std::vector<Point3> &peaks, size_t size, cv::Scalar color) {
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

            cv::drawMarker(target, Point((int)p.x, (int)p.y), color, cv::MarkerTypes::MARKER_TILTED_CROSS, std::clamp(50. * p.z, 1., 100.), 1);
        }
    }
    template<typename TP> void CalibrateMapper<TP>::drawBaseSquare(cv::Mat &target, const BaseSquare &square, cv::Scalar color) {
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

    template<typename TP> void CalibrateMapper<TP>::drawGrid(const cv::Mat &target, const std::vector<Point3> &grid, size_t w, size_t h, cv::Scalar color, int thickness) {

        for (int x = 0; x < w; ++x) {
            for (int y = 0; y < h; ++y) {
                auto p = grid[y * w + x];
                auto right = x + 1 == w ? Point3(0, 0, 0): grid[y * w + x + 1];
                auto bottom = y + 1 == h ? Point3(0, 0, 0) : grid[(y + 1) * w + x];

                if (x < w - 1) {
                    cv::line(target, Point((int) p.x, (int) p.y), Point((int) right.x, (int) right.y),
                             color, thickness);
                }

                if (y < h - 1) {
                    cv::line(target, Point((int) p.x, (int) p.y), Point((int) bottom.x, (int) bottom.y),
                             color, thickness);
                }
            }
        }
    }

    template<typename TP> void CalibrateMapper<TP>::drawGridCorrelation(cv::Mat &target, const std::vector<Point3> &imageGrid, const std::vector<Point3> &objectGrid, size_t w, size_t h, cv::Scalar color) {
        for (int x = 0; x < w; ++x) {
            for (int y = 0; y < h; ++y) {
                auto p = imageGrid[y * w + x];
                auto p2 = objectGrid[y * w + x];

                cv::line(target, Point((int) p.x, (int) p.y), Point((int) p2.x, (int) p2.y), color, 1);
            }
        }
    }

    template<typename TP> void CalibrateMapper<TP>::createCheckBoardPatterns(cv::Mat &t1) {
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
    template<typename TP> void CalibrateMapper<TP>::findPeaks(const cv::Mat &mat, std::vector<Point3> &points, size_t *size, int kernel, int noiseTolerance) {
        const auto data = (float *)mat.data;

        const cv::Mat mask = cv::Mat::zeros(mat.size(), CV_8SC1);

        const int kernelRadius = (kernel - 1) / 5;
        const int w = mat.cols;
        const auto step = 1;

        std::mutex pointsSetLock;
        std::multiset<Point3, Point3Compare> pointsSet;

#pragma omp parallel for default(shared)
        for (int i = kernelRadius; i < mat.rows - kernel; i += step) {
            for (int j = kernelRadius; j < w - kernel; j += step) {
                if (mask.at<char>(i, j) == 1) {
                    continue;
                }
                auto found = true;
                for (int k = 1; k <= kernelRadius; ++k) {
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
                    const cv::Rect2i &roi = cv::Rect(j - kernelRadius, i - kernelRadius, kernel, kernel);
                    auto processed = cv::Mat(mask, roi);
                    processed.setTo(1);
                    auto point = findMassCenter(mat, j, i, kernelRadius);
                    j+=kernel;

                    pointsSetLock.lock();
                    pointsSet.insert(point);
                    while (pointsSet.size() >= points.size()) {
                        pointsSet.erase(std::prev(pointsSet.end()));
                    }
                    pointsSetLock.unlock();
                }
            }
        }

        TP zMin = 1.0 / 0.0;
        TP zMax = 0.0;
        for (const auto &p : pointsSet) {
            zMin = std::min(p.z, zMin);
            zMax = std::max(p.z, zMax);
        }

        TP range = zMax - zMin;

        int i = 0;
        for (const auto &p : pointsSet) {
            points[i].x = p.x + (TP)kernel / 2;
            points[i].y = p.y + (TP)kernel / 2;
            const auto &z = (p.z - zMin) / range;
            points[i].z = z;
            if (z > 0.01) {
                i++;
            }

            if (i > points.size()) {
                break;
            }
        }

        *size = i;
    }

    template<typename TP> typename CalibrateMapper<TP>::Point3 CalibrateMapper<TP>::findMassCenter(const cv::Mat &mat, int x, int y, int searchRadius) {
        auto data = (float *)mat.data;
        auto w = mat.cols;
        auto mass = 0.0f;
        auto sumX = 0.0f;
        auto sumY = 0.0f;
        auto count = 0.0f;

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
                sumX += pixelValue * (TP)j;
                sumY += pixelValue * (TP)i;
                count++;
            }
        }

        auto cX = (TP)x + (sumX / mass);
        auto cY = (TP)y + (sumY / mass);

        auto height = 0.0f;
        if (mass > 0) {
            height = mass / (TP)searchRadius / (TP)searchRadius;
        }

        return {cX, cY, height};
    }

    template<typename TP> typename CalibrateMapper<TP>::Point3 CalibrateMapper<TP>::findPointsMassCenter(const std::vector<Point3> &points) {
        auto mass = 0.0f;
        auto sumX = 0.0f;
        auto sumY = 0.0f;

        for (const auto &p : points) {
            mass += p.z;
            sumX += p.z * p.x;
            sumY += p.z * p.y;
        }

        auto cX = sumX > 0 && mass > 0 ? (sumX / mass) : 0;
        auto cY = sumY > 0 && mass > 0 ? (sumY / mass) : 0;

        return {cX, cY, 0};
    }

// Быстрая проверка с использованием Bounding Box
    template<typename TP> bool CalibrateMapper<TP>::isInsideQuadSimple(const Point3& p, BaseSquare quad) {
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

    template<typename TP> TP CalibrateMapper<TP>::findSquareBy3Points(const std::vector<Point3> &points, size_t size, BaseSquare &result) {
        auto q = 1.0f / 0.0f;
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
                            auto z = p3.z * 4;
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

    template<typename TP> TP CalibrateMapper<TP>::findSquareByTop(const std::vector<Point3> &points, size_t size, BaseSquare &result) {
        auto q = 1.0f / 0.0f;
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

    template<typename TP> TP CalibrateMapper<TP>::findSquareByTopLeft(const std::vector<Point3> &points, size_t size, BaseSquare &result) {
        auto q = 1.0f / 0.0f;
        for (int i = 0; i < size; ++i) {
            auto p = points[i];
            auto dp = p - result.topLeft;
            double d = distance2(p, result.topLeft);
            if (d > (patternSize * 0.4) && d < (patternSize * 4) && dp.x > 0 && std::abs(dp.x / dp.y) > 1.5f) {
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

    template<typename TP> std::pair<TP, TP> CalibrateMapper<TP>::squareQualityNormRms(const BaseSquare &square) {
        auto left = distance2(square.topLeft, square.bottomLeft);
        auto right = distance2(square.topRight, square.bottomRight);
        auto top = distance2(square.topLeft, square.topRight);
        auto bottom = distance2(square.bottomLeft, square.bottomRight);
        auto d1 = (TP)(distance2(square.topLeft, square.bottomRight) / sqrt(2));
        auto d2 = (TP)(distance2(square.bottomLeft, square.topRight) / sqrt(2));
        auto avg = (left + right + top + bottom + d1 + d2) / 6;
        auto min = std::min({left, right, top, bottom, d1, d2});
        auto max = std::max({left, right, top, bottom, d1, d2});
        if (avg == 0) {
            return {(TP)1.0, max};
        }
        auto cost = (
                std::pow(avg - left, 2) + std::pow(avg - right, 2) +
                std::pow(avg - top, 2) + std::pow(avg - bottom, 2) +
                std::pow(avg - d1, 2) + std::pow(avg - d2, 2)
        ) / 6;



        return {sqrt(cost) / min + 1e-16, max};
    }

    template<typename TP> typename CalibrateMapper<TP>::Point3 CalibrateMapper<TP>::approximate(Point3 current, Point3 prev) {
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
    template<typename TP> typename CalibrateMapper<TP>::Point3 CalibrateMapper<TP>::approximate2(Point3 current, Point3 left, Point3 top) {
        auto x = current.x + (left.x - current.x) + (top.x - current.x);
        auto y = current.y + (left.y - current.y) + (top.y - current.y);

        return {x, y, (current.z + left.z + top.z) / 3};
    }

    template<typename TP> void CalibrateMapper<TP>::cropGrid(std::vector<Point3> &grid, size_t *w, size_t *h) {
        int top = 0, left = 0, right = 0, bottom = 0;
        size_t cW = *w / 2;
        size_t cH = *h / 2;
        int w4 = (int)*w / 4;
        int h4 = (int)*h / 4;

        auto wScore = 0.0f, hScore = 0.0f;
        auto threshold = 0.5f; // чем меньше - тем больше площадь сетки, но тем больше шанс захвата невалидных точек

        for (int x = w4; x < *w - w4; ++x) {
            wScore += std::max((TP)0., grid[cH * *w + x].z);
        }

        for (int y = h4; y < *h - h4; ++y) {
            hScore += std::max((TP)0., grid[y * *w + cW].z);
        }

        double wThreshold = wScore * threshold;
        double hThreshold = hScore * threshold;

        for (int y = 0; y < cH - 1; ++y) {
            top = y;
            auto rowScore = 0.0f;
            for (int x = w4; x < *w - w4; ++x) {
                rowScore += grid[y * *w + x].z;
            }

            if (rowScore > wThreshold) {
                break;
            }
        }

        for (int y = (int)*h - 1; y > cH; --y) {
            bottom = y + 1;
            auto rowScore = 0.0f;
            for (int x = w4; x < *w - w4; ++x) {
                rowScore += grid[y * *w + x].z;
            }

            if (rowScore > wThreshold) {
                break;
            }
        }

        for (int x = 0; x < cW - 1; ++x) {
            left = x;
            auto rowScore = 0.0f;
            for (int y = h4; y < *h - h4; ++y) {
                rowScore += grid[y * *w + x].z;
            }

            if (rowScore > hThreshold) {
                break;
            }
        }

        for (int x = (int)*w - 1; x > cW; --x) {
            right = x + 1;
            auto rowScore = 0.0f;
            for (int y = h4; y < *h - h4; ++y) {
                rowScore += grid[y * *w + x].z;
            }

            if (rowScore > hThreshold) {
                break;
            }
        }

        auto w0 = *w;
        *w = right - left;
        *h = bottom - top;

        for (int y = 0; y < *h; ++y) {
            for (int x = 0; x < *w; ++x) {
                grid[y * *w + x] = grid[(y + top) * w0 + x + left];
            }
        }
    }

    template<typename TP> void CalibrateMapper<TP>::fillGridRow(size_t w, int cH, int cW, int j, const std::vector<Point3> &peaks, std::vector<Point3> &grid) {
        for (auto i = 0; i < cW; i++) {
            for (auto s = -1; s <= 1; s += 2) {
                if (cW + s * i >= w || cW + s * i < 0) {
                    continue;
                }
                auto current = grid[(cH + j) * w + cW + s * i];
                auto prev = grid[(cH + j) * w + cW + s * (i - 1)];
                auto next = &grid[(cH + j) * w + cW + s * (i + 1)];
                auto searchRadius = distance2(current, prev) * 0.2f;
                auto nextApproximated = approximate(current, prev);
                *next = findNearestPoint(nextApproximated, peaks, searchRadius);
            }
        }
    }

    template<typename TP> typename CalibrateMapper<TP>::Point3 CalibrateMapper<TP>::findNearestPoint(const Point3 &point, const std::vector<Point3> &points, TP searchRadius) {
        auto found = Point3(0, 0, 0);
        auto foundDistance = 1e6;
        for (auto p : points) {
            if (std::abs(p.x - point.x) > searchRadius || std::abs(p.y - point.y) > searchRadius) {
                continue;
            }

            auto d = distance2(p, point);

            if (d < foundDistance) {
                foundDistance = d;
                found = p;
            }
        }

        if (found.z <= 0) {
            return {point.x, point.y, -1};
        }

        return found;
    }

    template<typename TP> TP CalibrateMapper<TP>::distance2(Point3 p1, Point3 p2) {
        return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
    }

    template<typename TP> TP CalibrateMapper<TP>::distanceSqr(Point3 p1, Point3 p2) {
        return std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2);
    }

    template<typename TP> TP CalibrateMapper<TP>::sign(TP val) {
        return TP((TP(0) < val) - (val < TP(0)));
    }

    template class CalibrateMapper<float>;
    template class CalibrateMapper<double>;
} // ecv