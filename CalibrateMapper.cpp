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
            this->patternSize = sz;
            this->skew = _skew;
            this->checkBoardCornerPattern = cv::Mat(this->patternSize, this->patternSize, CV_8U);
            createCheckBoardPatterns(this->checkBoardCornerPattern);
            auto half = (float) sz / 2;
            std::vector<cv::Point2f> src = {{half, half},
                                            {half, 0},
                                            {0,    half}};
            std::vector<cv::Point2f> dest = {{half,               half},
                                             {half + half * skew, half * skew},
                                             {half * skew,        half - half * skew}};
            auto rot = cv::getAffineTransform(src, dest);
            cv::warpAffine(this->checkBoardCornerPattern, this->checkBoardCornerPattern, rot,
                           this->checkBoardCornerPattern.size());
        }
    }

    template<typename TP>
    TP CalibrateMapper<TP>::detectFrameImagePointsGrid(const cv::Mat &frame, std::vector<Point3> &imageGrid,
                                                       size_t *w, size_t *h, cv::Mat &debugFrame) {
        std::vector<Point3> peaks(imageGrid.size());
        size_t size = peaks.size();
        BaseSquare square;
        detectPeaks(frame, peaks, &size);
        peaks.resize(size);
        auto squareRmse = detectBaseSquare(frame.size(), peaks , square);

        if (squareRmse > 0.1) {
            *w = 0;
            *h = 0;

            setPattern(prevPatternSize, prevSkew);

            return std::nan("1");
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
            drawPeaks(debugFrame, peaks, size, cv::Scalar(0, 255, 0));
            drawBaseSquare(debugFrame, square, squareRmse < 0.1f ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 0, 255));
            drawGrid(debugFrame, imageGrid, *w, *h, cv::Scalar(255, 0, 0));
        }

        if (updatePattern && *w > 4 && *h > 4) {
            setPattern(suggestPatternSize(imageGrid, square, *w, *h), suggestSkew(square));
            return result;
        }

        return 1. / 0.;
    }

    template<typename TP>
    TP CalibrateMapper<TP>::detectFrameImagePointsGrid(const cv::Mat &frame, std::vector<Point3> &imageGrid,
                                                       size_t *w, size_t *h) {
        cv::Mat nullFrame;

        return detectFrameImagePointsGrid(frame, imageGrid, w, h, nullFrame);
    }

    template<typename TP> void CalibrateMapper<TP>::detectPeaks(const cv::Mat &frame, std::vector<Point3> &peaks, size_t *size) {
        cv::UMat gray;
        cv::cvtColor(frame.getUMat(cv::AccessFlag::ACCESS_READ), gray, cv::ColorConversionCodes::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);
        auto matches = cv::UMat(gray.size(), CV_32F);
        cv::matchTemplate(gray, this->checkBoardCornerPattern, matches, cv::TemplateMatchModes::TM_CCOEFF_NORMED, cv::noArray());
        cv::multiply(matches, matches, matches);
        auto halfPattern = (int)patternSize / 2;
        cv::copyMakeBorder(matches, matches, halfPattern, halfPattern, halfPattern, halfPattern, cv::BorderTypes::BORDER_REPLICATE);

        findPeaks(matches.getMat(cv::AccessFlag::ACCESS_READ), peaks, size, halfPattern, 2);
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
        auto pSize = (size_t)(std::min(list) * 0.8f);
        pSize -= pSize % 2;
        pSize = std::min((size_t)256, std::max((size_t)24, pSize));

        return pSize;
    }

    template<typename TP> TP CalibrateMapper<TP>::detectFrameImagePointsGrid(const cv::Size &frameSize, const std::vector<Point3> &peaks, const BaseSquare &square, std::vector<Point3> &imageGrid, size_t *w, size_t *h) {
        auto pointsCount = peaks.size();

        if (pointsCount < 4 * 4) {
            return 1.0 / 0.0;
        }

        auto imageArea = frameSize.width * frameSize.height;
        auto pixelsPerPoint = std::sqrt(imageArea / pointsCount);
        auto _w = (size_t)(frameSize.width / pixelsPerPoint);
        auto _h = (size_t)(frameSize.height / pixelsPerPoint);
        *w = _w;
        *h = _h;

        CV_Assert(_w * _h <= imageGrid.size());
        CV_Assert(!frameSize.empty());

        for (int i = 0; i < imageGrid.size(); ++i) {
            imageGrid[i] = Point3(0, 0, -1);
        }

        auto gridMaxOffset = &imageGrid[imageGrid.size() - 1];

        auto cH = (int)(_h / 2);
        auto cW = (int)(_w / 2);

        imageGrid[cH * _w + cW] = square.topLeft;
        imageGrid[cH * _w + cW + 1] = square.topRight;
        imageGrid[(cH + 1) * _w + cW] = square.bottomLeft;
        imageGrid[(cH + 1) * _w + cW + 1] = square.bottomRight;

        // заполним 2 строки сетки влево и вправо от центрального квадрата
        for (auto j = 0; j < 2; j++) {
            fillGridRow(_w, cH, cW, j, peaks, imageGrid);
        }

        auto err = (TP)0.0;

        // заполним 2 столбца сетки вверх и вниз от центрального квадрата
        for (auto i = 0; i < cH; i++) {
            for (auto s = -1; s <= 1; s += 2) {
                for (auto j = 0; j < 2; j++) {
                    auto current = imageGrid[(cH + s * i) * _w + cW + j];
                    auto prev = imageGrid[(cH + s * (i - 1)) * _w + cW + j];
                    auto next = &imageGrid[(cH + s * (i + 1)) * _w + cW + j];
                    CV_Assert(next < gridMaxOffset);

                    auto searchRadius = distance2(current, prev) * 0.2f;
                    auto nextApproximated = approximate(current, prev);
                    *next = findNearestPoint(nextApproximated, peaks, searchRadius);

                    err += distanceSqr(*next, nextApproximated);
                }
                for (auto k = 0; k < cW; k++) {
                    for (auto ks = -1; ks <= 1; ks += 2) {
                        auto current = imageGrid[(cH + s * i) * _w + cW + ks * k];
                        auto left = imageGrid[(cH + s * (i + 1)) * _w + cW + ks * k];
                        auto top = imageGrid[(cH + s * i) * _w + cW + ks * (k + 1)];
                        auto next = &imageGrid[(cH + s * (i + 1)) * _w + cW + ks * (k + 1)];
                        CV_Assert(next < gridMaxOffset);
                        auto searchRadius = distance2(current, left) * 0.2f;
                        auto nextApproximated = approximate2(current, left, top);
                        *next = findNearestPoint(nextApproximated, peaks, searchRadius);

                        err += distanceSqr(*next, nextApproximated);
                    }
                }
            }
        }

        cropGrid(imageGrid, w, h);

        return std::sqrt(err / (TP)(_w * _h));
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

        for (int x = 0; x < w; ++x) {
            for (int y = 0; y < h; ++y) {
                o[y * w + x] = Point3(x, y, 1);
            }
        }

        const std::vector<cv::Point2f> src = {
            {(float)o[0 * w + 0].x, (float)o[0 * w + 0].y},
            {(float)o[0 * w + w - 1].x, (float)o[0 * w + w - 1].y},
            {(float)o[(h - 1) * w + 0].x, (float)o[(h - 1) * w + 0].y},
            {(float)o[(h - 1) * w + w - 1].x, (float)o[(h - 1) * w + w - 1].y},
        };
        const std::vector<cv::Point2f> dest = {
                {(float)imageGrid[0 * w + 0].x, (float)imageGrid[0 * w + 0].y},
                {(float)imageGrid[0 * w + w - 1].x, (float)imageGrid[0 * w + w - 1].y},
                {(float)imageGrid[(h - 1) * w + 0].x, (float)imageGrid[(h - 1) * w + 0].y},
                {(float)imageGrid[(h - 1) * w + w - 1].x, (float)imageGrid[(h - 1) * w + w - 1].y},
        };
        auto transform = cv::getPerspectiveTransform(src, dest);

        CV_Assert(transform.type() == CV_64FC1);

        cv::gemm(
                transform,
                cv::Mat(w * h, 3, CV_64FC1, o.data()),
                1,
                cv::Mat(),
                0,
                cv::Mat(3, w * h, CV_64FC1, o.data()),
                cv::GemmFlags::GEMM_2_T
        );

        cv::transpose(cv::Mat(3, w * h, CV_64FC1, o.data()), cv::Mat(w * h, 3, CV_64FC1, objectGrid.data()));

        TP rmse = 0;
        for (int i = 0; i < w * h; ++i) {
            rmse += distanceSqr(objectGrid[i], imageGrid[i]);
            objectGrid[i] /= objectGrid[i].z;
        }

        return std::sqrt(rmse / (w * h));
    }

    template<typename TP> bool CalibrateMapper<TP>::isGridValid(const cv::Size &frameSize, const std::vector<Point3> &gridPoints, size_t w, size_t h) {
        if (w < 5 || h < 5 || w > frameSize.width || h > frameSize.height || w * h >= gridPoints.size()) {
            return false;
        }

        return true;
    }

    template<typename TP> void CalibrateMapper<TP>::convertTo2dPoints(const std::vector<Point3> &points3d, std::vector<cv::Point2f> &points2d) {
        for (int j = 0; j < points2d.size(); ++j) {
            points2d[j] = Point(points3d[j].x, points3d[j].y);
        }
    }

    template<typename TP> void CalibrateMapper<TP>::convertToPlain3dPoints(const std::vector<Point3> &points1, std::vector<cv::Point3f> &points2) {
        for (int j = 0; j < points2.size(); ++j) {
            points2[j] = Point3(points1[j].x, points1[j].y, 0);
        }
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

            cv::drawMarker(target, Point((int)p.x, (int)p.y), color, cv::MarkerTypes::MARKER_TILTED_CROSS, std::max(1., 20. * p.z), 2);
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

        const int kernelRadius = (kernel - 1) / 2;
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

                    if (point.z > 1e-2) {
                        pointsSetLock.lock();
                        pointsSet.insert(point);
                        while (pointsSet.size() >= points.size()) {
                            pointsSet.erase(std::prev(pointsSet.end()));
                        }
                        pointsSetLock.unlock();
                    }

                }
            }
        }

        std::copy(pointsSet.begin(), pointsSet.end(), points.begin());

        *size = pointsSet.size();
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
            if (d > (patternSize * 0.4) && dp.x > 0 && std::abs(dp.x / dp.y) > 1.5f) {
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

        auto wScore = 0.0f, hScore = 0.0f;
        auto threshold = 0.5f;

        for (int x = 2; x < *w - 2; ++x) {
            wScore += std::max((TP)0., grid[cH * *w + x].z);
        }

        for (int y = 2; y < *h - 2; ++y) {
            hScore += std::max((TP)0., grid[y * *w + cW].z);
        }

        double wThreshold = wScore * threshold;
        double hThreshold = hScore * threshold;

        for (int y = 0; y < cH; ++y) {
            top = y;
            auto rowScore = 0.0f;
            for (int x = 2; x < *w - 2; ++x) {
                rowScore += grid[y * *w + x].z;
            }

            if (rowScore > wThreshold) {
                break;
            }
        }

        for (int y = (int)*h; y > cH + 1; --y) {
            bottom = y;
            auto rowScore = 0.0f;
            for (int x = 2; x < *w - 2; ++x) {
                rowScore += grid[(y - 1) * *w + x].z;
            }

            if (rowScore > wThreshold) {
                break;
            }
        }

        for (int x = 0; x < cW; ++x) {
            left = x;
            auto rowScore = 0.0f;
            for (int y = 2; y < *h - 2; ++y) {
                rowScore += grid[y * *w + x].z;
            }

            if (rowScore > hThreshold) {
                break;
            }
        }

        for (int x = (int)*w; x > cW + 1; --x) {
            right = x;
            auto rowScore = 0.0f;
            for (int y = 2; y < *h - 2; ++y) {
                rowScore += grid[y * *w + x - 1].z;
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

        if (found.z == 0) {
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