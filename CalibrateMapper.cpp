#include "CalibrateMapper.h"
#include <Eigen/Geometry>
#include <highgui.hpp>

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
            cv::imshow("pattern", this->checkBoardCornerPattern);
        }
    }

    template<typename TP>
    TP CalibrateMapper<TP>::detectFrameImagePointsGrid(const cv::Mat &frame, std::vector<Point3> &imageGrid,
                                                       size_t *w, size_t *h, cv::Mat &debugFrame) {
        std::vector<Point3> peaks(imageGrid.size());
        size_t size = peaks.size();
        BaseSquare square;
        detectPeaks(frame, peaks, &size);
        auto squareRmse = detectBaseSquare(frame.size(), peaks, square);

        if (std::isnan(squareRmse) || squareRmse > 0.4) {
            *w = 0;
            *h = 0;

            prevSquareRmse *= 1.2;
            setPattern(prevPatternSize, prevSkew);

            return std::nan("1");
        }

        auto result = detectFrameImagePointsGrid(frame.size(), peaks, square, imageGrid, w, h);

        if (!debugFrame.empty()) {
            drawPeaks(debugFrame, peaks, size, cv::Scalar(0, 255, 0));
            drawBaseSquare(debugFrame, square, squareRmse < 0.1f ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 0, 255));
            drawGrid(debugFrame, imageGrid, *w, *h, cv::Scalar(255, 0, 0));
        }

        if (squareRmse < prevSquareRmse && !std::isnan(result) && (*w * *h) > 0) {
            prevPatternSize = patternSize;
            prevSkew = skew;
            prevSquareRmse = squareRmse;
            setPattern(suggestPatternSize(imageGrid, square, *w, *h), suggestSkew(square));
        } else {
            setPattern(prevPatternSize, prevSkew);
            prevSquareRmse *= 1.2;
        }


        return result;
    }

    template<typename TP>
    TP CalibrateMapper<TP>::detectFrameImagePointsGrid(const cv::Mat &frame, std::vector<Point3> &imageGrid,
                                                       size_t *w, size_t *h) {
        cv::Mat nullFrame;

        return detectFrameImagePointsGrid(frame, imageGrid, w, h, nullFrame);
    }

    template<typename TP> void CalibrateMapper<TP>::detectPeaks(const cv::Mat &frame, std::vector<Point3> &peaks, size_t *size) {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::ColorConversionCodes::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);
        auto matches = cv::Mat(gray.size(), CV_32F);
        cv::matchTemplate(gray, this->checkBoardCornerPattern, matches, cv::TemplateMatchModes::TM_CCOEFF_NORMED, cv::noArray());
        cv::multiply(matches, matches, matches);
        auto halfPattern = (int)patternSize / 2;
        cv::copyMakeBorder(matches, matches, halfPattern, halfPattern, halfPattern, halfPattern, cv::BorderTypes::BORDER_REPLICATE);

        findPeaks(matches, peaks, size, halfPattern, 2);
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

        auto center = Point3((TP)frameSize.width / 2, (TP)frameSize.height / 2, 0);

        square = {Point3(0, 0, -1), Point3(0, 0, -1), Point3(0, 0, -1), Point3(0, 0, -1)};

        std::vector<Point3> centralPoints = peaks;

        auto roiSize = (TP)std::min(frameSize.width, frameSize.height) / 4;

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

        for (auto i = 0; i < currentSize; i++) {
            auto p = centralPoints[i];
            if (p.x <= center.x && p.y <= center.y) {
                BaseSquare quad;
                quad.topLeft = p;
                auto q2 = std::sqrt(distance(p, center)) * findSquareByTopLeft(centralPoints, currentSize, quad) / p.z / p.z;

                if (q2 < q) {
                    q = q2;
                    square = quad;
                }
            }
        }

        return squareQualityNormRms(square);
    }

    template<typename TP> size_t CalibrateMapper<TP>::suggestPatternSize(const std::vector<Point3> &imageGrid, const BaseSquare &square, size_t w, size_t h) {
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
        auto pSize = (size_t)(std::min(list) * 0.9f);
        pSize -= pSize % 2;
        pSize = std::min((size_t)256, std::max((size_t)24, pSize));

        return pSize;
    }

    template<typename TP> TP CalibrateMapper<TP>::detectFrameImagePointsGrid(const cv::Size &frameSize, const std::vector<Point3> &peaks, const BaseSquare &square, std::vector<Point3> &imageGrid, size_t *w, size_t *h) {
        auto pointsCount = peaks.size();
        auto imageArea = frameSize.width * frameSize.height;
        auto pixelsPerPoint = std::sqrt(imageArea / pointsCount);
        auto _w = (size_t)(frameSize.width / pixelsPerPoint);
        auto _h = (size_t)(frameSize.height / pixelsPerPoint);
        *w = _w;
        *h = _h;

        CV_Assert(_w * _h <= imageGrid.size());
        CV_Assert(!frameSize.empty());

        if (pointsCount < 4 * 4) {
            return 1.0 / 0.0;
        }

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

                    auto searchRadius = distance(current, prev) * 0.2f;
                    auto nextApproximated = approximate(current, prev);
                    *next = findNearestPoint(nextApproximated, peaks, searchRadius);
                    err += std::pow(distance(*next, nextApproximated), 2);
                }
                for (auto k = 0; k < cW; k++) {
                    for (auto ks = -1; ks <= 1; ks += 2) {
                        auto current = imageGrid[(cH + s * i) * _w + cW + ks * k];
                        auto left = imageGrid[(cH + s * (i + 1)) * _w + cW + ks * k];
                        auto top = imageGrid[(cH + s * i) * _w + cW + ks * (k + 1)];
                        auto next = &imageGrid[(cH + s * (i + 1)) * _w + cW + ks * (k + 1)];
                        CV_Assert(next < gridMaxOffset);
                        auto searchRadius = distance(current, left) * 0.2f;
                        auto nextApproximated = approximate2(current, left, top);
                        *next = findNearestPoint(nextApproximated, peaks, searchRadius);
                        err += std::pow(distance(*next, nextApproximated), 2);
                    }
                }
            }
        }

        cropGrid(imageGrid, w, h);

        return std::sqrt(err / (TP)(_w * _h));
    }

    /**
     * @tparam TP
     * @param frameSize
     * @param imageGrid
     * @param objectGrid
     * @param w
     * @param h
     * @return возвращает среднюю дистанцию между точкой объекта и точкой на изображении, скорректированную на площадь сетки на изображении.
     */
    template<typename TP> TP CalibrateMapper<TP>::generateFrameObjectPointsGrid(const cv::Size &frameSize, const std::vector<Point3> &imageGrid, std::vector<Point3> &objectGrid, size_t w, size_t h) {
        try {
            const auto cW = (int) w / 2;
            const auto cH = (int) h / 2;

            TP rmsE = 0.0;

            CV_Assert(w > 4);
            CV_Assert(h > 4);

            CV_Assert(imageGrid.size() >= w * h);
            CV_Assert(objectGrid.size() >= imageGrid.size());

            const auto c = imageGrid[cH * w + cW];
            objectGrid[cH * w + cW] = c;

            auto w0 = Point3(0, 0, 0);
            auto h0 = Point3(0, 0, 0);
            TP wn = 0;
            TP hn = 0;

            for (int i = -cH + 1; i < cW - 1; i++) {
                if (i == 0) {
                    continue;
                }
                const auto wr = i < 0 ? cW - 1 : -i;
                const auto hr = i < 0 ? i : cH - 1;
                const auto topLeft = imageGrid[(cH - hr) * w + cW - wr];
                const auto topRight = imageGrid[(cH - hr) * w + cW + wr];
                const auto bottomLeft = imageGrid[(cH + hr) * w + cW - wr];
                const auto bottomRight = imageGrid[(cH + hr) * w + cW + wr];

                if (i < 0) {
                    h0 += lineLineIntersection(topLeft, bottomLeft, topRight, bottomRight);
                    CV_Assert(!std::isnan(h0.x));
                    CV_Assert(!std::isnan(h0.y));
                    hn++;
                } else {
                    w0 += lineLineIntersection(topLeft, topRight, bottomLeft, bottomRight);
                    CV_Assert(!std::isnan(w0.x));
                    CV_Assert(!std::isnan(w0.y));
                    wn++;
                }
            }

            CV_Assert(wn > 0);
            CV_Assert(hn > 0);

            w0 /= wn;
            h0 /= hn;

            const auto top = imageGrid[(cH - 1) * w + cW];
            const auto left = imageGrid[cH * w + cW - 1];
            const auto bottom = imageGrid[(cH + 1) * w + cW];
            const auto right = imageGrid[cH * w + cW + 1];

            const auto gridWx = (right.x - left.x) / 2;
            const auto gridWy = (right.y - left.y) / 2;
            const auto gridHx = (bottom.x - top.x) / 2;
            const auto gridHy = (bottom.y - top.y) / 2;

            const auto gridWScale = distance(c, w0) / (distance(c, w0) + distance(left, right) / 2);
            const auto gridHScale = distance(c, h0) / (distance(c, h0) + distance(top, bottom) / 2);
            const auto gridWSign = -sign(c.x - w0.x);
            const auto gridHSign = -sign(c.y - h0.y);

            for (auto x0 = 0; x0 < w; ++x0) {
                for (auto y0 = 0; y0 < h; ++y0) {
                    auto dx = (TP) (x0 - cW);
                    auto wScale = std::pow(gridWScale, gridWSign * dx);
                    auto xw = c.x + dx * gridWx * wScale;
                    auto xh = c.y + dx * gridWy * wScale;

                    auto dy = (TP) (y0 - cH);
                    auto hScale = std::pow(gridHScale, gridHSign * dy);
                    auto yw = c.x + dy * gridHx * hScale;
                    auto yh = c.y + dy * gridHy * hScale;

                    auto p = lineLineIntersection(Point3(xw, xh, 0), h0, Point3(yw, yh, 0), w0);

                    if (std::isnan(p.x) || std::isnan(p.y)) {
                        continue;
                    }

                    objectGrid[y0 * w + x0] = p;
                    rmsE += distance(p, imageGrid[y0 * w + x0]);
                }
            }

            auto gridArea = autoFitGrid(objectGrid, imageGrid, w, h);
            auto imageArea = frameSize.width * frameSize.height;
            auto gridAreaRelative = gridArea / imageArea;

            return rmsE / TP(w * h) / gridAreaRelative;
        } catch (const std::exception &e) {
            setPattern(prevPatternSize, prevSkew);
            prevSquareRmse *= 1.2;
            return std::nan("3");
        }
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

            cv::drawMarker(target, Point((int)p.x, (int)p.y), color, cv::MarkerTypes::MARKER_TILTED_CROSS, 20 * p.z, 2);
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
                auto right = grid[y * w + x + 1];
                auto bottom = grid[(y + 1) * w + x];

                if (p.z < 0) {
                    continue;
                }

                if (right.z > 0 && x < w - 1) {
                    cv::line(target, Point((int) p.x, (int) p.y), Point((int) right.x, (int) right.y),
                             color, thickness);
                }

                if (bottom.z > 0 && y < h - 1) {
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
        auto data = (float *)mat.data;
        size_t p = 0;

        cv::Mat mask = cv::Mat::zeros(mat.size(), CV_8SC1);

        int kernelRadius = (kernel - 1) / 2;
        int w = mat.cols;
        auto step = 1;
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
                    points.at(p) = point;
                    p++;
                    if (p >= *size) {
                        return;
                    }
                }
            }
        }

        *size = p;
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
                auto q2 = squareQualityNormRms(quad) / p.z;

                if (!std::isnan(q2) && q2 < q) {
                    q = q2;
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

                if (!std::isnan(q2) && q2 < q) {
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
            double d = distance(p, result.topLeft);
            if (d > (patternSize * 0.4) && dp.x > 0 && std::abs(dp.x / dp.y) > 1.5f) {
                BaseSquare quad;
                quad.topLeft = result.topLeft;
                quad.topRight = p;
                auto q2 = d * findSquareByTop(points, size, quad) / p.z;

                if (!std::isnan(q2) && q2 < q) {
                    q = q2;
                    result = quad;
                }
            }
        }

        return q;
    }

    template<typename TP> TP CalibrateMapper<TP>::squareQualityNormRms(const BaseSquare &square) {
        auto left = distance(square.topLeft, square.bottomLeft);
        auto right = distance(square.topRight, square.bottomRight);
        auto top = distance(square.topLeft, square.topRight);
        auto bottom = distance(square.bottomLeft, square.bottomRight);
        auto d1 = (TP)(distance(square.topLeft, square.bottomRight) / sqrt(2));
        auto d2 = (TP)(distance(square.bottomLeft, square.topRight) / sqrt(2));
        auto avg = std::max({left + right, top + bottom, d1 + d2}) / 2;

        auto d = (TP)(
                (std::min(left, right) / avg)
                * (std::min(top, bottom) / avg)
                * (std::min(d1, d2) / avg)
                * (std::min(left, top) / std::max(left, top))
        );

        return 1 - d;
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
        auto threshold = 0.7f;

        for (int x = 0; x < *w; ++x) {
            wScore += grid[cH * *w + x].z;
        }

        for (int y = 0; y < *h; ++y) {
            hScore += grid[y * *w + cW].z;
        }

        double wThreshold = wScore * threshold;
        double hThreshold = hScore * threshold;

        for (int y = 0; y < cH - 1; ++y) {
            top = y;
            auto rowScore = 0.0f, rowScore1 = 0.0f;
            for (int x = 0; x < *w; ++x) {
                rowScore += grid[y * *w + x].z;
                rowScore1 += grid[(y + 1) * *w + x].z;
            }

            auto t1 = wThreshold - rowScore;
            auto t2 = rowScore1 - wThreshold;

            if (t1 < t2 && t2 > 0) {
                break;
            } else if (t1 >= t2 && t2 > 0) {
                top++;
                break;
            }
        }

        for (int y = (int)*h; y > cH + 1; --y) {
            bottom = y;
            auto rowScore = 0.0f, rowScore1 = 0.0f;
            for (int x = 0; x < *w; ++x) {
                rowScore += grid[(y - 1) * *w + x].z;
                rowScore1 += grid[(y - 2) * *w + x].z;
            }

            auto t1 = wThreshold - rowScore;
            auto t2 = rowScore1 - wThreshold;

            if (t1 < t2 && t2 > 0) {
                break;
            } else if (t1 >= t2 && t2 > 0) {
                bottom--;
                break;
            }
        }

        for (int x = 0; x < cW - 1; ++x) {
            left = x;
            auto rowScore = 0.0f, rowScore1 = 0.0f;
            for (int y = 0; y < *h; ++y) {
                rowScore += grid[y * *w + x].z;
                rowScore1 += grid[y * *w + x + 1].z;
            }

            auto t1 = hThreshold - rowScore;
            auto t2 = rowScore1 - hThreshold;

            if (t1 < t2 && t2 > 0) {
                break;
            } else if (t1 >= t2 && t2 > 0) {
                left++;
                break;
            }
        }

        for (int x = (int)*w; x > cW + 1; --x) {
            right = x;
            auto rowScore = 0.0f, rowScore1 = 0.0f;
            for (int y = 0; y < *h; ++y) {
                rowScore += grid[y * *w + x - 1].z;
                rowScore1 += grid[y * *w + x - 2].z;
            }

            auto t1 = hThreshold - rowScore;
            auto t2 = rowScore1 - hThreshold;

            if (t1 < t2 && t2 > 0) {
                break;
            } else if (t1 >= t2 && t2 > 0) {
                right--;
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
                auto searchRadius = distance(current, prev) * 0.2f;
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

            auto d = distance(p, point) / p.z / p.z;

            if (d < foundDistance) {
                foundDistance = d;
                found = p;
            }
        }

        if (found.z == 0) {
            return {point.x, point.y, 0};
        }

        return found;
    }


    template<typename TP> TP CalibrateMapper<TP>::autoFitGrid(std::vector<Point3> &grid, const std::vector<Point3> &fitTo, size_t w, size_t h) {
        if (grid.empty()) {
            return 0;
        }

        const auto cW = w / 2;
        const auto cH = h / 2;

        auto c1 = grid[cH * w + cW];
        auto left1 = grid[cH * w + 1];
        auto right1 = grid[cH * w + w - 2];
        auto top1 = grid[w + cW];
        auto bottom1 = grid[(h - 2) * w + cW];

        auto topLeft1 = grid[0];
        auto topRight1 = grid[w - 1];
        auto bottomLeft1 = grid[(h - 1) * w];
        auto bottomRight1 = grid[(h - 1) * w + w - 1];

        auto c2 = fitTo[cH * w + cW];
        auto left2 = fitTo[cH * w + 1];
        auto right2 = fitTo[cH * w + w - 2];
        auto top2 = fitTo[w + cW];
        auto bottom2 = fitTo[(h - 2) * w + cW];

        auto topLeft2 = fitTo[0];
        auto topRight2 = fitTo[w - 1];
        auto bottomLeft2 = fitTo[(h - 1) * w];
        auto bottomRight2 = fitTo[(h - 1) * w + w - 1];

        CV_Assert(right2.x > left2.x);
        CV_Assert(right1.x > left1.x);

        CV_Assert(bottom2.y > top2.y);
        CV_Assert(bottom1.y > top1.y);


        auto scaleX1 = (right2.x - c2.x) / (right1.x - c1.x);
        auto scaleX2 = (topRight2.x - c2.x) / (topRight1.x - c1.x);
        auto scaleX3 = (bottomRight2.x - c2.x) / (bottomRight1.x - c1.x);
        auto scaleX4 = (c2.x - left2.x) / (c1.x - left1.x);
        auto scaleX5 = (c2.x - topLeft2.x) / (c1.x - topLeft1.x);
        auto scaleX6 = (c2.x - bottomLeft2.x) / (c1.x - bottomLeft1.x);
        auto scaleY1 = (bottom2.y - c2.y) / (bottom1.y - c1.y);
        auto scaleY2 = (bottomLeft2.y - c2.y) / (bottomLeft1.y - c1.y);
        auto scaleY3 = (bottomRight2.y - c2.y) / (bottomRight1.y - c1.y);
        auto scaleY4 = (c2.y - top2.y) / (c1.y - top1.y);
        auto scaleY5 = (c2.y - topLeft2.y) / (c1.y - topLeft1.y);
        auto scaleY6 = (c2.y - topRight2.y) / (c1.y - topRight1.y);

        auto scaleX = std::min({scaleX1, scaleX2, scaleX3, scaleX4, scaleX5, scaleX6});
        auto scaleY = std::min({scaleY1, scaleY2, scaleY3, scaleY4, scaleY5, scaleY6});

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                auto p = &grid[y * w + x];
                p->x = (p->x - c1.x) * scaleX + c2.x;
                p->y = (p->y - c1.y) * scaleY + c2.y;
            }
        }

        return distance(right1, left1) * distance(bottom1, top1);
    }

    template<typename TP> typename CalibrateMapper<TP>::Point3 CalibrateMapper<TP>::lineLineIntersection(const Point3 &a1, const Point3 &a2, const Point3 &b1, const Point3 &b2) {
        Eigen::Vector2d av1 ((double)a1.x, (double)a1.y);
        Eigen::Vector2d av2 ((double)a2.x, (double)a2.y);
        Eigen::Vector2d bv1 ((double)b1.x, (double)b1.y);
        Eigen::Vector2d bv2 ((double)b2.x, (double)b2.y);
        auto a = Eigen::Hyperplane<double, 2>::Through(av1, av2);
        auto b = Eigen::Hyperplane<double, 2>::Through(bv1, bv2);

        auto k1 = (a2.x - a1.x) / (a2.y - a1.y);
        auto k2 = (b2.x - b1.x) / (b2.y - b1.y);

        if (std::abs(k1 - k2) < 1e-6) {
            auto x = 1e12 / std::sqrt(k1 * k1 + 1);
            auto y = x / k1;

            return Point3(x, y, 1);
        }

        auto v = a.intersection(b);

        return Point3(v.x(), v.y(), 1);
    }

    template<typename TP> TP CalibrateMapper<TP>::distance(Point3 p1, Point3 p2) {
        return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
    }

    template<typename TP> TP CalibrateMapper<TP>::sign(TP val) {
        return TP((TP(0) < val) - (val < TP(0)));
    }

    template class CalibrateMapper<float>;
    template class CalibrateMapper<double>;
} // ecv