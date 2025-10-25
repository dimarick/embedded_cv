//
// Created by dima on 19.10.25.
//

#ifndef EMBEDDED_CV_CALIBRATEMAPPER_H
#define EMBEDDED_CV_CALIBRATEMAPPER_H

#include <cstdlib>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

namespace ecv {
    template<typename TP> class CalibrateMapper {
    public:
        typedef cv::Point3_<TP> Point3;
        typedef cv::Point_<TP> Point;

        struct BaseSquare {
            Point3 topLeft;
            Point3 topRight;
            Point3 bottomLeft;
            Point3 bottomRight;
        };

        size_t patternSize = 64;
        float skew = 0;
    private:
        float prevSquareRmse = 64;
        size_t prevPatternSize = 64;
        float prevSkew = 0;
        cv::Mat checkBoardCornerPattern;
        void createCheckBoardPatterns(cv::Mat &t1);
        void findPeaks(const cv::Mat &mat, std::vector<Point3> &points, size_t *size, int kernel, int noiseTolerance);
        Point3 findMassCenter(const cv::Mat &mat, int x, int y, int searchRadius);
        TP squareQualityNormRms(const BaseSquare &square);
        TP findSquareBy3Points(const std::vector<Point3> &points, size_t size, BaseSquare &result);
        TP findSquareByTop(const std::vector<Point3> &points, size_t size, BaseSquare &result);
        TP findSquareByTopLeft(const std::vector<Point3> &points, size_t size, BaseSquare &result);
        Point3 approximate(Point3 current, Point3 prev);
        Point3 approximate2(Point3 current, Point3 left, Point3 top);
        void cropGrid(std::vector<Point3> &grid, size_t *w, size_t *h);
        void fillGridRow(size_t w, int cH, int cW, int j, const std::vector<Point3> &peaks, std::vector<Point3> &grid);
        Point3 findNearestPoint(const Point3 &point, const std::vector<Point3> &points, TP searchRadius);
        TP autoFitGrid(std::vector<Point3> &grid, const std::vector<Point3> &fitTo, size_t w, size_t h);
        Point3 lineLineIntersection(const Point3 &a1, const Point3 &a2, const Point3 &b1, const Point3 &b2);
        TP distance(Point3 p1, Point3 p2);
        TP sign(TP val);
    public:

        explicit CalibrateMapper();
        // Задает размер шаблона
        void setPattern(size_t patternSize, float _skew);
        void detectPeaks(const cv::Mat &frame, std::vector<Point3> &peaks, size_t *size);
        TP detectBaseSquare(const cv::Size &frameSize, const std::vector<Point3> &peaks, BaseSquare &square);
        TP detectFrameImagePointsGrid(const cv::Size &frameSize, const std::vector<Point3> &peaks, const BaseSquare &square, std::vector<Point3> &imageGrid, size_t *w, size_t *h);
        TP detectFrameImagePointsGrid(const cv::Mat &frame, std::vector<Point3> &imageGrid, size_t *w, size_t *h, cv::Mat &debugFrame);
        TP detectFrameImagePointsGrid(const cv::Mat &frame, std::vector<Point3> &imageGrid, size_t *w, size_t *h);
        size_t suggestPatternSize(const std::vector<Point3> &imageGrid, const BaseSquare &square, size_t w, size_t h);
        TP generateFrameObjectPointsGrid(const cv::Size &frameSize, const std::vector<Point3> &imageGrid, std::vector<Point3> &objectGrid, size_t w, size_t h);
        bool isGridValid(const cv::Size &frameSize, const std::vector<Point3> &gridPoints, size_t w, size_t h);
        void convertTo2dPoints(const std::vector<Point3> &points3d, std::vector<cv::Point2f> &points2d);
        void convertToPlain3dPoints(const std::vector<Point3> &points1, std::vector<cv::Point3f> &points2);

        void drawPeaks(cv::Mat &target, const std::vector<Point3> &peaks, size_t size, cv::Scalar color);
        void drawBaseSquare(cv::Mat &target, const BaseSquare &square, cv::Scalar color);
        void drawGrid(const cv::Mat &target, const std::vector<Point3> &grid, size_t w, size_t h, cv::Scalar color, int thickness = 5);
        void drawGridCorrelation(cv::Mat &target, const std::vector<Point3> &imageGrid, const std::vector<Point3> &objectGrid, size_t w, size_t h, cv::Scalar color);

        auto suggestSkew(BaseSquare square) {
            auto topVector = square.topRight - square.topLeft;
            return topVector.y / topVector .x;
        }
    };

} // ecv

#endif //EMBEDDED_CV_CALIBRATEMAPPER_H
