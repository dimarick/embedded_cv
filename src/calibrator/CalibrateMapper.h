#ifndef EMBEDDED_CV_CALIBRATEMAPPER_H
#define EMBEDDED_CV_CALIBRATEMAPPER_H

#include <cstdlib>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

namespace ecv {
    class CalibrateMapper {
    public:
        typedef cv::Point3_<double> Point3;
        typedef cv::Point_<double> Point;

        struct BaseSquare {
            Point3 topLeft;
            Point3 topRight;
            Point3 bottomLeft;
            Point3 bottomRight;
        };

        struct Point3Compare {
            bool operator()(const Point3& a, const Point3& b) const {
                if (a.z == b.z) return true;
                return a.z > b.z;
            }
        };

        size_t patternSize = 32;
        float skew = 0;
    private:
        double prevError = 1e10;
        cv::UMat checkBoardCornerPattern;
        void createCheckBoardPatterns(cv::Mat &t1);
        void findPeaks(const cv::Mat &mat, std::vector<Point3> &points, size_t *size, size_t kernel);
        Point3 findMassCenter(const cv::Mat &mat, int x, int y, int searchRadius);
        Point3 findPointsMassCenter(const std::vector<Point3> &points);
        std::pair<double, double> squareQualityNormRms(const BaseSquare &square);
        double findSquareBy3Points(const std::vector<Point3> &points, size_t size, BaseSquare &result);
        double findSquareByTop(const std::vector<Point3> &points, size_t size, BaseSquare &result);
        double findSquareByTopLeft(const std::vector<Point3> &points, size_t size, BaseSquare &result);
        Point3 approximate(Point3 current, Point3 prev);
        static Point3 approximate2(Point3 current, Point3 left, Point3 top);
        void fillGridRow(size_t w, size_t cH, size_t cW, int j, const std::vector<Point3> &peaks, std::vector<Point3> &grid);
        [[nodiscard]] Point3 findNearestPoint(const Point3 &point, const std::vector<Point3> &points, double searchRadius) const;
        [[nodiscard]] int findNearestPointId(const Point3 &point, const std::vector<Point3> &points, double searchRadius) const;
        [[nodiscard]] double distance2(Point3 p1, Point3 p2) const;
        [[nodiscard]] double sign(double val) const;
        [[nodiscard]] static double distanceSqr3(Point3 p1, Point3 p2);
        double cropGrid(std::vector<Point3> &grid, int *w, int *h, Point3 center) const;
    public:

        explicit CalibrateMapper();
        // Задает размер шаблона
        void setPattern(size_t patternSize, float _skew);
        void detectPeaks(const cv::UMat &frame, std::vector<Point3> &peaks, size_t *size);
        double detectBaseSquare(const cv::Size &frameSize, const std::vector<Point3> &peaks, BaseSquare &square);
        double detectFrameImagePointsGrid(const cv::Size &frameSize, const std::vector<Point3> &peaks, const BaseSquare &square, std::vector<Point3> &imageGrid, int *w, int *h);
        double detectFrameImagePointsGrid(const cv::UMat &frame, const std::vector<Point3>& peaks, std::vector<Point3> &imageGrid, int *w, int *h, cv::Mat &debugFrame);
        [[nodiscard]] size_t suggestPatternSize(const std::vector<Point3> &imageGrid, const BaseSquare &square, int w, int h) const;
        [[nodiscard]] double suggestSkew(const std::vector<Point3> &imageGrid, int w, int h) const;

        double generateFrameObjectPointsGrid(std::vector<Point3> &objectGrid, int w, int h);
        void drawPeaks(cv::Mat &target, const std::vector<Point3> &peaks, size_t size, const cv::Scalar& color);
        void drawBaseSquare(cv::Mat &target, const BaseSquare &square, const cv::Scalar& color);

        void drawGrid(const cv::Mat &target, const std::vector<Point3> &grid, int w, int h, const cv::Scalar& color, int thickness = 3);
        bool isInsideQuadSimple(const Point3 &p, BaseSquare quad);

        static double getGridCost(std::vector<Point3> &grid, int w, int h, int top = 0, int left = 0, int bottom = 0, int right = 0);
    };
} // ecv

#endif //EMBEDDED_CV_CALIBRATEMAPPER_H
