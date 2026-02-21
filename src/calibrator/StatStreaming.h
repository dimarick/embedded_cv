#ifndef EMBEDDED_CV_STATSTREAMING_H
#define EMBEDDED_CV_STATSTREAMING_H

#include <cmath>

namespace ecv {
    /**
     * Класс для потокового статистичекого анализа
     * Все операции O(1) по времени и по памяти
     * Допускается неточность вычисления стддев из-за того что среднее считается не по всей последовательности,
     * а только по "увиденной" ее части
     */
    class StatStreaming {
        double prev = 0;
        double sum = 0;
        double sqSum = 0;
        int count = 0;
    public:
        /**
         * Для работы с значениями
         */
        void addFirstValue(double v) {
            prev = std::nan("1");
            sum = v;
            count = 0;
            sqSum = v*v;
        }

        /**
         * Для работы с значениями
         */
        void addValue(double v) {
            if (!(v > -std::numeric_limits<double>::max() && v < std::numeric_limits<double>::max())) {
                return;
            }
            sum += v;
            count++;
            sqSum += std::pow(v - mean(), 2);
        }

        /**
         * Для работы с модулем производной
         */
        void addFirstDValue(double v) {
            prev = v;
            sum = 0;
            sqSum = 0;
            count = 0;
        }

        /** Для работы с модулем производной */
        void addDValue(double v) {
            if (!(v > -std::numeric_limits<double>::max() && v < std::numeric_limits<double>::max())) {
                return;
            }
            addValue(d(v));
            prev = v;
        }

        /** размер последовательности */
        [[nodiscard]] int n() const {
            return count;
        }

        /** модуль производной (отклонение от предыдущего) */
        [[nodiscard]] double d(double v) const {
            return std::abs(v - prev);
        }

        /** среднее */
        [[nodiscard]] double mean() const {
            return sum / count;
        }

        /** стандартное отклонение */
        [[nodiscard]] double stddev() const {
            return sqrt(dispersion());
        }

        /** дисперсия (квадрат стандартного отклонения) */
        [[nodiscard]] double dispersion() const {
            return sqSum / count;
        }

        /** значение в "сигмах" */
        [[nodiscard]] double sigmaValue(double v) const {
            return std::abs(v - mean()) / stddev() + EPS;
        }

        /** значение в "сигмах" */
        [[nodiscard]] double sigmaDValue(double v) const {
            return sigmaValue(d(v));
        }

        const double EPS = 1e-12;
    };
} // ecv

#endif //EMBEDDED_CV_STATSTREAMING_H
