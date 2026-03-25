// SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
// Copyright (c) 2026 Dmitrii Kosenok
//
// This file is part of EmbeddedCV.
//
// It is dual-licensed under the terms of the GNU General Public License v3
// and a commercial license. You can choose the license that fits your needs.
// For details, see the LICENSE file in the root of the repository.

#ifndef EMBEDDED_CV_STATSTREAMING_H
#define EMBEDDED_CV_STATSTREAMING_H

#include <cmath>

namespace ecv {
    /**
     * Класс для потокового статистического анализа
     * Все операции O(1) по времени и по памяти.
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
         * Для работы со значениями
         */
        void addFirstValue(double v) {
            prev = std::nan("1");
            sum = v;
            count = 0;
            sqSum = v*v;
        }

        /**
         * Для работы со значениями
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

        /** Размер последовательности */
        [[nodiscard]] int n() const {
            return count;
        }

        /** Модуль производной (отклонение от предыдущего) */
        [[nodiscard]] double d(double v) const {
            return std::abs(v - prev);
        }

        /** Среднее */
        [[nodiscard]] double mean() const {
            if (count == 0) {
                return 0;
            }
            return sum / count;
        }

        /** Стандартное отклонение */
        [[nodiscard]] double stddev() const {
            return sqrt(dispersion());
        }

        /** Дисперсия (квадрат стандартного отклонения) */
        [[nodiscard]] double dispersion() const {
            return sqSum / count;
        }

        /** Значение в "сигмах" */
        [[nodiscard]] double sigmaValue(double v) const {
            return std::abs(v - mean()) / stddev() + EPS;
        }

        /** Значение в "сигмах" */
        [[nodiscard]] double sigmaDValue(double v) const {
            return sigmaValue(d(v));
        }

        const double EPS = 1e-12;
    };
} // ecv

#endif //EMBEDDED_CV_STATSTREAMING_H
