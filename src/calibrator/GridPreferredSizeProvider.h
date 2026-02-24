#ifndef EMBEDDED_CV_GRIDPREFERREDSIZEPROVIDER_H
#define EMBEDDED_CV_GRIDPREFERREDSIZEPROVIDER_H

#include <cstddef>
#include <unordered_map>
#include <memory>
#include <algorithm>

namespace ecv {
    /**
     * Определяет предпочтительный размер сетки на основе голосования: предпочитается самая популярная
     */
    class GridPreferredSizeProvider {
    public:
        struct GridStat {
            size_t w;
            size_t h;
            int count;

            GridStat(size_t w, size_t h, int count) : w(w), h(h), count(count) {}
        };
    private:
        mutable std::mutex mutex;
        std::unordered_map<int, std::shared_ptr<GridStat>> gridSizeStat;
        std::shared_ptr<GridStat> gridSizeStatTop;
        void _registerFrameStat(size_t w, size_t h) {
            auto key = (int) w * 1000 + (int) h;
            auto statIt = gridSizeStat.find(key);
            if (statIt == gridSizeStat.end()) {
                if (gridSizeStat.empty()) {
                    gridSizeStatTop = std::shared_ptr<GridStat>(new GridStat(w, h, 1));
                    gridSizeStat.insert({key, gridSizeStatTop});
                } else {
                    gridSizeStat.insert({key, std::shared_ptr<GridStat>(new GridStat(w, h, 1))});
                }
            } else {
                statIt->second->count++;
                if (statIt->second->count > gridSizeStatTop->count) {
                    gridSizeStatTop = statIt->second;
                }
            }
        }

        void _unregisterFrameStat(size_t w, size_t h) {
            auto key = (int) w * 1000 + (int) h;
            const auto &compare = [](const std::pair<int, std::shared_ptr<GridStat>> &a,
                                     const std::pair<int, std::shared_ptr<GridStat>> &b) {
                return a.second->count > b.second->count;
            };

            auto statIt = gridSizeStat.find(key);
            if (statIt != gridSizeStat.end()) {
                statIt->second->count--;
                gridSizeStatTop = std::max_element(gridSizeStat.begin(), gridSizeStat.end(), compare)->second;
            }
        }
    public:
        const std::shared_ptr<GridStat> &getGridPreferredSize() const {
            std::lock_guard lock(mutex);
            return gridSizeStatTop;
        }

        void insertFrameStat(size_t w, size_t h) {
            std::lock_guard lock(mutex);
            _registerFrameStat(w, h);
        }
        void replaceFrameStat(size_t w1, size_t h1, size_t w0, size_t h0) {
            if (w1 == w0 && h1 == h0) {
                return;
            }

            std::lock_guard lock(mutex);
            _registerFrameStat(w1, h1);
            _unregisterFrameStat(w0, h0);
        }
    };
}
#endif //EMBEDDED_CV_GRIDPREFERREDSIZEPROVIDER_H
