#include "GridPreferredSizeProvider.h"

void ecv::GridPreferredSizeProvider::reset() {
    std::lock_guard lock(mutex);
    gridSizeStat.clear();
}
