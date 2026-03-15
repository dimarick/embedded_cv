#ifndef EMBEDDED_CV_DEFER_H
#define EMBEDDED_CV_DEFER_H

#include "functional"

namespace ecv {
    /**
     * GO-style defer
     */
    class Defer {
        std::vector<std::function<void()>> fns;
    public:
        void operator()(std::function<void()> fn) noexcept {
            fns.emplace_back(std::move(fn));
        }
        ~Defer() {
            for (size_t i = fns.size(); i > 0; --i) {
                fns.at(i - 1)();
            }
        }
    };
}

#define use_defer ecv::Defer __defer__storage
#define defer(code) __defer__storage([&]() {code;})

#endif //EMBEDDED_CV_DEFER_H
