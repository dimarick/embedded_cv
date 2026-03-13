#ifndef EMBEDDED_CV_DEFER_H
#define EMBEDDED_CV_DEFER_H

#include "functional"

namespace ecv {
    /**
     * GO-style defer
     */
    class Defer {
        std::function<void()> defer;
    public:
        explicit Defer(std::function<void()> defer) : defer(std::move(defer)) {}
        ~Defer() {
            defer();
        }
    };
}
#endif //EMBEDDED_CV_DEFER_H
