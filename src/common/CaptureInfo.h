#ifndef EMBEDDED_CV_CAPTUREINFO_H
#define EMBEDDED_CV_CAPTUREINFO_H

#include <vector>
#include <stdexcept>
#include <format>

namespace ecv {
    struct CaptureInfo {
        long created_at = 0;
        size_t size = 0;
        short w = 0;
        short h = 0;
        char channels = 0;

        [[nodiscard]] const CaptureInfo *getNextCaptureInfo() const {
            return reinterpret_cast<const CaptureInfo *>(reinterpret_cast<const char *>(this) + this->size +
                                                         sizeof(CaptureInfo));
        }

        [[nodiscard]] const char *getImageData() const {
            return reinterpret_cast<const char *>(this) + sizeof(CaptureInfo);
        }
    };

    struct CaptureBuffer {
        int bufferSize;
        int nCaptures;

        [[nodiscard]] static const CaptureBuffer *getHeader(const std::vector<char> &buffer) {
            return reinterpret_cast<const CaptureBuffer *>(buffer.data());
        }

        [[nodiscard]] static const CaptureBuffer *getHeader(const void *buffer, size_t size) {
            if (size < sizeof(CaptureBuffer)) {
                return nullptr;
            }
            auto header = reinterpret_cast<const CaptureBuffer *>(buffer);
            if (header->bufferSize > size) {
                throw std::runtime_error(std::format("Buffer header size mismatch: buffer size {}, but header contains {}", size, header->bufferSize));
            }
            return header;
        }

        [[nodiscard]] const CaptureInfo *getFirstCaptureInfo() const {
            if (this->nCaptures == 0) {
                return nullptr;
            }

            return reinterpret_cast<const CaptureInfo *>(reinterpret_cast<const char *>(this) + sizeof(CaptureBuffer));
        }
    };
}


#endif //EMBEDDED_CV_CAPTUREINFO_H
