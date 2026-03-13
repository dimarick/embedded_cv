#ifndef EMBEDDED_CV_ENCAPSULATION_H
#define EMBEDDED_CV_ENCAPSULATION_H

#include <memory>
#include <cstring>

namespace mini_server {
    class Encapsulation {
    public:
        template<typename T> static T *getHeader(std::vector<char> &frame) {
            return reinterpret_cast<T *>(frame.data());
        }

        template<typename T> static const char *getBody(const std::vector<char> &frame) {
            return &frame[sizeof (T)];
        }

        template<typename T> static std::vector<char> encapsulate(const void *buffer, size_t bufferSize, const T &header) {
            size_t frameSize = bufferSize + sizeof (T);
            std::vector<char> frame(frameSize);

            T *hdr = getHeader<T>(frame);
            *hdr = header;
            auto body = getBody<T>(frame);
            memcpy((void *) body, buffer, bufferSize);

            return frame;
        }
    };
}
#endif //EMBEDDED_CV_ENCAPSULATION_H
