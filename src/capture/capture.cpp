#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <csignal>
#include <glob.h>
#include <cpptrace/cpptrace.hpp>
#include <common/Defer.h>
#include <IpcServer.h>
#include <SocketFactory.h>
#include <common/CaptureInfo.h>

std::atomic running = true;

void onSignal(int) {
    running = false;
}

int64_t getMicroseconds();

template <class T> size_t writeBuffer(std::vector<char> &buffer, size_t offset, const T &value) {
    if (buffer.size() < offset + sizeof (T)) {
        buffer.resize(offset + sizeof (T));
    }
    *(reinterpret_cast<T *>(buffer.data() + offset)) = value;
    return sizeof (T);
}

template <class T> size_t reserveBuffer(std::vector<char> &buffer, size_t offset, size_t size) {
    if (buffer.size() < offset + sizeof (T) * size) {
        buffer.resize(offset + sizeof (T) * size);
    }
    return sizeof (T) * size;
}

int main(int argc, const char **argv) {
    cpptrace::register_terminate_handler();

    // Установка обработчика сигнала
    struct sigaction sa{};
    sa.sa_handler = onSignal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGHUP, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);

    if (argc < 2) {
        std::cout << "Usage " << argv[0] << " <output>[ input_options path]" << std::endl;
        std::cout << "Input options are:" << std::endl;
        std::cout << "\t--fps\tframes per second" << std::endl;
        std::cout << "\t--size\tframe size like 1920x1080" << std::endl;
        std::cout << "\t--format\tformat" << std::endl;

        return -1;
    }

    const char *output = nullptr;

    std::vector<FILE *> captures(0);
    std::vector<size_t> imageOffsets(0);
    std::vector<size_t> imageSizes(0);
    std::vector<size_t> captureInfoRef(0);
    std::vector<char> captureBuffer(0);

    Defer run([&captures] () {
        for (const auto f : captures) {
            fclose(f);
        }
    });

    size_t offset = 0;

    offset += writeBuffer(captureBuffer, offset, ecv::CaptureBuffer{0});

    int fps = 0;
    std::string size;
    std::string format;
    for (int i = 1; i < argc; ++i) {
        if (output == nullptr) {
            output = argv[i];
            continue;
        }

        if (argv[i][0] == '-' && i < argc - 1) {
            auto param = std::string(argv[i]);
            i++;
            auto value = std::string(argv[i]);
            size_t idx;
            if (param == "--fps") {
                if (fps != 0) {
                    throw std::runtime_error(std::format("--fps can be set only once {} {}", argv[i-1], argv[i]));
                }
                fps = stoi(value, &idx, 10);
            }
            if (param == "--size") {
                size = value;
            }
            if (param == "--format") {
                format = value;
            }
        } else {
            if (fps == 0) {
                throw std::runtime_error(std::format("--fps must be set for first camera {}", argv[i]));
            }
            if (size.empty()) {
                throw std::runtime_error(std::format("Frame size if required for {}", argv[i]));
            }

            std::istringstream s(size);

            short w, h;
            char x;

            s >> w;
            s >> x;
            s >> h;

            if (x != 'x') {
                throw std::runtime_error(std::format("Frame size format invalid: must be AxB, {} given", size));
            }

            const auto captureCommand = std::format(
                    "ffmpeg -loglevel fatal -f v4l2 {} {} {} -hwaccel auto -re -i {} -f rawvideo -filter:v 'format=bgr24' -fflags nobuffer -avioflags direct -",
                    !format.empty() ? std::format("-input_format {}", format) : "",
                    !size.empty() ? std::format("-s {}", size) : "",
                    fps > 0 ? std::format("-framerate {}", fps) : "",
                    argv[i]
            );

            auto file = popen(captureCommand.data(), "r");
            if (file == nullptr) {
                throw std::runtime_error(std::format("Error creating process {} {}", captureCommand, strerror(errno)));
            }

            captures.emplace_back(file);
            captureInfoRef.emplace_back(offset);
            auto imageSize = imageSizes.emplace_back(w * h * 3);
            offset += writeBuffer(captureBuffer, offset, ecv::CaptureInfo{0, imageSize, w, h, 3});
            imageOffsets.emplace_back(offset);
            offset += reserveBuffer<char>(captureBuffer, offset, imageSize);
        }
    }

    if (captures.empty()) {
        throw std::runtime_error(std::format("No captures accepted"));
    }
    writeBuffer(captureBuffer, 0, ecv::CaptureBuffer{(int)captureBuffer.size(), (int)captures.size()});

    mini_server::IpcServer captureServer;
    captureServer.setSocket(mini_server::SocketFactory::createServerSocket(output, 10));
    auto captureServerThread = std::thread([&captureServer](){captureServer.serve();});
    Defer stopServer([&captureServer, &captureServerThread]() {
        captureServer.stop();
        captureServerThread.join();
    });

    size_t captureCount = 0;

    while (running) {
        if (captureServer.getClientsCount() == 0) {
            // Если никто не читает наши данные, то экономим ресурсы.
            // Дочерние процессы заблокируются на пайпе stdout и не будут потреблять ресурсы.
            usleep(300e3);
            captureCount = 0;
            continue;
        }
//        auto now = std::chrono::system_clock::now();
//        std::cout << t(now) << "Waiting for coming frame" << std::endl;
//#pragma omp parallel for default(none) shared(captures, captureInfoRef, imageOffsets, imageSizes, captureBuffer, std::cout, now, coutMutex)
        for (int i = 0; i < captures.size(); i++) {
            const auto &capture = captures[i];
            auto captureInfo = reinterpret_cast<ecv::CaptureInfo *>(captureBuffer.data() + captureInfoRef[i]);
            auto imageOffset = imageOffsets[i];
            auto imageSize = imageSizes[i];

            while (imageSize > 0) {
                auto bytes = fread(captureBuffer.data() + imageOffset, sizeof(char), imageSize, capture);
                if (bytes == 0) {
                    if (feof(capture)) {
                        throw std::runtime_error(std::format("Error ffmpeg pipe: unexpected end of file"));
                    }
                    if (ferror(capture)) {
                        throw std::runtime_error(std::format("Error reading test.bin: ", strerror(errno)));
                    }
                }
                imageOffset += bytes;
                imageSize -= bytes;
            }
            captureInfo->created_at = getMicroseconds();
        }

        if (captureCount > fps / 4) { // пропускаем кадры во всевозможных буферах за четверть секунды
            captureServer.broadcast(captureBuffer.data(), captureBuffer.size());
        }
        captureCount++;
    }

    return 0;
}

int64_t getMicroseconds() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::system_clock::now().time_since_epoch()
                    ).count();
}
