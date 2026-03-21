#ifndef EMBEDDED_CV_PROCESSMANAGER_H
#define EMBEDDED_CV_PROCESSMANAGER_H
#include <atomic>
#include <functional>
#include <string>
#include <thread>
#include <vector>
#include <csignal>

class Process {
public:
    typedef std::function<void(int pid, int status)> OnExit;
private:
    std::string path;
    std::vector<std::string> args;
    OnExit onExit;
    std::atomic<int> pid = 0;
    std::atomic<int> exitStatus = 0;
    std::atomic<bool> stopping = false;
    std::thread waitForExit;
public:
    Process() = default;
    Process(std::string path, std::vector<std::string> args) :
        path(std::move(path)), args(std::move(args)) {}

    int run(OnExit _onExit);

    int getExitStatus() const {
        return exitStatus;
    }

    void stop(int signal = SIGINT) const;

    std::string strExitStatus() const;

    ~Process() {
        this->stop(SIGKILL);
        if (waitForExit.joinable()) {
            waitForExit.join();
        }
    }
};

#endif //EMBEDDED_CV_PROCESSMANAGER_H