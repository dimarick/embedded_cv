#include "Process.h"

#include <cstring>
#include <format>
#include <stdexcept>
#include <wait.h>

int Process::run(OnExit _onExit) {
    onExit = std::move(_onExit);
    if (pid > 0) {
        throw std::logic_error("Process already running");
    }

    auto forkPid = fork();
    if (forkPid < 0) {
        throw std::runtime_error(std::format("Fork failed {}", std::strerror(errno)));
    }

    if (forkPid == 0) {
        auto argsCopy = args;
        char *argv[argsCopy.size() + 1];

        for (int i = 0; i < argsCopy.size(); i++) {
            argv[i] = argsCopy[i].data();
        }

        argv[argsCopy.size()] = nullptr;

        execv(path.c_str(), argv);

        perror(std::format("execv {} {}", path, argsCopy).c_str());

        exit(EXIT_FAILURE);
    }

    pid = forkPid;

    waitForExit = std::thread([this] {
        int status;
        while (true) {
            auto result = waitpid(pid, &status, 0);

            if (result == -1) {
                throw std::runtime_error(std::format("Wait failed {}", std::strerror(errno)));
            }

            if (pid > 0) {
                break;
            }
        }

        exitStatus = status;
        onExit(pid, status);
        pid = 0;
    });

    return pid;
}

void Process::stop(int signal) const {
    if (pid > 0 && kill(pid, signal) < 0) {
        throw std::runtime_error(std::format("Kill failed {}", std::strerror(errno)));
    }
}

std::string Process::strExitStatus() const {
    auto status = getExitStatus();

    if (WIFEXITED(status)) {
        return std::format("Child exited with status {}", WEXITSTATUS(status));
    }

    if (WIFSTOPPED(status)) {
        return std::format("Child stopped by signal {} ({})", WSTOPSIG(status), strsignal(WSTOPSIG(status)));
    }

    if (WIFSIGNALED(status)) {
        return std::format("Child killed by signal {} ({})", WTERMSIG(status), strsignal(WTERMSIG(status)));
    }

    return std::format("Unknown child status");
}
