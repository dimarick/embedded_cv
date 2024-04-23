#include "CommandServer.h"

#include <string>
#include <stdexcept>
#include <cstring>
#include <unistd.h>
#include <netinet/in.h>
#include <thread>
#include <map>
#include <poll.h>
#include <iostream>

using namespace mini_server;

void CommandServer::interact(int socket, CommandServer *server, int threadId, std::map<int, std::thread> *threads) {

    char input[BUFFER_SIZE];
    struct pollfd fd = { .fd = socket, .events = POLLIN };

    while (server->running) {
        int pollStatus = poll(&fd, 1, 50);
        if (pollStatus < 0) {
            threads->erase(threadId);
            throw std::runtime_error(strerror(errno));
        }

        if (pollStatus == 0) {
            continue;
        }

        bzero(input,sizeof(input));
        auto n = read(socket, input, sizeof(input));

        if (n == BUFFER_SIZE) {
            threads->erase(threadId);
            throw std::runtime_error(strerror(errno));
        }

        if (n == 0) {
            break;
        }

        if (n < 0) {
            threads->erase(threadId);
            throw std::runtime_error(strerror(errno));
        }

        std::string in = input;
        std::string out;

        server->handler->handle(socket, in, out);

        n = write(socket, out.c_str(), out.size());

        if (n < 0) {
            threads->erase(threadId);
            throw std::runtime_error(strerror(errno));
        }
    }

    threads->erase(threadId);
}

void CommandServer::run() {
    struct pollfd fd = { .fd = socket, .events = POLLIN };
    std::map<int, std::thread> threads;
    int threadId = 0;

    running = true;

    while (running) {
        int pollStatus = poll(&fd, 1, 50);
        if (pollStatus < 0) {
            throw std::runtime_error(strerror(errno));
        }

        if (pollStatus == 0) {
            continue;
        }

        int acceptedSocket = accept(socket, nullptr, nullptr);
        threadId++;

        threads.insert({threadId, std::thread(CommandServer::interact, acceptedSocket, this, threadId, &threads)});
    }

    std::cerr << "Performing graceful shutdown of CommandServer..." << std::endl;

    for (auto &thread : threads) {
        thread.second.join();
    }

    std::cerr << "Done" << std::endl;
}
