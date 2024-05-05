#include "CommandServer.h"

#include <string>
#include <stdexcept>
#include <cstring>
#include <unistd.h>
#include <netinet/in.h>
#include <thread>
#include <poll.h>
#include <iostream>
#include <cassert>

using namespace mini_server;

void CommandServer::interact(int socket, CommandServer *server, int threadId) {
    class Finally
    {
        CommandServer *server;
        int threadId;
    public:
        Finally(CommandServer *server, int threadId) : server(server), threadId(threadId) {}
        ~Finally()
        {
            server->deadThreadsMutex.lock();
            server->deadThreads.insert(threadId);
            server->deadThreadsMutex.unlock();
        }
    };

    Finally finally(server, threadId);

    char input[BUFFER_SIZE];
    struct pollfd fd = { .fd = socket, .events = POLLIN };

    while (server->running) {
        int pollStatus = poll(&fd, 1, 50);
        if (pollStatus < 0) {
            throw std::runtime_error(strerror(errno));
        }

        if (pollStatus == 0) {
            continue;
        }

        bzero(input,sizeof(input));
        auto n = read(socket, input, sizeof(input));

        if (errno == EPIPE) {
            break;
        }

        if (n == BUFFER_SIZE) {
            throw std::runtime_error(strerror(errno));
        }

        if (n == 0) {
            break;
        }

        if (n < 0) {
            throw std::runtime_error(strerror(errno));
        }

        std::string in = input;
        std::string out;

        server->handler->handle(socket, in, out);

        if (out.empty()) {
            continue;
        }

        n = write(socket, out.c_str(), out.size());

        if (n < 0) {
            throw std::runtime_error(strerror(errno));
        }
    }
}

void CommandServer::run() {
    struct pollfd fd = { .fd = socket, .events = POLLIN };
    int threadId = 0;

    assert(running == false);

    running = true;

    while (running) {
        int pollStatus = poll(&fd, 1, 50);
        if (pollStatus < 0) {
            throw std::runtime_error(strerror(errno));
        }

        deadThreadsMutex.lock();
        for (const auto &deadThreadId : deadThreads) {
            const auto &iterator = threads.find(deadThreadId);
            iterator->second.join();
            threads.erase(deadThreadId);
            std::cerr << "Thread " << deadThreadId << " is dead. " << threads.size() << " is alive" << std::endl;
        }

        deadThreads.clear();
        deadThreadsMutex.unlock();

        if (pollStatus == 0) {
            continue;
        }

        int acceptedSocket = accept(socket, nullptr, nullptr);

        acceptedSocketsMutex.lock();
        acceptedSockets.insert(acceptedSocket);
        acceptedSocketsMutex.unlock();

        threadId++;

        threads.insert({threadId, std::thread(CommandServer::interact, acceptedSocket, this, threadId)});

        std::cerr << "New thread " << threadId << ". " << threads.size() << " is alive" << std::endl;
    }

    std::cerr << "Performing graceful shutdown of CommandServer..." << std::endl;

    for (auto &thread : threads) {
        thread.second.join();
        threads.erase(thread.first);
        std::cerr << "Thread " << threadId << "is dead. " << threads.size() << " is alive" << std::endl;
    }

    acceptedSocketsMutex.lock();
    acceptedSockets.clear();
    acceptedSocketsMutex.unlock();

    std::cerr << "Done" << std::endl;
}

void CommandServer::broadcast(const std::string &message) {
    acceptedSocketsMutex.lock();
    auto threadSafeAcceptedSockets = acceptedSockets;
    acceptedSocketsMutex.unlock();

    for (auto acceptedSocket : threadSafeAcceptedSockets) {
        auto n = send(acceptedSocket, message.c_str(), message.size(), MSG_NOSIGNAL);

        if (n < 0) {
            perror("ERROR writing to socket");

            acceptedSocketsMutex.lock();
            acceptedSockets.erase(acceptedSocket);
            acceptedSocketsMutex.unlock();

            std::cerr << "socket released: " << acceptedSocket << ". Connections count: " << acceptedSockets.size() << std::endl;
        }
        std::cerr << "Message : " << message << " sent to " << acceptedSocket << std::endl;
    }
}
