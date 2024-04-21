#ifndef HW_CTL_HANDLER_H
#define HW_CTL_HANDLER_H

#include <cstdlib>
#include <string>

namespace mini_server {
    class HandlerInterface {
    public:
        virtual void handle(int socket, const std::string &in, std::string &out) = 0;
    };
}

#endif //HW_CTL_HANDLER_H
