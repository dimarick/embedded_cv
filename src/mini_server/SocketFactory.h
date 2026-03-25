// SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
// Copyright (c) 2026 Dmitrii Kosenok
//
// This file is part of EmbeddedCV.
//
// It is dual-licensed under the terms of the GNU General Public License v3
// and a commercial license. You can choose the license that fits your needs.
// For details, see the LICENSE file in the root of the repository.

#ifndef EMBEDDED_CV_SOCKETFACTORY_H
#define EMBEDDED_CV_SOCKETFACTORY_H

#include <string>
#include <cstring>
#include <unistd.h>
#include <netinet/in.h>
#include <netinet/in.h>
#include <stdexcept>
#include "HandlerInterface.h"

namespace mini_server {
    class SocketFactory {
    public:
        static int createServerSocket(const std::string &name, int maxConnections);
        static int createClientSocket(const std::string &name);
    };
}

#endif //EMBEDDED_CV_SOCKETFACTORY_H
