// SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
// Copyright (c) 2026 Dmitrii Kosenok
//
// This file is part of EmbeddedCV.
//
// It is dual-licensed under the terms of the GNU General Public License v3
// and a commercial license. You can choose the license that fits your needs.
// For details, see the LICENSE file in the root of the repository.

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
