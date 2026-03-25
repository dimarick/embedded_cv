// SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
// Copyright (c) 2026 Dmitrii Kosenok
//
// This file is part of EmbeddedCV.
//
// It is dual-licensed under the terms of the GNU General Public License v3
// and a commercial license. You can choose the license that fits your needs.
// For details, see the LICENSE file in the root of the repository.

#include "GridPreferredSizeProvider.h"

void ecv::GridPreferredSizeProvider::reset() {
    std::lock_guard lock(mutex);
    gridSizeStat.clear();
}
