// SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
// Copyright (c) 2026 Dmitrii Kosenok
//
// This file is part of EmbeddedCV.
//
// It is dual-licensed under the terms of the GNU General Public License v3
// and a commercial license. You can choose the license that fits your needs.
// For details, see the LICENSE file in the root of the repository.

#include <catch2/catch_test_macros.hpp>
#include "../../src/calibrator/StatStreaming.h"

using namespace ecv;

TEST_CASE("StatStreaming::addValue", "[stat][value]") {

    StatStreaming s;

    s.addFirstValue(1);
    REQUIRE(s.mean() == 1);
    REQUIRE(s.stddev() == 0);
    REQUIRE(s.dispersion() == 0);

    s.addValue(2);
    REQUIRE(s.mean() == 1.5);
    REQUIRE(s.dispersion() == (0 + 0.5*0.5) / 2);
    REQUIRE(s.stddev() == std::sqrt((0 + 0.5*0.5) / 2));

    s.addValue(1.5);
    REQUIRE(s.mean() == 1.5);
    REQUIRE(s.dispersion() == (0 + 0.5*0.5) / 3);
    REQUIRE(s.stddev() == std::sqrt((0 + 0.5*0.5) / 3));

    s.addValue(0.5);
    REQUIRE(s.mean() == 1.25);
    REQUIRE(s.dispersion() == (0 + 0.5*0.5+0.75*0.75) / 4);
    REQUIRE(s.stddev() == std::sqrt((0 + 0.5*0.5+0.75*0.75) / 4));
}

TEST_CASE("StatStreaming::addDValue", "[stat][dvalue]") {

    StatStreaming s;

    s.addFirstDValue(1);
    REQUIRE(s.mean() == 1);
    REQUIRE(s.stddev() == 0);
    REQUIRE(s.dispersion() == 0);

    s.addDValue(3);
    REQUIRE(s.mean() == 1.5);
    REQUIRE(s.dispersion() == (0 + 0.5*0.5) / 2);
    REQUIRE(s.stddev() == std::sqrt((0 + 0.5*0.5) / 2));

    s.addDValue(1.5);
    REQUIRE(s.mean() == 1.5);
    REQUIRE(s.dispersion() == (0 + 0.5*0.5) / 3);
    REQUIRE(s.stddev() == std::sqrt((0 + 0.5*0.5) / 3));

    s.addDValue(2);
    REQUIRE(s.mean() == 1.25);
    REQUIRE(s.dispersion() == (0 + 0.5*0.5+0.75*0.75) / 4);
    REQUIRE(s.stddev() == std::sqrt((0 + 0.5*0.5+0.75*0.75) / 4));

    s.addFirstDValue(1);
    REQUIRE(s.mean() == 1);
    REQUIRE(s.stddev() == 0);
    REQUIRE(s.dispersion() == 0);

    s.addDValue(-2);
    REQUIRE(s.mean() == 1.5);
    REQUIRE(s.dispersion() == (0 + 0.5*0.5) / 2);
    REQUIRE(s.stddev() == std::sqrt((0 + 0.5*0.5) / 2));

    s.addDValue(-0.5);
    REQUIRE(s.mean() == 1.5);
    REQUIRE(s.dispersion() == (0 + 0.5*0.5) / 3);
    REQUIRE(s.stddev() == std::sqrt((0 + 0.5*0.5) / 3));

    s.addDValue(0);
    REQUIRE(s.mean() == 1.25);
    REQUIRE(s.dispersion() == (0 + 0.5*0.5+0.75*0.75) / 4);
    REQUIRE(s.stddev() == std::sqrt((0 + 0.5*0.5+0.75*0.75) / 4));
}