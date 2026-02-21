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