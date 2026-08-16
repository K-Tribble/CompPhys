#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "calculus/integration.hpp"

#include <cmath>

using calculus::integrate::IntegralResult;
using calculus::integrate::trapezoidal;
using calculus::integrate::simpsons;

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("trapezoidal calculates definite integrals",
          "[integrate][trapezoidal]") {

    SECTION("Constant function") {
        auto result = trapezoidal(
            [](d64 x) { return 2.0; },
            0.0,
            5.0
        );

        REQUIRE(result.converged);
        REQUIRE_THAT(result.value, WithinAbs(10.0, 1e-12));
    }

    SECTION("Linear function") {
        auto result = trapezoidal(
            [](d64 x) { return x; },
            0.0,
            2.0
        );

        // Trapezoidal rule is exact for linear functions.
        REQUIRE(result.converged);
        REQUIRE_THAT(result.value, WithinAbs(2.0, 1e-12));
    }

    SECTION("Quadratic function") {
        auto result = trapezoidal(
            [](d64 x) { return x * x; },
            0.0,
            1.0,
            1e-10
        );

        // Integral of x² from 0 to 1 = 1/3.
        REQUIRE(result.converged);
        REQUIRE_THAT(result.value, WithinAbs(1.0 / 3.0, 1e-9));
    }

    SECTION("Sine function") {
        auto result = trapezoidal(
            [](d64 x) { return std::sin(x); },
            0.0,
            M_PI,
            1e-10
        );

        // Integral of sin(x) from 0 to pi = 2.
        REQUIRE(result.converged);
        REQUIRE_THAT(result.value, WithinAbs(2.0, 1e-9));
    }
}

TEST_CASE("simpsons calculates definite integrals",
          "[integrate][simpsons]") {

    SECTION("Constant function") {
        auto result = simpsons(
            [](d64 x) { return 2.0; },
            0.0,
            5.0
        );

        REQUIRE(result.converged);
        REQUIRE_THAT(result.value, WithinAbs(10.0, 1e-12));
    }

    SECTION("Quadratic function") {
        auto result = simpsons(
            [](d64 x) { return x * x; },
            0.0,
            1.0,
            1e-10
        );

        // Simpson's rule is exact for polynomials up to degree 3.
        REQUIRE(result.converged);
        REQUIRE_THAT(result.value, WithinAbs(1.0 / 3.0, 1e-10));
    }

    SECTION("Cubic function") {
        auto result = simpsons(
            [](d64 x) { return x * x * x; },
            0.0,
            1.0
        );

        // Integral of x³ from 0 to 1 = 1/4.
        REQUIRE(result.converged);
        REQUIRE_THAT(result.value, WithinAbs(0.25, 1e-12));
    }

    SECTION("Sine function") {
        auto result = simpsons(
            [](d64 x) { return std::sin(x); },
            0.0,
            M_PI,
            1e-10
        );

        REQUIRE(result.converged);
        REQUIRE_THAT(result.value, WithinAbs(2.0, 1e-10));
    }
}

TEST_CASE("Simpson's method is more accurate than trapezoidal method",
          "[integrate][convergence]") {

    auto f = [](d64 x) {
        return x * x;
    };

    const d64 exact = 1.0 / 3.0;

    auto trap = trapezoidal(f, 0.0, 1.0, 1e-8);
    auto simp = simpsons(f, 0.0, 1.0, 1e-8);

    const d64 trapError = std::abs(trap.value - exact);
    const d64 simpError = std::abs(simp.value - exact);

    REQUIRE(simpError < trapError);
}

TEST_CASE("Integration returns convergence information",
          "[integrate]") {

    SECTION("Converged result") {
        auto result = trapezoidal(
            [](d64 x) { return x * x; },
            0.0,
            1.0,
            1e-8
        );

        REQUIRE(result.converged);
        REQUIRE(result.numIter > 0);
        REQUIRE(result.finalError >= 0.0);
        REQUIRE(result.finalError < 1e-8);
    }

    SECTION("Maximum iterations can prevent convergence") {
        auto result = trapezoidal(
            [](d64 x) { return std::sin(x); },
            0.0,
            M_PI,
            1e-15,
            1
        );

        REQUIRE_FALSE(result.converged);
        REQUIRE(result.numIter <= 1);
    }
}

TEST_CASE("Integration handles reversed bounds",
          "[integrate][bounds]") {

    SECTION("Trapezoidal") {
        auto result = trapezoidal(
            [](d64 x) { return x; },
            2.0,
            0.0
        );

        REQUIRE(result.converged);
        REQUIRE_THAT(result.value, WithinAbs(-2.0, 1e-12));
    }

    SECTION("Simpson") {
        auto result = simpsons(
            [](d64 x) { return x; },
            2.0,
            0.0
        );

        REQUIRE(result.converged);
        REQUIRE_THAT(result.value, WithinAbs(-2.0, 1e-12));
    }
}