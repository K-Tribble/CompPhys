#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "calculus/differentiation.hpp"

#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

using calculus::differentiate::DiffScheme;
using calculus::differentiate::firstDerivAt;
using calculus::differentiate::secondDerivAt;
using calculus::differentiate::differentiate;
using calculus::differentiate::secondDeriv;

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("firstDerivAt calculates first derivative",
          "[differentiate]") {

    auto f = [](d64 x) {
        return x * x;
    };

    const d64 x = 2.0;
    const d64 h = 1e-3;

    SECTION("Forward difference") {
        const d64 result =
            firstDerivAt(f, x, h, DiffScheme::Forward);

        // For x^2, forward difference = 4 + h
        REQUIRE_THAT(result, WithinAbs(4.0 + h, 1e-12));
    }

    SECTION("Backward difference") {
        const d64 result =
            firstDerivAt(f, x, h, DiffScheme::Backward);

        // For x^2, backward difference = 4 - h
        REQUIRE_THAT(result, WithinAbs(4.0 - h, 1e-12));
    }

    SECTION("Central difference") {
        const d64 result =
            firstDerivAt(f, x, h, DiffScheme::Central);

        // Central difference gives the exact derivative for x^2.
        REQUIRE_THAT(result, WithinAbs(4.0, 1e-10));
    }

    SECTION("Default scheme is central") {
        const d64 result = firstDerivAt(
            [](d64 x) { return std::sin(x); },
            0.7,
            1e-4
        );

        REQUIRE_THAT(result, WithinAbs(std::cos(0.7), 1e-8));
    }
}

TEST_CASE("secondDerivAt calculates second derivative",
          "[differentiate]") {

    SECTION("Quadratic") {
        const d64 result = secondDerivAt(
            [](d64 x) { return x * x; },
            2.0,
            1e-3
        );

        REQUIRE_THAT(result, WithinAbs(2.0, 1e-9));
    }

    SECTION("Sine") {
        const d64 x = 0.7;

        const d64 result = secondDerivAt(
            [](d64 x) { return std::sin(x); },
            x,
            1e-4
        );

        REQUIRE_THAT(result, WithinAbs(-std::sin(x), 1e-8));
    }
}

TEST_CASE("Finite difference schemes have expected convergence",
          "[differentiate][convergence]") {

    auto f = [](d64 x) {
        return std::sin(x);
    };

    const d64 x = 0.7;
    const d64 exact = std::cos(x);

    const d64 h = 1e-2;

    const d64 forwardError = std::abs(
        firstDerivAt(f, x, h, DiffScheme::Forward) - exact
    );

    const d64 forwardErrorHalf = std::abs(
        firstDerivAt(f, x, h / 2, DiffScheme::Forward) - exact
    );

    const d64 centralError = std::abs(
        firstDerivAt(f, x, h, DiffScheme::Central) - exact
    );

    const d64 centralErrorHalf = std::abs(
        firstDerivAt(f, x, h / 2, DiffScheme::Central) - exact
    );

    // Forward difference is O(h).
    REQUIRE_THAT(
        forwardErrorHalf / forwardError,
        WithinRel(0.5, 2e-3)
    );

    // Central difference is O(h^2).
    REQUIRE_THAT(
        centralErrorHalf / centralError,
        WithinRel(0.25, 1e-3)
    );
}

TEST_CASE("differentiate calculates derivatives of vector data",
          "[differentiate]") {

    std::vector<d64> x{
        1.0,
        1.01,
        1.02,
        1.03,
        1.04
    };

    std::vector<d64> y;

    for (d64 xi : x) {
        y.push_back(xi * xi);
    }

    SECTION("Forward") {
        auto result = differentiate(
            std::span<const d64>(x),
            std::span<const d64>(y),
            DiffScheme::Forward
        );

        REQUIRE(result.size() == x.size());

        // Forward difference:
        // (x+h)^2 - x^2 / h = 2x + h
        REQUIRE_THAT(result[0], WithinAbs(2.01, 1e-12));
        REQUIRE_THAT(result[2], WithinAbs(2.05, 1e-12));
        REQUIRE_THAT(result[4], WithinAbs(2.07, 1e-12));
    }

    SECTION("Backward") {
        auto result = differentiate(
            std::span<const d64>(x),
            std::span<const d64>(y),
            DiffScheme::Backward
        );

        REQUIRE(result.size() == x.size());

        // First point uses forward difference.
        REQUIRE_THAT(result[0], WithinAbs(2.01, 1e-12));

        // Remaining points use backward difference.
        REQUIRE_THAT(result[1], WithinAbs(2.01, 1e-12));
        REQUIRE_THAT(result[2], WithinAbs(2.03, 1e-12));
        REQUIRE_THAT(result[4], WithinAbs(2.07, 1e-12));
    }

    SECTION("Central") {
        auto result = differentiate(
            std::span<const d64>(x),
            std::span<const d64>(y),
            DiffScheme::Central
        );

        REQUIRE(result.size() == x.size());

        // Endpoints use first-order differences.
        REQUIRE_THAT(result[0], WithinAbs(2.01, 1e-12));
        REQUIRE_THAT(result[4], WithinAbs(2.07, 1e-12));

        // Interior points are exact for x².
        REQUIRE_THAT(result[1], WithinAbs(2.02, 1e-12));
        REQUIRE_THAT(result[2], WithinAbs(2.04, 1e-12));
        REQUIRE_THAT(result[3], WithinAbs(2.06, 1e-12));
    }
}

TEST_CASE("secondDeriv calculates second derivative of vector data",
          "[differentiate]") {

    const d64 h = 0.01;

    std::vector<d64> x{
        0.0,
        h,
        2.0 * h,
        3.0 * h,
        4.0 * h
    };

    std::vector<d64> y;

    for (d64 xi : x) {
        y.push_back(xi * xi);
    }

    auto result = secondDeriv(
        std::span<const d64>(x),
        std::span<const d64>(y)
    );

    REQUIRE(result.size() == x.size() - 2);

    // d^2/dx^2 x^2 = 2 exactly.
    for (d64 value : result) {
        REQUIRE_THAT(value, WithinAbs(2.0, 1e-10));
    }
}

TEST_CASE("Differentiation rejects invalid input",
          "[differentiate][validation]") {

    SECTION("Mismatched sizes") {
        std::vector<d64> x{0.0, 1.0, 2.0};
        std::vector<d64> y{0.0, 1.0};

        REQUIRE_THROWS_AS(
            differentiate(x, y),
            std::invalid_argument
        );

        REQUIRE_THROWS_AS(
            secondDeriv(x, y),
            std::invalid_argument
        );
    }

    SECTION("Too few points") {
        std::vector<d64> x{0.0};
        std::vector<d64> y{0.0};

        REQUIRE_THROWS_AS(
            differentiate(x, y),
            std::invalid_argument
        );

        REQUIRE_THROWS_AS(
            secondDeriv(x, y),
            std::invalid_argument
        );
    }

    SECTION("Central difference requires at least three points") {
        std::vector<d64> x{0.0, 0.01};
        std::vector<d64> y{0.0, 0.01};

        REQUIRE_THROWS_AS(
            differentiate(x, y, DiffScheme::Central),
            std::invalid_argument
        );
    }

    SECTION("Central difference rejects uneven spacing") {
        std::vector<d64> x{0.0, 0.01, 0.02, 0.035};
        std::vector<d64> y;

        for (d64 xi : x) {
            y.push_back(xi * xi);
        }

        REQUIRE_THROWS_AS(
            differentiate(x, y, DiffScheme::Central),
            std::invalid_argument
        );

        REQUIRE_THROWS_AS(
            secondDeriv(x, y),
            std::invalid_argument
        );
    }
}