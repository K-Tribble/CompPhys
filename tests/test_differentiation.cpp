// Catch2 (v3) test suite for calculus::integration functions
//
// Build notes:
//   - Requires Catch2 v3 (link against Catch2::Catch2WithMain) plus the
//     linalg library sources (matrix.cpp, vec.cpp, linalg_interop.cpp).
//   - Include paths must expose both "calculus/integration.hpp" and the
//     top-level "types.hpp" / "constants.hpp" used by those headers.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/catch_approx.hpp>

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

TEST_CASE("Gradient of multivariable quadratic matches analytic result", "[differentiation][gradient]") {
    // f(x, y, z) = x^2 + 2y^2 + 3z^2  =>  grad f = (2x, 4y, 6z)
    auto f = [](const linalg::Vec<d64>& v) {
        return v(0)*v(0) + 2*v(1)*v(1) + 3*v(2)*v(2);
    };

    linalg::Vec<d64> point{1.0, 2.0, 3.0};
    auto gradient = calculus::differentiate::grad(f, point, 1e-5);

    REQUIRE(gradient.size() == 3);
    REQUIRE(gradient(0) == Catch::Approx(2.0).epsilon(1e-4));
    REQUIRE(gradient(1) == Catch::Approx(8.0).epsilon(1e-4));
    REQUIRE(gradient(2) == Catch::Approx(18.0).epsilon(1e-4));
}

TEST_CASE("Hessian of quadratic form matches analytic result", "[differentiation][hessian]") {
    // f(x, y) = x^2 + xy + y^2  =>  H = [[2, 1], [1, 2]] everywhere
    auto f = [](const linalg::Vec<d64>& v) {
        return v(0)*v(0) + v(0)*v(1) + v(1)*v(1);
    };

    linalg::Vec<d64> point{1.0, 1.0};
    auto hessian = calculus::differentiate::hess(f, point, 1e-4);

    REQUIRE(hessian(0, 0) == Catch::Approx(2.0).epsilon(1e-3));
    REQUIRE(hessian(0, 1) == Catch::Approx(1.0).epsilon(1e-3));
    REQUIRE(hessian(1, 0) == Catch::Approx(1.0).epsilon(1e-3));
    REQUIRE(hessian(1, 1) == Catch::Approx(2.0).epsilon(1e-3));
}

TEST_CASE("Gradient on a non-quadratic function", "[differentiation][gradient]") {
    // f(x, y) = sin(x)cos(y)  =>  grad f = (cos(x)cos(y), -sin(x)sin(y))
    auto f = [](const linalg::Vec<d64>& v) {
        return std::sin(v(0)) * std::cos(v(1));
    };

    linalg::Vec<d64> point{0.5, 0.3};
    auto gradient = calculus::differentiate::grad(f, point, 1e-5);

    REQUIRE(gradient(0) == Catch::Approx(std::cos(0.5)*std::cos(0.3)).epsilon(1e-4));
    REQUIRE(gradient(1) == Catch::Approx(-std::sin(0.5)*std::sin(0.3)).epsilon(1e-4));
}