// Catch2 (v3) test suite for nonlin:: functions
//
// Build notes:
//   - Requires Catch2 v3 (link against Catch2::Catch2WithMain) plus the
//     linalg library sources (matrix.cpp, vec.cpp, linalg_interop.cpp).
//   - Include paths must expose both "nonlin_solve.hpp" and the
//     top-level "types.hpp" / "constants.hpp" used by those headers.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
#include <stdexcept>
#include "nonlin_solve.hpp"
#include <functional>
#include <cmath>

using namespace nonlin;
struct NewtonArgs {
    d64 x0;
    std::function<d64(d64)> derivf;
};

struct BisectArgs {
    d64 xl;
    d64 xr;
};

struct SecantArgs {
    d64 x0;
    d64 x1;
};

struct BrentArgs {
    d64 xl;
    d64 xr;
};

template <typename F>
static void testNonLinFuncs(F&& f, BisectArgs ba, NewtonArgs na, SecantArgs sa, BrentArgs brenta) {
    RootIterResult br = bisection(f, ba.xl, ba.xr);
    RootIterResult nr = newton(f, na.derivf, na.x0);
    RootIterResult sr = secant(f, sa.x0, sa.x1);
    RootIterResult brentr = brent(f, brenta.xl, brenta.xr);

    REQUIRE_THAT(br.function_val, WithinAbs(0.0, 1e-11));
    REQUIRE_THAT(nr.function_val, WithinAbs(0.0, 1e-11));
    REQUIRE_THAT(sr.function_val, WithinAbs(0.0, 1e-11));
    REQUIRE_THAT(brentr.function_val, WithinAbs(0.0, 1e-11));
    REQUIRE(br.converged);
    REQUIRE(nr.converged);
    REQUIRE(sr.converged);
    REQUIRE(brentr.converged);
    REQUIRE(br.foundRoot);
    REQUIRE(nr.foundRoot);
    REQUIRE(sr.foundRoot);
    REQUIRE(brentr.foundRoot);
}

auto f1 = [](d64 x) {
    return x * x - 17;
};

auto fprime1 = [](d64 x) {
    return 2 * x;
};

auto f2 = [](d64 x) {
    return std::sin(x*x) - 1 / (x + 5);
};

auto fprime2 = [](d64 x) {
    return std::cos(x*x) * 2 * x + 1 / ((x + 5) * (x + 5));
};

auto f3 = [](d64 x) -> d64 {
    return std::cos(x / 2.0) - x;
};

auto fprime3 = [](d64 x) -> d64 {
    return -std::sin(x / 2.0) / 2.0 - 1.0;
};

auto f4 = [](d64 x) -> d64 {
    return x * x - 2.0 * x - 4.0;
};

auto fprime4 = [](d64 x) -> d64 {
    return 2.0 * x - 2.0;
};

auto f5 = [](d64 x) -> d64 {
    return std::exp(x) - 4.0 * x;
};

auto fprime5 = [](d64 x) -> d64 {
    return std::exp(x) - 4.0;
};

TEST_CASE("x^2 - 17") {testNonLinFuncs(f1, {4.0, 20.0}, {24.0, fprime1}, {20.0, 18.0}, {4.0, 20.0});}
TEST_CASE("sin(x^2) - 1 / (x + 5)") {testNonLinFuncs(f2, {3.6, 4.0}, {1.5, fprime2}, {1.4, 1.6}, {3.6, 4.0});}
TEST_CASE("cos(x/2) - x") {testNonLinFuncs(f3, {-0.5, 2.0}, {0.5, fprime3}, {1.8, 1.3}, {-0.5, 2.0});}
TEST_CASE("x^2 - 2x - 4") {testNonLinFuncs(f4, {-1.36, -1.0}, {-5.5, fprime4}, {2.5, 2.8}, {-1.36, -1.0});}
TEST_CASE("e^x - 4x") {testNonLinFuncs(f5, {2.0, 3.0}, {24.0, fprime5}, {2.3, 2.4}, {2.0, 3.0});}