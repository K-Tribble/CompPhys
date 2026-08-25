#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>
#include <vector>
#include "calculus/optimize.hpp"
#include "linalg/vec.hpp"
#include "linalg/matrix.hpp"

TEST_CASE("Gradient descent finds minimum of a convex quadratic", "[optimize][gradient_descent]") {
    auto f = [](const linalg::Vec<d64>& v) {
        return (v(0)-3)*(v(0)-3) + (v(1)+1)*(v(1)+1);
    };

    linalg::Vec<d64> x0{0.0, 0.0};
    auto result = calculus::optimize::gradDescent(f, x0, 1e-8);

    REQUIRE(result.converged);
    REQUIRE(result.x_min(0) == Catch::Approx(3.0).epsilon(1e-4));
    REQUIRE(result.x_min(1) == Catch::Approx(-1.0).epsilon(1e-4));
}

TEST_CASE("Newton's method with LM regularization converges on Rosenbrock", "[optimize][newton]") {
    auto f = [](const linalg::Vec<d64>& v) {
        double a = 1 - v(0);
        double b = v(1) - v(0)*v(0);
        return a*a + 100*b*b;
    };

    linalg::Vec<d64> x0{-1.2, 1.0};  // standard Rosenbrock start point
    auto result = calculus::optimize::newtonsLM(f, x0, 1e-8);

    REQUIRE(result.converged);
    REQUIRE(result.x_min(0) == Catch::Approx(1.0).epsilon(1e-9));
    REQUIRE(result.x_min(1) == Catch::Approx(1.0).epsilon(1e-9));
}

TEST_CASE("Gradient descent line search never increases the objective", "[optimize][gradient_descent]") {
    auto f = [](const linalg::Vec<d64>& v) { return v(0)*v(0) + v(1)*v(1); };

    linalg::Vec<d64> x0{5.0, 5.0};
    auto result = calculus::optimize::gradDescent(f, x0, 1e-8);

    REQUIRE(result.converged);
    REQUIRE(f(result.x_min) <= f(x0));
}

TEST_CASE("Newton's method stays stable where the Hessian is not positive definite", "[optimize][newton]") {
    auto f = [](const linalg::Vec<d64>& v) {
        return v(0)*v(0)*v(0)*v(0) - v(0)*v(0);
    };

    linalg::Vec<d64> x0{0.1};  // starts in the concave region
    auto result = calculus::optimize::newtonsLM(f, x0, 1e-8);

    REQUIRE(result.converged);
    REQUIRE(std::abs(result.x_min(0)) == Catch::Approx(1.0/std::sqrt(2.0)).epsilon(1e-3));
}