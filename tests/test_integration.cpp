// Catch2 (v3) test suite for calculus::differentiation functions
//
// Build notes:
//   - Requires Catch2 v3 (link against Catch2::Catch2WithMain) plus the
//     linalg library sources (matrix.cpp, vec.cpp, linalg_interop.cpp).
//   - Include paths must expose both "calculus/differentiation.hpp" and the
//     top-level "types.hpp" / "constants.hpp" used by those headers.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "calculus/integration.hpp"
#include "calculus/target_distributions/gaussian_target.hpp"
#include "calculus/proposal/gaussian_proposal.hpp"

#include <cmath>
#include <random>
#include <vector>

using calculus::integrate::IntegralResult;
using calculus::integrate::trapezoidal;
using calculus::integrate::simpsons;
using calculus::integrate::mc;

using calculus::sample::GaussianTarget;
using calculus::sample::GaussianProposal;
using calculus::sample::ImportanceSampleResult;
using calculus::sample::importanceSample;

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

TEST_CASE("mc integrates constant functions exactly", "[mc][integrate]") {
    // A constant integrand has zero sample variance, so the Welford-based
    // stopping criterion is satisfied after the second draw regardless of
    // the RNG stream — this gives an exact, seed-independent check.
    SECTION("1D") {
        std::mt19937 gen(42);
        std::vector<d64> left{0.0};
        std::vector<d64> right{5.0};

        auto result = mc(
            [](const linalg::Vec<d64>& x) { return 2.0; },
            left, right, gen
        );

        REQUIRE(result.converged);
        REQUIRE(result.numIter == 2);
        REQUIRE_THAT(result.value, WithinAbs(10.0, 1e-9));
        REQUIRE_THAT(result.finalError, WithinAbs(0.0, 1e-9));
    }

    SECTION("multi-dimensional") {
        std::mt19937 gen(7);
        std::vector<d64> left{0.0, 0.0};
        std::vector<d64> right{2.0, 1.0};

        auto result = mc(
            [](const linalg::Vec<d64>& x) { return 3.0; },
            left, right, gen
        );

        REQUIRE(result.converged);
        REQUIRE_THAT(result.value, WithinAbs(6.0, 1e-9)); // volume (2*1) * 3
    }
}

TEST_CASE("mc converges to known integrals within its own error estimate", "[mc][integrate]") {
    SECTION("1D linear function") {
        std::mt19937 gen(123);
        std::vector<d64> left{0.0};
        std::vector<d64> right{2.0};

        auto result = mc(
            [](const linalg::Vec<d64>& x) { return x(0); },
            left, right, gen, kIterStopCondition, 20000
        );

        REQUIRE_FALSE(result.converged);
        REQUIRE(result.numIter == 20000);
        REQUIRE(std::abs(result.value - 2.0) < 5.0 * result.finalError);
    }

    SECTION("2D product function") {
        std::mt19937 gen(456);
        std::vector<d64> left{0.0, 0.0};
        std::vector<d64> right{1.0, 1.0};

        auto result = mc(
            [](const linalg::Vec<d64>& x) { return x(0) * x(1); },
            left, right, gen, kIterStopCondition, 20000
        );

        // Integral of x*y over the unit square = 1/4.
        REQUIRE(std::abs(result.value - 0.25) < 5.0 * result.finalError);
    }
}

TEST_CASE("mc is reproducible given a fixed-seed generator", "[mc][integrate][reproducibility]") {
    auto f = [](const linalg::Vec<d64>& x) { return std::sin(x(0)) + x(0) * x(0); };
    std::vector<d64> left{0.0};
    std::vector<d64> right{3.0};

    std::mt19937 gen1(2024);
    auto r1 = mc(f, left, right, gen1, kIterStopCondition, 500);

    std::mt19937 gen2(2024);
    auto r2 = mc(f, left, right, gen2, kIterStopCondition, 500);

    REQUIRE(r1.numIter == r2.numIter);
    REQUIRE_THAT(r1.value, WithinAbs(r2.value, 1e-15));
    REQUIRE_THAT(r1.finalError, WithinAbs(r2.finalError, 1e-15));
}

TEST_CASE("mc respects maxN when stopCondition is unreachable", "[mc][integrate]") {
    std::mt19937 gen(3);
    std::vector<d64> left{0.0};
    std::vector<d64> right{1.0};

    auto result = mc(
        [](const linalg::Vec<d64>& x) { return x(0); },
        left, right, gen,
        1e-15, // effectively unreachable with so few samples
        10     // tiny sample budget
    );

    REQUIRE_FALSE(result.converged);
    REQUIRE(result.numIter == 10);
}

TEST_CASE("mc validates its inputs", "[mc][integrate][validation]") {
    std::mt19937 gen(1);
    auto f = [](const linalg::Vec<d64>& x) { return 1.0; };

    SECTION("mismatched boundary array sizes") {
        std::vector<d64> left{0.0, 0.0};
        std::vector<d64> right{1.0};

        REQUIRE_THROWS_AS(mc(f, left, right, gen), std::invalid_argument);
    }

    SECTION("zero-width boundary") {
        std::vector<d64> left{1.0};
        std::vector<d64> right{1.0};

        REQUIRE_THROWS_AS(mc(f, left, right, gen), std::invalid_argument);
    }

    SECTION("reversed boundary") {
        std::vector<d64> left{2.0};
        std::vector<d64> right{0.0};

        REQUIRE_THROWS_AS(mc(f, left, right, gen), std::invalid_argument);
    }
}

TEST_CASE("mc convenience overload without an explicit generator works", "[mc][integrate]") {
    std::vector<d64> left{0.0};
    std::vector<d64> right{5.0};

    auto result = calculus::integrate::mc(
        [](const linalg::Vec<d64>& x) { return 2.0; },
        left, right
    );

    REQUIRE(result.converged);
    REQUIRE_THAT(result.value, WithinAbs(10.0, 1e-9));
}

TEST_CASE("importanceSample reduces to a plain sample mean when proposal matches target",
          "[importanceSample][sample]") {
    linalg::Vec<d64> mean{0.0};
    GaussianTarget target(mean, 1.0);
    GaussianProposal proposal(mean, 1.0);

    std::mt19937 gen(11);
    auto result = importanceSample(
        target, proposal,
        [](const linalg::Vec<d64>& x) { return x(0); },
        gen, kIterStopCondition, 5000
    );

    REQUIRE_THAT(result.effectiveSampleSize,
                 WithinRel(static_cast<d64>(result.numIter), 1e-8));
    REQUIRE(std::abs(result.value) < 5.0 * result.finalError);
}

TEST_CASE("importanceSample recovers known Gaussian moments with a mismatched proposal",
          "[importanceSample][sample]") {
    // Proposal deliberately wider than the target so importance weights stay
    // bounded (a "safe" importance sampling setup).
    linalg::Vec<d64> mean{0.0};
    GaussianTarget target(mean, 1.0);
    GaussianProposal proposal(mean, 2.0);

    SECTION("E[X] ~ 0") {
        std::mt19937 gen(99);
        auto result = importanceSample(
            target, proposal,
            [](const linalg::Vec<d64>& x) { return x(0); },
            gen, kIterStopCondition, 20000
        );

        REQUIRE(std::abs(result.value) < 5.0 * result.finalError);
    }

    SECTION("E[X^2] ~ target variance (1.0)") {
        std::mt19937 gen(99);
        auto result = importanceSample(
            target, proposal,
            [](const linalg::Vec<d64>& x) { return x(0) * x(0); },
            gen, kIterStopCondition, 20000
        );

        REQUIRE(std::abs(result.value - 1.0) < 5.0 * result.finalError);
    }
}

TEST_CASE("importanceSample works for multi-dimensional targets", "[importanceSample][sample]") {
    linalg::Vec<d64> mean{0.0, 0.0};
    GaussianTarget target(mean, 1.0);
    GaussianProposal proposal(mean, 2.0);

    std::mt19937 gen(31);
    auto result = importanceSample(
        target, proposal,
        [](const linalg::Vec<d64>& x) { return x(0) * x(0) + x(1) * x(1); },
        gen, kIterStopCondition, 20000
    );

    // E[||X||^2] for an isotropic N(0, sigma^2 I) in `dim` dimensions is
    // dim * sigma^2 = 2 * 1.0 = 2.0.
    REQUIRE(std::abs(result.value - 2.0) < 5.0 * result.finalError);
}

TEST_CASE("importanceSample's effective sample size degrades under proposal/target mismatch",
          "[importanceSample][sample][diagnostics]") {
    linalg::Vec<d64> mean{0.0};
    GaussianTarget target(mean, 1.0);

    GaussianProposal matchedProposal(mean, 1.0);       // constant weights, ESS == N
    GaussianProposal mismatchedProposal(mean, 0.05);    // much narrower than the target

    auto f = [](const linalg::Vec<d64>& x) { return x(0) * x(0); };

    std::mt19937 gen1(5);
    auto matchedResult = importanceSample(target, matchedProposal, f, gen1, kIterStopCondition, 5000);

    std::mt19937 gen2(5);
    auto mismatchedResult = importanceSample(target, mismatchedProposal, f, gen2, kIterStopCondition, 5000);

    REQUIRE(mismatchedResult.effectiveSampleSize <= static_cast<d64>(mismatchedResult.numIter));

    d64 matchedRatio = matchedResult.effectiveSampleSize / static_cast<d64>(matchedResult.numIter);
    d64 mismatchedRatio = mismatchedResult.effectiveSampleSize / static_cast<d64>(mismatchedResult.numIter);

    REQUIRE(mismatchedRatio < matchedRatio);
}

TEST_CASE("importanceSample respects maxN when stopCondition is unreachable",
          "[importanceSample][sample]") {
    linalg::Vec<d64> mean{0.0};
    GaussianTarget target(mean, 1.0);
    GaussianProposal proposal(mean, 1.5);

    std::mt19937 gen(21);
    auto result = importanceSample(
        target, proposal,
        [](const linalg::Vec<d64>& x) { return x(0); },
        gen, 1e-15, 15
    );

    REQUIRE_FALSE(result.converged);
    REQUIRE(result.numIter == 15);
}

TEST_CASE("importanceSample is reproducible given a fixed-seed generator",
          "[importanceSample][sample][reproducibility]") {
    linalg::Vec<d64> mean{0.0};
    GaussianTarget target(mean, 1.0);
    GaussianProposal proposal(mean, 1.5);
    auto f = [](const linalg::Vec<d64>& x) { return x(0) * x(0); };

    std::mt19937 gen1(777);
    auto r1 = importanceSample(target, proposal, f, gen1, kIterStopCondition, 500);

    std::mt19937 gen2(777);
    auto r2 = importanceSample(target, proposal, f, gen2, kIterStopCondition, 500);

    REQUIRE(r1.numIter == r2.numIter);
    REQUIRE_THAT(r1.value, WithinAbs(r2.value, 1e-15));
    REQUIRE_THAT(r1.effectiveSampleSize, WithinAbs(r2.effectiveSampleSize, 1e-9));
}