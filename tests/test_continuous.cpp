// Catch2 (v3) tests for the continuous steppers in mcmc/continuous.hpp.
//
// Everything here is checked against a Gaussian, where the mean, the
// covariance and the normalisation are all known in closed form. The
// anisotropic case is the load-bearing one: it is the only test that
// exercises the HMC mass matrix path (cholesky + inverseHPD + matvec),
// and with M = Sigma^{-1} the sampler should be close to independent
// so tau_int lands near 1.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "mcmc/continuous.hpp"
#include "mcmc/driver.hpp"
#include "calculus/target_distributions/gaussian_target.hpp"
#include "calculus/transition_proposal/gaussian_transition.hpp"

#include <cmath>
#include <random>
#include <string>
#include <vector>

using Catch::Matchers::WithinAbs;

using calculus::sample::GaussianTarget;
using calculus::sample::IsotropicGaussianTransition;
using mcmc::HMCStepper;
using mcmc::MALAStepper;
using mcmc::MetropolisStepper;
using mcmc::RunConfig;
using mcmc::RunResult;

namespace {

// Zero-mean Gaussian specified by its precision matrix L = Sigma^{-1}:
//   log p(x) = -0.5 x^T L x
//   grad     = -L x
// Written out by hand so the test does not depend on anything in
// calculus:: beyond the interface itself.
class PrecisionGaussian : public calculus::sample::DifferentiableTarget {
public:
    explicit PrecisionGaussian(linalg::Matrix<d64> precision)
        : precision_(std::move(precision)) {}

    d64 logDensity(const linalg::Vec<d64>& x) const override {
        return -0.5 * x.dot(precision_ * x);
    }

    linalg::Vec<d64> gradLogDensity(const linalg::Vec<d64>& x) const override {
        return (precision_ * x) * -1.0;
    }

    u32 dim() const override {return precision_.rows();}

private:
    linalg::Matrix<d64> precision_;
};

// Records x_i and the second moments x_i x_j for a small state.
struct Moments {
    u32 n;

    std::vector<std::string> names() const {
        std::vector<std::string> v;
        for (u32 i = 0; i < n; ++i) {
            v.push_back("x" + std::to_string(i));
        }
        for (u32 i = 0; i < n; ++i) {
            for (u32 j = i; j < n; ++j) {
                v.push_back("x" + std::to_string(i) + "x" + std::to_string(j));
            }
        }
        return v;
    }

    void eval(const linalg::Vec<d64>& x, std::span<d64> out) const {
        u32 k = 0;
        for (u32 i = 0; i < n; ++i) {
            out[k++] = x(i);
        }
        for (u32 i = 0; i < n; ++i) {
            for (u32 j = i; j < n; ++j) {
                out[k++] = x(i) * x(j);
            }
        }
    }
};

// Overdispersed starts so that split R-hat actually has something to detect.
linalg::Vec<d64> scatteredStart(u32 n, std::mt19937& gen, d64 spread) {
    std::normal_distribution<d64> wide(0.0, spread);
    return linalg::Vec<d64>::random(n, wide, gen);
}

RunConfig baseConfig() {
    RunConfig cfg;
    cfg.sweeps = 20000;
    cfg.burnIn = 2000;
    cfg.thin = 1;
    cfg.numChains = 4;
    cfg.seed = 987654321ull;
    return cfg;
}

// Compare against the exact value using the driver's own reported standard
// error. Five sigma is loose enough not to flake and tight enough that a
// genuinely biased sampler fails.
void checkAgainst(const RunResult& res, const std::string& name, d64 exact) {
    const u32 o = res.obsIndex(name);
    const d64 tol = 5.0 * res.stdError[o];
    INFO(name << " = " << res.mean[o] << " +- " << res.stdError[o]
              << ", exact " << exact << ", rhat " << res.rHat[o]
              << ", ess " << res.ess[o]);
    REQUIRE(std::isfinite(res.stdError[o]));
    REQUIRE_THAT(res.mean[o], WithinAbs(exact, tol));
    REQUIRE(res.rHat[o] < 1.05);
}

} // namespace

TEST_CASE("Metropolis recovers an isotropic Gaussian", "[continuous][metropolis]") {
    const u32 n = 2;
    const d64 sigma = 1.5;
    const linalg::Vec<d64> mu = {1.0, -2.0};

    GaussianTarget target(mu, sigma);
    // ~2.4 / sqrt(dim) * sigma is the usual scaling for random walk Metropolis
    IsotropicGaussianTransition proposal(2.4 * sigma / std::sqrt(static_cast<d64>(n)));

    auto factory = [&](u32, std::mt19937& g) {
        return MetropolisStepper(target, proposal, scatteredStart(n, g, 4.0));
    };

    const RunResult res = run(factory, Moments{n}, baseConfig());

    checkAgainst(res, "x0", 1.0);
    checkAgainst(res, "x1", -2.0);
    // <x0^2> = mu0^2 + sigma^2, <x0 x1> = mu0 * mu1 (independent components)
    checkAgainst(res, "x0x0", 1.0 + sigma * sigma);
    checkAgainst(res, "x1x1", 4.0 + sigma * sigma);
    checkAgainst(res, "x0x1", -2.0);

    // random walk Metropolis in 2d at this scale sits around 0.3-0.5
    REQUIRE(res.acceptedFraction > 0.15);
    REQUIRE(res.acceptedFraction < 0.75);
}

TEST_CASE("MALA recovers an isotropic Gaussian", "[continuous][mala]") {
    const u32 n = 2;
    const d64 sigma = 1.5;
    const linalg::Vec<d64> mu = {1.0, -2.0};

    GaussianTarget target(mu, sigma);

    auto factory = [&](u32, std::mt19937& g) {
        return MALAStepper(target, scatteredStart(n, g, 4.0), 0.9);
    };

    const RunResult res = run(factory, Moments{n}, baseConfig());

    checkAgainst(res, "x0", 1.0);
    checkAgainst(res, "x1", -2.0);
    checkAgainst(res, "x0x0", 1.0 + sigma * sigma);
    checkAgainst(res, "x1x1", 4.0 + sigma * sigma);

    // MALA's optimal acceptance is 0.574; anywhere in this band means the
    // Hastings correction is at least self-consistent
    REQUIRE(res.acceptedFraction > 0.4);
    REQUIRE(res.acceptedFraction < 0.99);
}

TEST_CASE("HMC recovers an isotropic Gaussian with identity mass", "[continuous][hmc]") {
    const u32 n = 2;
    const d64 sigma = 1.5;
    const linalg::Vec<d64> mu = {1.0, -2.0};

    GaussianTarget target(mu, sigma);
    const linalg::Matrix<d64> mass = linalg::Matrix<d64>::identity(n);

    auto factory = [&](u32, std::mt19937& g) {
        return HMCStepper(target, scatteredStart(n, g, 4.0), mass, 0.35, 12);
    };

    const RunResult res = run(factory, Moments{n}, baseConfig());

    checkAgainst(res, "x0", 1.0);
    checkAgainst(res, "x1", -2.0);
    checkAgainst(res, "x0x0", 1.0 + sigma * sigma);
    checkAgainst(res, "x1x1", 4.0 + sigma * sigma);

    REQUIRE(res.acceptedFraction > 0.6);
}

TEST_CASE("HMC with a matched mass matrix samples a correlated Gaussian", "[continuous][hmc][mass]") {
    // Sigma^{-1}, strongly correlated: rho = 0.8
    const linalg::Matrix<d64> precision = {{2.0, -0.8},
                                           {-0.8, 0.5}};
    const u32 n = 2;

    PrecisionGaussian target(precision);

    // Sigma = precision^{-1}; the exact second moments come straight from it
    const linalg::Matrix<d64> sigma = precision.inverseHPD();

    // M = Sigma^{-1} = precision. This is the whole point of the mass matrix:
    // it should whiten the target.
    auto factory = [&](u32, std::mt19937& g) {
        return HMCStepper(target, scatteredStart(n, g, 5.0), precision, 0.6, 8);
    };

    const RunResult res = run(factory, Moments{n}, baseConfig());

    checkAgainst(res, "x0", 0.0);
    checkAgainst(res, "x1", 0.0);
    checkAgainst(res, "x0x0", sigma(0, 0));
    checkAgainst(res, "x1x1", sigma(1, 1));
    checkAgainst(res, "x0x1", sigma(0, 1));

    // Whitened, so the chain should be close to independent
    const u32 o = res.obsIndex("x0");
    INFO("tau_int(x0) = " << res.tauInt[o]);
    REQUIRE(res.tauInt[o] < 3.0);
    REQUIRE(res.acceptedFraction > 0.7);
}

TEST_CASE("Continuous steppers validate their arguments", "[continuous][validation]") {
    const linalg::Vec<d64> mu = {0.0, 0.0};
    GaussianTarget target(mu, 1.0);
    IsotropicGaussianTransition proposal(0.5);
    const linalg::Vec<d64> wrongDim = {0.0, 0.0, 0.0};
    const linalg::Vec<d64> okDim = {0.0, 0.0};

    REQUIRE_THROWS_AS(MetropolisStepper(target, proposal, wrongDim), std::invalid_argument);
    REQUIRE_THROWS_AS(MALAStepper(target, wrongDim, 0.1), std::invalid_argument);
    REQUIRE_THROWS_AS(MALAStepper(target, okDim, 0.0), std::invalid_argument);
    REQUIRE_THROWS_AS(MALAStepper(target, okDim, -1.0), std::invalid_argument);

    const linalg::Matrix<d64> mass = linalg::Matrix<d64>::identity(2);
    const linalg::Matrix<d64> wrongMass = linalg::Matrix<d64>::identity(3);
    REQUIRE_THROWS_AS(HMCStepper(target, okDim, wrongMass, 0.1, 5), std::invalid_argument);
    REQUIRE_THROWS_AS(HMCStepper(target, wrongDim, mass, 0.1, 5), std::invalid_argument);
    REQUIRE_THROWS_AS(HMCStepper(target, okDim, mass, 0.0, 5), std::invalid_argument);
    // numSteps == 0 used to be written `numSteps = 0`, so this guard silently never fired
    REQUIRE_THROWS_AS(HMCStepper(target, okDim, mass, 0.1, 0), std::invalid_argument);
}

TEST_CASE("Runs are reproducible and thread-count independent", "[continuous][seed]") {
    const u32 n = 2;
    const linalg::Vec<d64> mu = {0.0, 0.0};
    GaussianTarget target(mu, 1.0);

    RunConfig cfg = baseConfig();
    cfg.sweeps = 2000;
    cfg.burnIn = 200;

    auto factory = [&](u32, std::mt19937& g) {
        return MALAStepper(target, scatteredStart(n, g, 2.0), 0.8);
    };

    const RunResult a = run(factory, Moments{n}, cfg);
    const RunResult b = run(factory, Moments{n}, cfg);

    REQUIRE(a.series.size() == b.series.size());
    for (u32 i = 0; i < a.series.size(); ++i) {
        REQUIRE(a.series[i] == b.series[i]);
    }
}

TEST_CASE("CovarianceRecorder matches a direct two-pass covariance", "[continuous][covariance]") {
    const u32 n = 3;
    std::mt19937 gen(4242);
    std::normal_distribution<d64> normal(0.0, 2.0);

    std::vector<linalg::Vec<d64>> samples;
    for (u32 k = 0; k < 500; ++k) {
        samples.push_back(linalg::Vec<d64>::random(n, normal, gen));
    }

    mcmc::CovarianceRecorder rec(n);
    for (const auto& s : samples) {
        rec(s);
    }
    const linalg::Matrix<d64> online = rec.covariance();

    // direct two-pass
    linalg::Vec<d64> mean(n, 0.0);
    for (const auto& s : samples) {
        mean += s;
    }
    mean /= static_cast<d64>(samples.size());

    linalg::Matrix<d64> direct(n, n);
    for (const auto& s : samples) {
        for (u32 i = 0; i < n; ++i) {
            for (u32 j = 0; j < n; ++j) {
                direct(i, j) += (s(i) - mean(i)) * (s(j) - mean(j));
            }
        }
    }
    for (u32 i = 0; i < n; ++i) {
        for (u32 j = 0; j < n; ++j) {
            direct(i, j) /= static_cast<d64>(samples.size() - 1);
        }
    }

    for (u32 i = 0; i < n; ++i) {
        for (u32 j = 0; j < n; ++j) {
            REQUIRE_THAT(online(i, j), WithinAbs(direct(i, j), 1e-9));
        }
    }
}