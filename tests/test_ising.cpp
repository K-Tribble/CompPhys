// Catch2 (v3) tests for the mcmc:: lattice pipeline.
//
// The load-bearing test here is the comparison against exact enumeration of
// the 4x4 periodic lattice. 2^16 configurations is small enough to sum
// directly.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "mcmc/chain_stats.hpp"
#include "mcmc/driver.hpp"
#include "mcmc/models/ising2d.hpp"
#include "mcmc/steppers/ising_stepper.hpp"

#include <cmath>
#include <random>
#include <vector>

using Catch::Matchers::WithinAbs;

using mcmc::Ising2D;
using mcmc::IsingMetropolis;
using mcmc::IsingObservables;
using mcmc::RunConfig;
using mcmc::RunResult;

namespace {

struct ExactIsing {
    d64 energyPerSite;
    d64 absMagPerSite;
};

// Direct sum over all 2^(L*L) configurations. Only tractable for L = 4.
ExactIsing exact4x4(d64 beta, d64 J = 1.0) {
    constexpr u32 L = 4;
    constexpr u32 N = 16;

    std::vector<d64> E(1u << N), M(1u << N);
    d64 minE = 1e300;

    for (u32 c = 0; c < (1u << N); ++c) {
        int s[N];
        int m = 0;
        for (u32 i = 0; i < N; ++i) {
            s[i] = ((c >> i) & 1u) ? 1 : -1;
            m += s[i];
        }
        d64 bonds = 0.0;
        for (u32 y = 0; y < L; ++y) {
            for (u32 x = 0; x < L; ++x) {
                const u32 i = y * L + x;
                bonds += s[i] * (s[y * L + (x + 1) % L] + s[((y + 1) % L) * L + x]);
            }
        }
        E[c] = -J * bonds;
        M[c] = static_cast<d64>(m);
        minE = std::min(minE, E[c]);
    }

    d64 Z = 0.0, sumE = 0.0, sumAbsM = 0.0;
    for (u32 c = 0; c < (1u << N); ++c) {
        const d64 w = std::exp(-beta * (E[c] - minE));   // shift for stability
        Z += w;
        sumE += w * E[c];
        sumAbsM += w * std::abs(M[c]);
    }
    return {sumE / Z / N, sumAbsM / Z / N};
}

} // namespace

TEST_CASE("Ising2D bookkeeping stays exact under flips", "[ising][model]") {
    std::mt19937 gen(20260905);

    SECTION("cached energy and magnetization track single flips") {
        Ising2D m(8, 1.0, 0.3);
        m.randomise(gen);

        std::uniform_int_distribution<u32> site(0, m.size() - 1);
        for (u32 k = 0; k < 5000; ++k) {
            m.flip(site(gen));
        }

        REQUIRE_THAT(m.energy(), WithinAbs(m.computeEnergy(), 1e-9));
        REQUIRE(m.magnetization() == m.computeMagnetization());
    }

    SECTION("energy of the fully ordered state is -2JN") {
        Ising2D m(10);
        m.setAllUp();
        REQUIRE_THAT(m.energyPerSite(), WithinAbs(-2.0, 1e-12));
        REQUIRE_THAT(m.magnetizationPerSite(), WithinAbs(1.0, 1e-12));
    }
}

TEST_CASE("Ising samplers reproduce exact 4x4 enumeration",
          "[ising][sampler][exact]") {
    RunConfig cfg;
    cfg.numChains = 4;
    cfg.burnIn = 2000;
    cfg.sweeps = 100000;
    cfg.seed = 99;

    // beta = 0.44 is close to the bulk critical coupling, so this is the
    // hardest of the three for a local sampler.
    for (const d64 beta : {0.2, 0.44, 0.7}) {
        const ExactIsing ex = exact4x4(beta);

        RunResult metro = mcmc::run(
            [&](u32, std::mt19937& g) {
                Ising2D m(4);
                m.randomise(g);
                return IsingMetropolis(std::move(m), beta);
            },
            IsingObservables{}, cfg);

        const u32 ie = metro.obsIndex("e");
        const u32 im = metro.obsIndex("abs_m");

        // Five sigma, floored so a tiny error bar can't make this brittle.
        auto tol = [](d64 err) { return std::max(5.0 * err, 2e-3); };

        REQUIRE_THAT(metro.mean[ie],
                     WithinAbs(ex.energyPerSite, tol(metro.stdError[ie])));
        REQUIRE_THAT(metro.mean[im],
                     WithinAbs(ex.absMagPerSite, tol(metro.stdError[im])));
    }
}

TEST_CASE("autocorrelation and tau_int estimators", "[mcmc][stats]") {
    std::mt19937 gen(4242);
    std::normal_distribution<d64> nd(0.0, 1.0);

    SECTION("FFT estimate matches the direct O(N*lag) sum") {
        // Deliberately not a power of two: this is the length at which the
        // two backends pad differently (2^17 for the built-in radix-2
        // transform, 7-smooth for FFTW), so it exercises the padding rather
        // than sidestepping it. Whichever backend is compiled in must agree
        // with the direct sum.
        std::vector<d64> x(33000);
        d64 prev = 0.0;
        for (auto& v : x) {
            prev = 0.8 * prev + nd(gen);
            v = prev;
        }

        const u32 maxLag = 64;
        const auto fast = mcmc::stats::autocorrelation(x, maxLag);

        d64 mean = 0.0;
        for (const d64 v : x) mean += v;
        mean /= static_cast<d64>(x.size());

        d64 g0 = 0.0;
        for (const d64 v : x) g0 += (v - mean) * (v - mean);
        g0 /= static_cast<d64>(x.size());

        for (u32 k = 0; k <= maxLag; ++k) {
            d64 g = 0.0;
            for (u32 i = 0; i + k < x.size(); ++i) {
                g += (x[i] - mean) * (x[i + k] - mean);
            }
            REQUIRE_THAT(fast[k], WithinAbs((g / x.size()) / g0, 1e-12));
        }
    }

    SECTION("AR(1) with phi = 0.8 gives tau_int near (1+phi)/(1-phi) = 9") {
        std::vector<d64> x(1u << 16);
        d64 prev = 0.0;
        for (auto& v : x) {
            prev = 0.8 * prev + nd(gen);
            v = prev;
        }
        const d64 tau = mcmc::stats::geyerTau(mcmc::stats::autocorrelation(x, 512));
        REQUIRE(tau > 7.5);
        REQUIRE(tau < 10.5);
    }

    SECTION("iid data gives tau_int near 1") {
        std::vector<d64> x(20000);
        for (auto& v : x) v = nd(gen);
        const d64 tau = mcmc::stats::geyerTau(mcmc::stats::autocorrelation(x, 256));
        REQUIRE(tau >= 1.0);
        REQUIRE(tau < 1.5);
    }

    SECTION("tau_int is clamped at 1 for an anti-correlated series") {
        // Without the clamp this returns a value below 1, giving ESS > N and,
        // if Gamma_0 goes negative, a negative tau and a NaN error bar.
        std::vector<d64> x(4096);
        for (u32 i = 0; i < x.size(); ++i) {
            x[i] = (i % 2 ? 1.0 : -1.0) + 0.01 * nd(gen);
        }
        REQUIRE(mcmc::stats::geyerTau(mcmc::stats::autocorrelation(x, 256)) >= 1.0);
    }
}

TEST_CASE("split-Rhat separates agreeing from disagreeing chains",
          "[mcmc][stats][rhat]") {
    std::mt19937 gen(7);
    std::normal_distribution<d64> nd(0.0, 1.0);

    std::vector<std::vector<d64>> same(4, std::vector<d64>(5000));
    for (auto& c : same) {
        for (auto& v : c) v = nd(gen);
    }
    std::vector<std::span<const d64>> a;
    for (const auto& c : same) a.emplace_back(c);
    REQUIRE(mcmc::stats::splitRHat(a) < 1.01);

    std::vector<std::vector<d64>> shifted(4, std::vector<d64>(5000));
    for (u32 c = 0; c < shifted.size(); ++c) {
        std::normal_distribution<d64> off(static_cast<d64>(c), 1.0);
        for (auto& v : shifted[c]) v = off(gen);
    }
    std::vector<std::span<const d64>> b;
    for (const auto& c : shifted) b.emplace_back(c);
    REQUIRE(mcmc::stats::splitRHat(b) > 1.2);
}

TEST_CASE("driver contract", "[mcmc][driver]") {
    RunConfig cfg;
    cfg.numChains = 3;
    cfg.burnIn = 100;
    cfg.sweeps = 2000;
    cfg.seed = 777;

    auto factory = [](u32, std::mt19937& g) {
        Ising2D m(8);
        m.randomise(g);
        return IsingMetropolis(std::move(m), 0.44);
    };

    SECTION("a fixed seed reproduces the run exactly, thread count aside") {
        const RunResult a = mcmc::run(factory, IsingObservables{}, cfg);
        const RunResult b = mcmc::run(factory, IsingObservables{}, cfg);
        REQUIRE(a.series == b.series);
    }

    SECTION("thinning reduces the recorded draws but not the sweeps") {
        RunConfig thinned = cfg;
        thinned.thin = 4;
        const RunResult r = mcmc::run(factory, IsingObservables{}, thinned);
        REQUIRE(r.numDraws == cfg.sweeps / 4);
        // tau in draws shrinks by the thinning factor; tau in sweeps does not.
        REQUIRE_THAT(r.tauSweeps[0], WithinAbs(r.tauInt[0] * 4.0, 1e-12));
    }

    SECTION("harvest sees every chain's final configuration") {
        std::vector<u32> seen(cfg.numChains, 0);
        mcmc::run(factory, IsingObservables{}, cfg,
                  [&](u32 c, const IsingMetropolis& s) {
                      seen[c] = s.state().size();
                  });
        for (const u32 v : seen) {
            REQUIRE(v == 64);
        }
    }

    SECTION("invalid configuration is rejected") {
        RunConfig bad = cfg;
        bad.numChains = 0;
        REQUIRE_THROWS_AS(mcmc::run(factory, IsingObservables{}, bad),
                          std::invalid_argument);
    }
}