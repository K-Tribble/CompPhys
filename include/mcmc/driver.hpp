#pragma once

#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <omp.h>
 
#include "types.hpp"
#include "mcmc/chain_stats.hpp"
#include "mcmc/io.hpp"
#include "mcmc/stepper.hpp"

namespace mcmc {

struct RunConfig {
    u32 sweeps = 10000; // measured sweeps per chain, before thinning
    u32 burnIn = 1000; // discarded sweeps per chain
    u32 thin = 1; // record every thin-th sweep
    u32 numChains = 4;
    u32 maxLag = 0; // 0 -> min(numDraws / 2, 4096)
    std::uint64_t seed = 20260905ull;
};

struct ChainSummary {
    d64 acceptedFraction = std::numeric_limits<d64>::quiet_NaN();
    std::vector<d64> mean;
    std::vector<d64> variance;
    std::vector<d64> tauInt; // in unites of recorded draws
    std::vector<d64> ess;
};

struct RunResult {
    std::vector<std::string> names;
    u32 numChains = 0;
    u32 numDraws = 0; // per chain, after thinning
    u32 numObs = 0;
    u32 thin = 1;

    // Flat, C, order, shpae (numChains, numDraws, numObs). Written to .npy file
    std::vector<d64> series;

    std::vector<ChainSummary> chains;

    // Pooled across chains, one entry per observable
    std::vector<d64> mean;
    std::vector<d64> stdError;
    std::vector<d64> tauInt; // mean over chains, in draws
    std::vector<d64> tauSweeps; // tauInt * thin
    std::vector<d64> ess; // summed over chains
    std::vector<d64> rHat;

    d64 timeTaken = 0.0;

    d64 at(u32 chain, u32 draw, u32 obs) const {
        return series[(chain * numDraws + draw) * numObs + obs];
    }

    std::vector<d64> extract(u32 chain, u32 obs) const {
        std::vector<d64> out(numDraws);
        for (u32 d = 0; d < numDraws; ++d) {
            out[d] = at(chain, d, obs);
        }

        return out;
    }

    u32 obsIndex(const std::string& name) const {
        for (u32 i = 0; i < names.size(); ++i) {
            if (names[i] == name) {
                return i;
            }
        }
        throw std::invalid_argument("RunResult: no observable named " + name);
    }
};

namespace detail {
struct NoHarvest {
    template <typename Stepper>
    void operator()(u32, const Stepper&) const {}
};
} // namespace detail

// Run cfg.numChains independant chains and collect per-observable 
// series and diagnostics.
// makeStepper

template <typename Factory, typename Obs, typename Harvest = detail::NoHarvest>
RunResult run(Factory&& makeStepper, const Obs& observables, const RunConfig& cfg, Harvest&& harvest = {}) {
    using Stepper = std::invoke_result_t<Factory&, u32, std::mt19937&>;
    using State = typename Stepper::State;

    static_assert(ChainStepper<Stepper>, "makeStepper must return a type satisfying ChainStepper");
    static_assert(Observables<Obs, State>, "observables must satisfy Observables<State>");

    if (cfg.numChains == 0) {
        throw std::invalid_argument("run: need at least one chain");
    }
    if (cfg.thin == 0) {
        throw std::invalid_argument("run: thin must be at least 1");
    }

    const auto t0 = std::chrono::steady_clock::now();

    RunResult res;
    res.names = observables.names();
    res.numObs = res.names.size();
    res.numChains = cfg.numChains;
    res.numDraws = cfg.sweeps / cfg.thin;
    res.thin = cfg.thin;

    if (res.numObs == 0) {
        throw std::invalid_argument("run: observable set is empty");
    }
    if (res.numDraws < 2) {
        throw std::invalid_argument("run: need at least 2 recorded draws");
    }

    res.series.assign(static_cast<std::size_t>(res.numChains) * res.numDraws * res.numObs, 0.0);
    res.chains.resize(res.numChains);

    std::exception_ptr firstError = nullptr;

    #pragma omp parallel for schedule(dynamic)
    for (u32 c = 0; c < cfg.numChains; ++c) {
        try {
            // each chain gets an independant stream from a seed_seq mixing the
            // run seed with chain index. Results are reproducible and independant
            // of how many threads are running them
            std::seed_seq seq{
                static_cast<std::uint32_t>(cfg.seed & 0xffffffffull),
                static_cast<std::uint32_t>(cfg.seed >> 32),
                static_cast<std::uint32_t>(c)};
            std::mt19937 gen(seq);

            Stepper stepper = makeStepper(c, gen);

            // Burn in phase
            for (u32 i = 0; i < cfg.burnIn; ++i) {
                stepper.sweep(gen);
            }
            stepper.resetCounters();

            d64* base = res.series.data() + static_cast<u32>(c) * res.numDraws * res.numObs;

            for (u32 d = 0; d < res.numDraws; ++d) {
                for (u32 t = 0; t < cfg.thin; ++t) {
                    stepper.sweep(gen);
                }
                observables.eval(stepper.state(), std::span<d64>(base + static_cast<u32>(d) * res.numObs, res.numObs));
            }

            res.chains[c].acceptedFraction = stepper.acceptedFraction();
            
            #pragma omp critical
            {
                harvest(c, stepper);
            }
        } catch (...) {
            #pragma omp critical
            {
                if (!firstError) {
                    firstError = std::current_exception();
                }
            }
        }
    }

    if (firstError) {
        std::rethrow_exception(firstError);
    }

    u32 maxLag = cfg.maxLag;
    if (maxLag == 0) {
        maxLag = std::min<u32>(res.numDraws / 2, 4096);
    }

    res.mean.assign(res.numObs, 0.0);
    res.stdError.assign(res.numObs, 0.0);
    res.tauInt.assign(res.numObs, 0.0);
    res.tauSweeps.assign(res.numObs, 0.0);
    res.ess.assign(res.numObs, 0.0);
    res.rHat.assign(res.numObs, 0.0);

    for (auto& cs : res.chains) {
        cs.mean.assign(res.numObs, 0.0);
        cs.variance.assign(res.numObs, 0.0);
        cs.tauInt.assign(res.numObs, 0.0);
        cs.ess.assign(res.numObs, 0.0);
    }

    std::vector<std::vector<d64>> perChain(res.numChains);

    for (u32 o = 0; o < res.numObs; ++o) {
        d64 essTotal = 0.0;
        d64 pooledMean = 0.0;
        d64 tauSum = 0.0;

        for (u32 c = 0; c < res.numChains; ++c) {
            perChain[c] = res.extract(c, o);

            d64 mu = 0.0;
            for (const d64 v : perChain[c]) {
                mu += v;
            }
            mu /= static_cast<d64>(res.numDraws);

            d64 ss = 0.0;
            for (const d64 v : perChain[c]) {
                ss += (v - mu) * (v - mu);
            }
            const d64 var = ss / static_cast<d64>(res.numDraws - 1);

            const std::vector<d64> rho = stats::autocorrelation(perChain[c], maxLag);
            const d64 tau = stats::geyerTau(rho);

            res.chains[c].mean[o] = mu;
            res.chains[c].variance[o] = var;
            res.chains[c].tauInt[o] = tau;
            res.chains[c].ess[o] = static_cast<d64>(res.numDraws) / tau;

            pooledMean += mu;
            tauSum += tau;
            essTotal += static_cast<d64>(res.numDraws) / tau;
        }

        pooledMean /= static_cast<d64>(res.numChains);

        // Variance about the pooled mean
        d64 ss = 0.0;
        for (u32 c = 0; c < res.numChains; ++c) {
            for (const d64 v : perChain[c]) {
                ss += (v - pooledMean) * (v - pooledMean);
            }
        }
        const u32 total = res.numChains * res.numDraws;
        const d64 pooledVar = ss / static_cast<d64>(total - 1);

        std::vector<std::span<const d64>> spans;
        spans.reserve(res.numChains);
        for (u32 c = 0; c < res.numChains; ++c) {
            spans.emplace_back(perChain[c]);
        }

        res.mean[o] = pooledMean;
        res.tauInt[o] = tauSum / static_cast<d64>(res.numChains);
        res.tauSweeps[o] = res.tauInt[o] * static_cast<d64>(res.thin);
        res.ess[o] = essTotal;
        res.stdError[o] = std::sqrt(pooledVar / essTotal);
        res.rHat[o] = stats::splitRHat(spans);
    }

    res.timeTaken = std::chrono::duration<d64>(std::chrono::steady_clock::now() - t0).count();
    return res;
}

// Write <stem>.npy (float64, shape (chains, draws observables)) plus
// <stem>.json carrying everything needed to interpret it. Pass 'extra'
// to add the physics parameters -- L, beta, algorithm, whatever the scan varies
inline void save(const RunResult& res, const std::string& stem, const RunConfig& cfg, const io::JsonWriter& extra = io::JsonWriter{}) {
    const u32 shape[3] = {res.numChains, res.numDraws, res.numObs};
    io::writeNpy<d64>(stem + ".npy", res.series, std::span<const u32>(shape, 3));

    io::JsonWriter j;
    j.add("data_file", stem + ".npy")
        .add("observables", res.names)
        .add("num_chains", res.numChains)
        .add("num_draws", res.numDraws)
        .add("thin", res.thin)
        .add("burn_in", cfg.burnIn)
        .add("sweeps", cfg.sweeps)
        .add("mean", res.mean)
        .add("std_error", res.stdError)
        .add("tau_int_sweeps", res.tauSweeps)
        .add("ess", res.ess)
        .add("r_hat", res.rHat)
        .add("timeTaken", res.timeTaken);

    std::vector<d64> acc;
    for (const auto& c : res.chains) {
        acc.push_back(c.acceptedFraction);
    }
    j.add("accepted_fraction", acc);

    std::string body = j.str();
    const std::string tail = extra.str();
    const u32 open = tail.find('\n');
    const u32 close = tail.rfind('}');
    if (open != std::string::npos && close != std::string::npos && close > open + 2) {
        const std::string inner = tail.substr(open + 1, close - open - 2);
        const std::size_t bodyClose = body.rfind('}');
        body = body.substr(0, bodyClose) + ",\n" + inner + "\n}\n";
    }

    std::ofstream out(stem + ".json");
    if (!out) {
        throw std::runtime_error("save: cannot open " + stem + ".json");
    }
    out << body;
}

} // namespace mcmc