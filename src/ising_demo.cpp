#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "mcmc/driver.hpp"
#include "mcmc/models/ising2d.hpp"
#include "mcmc/steppers/ising_stepper.hpp"

using namespace mcmc;

int main(int argc, char** argv) {
    const u32 L = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 16;
    const u32 sweeps = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 20000;
    const u32 numChains = argc > 3 ? std::strtoul(argv[3], nullptr, 10) : 4;
    const std::string dir = argc > 4 ? argv[4] : "ising_data";

    const d64 Tc = 2.0 / std::log(1.0 + std::sqrt(2.0)); // 2.269185..

    std::vector<d64> temps;
    for (u32 i = 0; i <= 58; ++i) {
        temps.push_back(3.4 - 0.05 * static_cast<d64>(i));
    }


    const std::string mk = "mkdir -p " + dir;
    if (std::system(mk.c_str()) != 0) {
        std::fprintf(stderr, "could not create %s\n", dir.c_str());
        return 1;
    }

    std::printf("2D Ising, L=%zu (N=%zu), %s, %zu chains x %zu sweeps\n", L, L * L, "metropolis", numChains, sweeps);
    std::printf("exact T_c = %.6f\n\n", Tc);
    std::printf("%7s %10s %10s %10s %10s %9s %8s %9s %9s %7s\n",
                "T", "e", "m", "|m|", "err(|m|)", "tau(|m|)", "acc",
                "Rhat(|m|)", "Rhat(m)", "sec");

    // Warm start, each temp begins from previous ones final config, chain for chain.
    // Decreases burn-in near Tc, but does correlate successive temp points
    std::vector<std::vector<Ising2D::Spin>> warm;

    for (const d64 T : temps) {
        const d64 beta = 1.0 / T;

        RunConfig cfg;
        cfg.numChains = numChains;
        cfg.sweeps = sweeps;
        cfg.burnIn = warm.empty() ? sweeps / 4 : sweeps / 20;
        cfg.thin = 1;
        cfg.seed = 20260905ull + static_cast<std::uint64_t>(1000.0 * T);

        std::vector<std::vector<Ising2D::Spin>> next(numChains);

        auto makeModel = [&](u32 c, std::mt19937& g) {
            Ising2D m(L, 1.0, 0.0);
            if (c < warm.size()) {
                m.setSpins(warm[c]);
            } else {
                m.randomise(g);
            }
            return m;
        };

        auto harvest = [&](u32 c, const auto& stepper) {
            next[c] = stepper.state().spins();
        };

        RunResult res = run([&](u32 c, std::mt19937& g) { return IsingMetropolis(makeModel(c, g), beta);}, IsingObservables{}, cfg, harvest);

        warm = std::move(next);

        const u32 ie = res.obsIndex("e");
        const u32 imm = res.obsIndex("m");
        const u32 im = res.obsIndex("abs_m");

        char stem[512];
        std::snprintf(stem, sizeof(stem), "%s/ising_L%zu_T%.4f", dir.c_str(), L, T);

        io::JsonWriter extra;
        extra.add("model", std::string("ising2d"))
            .add("algorithm", "metropolis")
            .add("L", L)
            .add("N", L * L)
            .add("J", 1.0)
            .add("field", 0.0)
            .add("T", T)
            .add("beta", beta)
            .add("T_c_exact", Tc)
            .add("warm_started", true);

        save(res, stem, cfg, extra);

        std::printf("%7.4f %10.5f %10.5f %10.5f %10.5f %9.2f %8.2e %9.4f %9.4f %7.2f\n",
            T, res.mean[ie], res.mean[imm], res.mean[im],
            res.stdError[im], res.tauSweeps[im],
            res.chains[0].acceptedFraction,
            res.rHat[im], res.rHat[imm], res.timeTaken);

        std::fflush(stdout);
    }

    std::printf("\nwrote %zu runs to %s/\n", temps.size(), dir.c_str());
    std::printf("analyse with: python3 analysis/analyse_ising.py %s\n", dir.c_str());
    return 0;
}