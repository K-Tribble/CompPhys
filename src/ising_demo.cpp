#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "mcmc/driver.hpp"
#include "mcmc/models/ising2d.hpp"
#include "mcmc/steppers/ising_stepper.hpp"

using namespace mcmc;

namespace {
 
const d64 kTc = 2.0 / std::log(1.0 + std::sqrt(2.0)); // 2.269185..
 
struct Args {
    std::vector<u32> sizes{16};
    u32 baseSweeps = 20000; // floor on the measured sweeps per chain
    u32 maxSweeps = 0;      // 0 -> 32 * baseSweeps
    u32 numChains = 4;
    u32 targetEss = 200;    // pooled ESS of |m| the adaptive sweep count aims for
    u32 maxDraws = 50000;   // recorded draws per chain, thinning absorbs the rest
    std::uint64_t seed = 20260905ull;
    std::string dir = "ising_data";
    std::string direction = "cooling"; // cooling | heating | both
    int tminH = 50;   // temperatures in hundredths, so the grid carries no
    int tmaxH = 340;  // floating point drift into seeds or filenames
};
 
void usage() {
    std::printf(
        "usage: ising [options]\n"
        "  --L 16,32,64        lattice sizes, comma separated (default 16)\n"
        "  --sweeps N          floor on measured sweeps per chain (default 20000)\n"
        "  --max-sweeps N      ceiling on the adaptive sweep count (default 32x floor)\n"
        "  --chains N          chains per temperature (default 4)\n"
        "  --target-ess N      pooled ESS of |m| to aim for (default 200)\n"
        "  --max-draws N       recorded draws per chain before thinning (default 50000)\n"
        "  --direction WHICH   cooling | heating | both (default cooling)\n"
        "  --tmin T --tmax T   temperature range (default 0.5 to 3.4)\n"
        "  --seed N            base seed (default 20260905)\n"
        "  --dir PATH          output directory (default ising_data)\n");
}
 
bool parseArgs(int argc, char** argv, Args& a) {
    auto next = [&](int& i) -> const char* {
        if (i + 1 >= argc) {
            std::fprintf(stderr, "%s needs a value\n", argv[i]);
            return nullptr;
        }
        return argv[++i];
    };
 
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        const char* v = nullptr;
 
        if (flag == "--help" || flag == "-h") {
            usage();
            return false;
        } else if (flag == "--L") {
            if (!(v = next(i))) return false;
            a.sizes.clear();
            char* p = const_cast<char*>(v);
            while (*p) {
                a.sizes.push_back(std::strtoul(p, &p, 10));
                if (*p == ',') ++p;
                else break;
            }
        } else if (flag == "--sweeps") {
            if (!(v = next(i))) return false;
            a.baseSweeps = std::strtoul(v, nullptr, 10);
        } else if (flag == "--max-sweeps") {
            if (!(v = next(i))) return false;
            a.maxSweeps = std::strtoul(v, nullptr, 10);
        } else if (flag == "--chains") {
            if (!(v = next(i))) return false;
            a.numChains = std::strtoul(v, nullptr, 10);
        } else if (flag == "--target-ess") {
            if (!(v = next(i))) return false;
            a.targetEss = std::strtoul(v, nullptr, 10);
        } else if (flag == "--max-draws") {
            if (!(v = next(i))) return false;
            a.maxDraws = std::strtoul(v, nullptr, 10);
        } else if (flag == "--direction") {
            if (!(v = next(i))) return false;
            a.direction = v;
        } else if (flag == "--tmin") {
            if (!(v = next(i))) return false;
            a.tminH = static_cast<int>(std::lround(100.0 * std::atof(v)));
        } else if (flag == "--tmax") {
            if (!(v = next(i))) return false;
            a.tmaxH = static_cast<int>(std::lround(100.0 * std::atof(v)));
        } else if (flag == "--seed") {
            if (!(v = next(i))) return false;
            a.seed = std::strtoull(v, nullptr, 10);
        } else if (flag == "--dir") {
            if (!(v = next(i))) return false;
            a.dir = v;
        } else {
            std::fprintf(stderr, "unknown option %s\n", flag.c_str());
            usage();
            return false;
        }
    }
 
    if (a.sizes.empty() || a.numChains == 0 || a.baseSweeps < 100) {
        std::fprintf(stderr, "invalid --L, --chains or --sweeps\n");
        return false;
    }
    if (a.direction != "cooling" && a.direction != "heating" && a.direction != "both") {
        std::fprintf(stderr, "--direction must be cooling, heating or both\n");
        return false;
    }
    if (a.maxSweeps == 0) {
        a.maxSweeps = 32 * a.baseSweeps;
    }
    return true;
}

// Temperatures in hundreds ascending, coarser in tails and finer around Tc
std::vector<int> grid(int low, int high) {
    std::vector<int> t;
    auto push = [&](int v) {
        if (v >= low && v <= high) t.push_back(v);
    };

    for (int v = 50; v < 200; v += 10) push(v);
    for (int v = 200; v <= 260; v += 2) push(v);
    for (int v = 270; v <= 400; v += 10) push(v);
    return t;
}

// seeds from run parameters
std::uint64_t mixSeed(std::uint64_t base, u32 L, int hundredths, int dirCode, int phase) {
    std::uint64_t x = base;
    for (const std::uint64_t v : {static_cast<std::uint64_t>(L),
                                  static_cast<std::uint64_t>(hundredths),
                                  static_cast<std::uint64_t>(dirCode),
                                  static_cast<std::uint64_t>(phase)}) {
        x ^= v + 0x9e3779b97f4a7c15ull + (x << 6) + (x >> 2);
    }
    return x;
}

struct Point {
    d64 e = 0.0, eErr = 0.0, absm = 0.0, absmErr = 0.0;
};

// Everything needed for one temp run
struct Site {
    u32 L;
    d64 beta;
    const std::vector<std::vector<Ising2D::Spin>>* warm;
    std::vector<std::vector<Ising2D::Spin>>* out;
    d64 drift = 0.0;
};

RunResult runChains(Site& site, const RunConfig& cfg) {
    auto factory = [&](u32 c, std::mt19937& g) {
        Ising2D m(site.L, 1.0, 0.0);
        if (c < site.warm->size() && !(*site.warm)[c].empty()) {
            m.setSpins((*site.warm)[c]);
        } else {
            m.randomise(g);
        }

        return IsingMetropolis(std::move(m), site.beta);
    };

    auto harvest = [&](u32 c, const auto& stepper) {
        const Ising2D& s = stepper.state();
        // flip() tracks the energy incrementally, so this checks it against
        // the total energy computed at the end
        site.drift = std::max(site.drift, std::abs(s.energy() - s.computeEnergy()));
        (*site.out)[c] = s.spins();
    };

    return run(factory, IsingObservables{}, cfg, harvest);
}

} // namespace 

int main(int argc, char** argv) {
    Args args;
    if (!parseArgs(argc, argv, args)) {
        return 1;
    }

    const std::vector<int> ascending = grid(args.tminH, args.tmaxH);
    if (ascending.size() < 2) {
        std::fprintf(stderr, "temperature range too narrow\n");
        return 1;
    }

    std::vector<std::string> directions;
    if (args.direction == "both") {
        directions = {"cooling", "heating"};
    } else {
        directions = {args.direction};
    }

    std::printf("2D Ising, single spin flip metropolis\n");
    std::printf("exact T_c = %.6f\n", kTc);
    std::printf("%zu temperatures from %.2f to %.2f, %zu chains\n",
                ascending.size(), args.tminH / 100.0, args.tmaxH / 100.0,
                args.numChains);
    std::printf("burn-in 20 tau, sweeps chosen for ESS(|m|) >= %zu, "
                "clamped to [%zu, %zu]\n\n",
                args.targetEss, args.baseSweeps, args.maxSweeps);
 
    std::map<std::pair<u32, std::string>, std::map<int, Point>> scans;
    for (const u32 L : args.sizes) {
        for (const std::string& dirName : directions) {
            const bool cooling = dirName == "cooling";
 
            std::vector<int> temps = ascending;
            if (cooling) {
                std::reverse(temps.begin(), temps.end());
            }
 
            std::string outDir = args.dir;
            if (directions.size() > 1) {
                outDir += "/" + dirName;
            }
            const std::string mk = "mkdir -p " + outDir;
            if (std::system(mk.c_str()) != 0) {
                std::fprintf(stderr, "could not create %s\n", outDir.c_str());
                return 1;
            }
 
            std::printf("L = %zu (N = %zu), %s, writing to %s/\n",
                        L, L * L, dirName.c_str(), outDir.c_str());
            std::printf("%7s %10s %10s %10s %9s %9s %8s %7s %8s %5s %8s %7s  %s\n",
                        "T", "e", "m", "|m|", "err(|m|)", "tau(|m|)", "ESS",
                        "burn", "sweeps", "thin", "acc", "Rhat", "flags");
 
            std::vector<std::vector<Ising2D::Spin>> warm;
            d64 worstDrift = 0.0;
            u32 flagged = 0;
 
            for (const int h : temps) {
                const d64 T = h / 100.0;
                const d64 beta = 1.0 / T;
 
                std::vector<std::vector<Ising2D::Spin>> next(args.numChains);
                Site site{L, beta, &warm, &next, 0.0};
 
                // Pilot: short runs purely to measure tau here, so burn-in and
                // length follow the correlation time at this temperature
                // rather than a fixed fraction that happened to be smallest
                // where tau is largest. A tau measured on a chain shorter than
                // ~50 tau is only a lower bound, so the pilot extends itself
                // until it can resolve what it found, or hits the ceiling.
                u32 pilotSweeps = std::max<u32>(1000, args.baseSweeps / 10);
                d64 tau = 1.0;
                bool tauUnreliable = true;
 
                for (u32 attempt = 0; attempt < 4; ++attempt) {
                    RunConfig pc;
                    pc.numChains = args.numChains;
                    pc.sweeps = pilotSweeps;
                    pc.burnIn = warm.empty() ? pilotSweeps : pilotSweeps / 2;
                    pc.thin = 1;
                    pc.seed = mixSeed(args.seed, L, h, cooling ? 0 : 1, attempt);
 
                    const RunResult pilot = runChains(site, pc);
                    tau = pilot.tauSweeps[pilot.obsIndex("abs_m")];
 
                    // the pilot's endpoint is better equilibrated than the
                    // point it began from, so the next run continues from it
                    warm = next;
                    next.assign(args.numChains, {});
 
                    tauUnreliable = tau * 50.0 > static_cast<d64>(pilotSweeps);
                    if (!tauUnreliable || pilotSweeps >= args.maxSweeps) {
                        break;
                    }
                    pilotSweeps = std::min<u32>(
                        args.maxSweeps, static_cast<u32>(std::ceil(50.0 * tau)));
                }
 
                const u32 burnIn = std::clamp<u32>(
                    static_cast<u32>(std::ceil(20.0 * tau)), 500, args.maxSweeps / 2);
 
                u32 sweeps = std::clamp<u32>(
                    static_cast<u32>(std::ceil(tau * args.targetEss / args.numChains)),
                    args.baseSweeps, args.maxSweeps);
 
                const u32 thin = std::max<u32>(
                    1, (sweeps + args.maxDraws - 1) / args.maxDraws);
                sweeps = (sweeps / thin) * thin; // run() requires a whole multiple
 
                RunConfig cfg;
                cfg.numChains = args.numChains;
                cfg.sweeps = sweeps;
                cfg.burnIn = burnIn;
                cfg.thin = thin;
                cfg.seed = mixSeed(args.seed, L, h, cooling ? 0 : 1, 100);
 
                const RunResult res = runChains(site, cfg);
                warm = std::move(next);
                worstDrift = std::max(worstDrift, site.drift);
 
                const u32 ie = res.obsIndex("e");
                const u32 im = res.obsIndex("m");
                const u32 iam = res.obsIndex("abs_m");
 
                char stem[512];
                std::snprintf(stem, sizeof(stem), "%s/ising_L%zu_T%d.%02d",
                              outDir.c_str(), L, h / 100, h % 100);
 
                const bool frozen = res.acceptedFraction < 1e-6;
                const bool shortEss = res.ess[iam] < static_cast<d64>(args.targetEss);
 
                io::JsonWriter extra;
                extra.add("model", std::string("ising2d"))
                    .add("algorithm", std::string("metropolis"))
                    .add("L", L)
                    .add("N", L * L)
                    .add("J", 1.0)
                    .add("field", 0.0)
                    .add("T", T)
                    .add("beta", beta)
                    .add("T_c_exact", kTc)
                    .add("direction", dirName)
                    .add("warm_started", true)
                    .add("tau_pilot_sweeps", tau)
                    .add("tau_pilot_unreliable", tauUnreliable)
                    .add("target_ess", args.targetEss)
                    .add("ess_below_target", shortEss)
                    .add("frozen", frozen)
                    .add("energy_drift", site.drift);
 
                save(res, stem, cfg, extra);
 
                std::string flags;
                if (frozen) flags += " frozen";
                if (tauUnreliable) flags += " tau-lower-bound";
                if (shortEss) flags += " ESS<target";
                if (res.rHat[iam] > 1.01) flags += " Rhat";
                if (!flags.empty()) ++flagged;
 
                std::printf("%7.2f %10.5f %10.5f %10.5f %9.5f %9.2f %8.0f "
                            "%7zu %8zu %5zu %8.2e %7.4f %s\n",
                            T, res.mean[ie], res.mean[im], res.mean[iam],
                            res.stdError[iam], res.tauSweeps[iam], res.ess[iam],
                            burnIn, sweeps, thin, res.acceptedFraction,
                            res.rHat[iam], flags.c_str());
                std::fflush(stdout);
 
                Point pt;
                pt.e = res.mean[ie];
                pt.eErr = res.stdError[ie];
                pt.absm = res.mean[iam];
                pt.absmErr = res.stdError[iam];
                scans[{L, dirName}][h] = pt;
            }
 
            std::printf("  worst |energy - recomputed| = %.3e over the scan\n",
                        worstDrift);
            if (flagged) {
                std::printf("  %zu of %zu temperatures carry a flag\n",
                            flagged, temps.size());
            }
            std::printf("\n");
        }
    }

    // Cooling drags every point towards its hotter neighbour and heating
    // towards its colder one, so agreement between the two is the test
    // that the warm start is not biasing the scan.
    if (directions.size() > 1) {
        std::printf("hysteresis check, cooling against heating:\n");
        for (const u32 L : args.sizes) {
            const auto& cool = scans[{L, std::string("cooling")}];
            const auto& heat = scans[{L, std::string("heating")}];
 
            d64 worstE = 0.0, worstM = 0.0;
            int atE = 0, atM = 0;
 
            for (const auto& entry : cool) {
                const auto it = heat.find(entry.first);
                if (it == heat.end()) continue;
 
                const Point& c = entry.second;
                const Point& w = it->second;
 
                const d64 sE = std::hypot(c.eErr, w.eErr);
                const d64 sM = std::hypot(c.absmErr, w.absmErr);
                if (sE > 0.0 && std::abs(c.e - w.e) / sE > worstE) {
                    worstE = std::abs(c.e - w.e) / sE;
                    atE = entry.first;
                }
                if (sM > 0.0 && std::abs(c.absm - w.absm) / sM > worstM) {
                    worstM = std::abs(c.absm - w.absm) / sM;
                    atM = entry.first;
                }
            }
 
            std::printf("  L = %zu: largest gap %.1f sigma in e at T = %.2f, "
                        "%.1f sigma in |m| at T = %.2f\n",
                        L, worstE, atE / 100.0, worstM, atM / 100.0);
        }
        std::printf("\n");
    }
 
    std::printf("analyse with: python3 analysis/analyze_ising.py %s\n",
                args.dir.c_str());
    return 0;
}
