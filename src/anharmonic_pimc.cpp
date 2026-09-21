// This uses the full Markov Chain Monte Carlo suite in mcmc/ to simulate the quantum anharmonic oscillator
// the output data is analyzed in analysis/analyze_anharmonic.py

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <span>
#include <string>
#include <vector>
#include <fftw3.h>
#include <stdexcept>
#include <algorithm>
#include <cstdint>
 
#include "types.hpp"
#include "linalg/vec.hpp"
#include "linalg/matrix.hpp"
#include "mcmc/continuous.hpp"
#include "mcmc/driver.hpp"
#include "calculus/target_distributions/differentiable_target.hpp"
 
using namespace mcmc;

namespace {

// parameters
struct Params {
    u32 N = 200; // Imaginary time slices
    d64 m = 1.0; // mass
    d64 omega = 1.0; // frequency
    d64 lambda = 0.1; // quartic coupling
    d64 beta = 10.0; // inverse temp

    d64 dtau() const {return beta / static_cast<d64>(N);}
};

// Target

class AnharmonicPath : public calculus::sample::DifferentiableTarget {
public:
    explicit AnharmonicPath(const Params& p) : p_(p) {}

    d64 logDensity(const linalg::Vec<d64>& x) const override {
        const u32 N = p_.N;
        const d64 dt = p_.dtau();
        d64 action = 0.0;
        for (u32 i = 0; i < N; ++i) {
            const d64 xi = x(i);
            const d64 dx = x((i + 1) % N) - xi;
            action += (p_.m / (2.0 * dt)) * dx * dx
                    + dt * (0.5 * p_.m * p_.omega * p_.omega * xi * xi
                            + p_.lambda * xi * xi * xi * xi);
        }
        return -action;
    }

    linalg::Vec<d64> gradLogDensity(const linalg::Vec<d64>& x) const override {
        const u32 N = p_.N;
        const d64 dt = p_.dtau();
        linalg::Vec<d64> g(N);
        for (u32 i = 0; i < N; ++i) {
            const d64 xi = x(i);
            const d64 lap = x((i + 1) % N) - 2.0 * xi + x((i + N - 1) % N);
            const d64 dSkin = -(p_.m / dt) * lap;
            const d64 dSpot = dt * (p_.m * p_.omega * p_.omega * xi
                                    + 4.0 * p_.lambda * xi * xi * xi);
            g(i) = -(dSkin + dSpot);
        }
        return g;
    }
 
    u32 dim() const override {return p_.N;}

private:
    Params p_;
};

// observables
struct PIMCObservables {
    Params p;
    bool recordPath = true;

    static constexpr u32 numScalars = 8;
    std::vector<std::string> names () const {
        std::vector<std::string> v = {"E_vir", "E_prim", "E_vir_E_prim", "E_vir_sq",
                                        "m2", "m4", "xbar", "Rg2"};

        if (recordPath) {
            for (u32 i = 0; i < p.N; ++i) {
                v.push_back("x" + std::to_string(i));
            }
        }
        return v;
    }

    void eval(const linalg::Vec<d64>& x, std::span<d64> out) const {
        const u32 N = p.N;
        const d64 dN = static_cast<d64>(N);

        d64 m2 = 0.0, m4 = 0.0, xbar = 0.0, spring = 0.0;
        for (u32 i = 0; i < N; ++i) {
            const d64 xi = x(i);
            const d64 x2 = xi * xi;
            m2 += x2;
            m4 += x2 * x2;
            xbar += xi;
            const d64 dx = x((i + 1) % N) - xi;
            spring += dx * dx;
        }
        m2 /= dN;
        m4 /= dN;
        xbar /= dN;

        const d64 meanV = 0.5 * p.m * p.omega * p.omega * m2 + p.lambda * m4;
        const d64 eVir  = p.m * p.omega * p.omega * m2 + 3.0 * p.lambda * m4;
        const d64 ePrim = dN / (2.0 * p.beta)
                        - (p.m * dN / (2.0 * p.beta * p.beta)) * spring
                        + meanV;

        out[0] = eVir;
        out[1] = ePrim;
        out[2] = eVir * ePrim;
        out[3] = eVir * eVir;
        out[4] = m2;
        out[5] = m4;
        out[6] = xbar;
        out[7] = m2 - xbar * xbar;   // radius of gyration squared

        if (recordPath) {
            for (u32 i = 0; i < N; ++i) {
                out[numScalars + i] = x(i);
            }
        }
    }
};

// Tune step size with Robbins-Monro to target acceptance
template <typename Stepper>
d64 tuneScale(Stepper& s, std::mt19937& gen, u32 rounds = 25, u32 perRound = 60) {
    const d64 target = s.targetAcceptance();
    for (u32 r = 0; r < rounds; ++r) {
        s.resetCounters();
        for (u32 i = 0; i < perRound; ++i) {
            s.sweep(gen);
        }
        const d64 acc = s.acceptedFraction();
        const d64 gain = 3.0 / (1.0 + static_cast<d64>(r) / 4.0);
        d64 h = s.scale() * std::exp(gain * (acc - target));
        h = std::min(std::max(h, 1e-10), 1e4);
        s.setScale(h);
    }
    s.resetCounters();
    return s.scale();

}

// Covariance estimate of target distribution to estimate mass matrix.
struct MassEstimate {
    std::vector<d64> spectrum;
    std::vector<d64> autocov;
    std::vector<d64> massRow;
    linalg::Matrix<d64> mass;

    d64 globalMean = 0.0;
    d64 relFrobenius = 0.0;
    d64 diagSpread = 0.0;
    u32 denseDraws = 0;
    d64 analyticRelDiff = 0.0;
};

MassEstimate estimateMass(const RunResult& res, const Params& p, u32 pathOffset, u32 maxDenseDraws) {
    const u32 N = p.N;
    const u32 nq = N / 2 + 1;
    MassEstimate est;

    const u32 totalDraws = res.numChains * res.numDraws;

    d64 sum = 0.0;
    for (u32 c = 0; c < res.numChains; ++c) {
        for (u32 d = 0; d < res.numDraws; ++d) {
            for (u32 i = 0; i < N; ++i) {
                sum += res.at(c, d, pathOffset + i);
            }
        }
    }
    est.globalMean = sum / (static_cast<d64>(totalDraws) * static_cast<d64>(N));

    std::vector<d64> in(N);
    std::vector<c64> out(nq);
    fftw_plan plan = fftw_plan_dft_r2c_1d(static_cast<int>(N), in.data(),
                                          reinterpret_cast<fftw_complex*>(out.data()),
                                          FFTW_ESTIMATE);

    est.spectrum.assign(nq, 0.0);
    for (u32 c = 0; c < res.numChains; ++c) {
        for (u32 d = 0; d < res.numDraws; ++d) {
            for (u32 i = 0; i < N; ++i) {
                in[i] = res.at(c, d, pathOffset + i) - est.globalMean;
            }
            fftw_execute(plan);
            for (u32 q = 0; q < nq; ++q) {
                est.spectrum[q] += std::norm(out[q]);
            }
        }
    }
    fftw_destroy_plan(plan);

    const d64 norm = static_cast<d64>(totalDraws) * static_cast<d64>(N);
    for (u32 q = 0; q < nq; ++q) {
        est.spectrum[q] /= norm;
        if (!(est.spectrum[q] > 0.0)) {
            throw std::runtime_error("estimateMass: non-positive spectrum, the warm-up run is degenerate");
        }
    }

    auto inverseTransform = [&](auto valueAt) {
        std::vector<d64> row(N, 0.0);
        const d64 dN = static_cast<d64>(N);
        for (u32 k = 0; k < N; ++k) {
            d64 acc = valueAt(0u);
            u32 qHi = nq - 1;
            const bool nyquist = (N % 2 == 0);
            if (nyquist) {
                acc += valueAt(qHi) * ((k % 2 == 0) ? 1.0 : -1.0);
            }
            const u32 last = nyquist ? qHi - 1 : qHi;
            for (u32 q = 1; q <= last; ++q) {
                acc += 2.0 * valueAt(q)
                     * std::cos(2.0 * M_PI * static_cast<d64>(q) * static_cast<d64>(k) / dN);
            }
            row[k] = acc / dN;
        }
        return row;
    };

    est.autocov = inverseTransform([&](u32 q) {return est.spectrum[q];});
    est.massRow = inverseTransform([&](u32 q) {return 1.0 / est.spectrum[q];});

    est.mass = linalg::Matrix<d64>(N, N);
    for (u32 i = 0; i < N; ++i) {
        for (u32 j = 0; j < N; ++j) {
            est.mass(i, j) = est.massRow[(j + N - i) % N];
        }
    }

    const u32 stride = std::max<u32>(1, totalDraws / std::max<u32>(1, maxDenseDraws));
    linalg::Matrix<d64> S(N, N);
    u32 used = 0;
    std::vector<d64> dx(N);
    for (u32 c = 0; c < res.numChains; ++c) {
        for (u32 d = 0; d < res.numDraws; d += stride) {
            for (u32 i = 0; i < N; ++i) {
                dx[i] = res.at(c, d, pathOffset + i) - est.globalMean;
            }
            for (u32 i = 0; i < N; ++i) {
                for (u32 j = 0; j < N; ++j) {
                    S(i, j) += dx[i] * dx[j];
                }
            }
            ++used;
        }
    }
    est.denseDraws = used;
    if (used > 1) {
        const d64 den = static_cast<d64>(used - 1);
        d64 num = 0.0, tot = 0.0;
        for (u32 i = 0; i < N; ++i) {
            for (u32 j = 0; j < N; ++j) {
                S(i, j) /= den;
            }
        }
        for (u32 i = 0; i < N; ++i) {
            for (u32 j = 0; j < N; ++j) {
                const d64 diff = S(i, j) - est.autocov[(j + N - i) % N];
                num += diff * diff;
                tot += S(i, j) * S(i, j);
            }
        }
        est.relFrobenius = (tot > 0.0) ? std::sqrt(num / tot) : 0.0;
 
        d64 meanDiag = 0.0;
        for (u32 i = 0; i < N; ++i) {
            meanDiag += S(i, i);
        }
        meanDiag /= static_cast<d64>(N);
        for (u32 i = 0; i < N; ++i) {
            est.diagSpread = std::max(est.diagSpread,
                                      std::abs(S(i, i) - meanDiag) / meanDiag);
        }
    }
 
    const d64 dt = p.dtau();
    std::vector<d64> hRow(N, 0.0);
    hRow[0] = 2.0 * p.m / dt + dt * p.m * p.omega * p.omega;
    hRow[1] = -p.m / dt;
    hRow[N - 1] = -p.m / dt;
    d64 num = 0.0, tot = 0.0;
    for (u32 k = 0; k < N; ++k) {
        const d64 diff = est.massRow[k] - hRow[k];
        num += static_cast<d64>(N) * diff * diff;
        tot += static_cast<d64>(N) * hRow[k] * hRow[k];
    }
    est.analyticRelDiff = std::sqrt(num / tot);
 
    return est;
}

// Use blocked jackknife to calculate Cv since its a covariance for E_prim

struct Jackknife {
    d64 value = 0.0;
    d64 error = 0.0;
    u32 blocks = 0;
    u32 blockLen = 0;
};

Jackknife heatCapacity(const RunResult& res, d64 beta,
                       u32 iVir, u32 iPrim, u32 iCross, d64 tau) {
    Jackknife out;
    out.blockLen = std::max<u32>(1, static_cast<u32>(std::ceil(10.0 * tau)));
    if (res.numDraws < 4 * out.blockLen) {
        out.blockLen = std::max<u32>(1, res.numDraws / 4);
    }
 
    std::vector<d64> sa, sb, sab;
    std::vector<d64> cnt;
    for (u32 c = 0; c < res.numChains; ++c) {
        for (u32 start = 0; start + out.blockLen <= res.numDraws; start += out.blockLen) {
            d64 a = 0.0, b = 0.0, ab = 0.0;
            for (u32 d = start; d < start + out.blockLen; ++d) {
                a += res.at(c, d, iVir);
                b += res.at(c, d, iPrim);
                ab += res.at(c, d, iCross);
            }
            sa.push_back(a);
            sb.push_back(b);
            sab.push_back(ab);
            cnt.push_back(static_cast<d64>(out.blockLen));
        }
    }
 
    out.blocks = sa.size();
    if (out.blocks < 2) {
        return out;
    }
 
    d64 SA = 0.0, SB = 0.0, SAB = 0.0, NT = 0.0;
    for (u32 b = 0; b < out.blocks; ++b) {
        SA += sa[b]; SB += sb[b]; SAB += sab[b]; NT += cnt[b];
    }
    out.value = beta * beta * (SAB / NT - (SA / NT) * (SB / NT));
 
    std::vector<d64> leave(out.blocks);
    d64 mean = 0.0;
    for (u32 b = 0; b < out.blocks; ++b) {
        const d64 n = NT - cnt[b];
        const d64 a = (SA - sa[b]) / n;
        const d64 bb = (SB - sb[b]) / n;
        const d64 ab = (SAB - sab[b]) / n;
        leave[b] = beta * beta * (ab - a * bb);
        mean += leave[b];
    }
    mean /= static_cast<d64>(out.blocks);
 
    d64 ss = 0.0;
    for (const d64 v : leave) {
        ss += (v - mean) * (v - mean);
    }
    out.error = std::sqrt(ss * static_cast<d64>(out.blocks - 1)
                          / static_cast<d64>(out.blocks));
    return out;
}

// Command line
struct Args {
    Params p;
    u32 numChains = 4;
    u32 malaSweeps = 20000;
    u32 malaBurn = 5000;
    d64 malaH = 0.01;
    u32 sweeps = 50000;
    u32 burnIn = 5000;
    u32 thin = 1;
    d64 eps = 0.3;
    d64 epsMax = 0.6;
    d64 traj = M_PI / 2.0;   // whitened modes have unit frequency, period 2 pi
    u32 leapSteps = 0;       // 0 = derive from traj / eps
    u32 denseDraws = 20000;
    bool recordPath = true;
    bool tune = true;
    bool analyticMass = false;
    std::uint64_t seed = 20260921ull;
    std::string dir = "data";
};
 
void usage() {
    std::fprintf(stderr,
        "anharmonic_pimc [options]\n"
        "  --N <int>          imaginary time slices (default 200)\n"
        "  --beta <f>         inverse temperature (default 10)\n"
        "  --omega <f>        harmonic frequency (default 1)\n"
        "  --lambda <f>       quartic coupling (default 0.1; 0 gives the exact HO)\n"
        "  --mass <f>         particle mass (default 1)\n"
        "  --chains <int>     chains per phase (default 4)\n"
        "  --mala-sweeps <int>  MALA warm-up sweeps (default 20000)\n"
        "  --mala-h <f>       MALA step size seed (default 0.01)\n"
        "  --sweeps <int>     HMC production sweeps (default 50000)\n"
        "  --burnin <int>     HMC burn-in (default 5000)\n"
        "  --thin <int>       record every n-th sweep (default 1)\n"
        "  --eps <f>          leapfrog step size seed (default 0.3)\n"
        "  --eps-max <f>      cap on the tuned step size (default 0.6)\n"
        "  --traj <f>         trajectory length eps*L (default pi/2)\n"
        "  --steps <int>      fix leapfrog steps; 0 derives L from --traj (default 0)\n"
        "  --dense-draws <int>  draws used for the dense covariance check (default 20000)\n"
        "  --no-path          do not record bead positions in the production run\n"
        "  --no-tune          skip step size tuning, use the seeds as given\n"
        "  --analytic-mass    skip the MALA phase, use the harmonic Hessian\n"
        "  --seed <int>       base seed\n"
        "  --dir <path>       output directory (default data)\n");
}
 
bool parseArgs(int argc, char** argv, Args& a) {
    auto next = [&](int& i) -> const char* {
        return (i + 1 < argc) ? argv[++i] : nullptr;
    };
    for (int i = 1; i < argc; ++i) {
        const std::string f = argv[i];
        const char* v = nullptr;
        if (f == "--N") {if (!(v = next(i))) return false; a.p.N = std::strtoul(v, nullptr, 10);}
        else if (f == "--beta") {if (!(v = next(i))) return false; a.p.beta = std::strtod(v, nullptr);}
        else if (f == "--omega") {if (!(v = next(i))) return false; a.p.omega = std::strtod(v, nullptr);}
        else if (f == "--lambda") {if (!(v = next(i))) return false; a.p.lambda = std::strtod(v, nullptr);}
        else if (f == "--mass") {if (!(v = next(i))) return false; a.p.m = std::strtod(v, nullptr);}
        else if (f == "--chains") {if (!(v = next(i))) return false; a.numChains = std::strtoul(v, nullptr, 10);}
        else if (f == "--mala-sweeps") {if (!(v = next(i))) return false; a.malaSweeps = std::strtoul(v, nullptr, 10);}
        else if (f == "--mala-h") {if (!(v = next(i))) return false; a.malaH = std::strtod(v, nullptr);}
        else if (f == "--sweeps") {if (!(v = next(i))) return false; a.sweeps = std::strtoul(v, nullptr, 10);}
        else if (f == "--burnin") {if (!(v = next(i))) return false; a.burnIn = std::strtoul(v, nullptr, 10);}
        else if (f == "--thin") {if (!(v = next(i))) return false; a.thin = std::strtoul(v, nullptr, 10);}
        else if (f == "--eps") {if (!(v = next(i))) return false; a.eps = std::strtod(v, nullptr);}
        else if (f == "--eps-max") {if (!(v = next(i))) return false; a.epsMax = std::strtod(v, nullptr);}
        else if (f == "--traj") {if (!(v = next(i))) return false; a.traj = std::strtod(v, nullptr);}
        else if (f == "--steps") {if (!(v = next(i))) return false; a.leapSteps = std::strtoul(v, nullptr, 10);}
        else if (f == "--dense-draws") {if (!(v = next(i))) return false; a.denseDraws = std::strtoul(v, nullptr, 10);}
        else if (f == "--seed") {if (!(v = next(i))) return false; a.seed = std::strtoull(v, nullptr, 10);}
        else if (f == "--dir") {if (!(v = next(i))) return false; a.dir = v;}
        else if (f == "--no-path") {a.recordPath = false;}
        else if (f == "--no-tune") {a.tune = false;}
        else if (f == "--analytic-mass") {a.analyticMass = true;}
        else {std::fprintf(stderr, "unknown option %s\n", f.c_str()); usage(); return false;}
    }
    if (a.p.N < 8 || a.numChains == 0 || a.sweeps < 100 || a.thin == 0) {
        std::fprintf(stderr, "invalid --N, --chains, --sweeps or --thin\n");
        return false;
    }
    if (a.sweeps % a.thin != 0) {
        a.sweeps = (a.sweeps / a.thin) * a.thin;   // run() requires a whole multiple
    }
    return true;
}

// Small thermal noise about classical min
linalg::Vec<d64> makeStart(u32 N, std::mt19937& g, d64 spread) {
    std::normal_distribution<d64> noise(0.0, spread);
    return linalg::Vec<d64>::random(N, noise, g);
}

// Exact reference at lambda = 0
struct ExactHO {
    d64 energy, x2, heatCapacity;
    d64 contEnergy, contX2, contHeatCapacity;
};

d64 latticeX2(const Params& p, d64 beta) {
    const d64 dt = beta / static_cast<d64>(p.N);
    d64 s = 0.0;
    for (u32 q = 0; q < p.N; ++q) {
        const d64 h = (2.0 * p.m / dt)
                        * (1.0 - std::cos(2.0 * M_PI * static_cast<d64>(q)
                                          / static_cast<d64>(p.N)))
                    + dt * p.m * p.omega * p.omega;
        s += 1.0 / h;
    }
    return s / static_cast<d64>(p.N);
}

ExactHO exactHO(const Params& p) {
    ExactHO e;
    const d64 w = p.omega, b = p.beta;
 
    e.x2 = latticeX2(p, b);
    e.energy = p.m * w * w * e.x2;
    const d64 hb = 1e-5 * b;
    const d64 ep = p.m * w * w * latticeX2(p, b + hb);
    const d64 em = p.m * w * w * latticeX2(p, b - hb);
    e.heatCapacity = -b * b * (ep - em) / (2.0 * hb);
 
    const d64 s = std::sinh(b * w / 2.0);
    e.contEnergy = 0.5 * w / std::tanh(b * w / 2.0);
    e.contX2 = e.contEnergy / (p.m * w * w);
    e.contHeatCapacity = b * b * w * w / (4.0 * s * s);
    return e;
}

} // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parseArgs(argc, argv, a)) {
        return 1;
    }
 
    const Params p = a.p;
    const AnharmonicPath target(p);
 
    std::printf("anharmonic oscillator PIMC\n");
    std::printf("  N = %zu, beta = %.4g, dtau = %.4g, m = %.4g, omega = %.4g, lambda = %.4g\n",
                p.N, p.beta, p.dtau(), p.m, p.omega, p.lambda);
    std::printf("  beta*omega = %.3g, exp(-beta*dE) ~ %.2e\n\n",
                p.beta * p.omega, std::exp(-p.beta * p.omega));
 
    linalg::Matrix<d64> mass;
    MassEstimate est;
    d64 malaH = a.malaH;

    // Phase 1, do MALA warm-up run, and use it to estimate covariance
    if (a.analyticMass) {
        std::printf("phase 1: skipped, using the analytic harmonic Hessian\n\n");
        const d64 dt = p.dtau();
        mass = linalg::Matrix<d64>(p.N, p.N);
        for (u32 i = 0; i < p.N; ++i) {
            mass(i, i) = 2.0 * p.m / dt + dt * p.m * p.omega * p.omega;
            mass(i, (i + 1) % p.N) = -p.m / dt;
            mass(i, (i + p.N - 1) % p.N) = -p.m / dt;
        }
    } else {
        std::printf("phase 1: MALA warm-up, estimating the path covariance\n");
        
        if (a.tune) {
            std::mt19937 tuneGen(a.seed ^ 0x9e3779b9ull);
            MALAStepper probe(target, makeStart(p.N, tuneGen, 0.3), a.malaH);
            malaH = tuneScale(probe, tuneGen);
            std::printf(". tuned MALA h: %.4g -> %.4g\n", a.malaH, malaH);
        }

        RunConfig mcfg;
        mcfg.sweeps = a.malaSweeps;
        mcfg.burnIn = a.malaBurn;
        mcfg.thin = 1;
        mcfg.numChains = a.numChains;
        mcfg.seed = a.seed;

        auto malaFactory = [&](u32, std::mt19937& g) {
            return MALAStepper(target, makeStart(p.N, g, 0.5), malaH);
        };

        const RunResult mres = run(malaFactory, PIMCObservables{p, true}, mcfg);
 
        std::printf("  acceptance %.3f, <E_vir> = %.6f +- %.6f, tau = %.1f, rhat = %.4f, %.1fs\n",
                    mres.acceptedFraction,
                    mres.mean[0], mres.stdError[0], mres.tauInt[0], mres.rHat[0],
                    mres.timeTaken);
 
        est = estimateMass(mres, p, PIMCObservables::numScalars, a.denseDraws);
        mass = est.mass;
 
        std::printf("  spectrum chat(q): min %.4g at q=%zu, max %.4g at q=0\n",
                    est.spectrum.back(), est.spectrum.size() - 1, est.spectrum[0]);
        std::printf("  cross-check, dense vs circulant: relative Frobenius %.4f, "
                    "diagonal spread %.4f, from %zu draws\n",
                    est.relFrobenius, est.diagSpread, est.denseDraws);
        std::printf("  cross-check, M vs analytic harmonic Hessian: relative Frobenius %.4f\n\n",
                    est.analyticRelDiff);
 
        mcmc::io::JsonWriter mextra;
        mextra.add("phase", std::string("mala"))
              .add("N", p.N).add("beta", p.beta).add("omega", p.omega)
              .add("lambda", p.lambda).add("mass_param", p.m).add("dtau", p.dtau())
              .add("mala_h", malaH)
              .add("spectrum", est.spectrum)
              .add("autocov", est.autocov)
              .add("mass_row", est.massRow)
              .add("dense_vs_circulant_rel_frobenius", est.relFrobenius)
              .add("dense_diag_spread", est.diagSpread)
              .add("mass_vs_analytic_rel_frobenius", est.analyticRelDiff);
        mcmc::save(mres, a.dir + "/anharmonic_mala", mcfg, mextra);
    }

    // Phase 2
    std::printf("phase 2: HMC production\n");
    d64 eps = a.eps;
    u32 leapSteps = (a.leapSteps > 0) ? a.leapSteps : 8;

    if (a.tune) {
        std::mt19937 tuneGen(a.seed ^ 0xc2b2ae35ull);
        HMCStepper probe(target, makeStart(p.N, tuneGen, 0.3), mass, a.eps, leapSteps);
        eps = std::min(tuneScale(probe, tuneGen), a.epsMax);
    }
    if (a.leapSteps == 0) {
        leapSteps = std::max<u32>(1, static_cast<u32>(std::llround(a.traj / eps)));
    }
    std::printf("  leapfrog eps %.4g (seed %.4g, cap %.4g), L = %zu, trajectory %.4g\n",
                eps, a.eps, a.epsMax, leapSteps, eps * static_cast<d64>(leapSteps));

    RunConfig cfg;
    cfg.sweeps = a.sweeps;
    cfg.burnIn = a.burnIn;
    cfg.thin = a.thin;
    cfg.numChains = a.numChains;
    cfg.seed = a.seed + 1;
 
    u32 divergences = 0;
    auto hmcFactory = [&](u32, std::mt19937& g) {
        return HMCStepper(target, makeStart(p.N, g, 0.5), mass, eps, leapSteps);
    };
    auto harvest = [&](u32, const HMCStepper& s) {
        divergences += s.divergences();
    };
 
    const RunResult res = run(hmcFactory, PIMCObservables{p, a.recordPath}, cfg, harvest);

    // report
    const u32 iVir = res.obsIndex("E_vir");
    const u32 iPrim = res.obsIndex("E_prim");
    const u32 iCross = res.obsIndex("E_vir_E_prim");
    const u32 iM2 = res.obsIndex("m2");
    const u32 iM4 = res.obsIndex("m4");
    const u32 iRg = res.obsIndex("Rg2");
 
    const d64 tauMax = std::max(res.tauInt[iVir], res.tauInt[iPrim]);
    const Jackknife cvJack = heatCapacity(res, p.beta, iVir, iPrim, iCross, tauMax);
    const d64 cv = cvJack.value;
    const d64 kurtosis = res.mean[iM4] / (res.mean[iM2] * res.mean[iM2]);
 
    std::printf("  acceptance %.3f, divergences %zu, %.1fs\n",
                res.acceptedFraction, divergences, res.timeTaken);
    std::printf("  %-14s %12s %12s %10s %10s %8s\n",
                "observable", "mean", "std err", "tau", "ess", "rhat");
    for (const char* n : {"E_vir", "E_prim", "m2", "m4", "Rg2", "xbar"}) {
        const u32 o = res.obsIndex(n);
        std::printf("  %-14s %12.6f %12.6f %10.2f %10.0f %8.4f\n",
                    n, res.mean[o], res.stdError[o], res.tauInt[o],
                    res.ess[o], res.rHat[o]);
    }
    std::printf("\n  Cv = beta^2 Cov(E_vir, E_prim) = %.6f +- %.6f"
                "  (jackknife, %zu blocks of %zu)\n",
                cv, cvJack.error, cvJack.blocks, cvJack.blockLen);
    std::printf("  <x^4>/<x^2>^2 = %.6f (3 = gaussian, excess %.6f)\n",
                kurtosis, kurtosis - 3.0);
    std::printf("  estimator variance ratio, primitive/virial = %.1f\n",
                res.chains[0].variance[iPrim] / res.chains[0].variance[iVir]);
    std::printf("  <Rg^2> = %.6f\n", res.mean[iRg]);
 
    if (p.lambda == 0.0) {
        const ExactHO e = exactHO(p);
        std::printf("\n  lambda = 0, exact reference (lattice at this N, then continuum):\n");
        std::printf("    E     lattice %.8f  sampled %.8f  dev %+.2f sigma   (continuum %.8f)\n",
                    e.energy, res.mean[iVir],
                    (res.mean[iVir] - e.energy) / res.stdError[iVir], e.contEnergy);
        std::printf("    <x^2> lattice %.8f  sampled %.8f  dev %+.2f sigma   (continuum %.8f)\n",
                    e.x2, res.mean[iM2],
                    (res.mean[iM2] - e.x2) / res.stdError[iM2], e.contX2);
            std::printf("    Cv    lattice %.8f  sampled %.8f +- %.8f  dev %+.2f sigma   (continuum %.8f)\n",
                    e.heatCapacity, cv, cvJack.error,
                    cvJack.error > 0.0 ? (cv - e.heatCapacity) / cvJack.error : 0.0,
                    e.contHeatCapacity);
    }
 
    mcmc::io::JsonWriter extra;
    extra.add("phase", std::string("hmc"))
         .add("N", p.N).add("beta", p.beta).add("omega", p.omega)
         .add("lambda", p.lambda).add("mass_param", p.m).add("dtau", p.dtau())
         .add("leapfrog_eps", eps).add("leapfrog_steps", leapSteps)
         .add("trajectory_length", eps * static_cast<d64>(leapSteps))
         .add("divergences", static_cast<d64>(divergences))
         .add("num_scalars", PIMCObservables::numScalars)
         .add("path_recorded", a.recordPath)
         .add("heat_capacity", cv)
         .add("heat_capacity_error", cvJack.error)
         .add("kurtosis", kurtosis)
         .add("mass_source", std::string(a.analyticMass ? "analytic" : "mala_circulant"));
    if (!a.analyticMass) {
        extra.add("autocov", est.autocov).add("mass_row", est.massRow);
    }
    mcmc::save(res, a.dir + "/anharmonic_hmc", cfg, extra);
 
    std::printf("\nwrote %s/anharmonic_hmc.{npy,json}\n", a.dir.c_str());
    return 0;
}