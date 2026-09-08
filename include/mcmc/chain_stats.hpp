#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <mutex>
#include <span>
#include <stdexcept>
#include <vector>
#include <fftw3.h>

#include "types.hpp"

// autocorrelation and convergence diagnostics for MCMC chains.

namespace mcmc {
 
namespace stats {
 
namespace detail {

    inline u32 nextPow2(u32 n) {
        u32 m = 1;
        while (m < n) {
            m <<= 1;
        }
        return m;
    }

    inline u32 goodFFTSize(u32 n) {
        if (n <= 4) {
            return std::max<u32>(n, 1);
        }

        const u32 limit = nextPow2(n);
        u32 best = limit;
        for (u32 p7 = 1; p7 < limit; p7 *= 7) {
            for (u32 p5 = p7; p5 <= limit; p5 *= 5) {
                for (u32 p3 = p5; p3 <= limit; p3 *= 3) {
                    u32 v = p3;
                    while (v < n) {
                        v *= 2;
                    }
                    if (v <= limit) {
                        best = std::min(best, v);
                    }
                    if (p3 > limit / 3) {
                        break;
                    }
                }
                if (p5 > limit / 5) {
                    break;
                }
            }
            if (p7 > limit / 7) {
                break;
            }
        }
        return best;
    }

    inline std::mutex& planMutex() {
        static std::mutex m;
        return m;
    }

    inline std::vector<d64> autocovariance(std::span<const d64> centred, u32 maxLag) {
        const u32 n = centred.size();
        const u32 m = goodFFTSize(2 * n);

        std::vector<d64> real(m, 0.0);
        std::vector<c64> freq(m / 2 + 1);

        fftw_plan forward = nullptr;
        fftw_plan backward = nullptr;
        {
            std::lock_guard<std::mutex> lock(planMutex());
            forward = fftw_plan_dft_r2c_1d(
                static_cast<int>(m), real.data(),
                reinterpret_cast<fftw_complex*>(freq.data()), FFTW_ESTIMATE);
            backward = fftw_plan_dft_c2r_1d(
                static_cast<int>(m), reinterpret_cast<fftw_complex*>(freq.data()),
                real.data(), FFTW_ESTIMATE);
        }

        if (!forward || !backward) {
            std::lock_guard<std::mutex> lock(planMutex());
            if (forward) {
                fftw_destroy_plan(forward);
            }
            if (backward) {
                fftw_destroy_plan(backward);
            }
            throw std::runtime_error("autocovariance: FFTW planning failed");
        }

        std::copy(centred.begin(), centred.end(), real.begin());

        fftw_execute(forward);
        for (c64& z : freq) {
            z = c64(std::norm(z), 0.0);
        }
        fftw_execute(backward);

        std::vector<d64> gamma(maxLag + 1, 0.0);

        for (u32 k = 0; k <= maxLag; ++k) {
            gamma[k] = real[k] / (static_cast<d64>(m) * static_cast<d64>(n));
        }

        {
            std::lock_guard<std::mutex> lock(planMutex());
            fftw_destroy_plan(forward);
            fftw_destroy_plan(backward);
        }

        return gamma;
    }

} // namespace detail

// Normalized auutocorrelation function rho(k) = gamma(k) / gamma(0)
// k = 0 ... maxLag
// Uses 1 / N biased normalization
inline std::vector<d64> autocorrelation(std::span<const d64> x, u32 maxLag = 0) {
    const u32 n = x.size();
    if (n < 2) {
        throw std::invalid_argument("autocorrelation: need at least 2 samples");
    }
    if (maxLag == 0 || maxLag > n - 1) {
        maxLag = n - 1;
    }

    d64 mean  = 0.0;
    for (const d64 v : x) {
        mean += v;
    }
    mean /= static_cast<d64>(n);

    std::vector<d64> centred(n);
    for (u32 i = 0; i < n; ++i) {
        centred[i] = x[i] - mean;
    }

    const std::vector<d64> gamma = detail::autocovariance(centred, maxLag);

    std::vector<d64> rho(maxLag + 1, 0.0);
    if (gamma[0] <= 0.0) {
        // constant series: no variance, so the chain carries no information
        // about correlation. set rho(0) = 1
        rho[0] = 1.0;
        return rho;
    }

    for (u32 k = 0; k <= maxLag; ++k) {
        rho[k] = gamma[k] / gamma[0];
    }
    return rho;
}

// Geyers estimator of tau_int
// clamped at tau >= 1
inline d64 geyerTau(std::span<const d64> rho) {
    std::vector<d64> gamma;
    gamma.reserve(rho.size() / 2);

    for (u32 m = 0; 2 * m + 1 < rho.size(); ++m) {
        const d64 g = rho[2 * m] + rho[2 * m + 1];
        if (g <= 0.0) {
            break;
        }
        gamma.push_back(g);
    }

    for (u32 m = 1; m < gamma.size(); ++m) {
        gamma[m] = std::min(gamma[m], gamma[m - 1]);
    }

    d64 sumGamma = 0.0;
    for (const d64 g : gamma) {
        sumGamma += g;
    }

    return std::max(2.0 * sumGamma - 1.0, 1.0);
}

// Sokal's tau_int. Smallest M st M >= C * tau_int(M)
// If there is no such M then fall back to full windowing.
inline d64 sokalTau(std::span<d64> rho, d64 C = 5.0) {
    d64 runningSum = 0.0;
    for (u32 m = 1; m < rho.size(); ++m) {
        runningSum += rho[m];
        const d64 tau = 1.0 + 2.0 * runningSum;
        if (static_cast<d64>(m) >= C * tau) {
            return std::max(tau, 1.0);
        }
    }

    return std::max(1.0 + 2.0 * runningSum, 1.0);
}

// Gelman-Rubin split R^hat
// Cut each chain in half and look at how they drift
inline d64 splitRHat(const std::vector<std::span<const d64>>& chains) {
    if (chains.empty()) {
        return std::numeric_limits<d64>::quiet_NaN();
    }

    u32 n = std::numeric_limits<u32>::max();
    for (const std::span<const d64>& c : chains) {
        n = std::min<u32>(n, c.size() / 2);
    }

    if (n < 2) {
        return std::numeric_limits<d64>::quiet_NaN();
    }

    std::vector<d64> segMean;
    std::vector<d64> segVar;

    for (const std::span<const d64>& c : chains) {
        for (u32 half = 0; half < 2; ++half) {
            const d64* p = c.data() + half * n;

            d64 mu = 0.0;
            for (u32 i = 0; i < n; ++i) {
                mu += p[i];
            }

            mu /= static_cast<d64>(n);

            d64 ss = 0.0;
            for (u32 i = 0; i < n; ++i) {
                ss += (p[i] - mu) * (p[i] - mu);
            }

            segMean.push_back(mu);
            segVar.push_back(ss / static_cast<d64>(n - 1));
        }
    }

    const u32 m = segMean.size();

    d64 W = 0.0;
    for (const d64 v : segVar) {
        W += v;
    }
    W /= static_cast<d64>(m);

    d64 grand = 0.0;
    for (const d64 v : segMean) {
        grand += v;
    }
    grand /= static_cast<d64>(m);

    d64 B = 0.0;
    for (const d64 v : segMean) {
        B += (v - grand) * (v - grand);
    }
    B *= static_cast<d64>(n) / static_cast<d64>(m - 1);

    if (W <= 0.0) {
        return std::numeric_limits<d64>::quiet_NaN();
    }

    const d64 varPlus = (static_cast<d64>(n - 1) * W + B) / static_cast<d64>(n);
    return std::sqrt(varPlus / W);
}

} // namespace stats

} // namespace mcmc