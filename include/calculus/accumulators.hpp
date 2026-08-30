#pragma once

#include <limits>
#include <cmath>
#include <optional>
#include <stdexcept>
#include "types.hpp"
#include "linalg/vec.hpp"

// Welford's online algorithm: updates a running mean and sum-of-squared-deviations
// (M2) one sample at a time, in a single pass.
struct WelfordAccumulator {
    u32 count = 0;
    d64 mean = 0.0;
    d64 M2 = 0.0;

    inline void update(d64 x) {
        ++count;
        d64 delta = x - mean;
        mean += delta / count;
        d64 delta2 = x - mean;
        M2 += delta * delta2;
    }

    // Sample variance (Bessel-corrected). Undefined for count < 2, so callers
    // should guard on count before calling this.
    inline d64 variance() const {
        return M2 / (count - 1);
    }
 };

// Accumulator for self-normalized importance sampling:
// mu_hat = sum(w_i * f_i) / sum(w_i), w_i = P(x_i) / q(x_i)
struct ImportanceAccumulator {
    u32 count = 0;
    d64 maxLogWeight = -std::numeric_limits<d64>::infinity();
    d64 sumW = 0.0;
    d64 sumWF = 0.0;
    d64 sumW2 = 0.0;
    d64 sumW2F = 0.0;
    d64 sumW2F2 = 0.0;

    inline void update(d64 logWeight, d64 f) {
        ++count;

        if (logWeight > maxLogWeight) {
            if (sumW > 0.0) {
                d64 rescale1 = std::exp(maxLogWeight - logWeight);
                d64 rescale2 = rescale1 * rescale1;

                sumW *= rescale1;
                sumWF *= rescale1;
                sumW2 *= rescale2;
                sumW2F *= rescale2;
                sumW2F2 *= rescale2;
            }
            maxLogWeight = logWeight;
        }

        d64 w = std::exp(logWeight - maxLogWeight);
        d64 w2 = w * w;

        sumW += w;
        sumWF += w * f;
        sumW2 += w2;
        sumW2F += w2 * f;
        sumW2F2 += w2 * f * f;
    }

    inline d64 mean() const {
        return sumWF / sumW;
    }

    // Delta-method variance estimate of mean() itself:
    //   Var(mu_hat) ~= sum(w_i^2 (f_i - mu_hat)^2) / (sum w_i)^2
    inline d64 varianceEstimate() const {
        d64 mu = mean();
        d64 num = sumW2F2 - 2.0 * mu * sumW2F + mu * mu * sumW2;
        return num / (sumW * sumW);
    }

    // How many equally-weighted samples this weighted sample is
    // "worth". ESS << count signals weight degeneracy — a handful of
    // samples dominating the estimate, usually because g is a poor
    // match to P.
    inline d64 effectiveSampleSize() const {
        return (sumW * sumW) / sumW2;
    }
};

enum class ESSMethod {
    Sokal, Geyer
};
 
struct MCMCAccumulator {
    u32 count = 0;
    std::vector<d64> samples;
    std::vector<d64> autoCorrelation;
    d64 sum = 0.0;
    d64 mean = 0.0;
    d64 variance = 0.0;
 
    inline void update(d64 newSample) {
        samples.push_back(newSample);
        ++count;
        sum += newSample;
        mean = sum / count;
    }
 
    // maxLag caps how far out the ACF is estimated. Every windowed ESS
    // method only ever needs a small window relative to N (a few hundred to
    // a couple thousand lags is generous for any chain that's actually
    // mixing), and estimating out to lag ~N is both meaningless (too few
    // pairs contribute per lag) and O(N^2) to compute, so there's no reason
    // to default to the full range.
    inline void finalize(u32 maxLag = 0) {
        if (count < 2) {
            throw std::runtime_error("MCMCAccumulator::finalize: need at least 2 samples");
        }
 
        d64 sumSq = 0.0;
        for (const auto& s : samples) {
            sumSq += (s - mean) * (s - mean);
        }
        variance = sumSq / (count - 1);
 
        if (maxLag == 0) {
            maxLag = std::min<u32>(count / 2, 2000);
        }
        maxLag = std::min<u32>(maxLag, count - 1);
 
        autoCorrelation.clear();
        autoCorrelation.reserve(maxLag + 1);
        for (u32 lag = 0; lag <= maxLag; ++lag) {
            d64 ac = 0.0;
            for (u32 i = 0; i < count - lag; ++i) {
                ac += (samples[i] - mean) * (samples[i + lag] - mean);
            }
            ac /= static_cast<d64>(count - lag);
            autoCorrelation.push_back(ac / variance);
        }
    }
 
    inline d64 t_int(ESSMethod method = ESSMethod::Geyer, std::optional<d64> C = std::nullopt) const {
        switch (method) {
            case ESSMethod::Geyer:
                return geyerTime();
            case ESSMethod::Sokal:
                if (!C) {
                    throw std::invalid_argument("Sokal method requires C parameter");
                }
                return sokalTime(*C);
        }
        throw std::logic_error("unreachable");
    }

    inline d64 effectiveSampleSize(ESSMethod method = ESSMethod::Geyer, std::optional<d64> C = std::nullopt) const {
        d64 t_int_val = t_int(method, C);
        return static_cast<d64>(count) / t_int_val;
    }
 
    private:
        // Known limitation of sokals method: the M >= C*tau_int(M) criterion can trigger
        // prematurely on a non-monotonic/oscillating ACF, since the running
        // sum can dip near zero well before the tail has actually decayed.
        // Sokal windowing is designed around smoothly, monotonically
        // decaying positive autocorrelation, so it can fail on chains with oscillating ACFs. 
        // Geyer's method is more robust to this, but requires a reversible chain.
        inline d64 sokalTime(d64 C) const {
            d64 runningSum = 0.0;
            for (u32 m = 1; m < autoCorrelation.size(); ++m) {
                runningSum += autoCorrelation[m];
                d64 t_int = 1.0 + 2.0 * runningSum;
                if (static_cast<d64>(m) >= C * t_int) {
                    return t_int;
                }
            }
            throw std::runtime_error(
                "sokalTime: no self-consistent window found within maxLag; "
                "increase maxLag passed to finalize()");
        }
 
        // requires reversible chain
        inline d64 geyerTime() const {
            d64 sumGamma = 0.0;
            for (u32 m = 0; 2 * m + 1 < autoCorrelation.size(); ++m) {
                d64 gamma_m = autoCorrelation[2 * m] + autoCorrelation[2 * m + 1];
                if (gamma_m < 0.0) {
                    break;
                }
                sumGamma += gamma_m;
            }
            return 2.0 * sumGamma - 1.0;
        }
};