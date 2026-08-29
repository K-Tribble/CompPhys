#pragma once

#include <cmath>
#include <vector>
#include <random>
#include <span>
#include <limits>
#include <stdexcept>
#include "types.hpp"
#include "constants.hpp"
#include "linalg/vec.hpp"
#include "calculus/target_distributions/target_distribution.hpp"
#include "calculus/proposal/proposal.hpp"
#include "calculus/transition_proposal/transition_proposal.hpp"

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

// Accumulator for self-normalized importance sampleing:
// mu_hat = sum(w_i * f_i) / sum(w_i),   w_i = P(x_i) / q(x_i)
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

namespace calculus {

namespace integrate {

    struct IntegralResult {
        bool converged;
        d64 finalError;
        d64 value;
        u32 numIter;
    };

    template <typename F>
    inline IntegralResult trapezoidal(F&& func, const d64 a, const d64 b, d64 stopCondition = kIterStopCondition, u32 maxIter = 1000) {
        d64 h = b - a;

        d64 fa = func(a);
        d64 fb = func(b);

        d64 I1 = h * 0.5 * (fa + fb);
        d64 Ij = I1;

        u32 numIter = 0;
        u32 j = 1; // first iteration in loop is to evaluate I2
        d64 err = 0.0;
        bool converged = false;

        while (numIter < maxIter) {
            h /= 2;

            d64 I_jp1 = 0.5 * Ij;

            for (u32 i = 1; i <= (1u << (j - 1)); ++i) {
                I_jp1 += h * func(a + (2 * i - 1) * h);
            }

            err = std::fabs(I_jp1 - Ij) / std::max(std::fabs(Ij), 1.0); // use absolute error in case I_j is close to zero.
            Ij = I_jp1;
            if (err < stopCondition) {
                converged = true;
                break;
            }

            ++numIter;
            ++j;
        }

        IntegralResult res;
        res.converged = converged;
        res.numIter = numIter;
        res.finalError = err;
        res.value = Ij;

        return res;
    }

    template <typename F>
    inline IntegralResult simpsons(F&& func, const d64 a, const d64 b, d64 stopCondition = kIterStopCondition, u32 maxIter = 1000) {
        // Tj is the jth value of the integral as evaluated by the trapzoidal method. 
        // Sj is the jth value of the integral as evaluated by simpsons method, which can be calcualted with succesive Tj
        d64 h = b - a;

        d64 fa = func(a);
        d64 fb = func(b);

        d64 T1 = h * 0.5 * (fa + fb);
        d64 Tj = T1;

        u32 numIter = 0;
        u32 j = 1; // first iteration in loop is to evaluate I2
        d64 err = 0.0;
        bool converged = false;
        d64 T_jm1 = 0.0;
        d64 Sj = T1;

        while (numIter < maxIter) {
            h /= 2;

            d64 T_jp1 = 0.5 * Tj;

            for (u32 i = 1; i <= (1u << (j - 1)); ++i) {
                T_jp1 += h * func(a + (2 * i - 1) * h);
            }

            Sj = 4.0 * T_jp1 / 3.0 - Tj / 3.0;
            
            if (numIter >= 1) {
                d64 S_prev = (4.0 * Tj - T_jm1) / 3.0;
                err = std::fabs((4.0 * T_jp1 / 3.0 - 5.0 * Tj / 3.0 + T_jm1 / 3.0) / std::max(std::fabs(S_prev), 1.0)); // prevent division by something close to zero
                if (err < stopCondition) {
                    converged = true;
                    break;
                }
            }

            T_jm1 = Tj;
            Tj = T_jp1;
            ++numIter;
            ++j;
        }

        IntegralResult res;
        res.converged = converged;
        res.numIter = numIter;
        res.finalError = err;
        res.value = Sj;

        return res;
    }

    // Here F needs to be a function that accepts a linalg::Vec<d64>.
    // Generator is a template parameter (constrained to a real random-bit
    // generator) rather than owned internally, so callers can inject their
    // own engine. That matters for two reasons: it makes runs reproducible
    // for testing against known integrals, and it avoids the surprise that a
    // function-local static RNG inside a template is actually instantiated
    // once *per distinct lambda type F*, not shared globally.
    template <typename F, std::uniform_random_bit_generator Generator>
    inline IntegralResult mc(F&& func, std::span<const d64> leftBoundaries, std::span<const d64> rightBoundaries, Generator& gen, d64 stopCondition = kMonteCarloStopCondition, u32 maxN = 1000) {
        const u32 n = leftBoundaries.size();
        if (rightBoundaries.size() != n) {
            throw std::invalid_argument("Boundary arrays must be same size");
        }

        d64 V = 1.0;

        std::vector<std::uniform_real_distribution<d64>> dists;
        dists.reserve(n);
        for (u32 i = 0; i < n; ++i) {
            const d64 width = rightBoundaries[i] - leftBoundaries[i];
            if (width <= 0.0) {
                throw std::invalid_argument("Each right boundary must exceed its matching left boundary");
            }
            V *= width;
            dists.emplace_back(leftBoundaries[i], rightBoundaries[i]);
        }

        IntegralResult res;
        bool converged = false;

        linalg::Vec<d64> xj(n);
        WelfordAccumulator acc;
        d64 Ival = 0.0;
        d64 finalErr = 0.0;

        u32 N = 0;
        while (N < maxN) {
            ++N;

            for (u32 j = 0; j < n; ++j) {
                xj(j) = dists[j](gen);
            }
            acc.update(func(xj));

            if (N < 2) {
                continue; // need at least 2 samples for a variance estimate
            }

            Ival = V * acc.mean;
            finalErr = V * std::sqrt(acc.variance() / N);

            if (finalErr < stopCondition) {
                converged = true;
                break;
            }
        }

        res.finalError = finalErr;
        res.value = Ival;
        res.numIter = N;
        res.converged = converged;

        return res;
    }

    // Convenience overload: owns its own engine, seeded from random_device,
    // when reproducibility isn't needed.
    template <typename F>
    inline IntegralResult mc(F&& func, std::span<const d64> leftBoundaries, std::span<const d64> rightBoundaries, d64 stopCondition = kMonteCarloStopCondition, u32 maxN = 1000) {
        thread_local std::mt19937 gen(std::random_device{}());
        return mc(std::forward<F>(func), leftBoundaries, rightBoundaries, gen, stopCondition, maxN);
    }
} // namespace integrate

namespace sample {

    struct ImportanceSampleResult {
        bool converged;
        d64 finalError;
        d64 value;                // self-normalized estimate of E_P[f]
        u32 numIter;
        d64 effectiveSampleSize;  // diagnostic to compare against numIter
    };
        
    // Self-normalized important sampling estimate of E_P[f(X)], X ~ P,
    // using samples drawn from posposal g. 
    template <typename F>
    inline ImportanceSampleResult importanceSample(const TargetDistribution& target, const Proposal& proposal, F&& f,
        std::mt19937& gen, d64 stopCondition = kMonteCarloStopCondition, u32 maxN = 1000) {
            ImportanceAccumulator acc;
            bool converged = false;

            u32 N = 0;

            while (N < maxN) {
                ++N; 

                linalg::Vec<d64> x = proposal.sample(gen);
                d64 logWeight = target.logDensity(x) - proposal.logDensity(x);

                acc.update(logWeight, f(x));

                if (N < 2) {
                    continue; // need 2 weighted samples for variance estimate
                }

                d64 err = std::sqrt(acc.varianceEstimate());
                if (err < stopCondition) {
                    converged = true;
                    break;
                }
            }

            ImportanceSampleResult res;
            res.converged = converged;
            res.finalError = std::sqrt(acc.varianceEstimate());
            res.value = acc.mean();
            res.numIter = N;
            res.effectiveSampleSize = acc.effectiveSampleSize();

            return res;
        }


    // Convenience overload: owns its own engine, seeded from random_device,
    // when reproducibility isn't needed.
    template <typename F>
    inline ImportanceSampleResult importanceSample(const TargetDistribution& target, const Proposal& proposal, F&& f,
        d64 stopCondition = kMonteCarloStopCondition, u32 maxN = 1000) {
            thread_local std::mt19937 gen(std::random_device{}());
            return importanceSample(target, proposal, std::forward<F>(f), gen, stopCondition, maxN);
        }

    struct MetropolisResult {
        bool converged;
        d64 finalError;
        d64 value;
        u32 numIter;
        d64 acceptanceRate;
        d64 effectiveSampleSize
    }; 

    inline MetropolisResult metropolis(const TargetDistribution& target, const TransitionProposal& proposal, F&& f,
        const linalg::Vec<d64>& initial, std::mt19937& gen, d64 stopCondition = kMonteCarloStopCondition,
        u32 maxN = 1000, u32 minIter = 100) {
 
            WelfordAccumulator acc;
            bool converged = false;
            u32 accepted = 0;
            u32 N = 0;
            linalg::Vec<d64> currentSample = initial;
            d64 logDensityCurrent = target.logDensity(currentSample);
 
            while (N < maxN) {
                ++N;
 
                linalg::Vec<d64> proposedSample = proposal.sample(currentSample, gen);
                d64 logDensityProposed = target.logDensity(proposedSample);
 
                d64 logAcceptanceRatio = logDensityProposed - logDensityCurrent
                    + proposal.logDensity(proposedSample, currentSample)   
                    - proposal.logDensity(currentSample, proposedSample);  
 
                bool accept = (logAcceptanceRatio >= 0.0) ||
                    (std::uniform_real_distribution<d64>(0.0, 1.0)(gen) < std::exp(logAcceptanceRatio));
 
                if (accept) {
                    currentSample = proposedSample;
                    logDensityCurrent = logDensityProposed;
                    ++accepted;
                }

                acc.update(f(currentSample));
 
                if (N < std::max(static_cast<u32>(2), minIter)) {
                    // Consecutive rejections leave the chain stuck at the same
                    // point, which drives the naive (autocorrelation-blind)
                    // variance estimate straight to zero. This makes it look
                    // "converged" after just a couple of samples despite the
                    // chain never having moved. Waiting for minIter samples
                    // before ever trusting the stop condition doesn't fix the
                    // underlying autocorrelation problem, but it rules out
                    // this specific, likely, and misleading failure mode.
                    continue;
                }
 
                d64 err = std::sqrt(acc.variance() / N);
                if (err < stopCondition) {
                    converged = true;
                    break;
                }
            }
 
            MetropolisResult res;
            res.converged = converged;
            res.acceptanceRate = static_cast<d64>(accepted) / N;
            res.value = acc.mean;
            res.finalError = std::sqrt(acc.variance() / N);
            res.effectiveSampleSize = 0.0; // placeholder for now until integrated-autocorrelation-time estimate is made
            res.numIter = N;
 
            return res;
    }
    } // namespace sample
    
} // namespace calculus