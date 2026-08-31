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
#include "linalg/matrix.hpp"
#include "linalg/linalg_solve.hpp"
#include "linalg/linalg_interop.hpp"
#include "calculus/target_distributions/target_distribution.hpp"
#include "calculus/proposal/proposal.hpp"
#include "calculus/transition_proposal/transition_proposal.hpp"
#include "calculus/integral_results.hpp"
#include "calculus/accumulators.hpp"
#include "calculus/transition_proposal/mala_transition.hpp"
#include "calculus/target_distributions/differentiable_target.hpp"
#include "calculus/target_distributions/gaussian_target.hpp"
#include "calculus/transition_proposal/gaussian_transition.hpp"

namespace calculus {

namespace integrate {

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

    // Integrates a function with metropolis-hastings MCMC, it can accept a general non-symmetric transition proposal
    template <typename F>
    inline MCMCResult metropolisHastings(const TargetDistribution& target, const TransitionProposal& proposal, F&& f,
        const linalg::Vec<d64>& initial, std::mt19937& gen, u32 maxN = 10000, u32 maxLag = 0, 
        ESSMethod essMethod = ESSMethod::Geyer, std::optional<d64> C = std::nullopt) {

            if (target.dim() != initial.size()) {
                throw std::invalid_argument("Initial sample dimension does not match target distribution dimension");
            }
 
            MCMCAccumulator acc;
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
            }

            acc.finalize(maxLag); // compute autocorrelation and effective sample size
            d64 ess = acc.effectiveSampleSize(essMethod, C);
 
            MCMCResult res;
            res.converged = converged;
            res.acceptanceRate = static_cast<d64>(accepted) / N;
            res.value = acc.mean;
            res.effectiveSampleSize = ess; 
            res.finalError = std::sqrt(acc.variance / ess);
            res.method = essMethod;
            res.numIter = N;
 
            return res;
    }
    
    // Convenience overload for metropolisHastings: owns its own engine, seeded from random_device, 
    // when reproducibility isn't needed.
    template <typename F>
    inline MCMCResult metropolisHastings(const TargetDistribution& target, const Proposal& proposal, F&& f,
        const linalg::Vec<d64>& initial, u32 maxN = 10000, u32 maxLag = 0,
        ESSMethod essMethod = ESSMethod::Geyer, std::optional<d64> C = std::nullopt) {
            thread_local std::mt19937 gen(std::random_device{}());
            return metropolisHastings(target, proposal, std::forward<F>(f), initial, gen, maxN, maxLag, essMethod, C);
    }

    // Convenience function for metropolisHastings for when the proposal is an isotropic gaussian
    template <typename F> 
    inline MCMCResult gaussianMH(const TargetDistribution& target, F&& f, d64 stdev, const linalg::Vec<d64>& initial,
        std::mt19937& gen, u32 maxN = 10000, u32 maxLag = 0, ESSMethod essMethod = ESSMethod::Geyer, std::optional<d64> C = std::nullopt) {
            IsotropicGaussianTransition proposal(stdev);
            return metropolisHastings(target, proposal, std::forward<F>(f), initial, gen, maxN, maxLag, essMethod, C);
    }

    // Integrates a function with MALA, using the metropolis-hastings function with a specific transition proposal
    template <typename F>
    inline MCMCResult mala(const DifferentiableTarget& target, F&& f, const linalg::Vec<d64>& initial, d64 h, std::mt19937& gen, 
        u32 maxN = 10000, u32 maxLag = 0, ESSMethod essMethod = ESSMethod::Geyer, std::optional<d64> C = std::nullopt) {
            MALATransition proposal(h, target);
            return metropolisHastings(target, proposal, std::forward<F>(f), initial, gen, maxN, maxLag, essMethod, C);
    }

    // Convenience overload for mala: owns its own engine, seeded from random_device,
    // when reproducibility isn't needed.
    template <typename F>
    inline MCMCResult mala(const DifferentiableTarget& target, F&& f, const linalg::Vec<d64>& initial, d64 h, 
        u32 maxN = 10000, u32 maxLag = 0, ESSMethod essMethod = ESSMethod::Geyer, std::optional<d64> C = std::nullopt) {
            thread_local std::mt19937 gen(std::random_device{}());
            return mala(target, std::forward<F>(f), initial, h, gen, maxN, maxLag, essMethod, C);
    }

    // Integrates a function using Hamiltonian Monte Carlo
    // The mass matrix must be symmetric positive definite, but this is not enforced in code.
    // The mass matrix is usually chosen to be the covariance of the target distribution, but this is not required.
    // The step size and number of steps must be chosen to balance exploration and acceptance rate.
    // The step size should be small enough to ensure that the acceptance rate is high, but not so small that the number of steps is 
    // too large and the exploration is too slow. 
    template <typename F>
    inline MCMCResult hmc(const DifferentiableTarget& target, F&& f, const linalg::Vec<d64>& initial, const linalg::Matrix<d64> massMatrix, 
        d64 stepSize, u32 numSteps, std::mt19937& gen, u32 maxN = 10000, u32 maxLag = 0, ESSMethod essMethod = ESSMethod::Geyer, std::optional<d64> C = std::nullopt) {
            u32 n = initial.size();

            if (massMatrix.rows() != n || massMatrix.cols() != n) {
                throw std::invalid_argument("Mass matrix must be square and match the dimension of the initial position");
            }

            if (target.dim() != n) {
                throw std::invalid_argument("Target distribution dimension must match the dimension of the initial position");
            }

            MCMCAccumulator acc;
            bool converged = false;
            u32 accepted = 0;
            u32 N = 0;
            linalg::Vec<d64> currentPos = initial;

            linalg::Matrix<d64> mInverse = massMatrix.inverseHPD();

            linalg::Matrix<d64> L = massMatrix.choleskyDecomp();
            
            while (N < maxN) {
                ++N;

                linalg::Vec<d64> currentMomentum = L * linalg::Vec<d64>::random(n, std::normal_distribution<d64>(0.0, 1.0), gen);

                d64 currentKineticEnergy = 0.5 * (currentMomentum.dot(mInverse * currentMomentum));
                d64 currentPotentialEnergy = -target.logDensity(currentPos);

                // current value of the hamiltonian
                d64 currentHamiltonian = currentPotentialEnergy + currentKineticEnergy;

                // rate of change of position
                linalg::Vec<d64> dxdt = mInverse * currentMomentum;
                // rate of change of momentum is negative gradient of potential energy, 
                // which is the grad log density of target at the current position.
                linalg::Vec<d64> dpdt = target.gradLogDensity(currentPos);

                // Update momentum half step
                linalg::Vec<d64> proposedMomentum = currentMomentum + dpdt * 0.5 * stepSize;
                // Update position full step
                linalg::Vec<d64> proposedPos = currentPos + dxdt * stepSize;
                // Update momentum last half step
                proposedMomentum += target.gradLogDensity(proposedPos) * 0.5 * stepSize;

                // Calculate proposed potential energy
                d64 proposedPotentialEnergy = -target.logDensity(proposedPos);
                // Calculate proposed kinetic energy
                d64 proposedKineticEnergy = 0.5 * (proposedMomentum.dot(mInverse * proposedMomentum));

                d64 proposedHamiltonian = proposedPotentialEnergy + proposedKineticEnergy;

                d64 deltaH = proposedHamiltonian - currentHamiltonian;

                d64 logAcceptanceProb = -deltaH;

                bool accept = (logAcceptanceProb >= 0.0) ||
                    (std::uniform_real_distribution<d64>(0.0, 1.0)(gen) < std::exp(logAcceptanceProb));

                if (accept) {
                    currentPos = proposedPos;
                    ++accepted;
                }

                acc.update(f(currentPos));
            }

            acc.finalize(maxLag);
            d64 ess = acc.effectiveSampleSize(essMethod, C);

            MCMCResult res;
            res.converged = converged;
            res.acceptanceRate = static_cast<d64>(accepted) / N;
            res.value = acc.mean;
            res.effectiveSampleSize = ess;
            res.finalError = std::sqrt(acc.variance / ess);
            res.method = essMethod;
            res.numIter = N;

            return res;
        }

} // namespace sample
    
} // namespace calculus