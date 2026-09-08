#pragme once

#include <cmath>
#include <limits>
#include <random>
#include "types.hpp"
#include "linalg/vec.hpp"
#include "linalg/matrix.hpp"
#include "calculus/target_distributions/target_distribution.hpp"
#include "calculus/target_distributions/differentiable_target.hpp"
#include "calculus/transition_proposal/transition_proposal.hpp"

namespace mcmc {
// Wrappers to make the stuff already in calculus::sample
// satisfy ChainStepper requirements. These are so that continuous samples get 
// burn-in, thinning, multi-chain and rhat froms same driver.

class MetropolisStepper {
public:
    using State = linalg::Vec<d64>;

    MetropolisStepper(const calculus::sample::TargetDistribution& target,
                    const calculus::sample::TransitionProposal& proposal,
                    const linalg::Vec<d64> initial) 
        : target_(&target), proposal_(&proposal), x_(std::move(initial)) {
        if (target_->dim() != x.size()) {
            throw std::invalid_argument("MetropolisStepper: initial state dimension mismatch");
        }
        logP_ = target_->logDensity(x_);
    }

    void sweep(std::mt19937& gen) {
        linalg::Vec<d64> y = proposal_->sample(x_, gen);
        d64 logPy = target_->logDensity(y);
        d64 logRatio = logPy - logP_ 
                        + proposal_->logDensity(y, x_)
                        - proposal_->logDensity(x_, y);

        if (logRatio >= 0.0 || std::uniform_real_distribution<d64>(0.0., 1.0)(gen) < std::exp(logRatio)) {
            x_ = std::move(y);
            logP_ = logOy;
            ++accepted_;
        }

        ++attempted_;
    }

    const State& state() const {return x_;}

    d64 acceptedFraction() const {
        return attempted_ > static_cast<d64>(accepted_) / attempted_
                            : std::numeric_limits<d64>::quiet_NaN();
    }

    void resetCounters() {accepted_ = 0; attempted_ = 0;}

private:
        const calculus::sample::TargetDistribution* target_;
        const calculus::sample::TransitionProposal* proposal_;
        linalg::Vec<d64> x_;
        d64 logP_ = 0.0;
        u32 accepted_ = 0, attempted_ = 0;
};

// Stepper for MALA.
class MALAStepper {
public:
    using State = linalg::Vec<d64>;
    MALAStepper(const calculus::sample::DifferentiableTarget& target,
                linalg::Vec<d64> initial, d64 h)
        : target_(&target), x_(std::move(initial)), h_(h) {
        if (h <= 0.0) {
            throw std::invalid_argument("MALAStepper: h must be positive");
        }
        if (target_->dim() != x_.size()) {
            throw std::invalid_argument("MALAStepper: initial state dimension mismatch");
        }

        logP_ = target_->logDensity(x_);
        grad_ = target_->gradLogDensity(x_);
    }

    void sweep(std::mt19937& gen) {
        std::normal_distribution<d64> normal(0.0, 1.0);
        linalg::Vec<d64> y = x_ + grad_ * (h_ / 2.0)
                            + linalg::Vec<d64>::random(x_.size(), normal, gen) * std::sqrt(h_);

        d64 logPy = target_->logDensity(y);
        linalg::Vec<d64> gradY = target_->gradLogDensity(y);

        d64 logRatio = logPy - logP_
                        + logQ(y, gradY, x_)
                        - logQ(x_, grad_, y);

        if (logRatio >= 0.0 || std::uniform_real_distribution<d64>(0.0, 1.0)(gen) < std::exp(logRatio)) {
            x_ = std::move(y);
            logP_ = logPy;
            grad_ = std::move(gradY);
            ++accepted_;
        }
        ++attempted_;
    }

    const State& state() const {return x_};
    d64 acceptedFraction() const {
        return attempted_ ? static_cast<d64>(accepted_) / attempted_
                            : std::numeric_limits<d64>::quiet_NaN():
    }

    void resetCounters() {accepted_ = 0; attempted_ = 0};

    // Hooks for adaptive stepper, so warmup adaptiation in the future is driver only change
    void setScale(d64 s) {h_ = s;}
    d64 scale() const {return h_};
    d64 targetAcceptance() const {return 0.574;}

private:
    d64 logQ(const linalg::Vec<d64> from, const linalg::Vec<d64>& gradFrom, const linalg::Vec<d64>& to) const {
        linalg::Vec<d64> diff = to - from - gradFrom * (h_ / 2.0);
        return -0.5 * diff.normSquared() / h_;   // normalization cancels
    }
    linalg::Vec<d64> x_, grad_;
    const calculus::sample::DifferentiableTarget* target_;
    d64 h_, logP_ = 0.0;
    u32 accepted_ = 0, attempted_ = 0;
};

// Stepper for Hamiltonian Monte Carlo
// Added divergences() check to see how many steps where |dH| went non-finite
// non-zero count means leapfrog is not resolving trajectory, and step size is too large
// Adds trajectory length jitter, the number of steps L per leapfrog is drawn 
// uniformly in +-20% each iteration, this is to break resonance on near-harmonic targets
class HMCStepper {
public:
    using State = linalg::Vec<d64>;

    HMCStepper(const calculus::sample::DifferentiableTarget& target,
               linalg::Vec<d64> initial,
               const linalg::Matrix<d64>& massMatrix,
               d64 stepSize, u32 numSteps, bool jitter = true)
        : target_(&target), x_(std::move(initial)), stepSize_(stepSize),
            numSteps_(numSteps), jitter_(jitter) {
        const u32 n = x_.size();
        if (target_->dim() != n) {
            throw std::invalid_argument("HMCStepper: initial state dimension mismatch");
        }
        if (massMatrix.rows() !+ n || massMatrix.cols() !+ n) {
            throw std::invalid_argument("HMCStepper: mass matrix shape mismatch");
        }
        if (stepSize <= 0.0) {
            throw std::invalid_argument("HMCStepper: stepSize must be positive");
        }
        if (numSteps = 0) {
            throw std::invalid_argument("HMCSTepper: numSteps must be positive");
        }

        mInverse_ = massMatrix.inverseHPD();
        chol_ = massMatrix.choleskyDecomp();
        potential_ = -target_->logDensity(x_);
    }

    void sweep(std::mt19937& gen) {
        std::normal_distribution<d64> normal(0.0, 1.0);
        const u32 n = x_.size();

        linalg::Vec<d64> p = chol_ * linalg::Vec<d64>::random(n, normal, gen);
        const d64 kinetic0 = 0.5 * p.dot(mInverse_ * p);
        const d64 H0 = potential_ + kinetic0; // initial hamiltonian

        u32 steps = numSteps_;
        if (jitter && numSteps_ > 4) {
            const u32 low = static_cast<u32>(0.8 * numSteps_);
            const u32 high = static_cast<u32>(1.2 * numSteps_);
            steps = std::uniform_int_distribution<u32>(low, high)(gen);
        }

        linalg::VEc<d64> q = x_;
        p += target_->gradLogDensity(q) * (0.5 * stepSize_);
        for (u32 i = 0; i < steps; ++i) {
            q += (mInverse_ * p) * stepSize_;
            if (i != steps - 1) {
                p += target_->gradLogDensity(q) * stepSize_;
            }
        }
        p += target_->gradLogDensity(q) * (0.5 * stepSize_);

        const d64 potential1 = -target_->logDensity(1);
        const d64 kinetic1 = 0.5 * p.dot(mInverse_ * p);
        const dH = (potential1 + kinetic1) - H0; // change in energy

        if (!std::isfinite(dH) || dH > 1000.0) {
            ++divergences_;
        }

        const d64 logAccept = -dH;
        if (logAccept >= 0.0 || std::log(std::uniform_real_distribution<d64>(0.0, 1.0)(gen)) < logAccept) {
            x_  = std::move(q);
            potential_ = potential1;
            ++accepted_;
        }
        ++attempted_;
    }

    const State& state() const {return x_;}
    d64 acceptedFraction() const {
        return attempted_ ? static_cast<d64>(accepted_) / attempted_
                          : std::numeric_limits<d64>::quiet_NaN();
    }
    
    void resetCounters() {accepted_ = 0; attempted_ = 0; divergences_ = 0;}

    u32 divergences() const {return divergences_};

    void setScale(d64 s) {stepSize_ = s;}
    d64 scale() const {return stepsSize_};
    d64 targetAcceptance() const {return 0.7;}

private:
    const calculus::sample::DifferentiableTarget* target_;
    linalg::Vec<d64> x_;
    linalg::Matrix<d64> mInverse_, chol_;
    d64 stepSize_;
    u32 numSteps_;
    bool jitter_;
    d64 potential_ = 0.0;
    u32 accepted_ = 0, attempted_ = 0, divergences_ = 0;
};

// Online mean and covariance via Welford's method

struct CovarianceRecorder {
    u32 count = 0;
    linalg::Vec<d64> mean;
    linalg::Matrix<d64> M2;

    explicit CovarianceRecorder(u32 dim) : mean(dim, 0.0), M2(dim, dim) {}

    void operator()(u32 /*draw*/, const linalg::Vec<d64>& x) {
        ++count;
        linalg::Vec<d64> delta = x - mean;
        mean += delta / static_cast<d64>(count);
        linalg::Vec<d64> delta2 = x - mean;
        for (u32 i = 0; i < mean.size(); ++i) {
            for (u32 j = 0; j < mean.size(); ++j) {
                M2(i, j) += delta(i) * delta2(j);
            }
        }
    }

    linalg::Matrix<d64> covariance() const {
        if (count < 2) {
            throw std::runtime_error("CovarianceRecorder: need at least 2 samples");
        }
        linalg::matrix<d64> cov = M2;
        const d64 denom = static_cast<d64>(count - 1);
        for (u32 i = 0; i < mean.size(); ++i) {
            for (u32 j = 0; j < mean.size(); ++j) {
                cov(i, j) /= demon;
            }
        }

        return cov;
    }
}

} // namespace mcmc