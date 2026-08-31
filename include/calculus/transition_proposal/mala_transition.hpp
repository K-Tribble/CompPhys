#pragma once

#include <cmath>
#include "calculus/transition_proposal/transition_proposal.hpp"
#include "calculus/target_distributions/differentiable_target.hpp"

namespace calculus {

namespace sample {

    class MALATransition : public TransitionProposal {
        public:
            MALATransition(d64 h, const DifferentiableTarget& target) : h_(h), target_(target) {}

            linalg::Vec<d64> sample(const linalg::Vec<d64>& current, std::mt19937& gen) const override {
                return sampleFromGrad(current, target_.gradLogDensity(current), gen);
            }

            d64 logDensity(const linalg::Vec<d64>& from, const linalg::Vec<d64>& to) const override {
                return logDensityFromGrad(from, target_.gradLogDensity(from), to);
            }

            linalg::Vec<d64> sampleFromgrad(const linalg::Vec<d64>& current, const linalg::Vec<d64>& gradCurrent, 
                std::mt19937& gen) const {
                std::normal_distribution<d64> dist(0.0, 1.0);
                linalg::Vec<d64> noise = gradCurrent * (h_ / 2.0) + 
                    linalg::Vec<d64>::random(current.size(), dist, gen);
                return current + noise;
            }
            // Normalization isnt strictly needed but is very easy and cheap to include here
            d64 logDensityFromGrad(const linalg::Vec<d64>& from, const linalg::Vec<d64>& gradFrom, 
                const linalg::Vec<d64>& to) const {
                linalg::Vec<d64> diff = to - from - gradFrom * (h_ / 2.0);
                d64 exponent = -0.5 * diff.normSquared() / h_;
                d64 normalization = -0.5 * from.size() * std::log(2.0 * M_PI * h_);
                return exponent + normalization;
            }
        
        private:
            d64 h_; // variance of the transition proposal
            const DifferentiableTarget& target_; // target distribution to sample from, MALA requires its gradient

    };
} // namespace sample

} // namespace calculus