#pragma once

#include <random>
#include <cmath>
#include "calculus/transition_proposal/transition_proposal.hpp"

namespace calculus {

namespace sample {

    class IsotropicGaussianTransition : public TransitionProposal {
        public:
            IsotropicGaussianTransition(d64 stdev) : stdev_(stdev) {}

            linalg::Vec<d64> sample(const linalg::Vec<d64>& current, std::mt19937& gen) const override {
                std::normal_distribution<d64> dist(0.0, stdev_);

                linalg::Vec<d64> noise = linalg::Vec<d64>::random(current.size(), dist, gen);
                return current + noise;
            }

            // Normalization isn't strictly needed, but again is very easy and cheap to include here
            d64 logDensity(const linalg::Vec<d64>& from, const linalg::Vec<d64>& to) const override {
                linalg::Vec<d64> diff = to - from;
                d64 exponent = -0.5 * diff.normSquared() / (stdev_ * stdev_);
                d64 normalization = -0.5 * from.size() * std::log(2.0 * M_PI * stdev_ * stdev_);
                return exponent + normalization;
            }
            
        private:
            d64 stdev_;
    };

} // namespace sample

} // namespace calculus