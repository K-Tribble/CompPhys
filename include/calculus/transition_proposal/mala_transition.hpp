#pragma once

#include <cmath>
#include "calculus/transition_proposal/transition_proposal.hpp"
#include "calculus/target_distributions/differentiable_target.hpp"

namespace calculus {

namespace sample {

    class MALATransition {
        public:
            MALATransition(d64 h, DifferentiableTarget target) : h_(h), target_(target) {}

            linalg::Vec<d64> sample(const linalg::Vec<d64> current, std::mt19937& gen) const override {
                std::normal_distribution<d64> dist(0.0, 1.0);

                linalg::Vec<d64> noise = h_ * target_.gradLogDensity(current) + 
                    std::sqrt(h_) * linalg::Vec<d64>::random(current.size(), dist, gen);
                
                return current + noise;
            }

            // Normalization isnt strictly needed but is very easy and cheap to include here
            d64 logDensity(const linalg::Vec<d64>& from, const linalg::Vec<d64>& to) const override {
                linalg::Vec<d64> diff = to - from - h_ * target_.gradLogDensity(from) / 2;
                d64 exponent = -0.5 * diff.normSquared() / h_;
                d64 normalizaton = -0.5 * from.size() * std::log(2.0 * M_PI * h_)
                return exponent + normalizaton;
            }
        
        private:
            d64 h_; // variance of the transition proposal
            DifferentiableTarget target_; // target distribution to sample from, MALA requires its gradient

    };
} // namespace sample

} // namespace calculus