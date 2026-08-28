#pragma once

#include "calculus/proposal/proposal.hpp"

#include <cmath>
#include <numbers>
#include <random>

namespace calculus {
    
namespace sample {

    // Isotropic Gaussian Proposal g(x) = N(mean, sigma&2 I).

    class GaussianProposal : public Proposal {
        public:
            GaussianProposal(linalg::Vec<d64> mean, d64 sigma) 
                : mean_(std::move(mean)), sigma_(sigma) {}

            d64 logDensity(const linalg::Vec<d64>& x) const override {
                u32 n = x.size();
                d64 sumSq = 0.0;
                for (u32 i = 0; i < n; ++i) {
                    d64 diff = x(i) - mean_(i);
                    sumSq += diff * diff;
                }

                // not needed but cheap to include
                d64 logNormConst = -0.5 * n * std::log(2.0 * std::numbers::pi_v<d64> * sigma_ * sigma_);
                return logNormConst - 0.5 * sumSq / (sigma_ * sigma_);
            }

            linalg::Vec<d64> sample(std::mt19937& gen) const override {
                std::normal_distribution<d64> dist(0.0, sigma_);
                u32 n = mean_.size();
                linalg::Vec<d64> x(n);
                for (u32 i = 0; i < n; ++i) {
                    x(i) = mean_(i) + dist(gen);
                }
                return x;
            }

            u32 dim() const override {
                return mean_.size();
            }
        
        private:
            linalg::Vec<d64> mean_;
            d64 sigma_;
    };

} // namespace sample

} // namespace calculus