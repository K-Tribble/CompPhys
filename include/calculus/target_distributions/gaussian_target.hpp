#pragma once

#include "calculus/target_distributions/target_distribution.hpp"

namespace calculus {

namespace sample {

    // N-dimensional isotropic gaussian, mean mu, std-ev sigma

    class Gaussian : public DifferentiableTarget {
        public:
            Gaussian(linalg::Vec<d64> mean, d64 sigma) 
                : mean_(std::move(mean)), sigma_(sigma) {}

            d64 logDensity(const linalg::Vec<d64>& x) const override {
                d64 sumSq = 0.0;

                for (u32 i = 0; i < x.size(); ++i) {
                    d64 diff = x(i) - mean_(i);
                    sumSq += diff * diff;
                }
                return -0.5 * sumSq / (sigma_ * sigma_);
            }

            linalg::Vec<d64> gradLogDensity(const linalg::Vec<d64>& x) const override {
                linalg::Vec<d64> g(x.size());

                for (u32 i = 0; i < x.size(); ++i) {
                    g(i) = -(x(i) - mean_(i)) / (sigma_ * sigma_);
                }

                return g;
            }

            u32 dim() const override {
                return mean_.size();
            }


        private:
            linalg::Vec<d64> mean_;
            d64 sigma_;
    }

} // namespace sample

} // namespace calculus