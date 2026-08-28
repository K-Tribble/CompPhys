#pragma once

#include "calculus/target_distributions/target_distribution.hpp"
#include "calculus/differentiation.hpp"

namespace calculus {

namespace sample {

    // Adds derivative of log density, required for MALA, HMC, and NUTS
    class DifferentiableTarget : public TargetDistribution {
        public:
            // Gradient of logDensity(x)
            virtual linalg::Vec<d64> gradLogDensity(const linalg::Vec<d64>& x) const = 0;
    };


    // Upgrades TargetDistribution into DifferentiableTarget by implementing gradLogDensity()
    // using grad function implemented in differentiation.hpp

    // logDensity() must not mutate shared/global state due to OpenMP parallelization

    // Ecah gradient evaluation is O(dim) calls to logDensity

    class FiniteDifferenceTarget : public DifferentiableTarget {
        public:
            explicit FiniteDifferenceTarget(const TargetDistribution& base, d64 h = 1e-5) 
                : base_(base), h_(h) {}

            d64 logDensity(const linalg::Vec<d64>& x) const override {
                return base_.logDensity(x);
            }

            linalg::Vec<d64> gradLogDensity(const linalg::Vec<d64>& x) const override {
                return differentiate::grad([this](const linalg::Vec<d64>& xi) {return base_.logDensity(xi);}, x, h_);
            }

            u32 dim() const override {
                return base_.dim();
            }

        private:
            const TargetDistribution& base_;
            d64 h_;
    };

} // namespace sample

} // namespace calculus