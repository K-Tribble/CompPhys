#pragma once

#include "types.hpp"
#include "linalg/vec.hpp"

namespace calculus {

namespace sample {

    // Base interface, anything that an unnormalized log density can be evaluated on

    class TargetDistribution {
        public:
            virtual ~TargetDistribution() = default;

            // log(p(x)) up to additive constant since MCMC only needs ratios of densities
            virtual d64 logDensity(const linalg::Vec<d64>& x) const = 0;

            // Dimensionality of space that x is in
            virtual u32 dim() const = 0;
    }

    // Adds derivative of log density, required for MALA, HMC, and NUTS
    class DifferentiableTarget : public TargetDistribution {
        public:
            // Gradient of logDensity(x)
            virtual linalg::Vec<d64> gradLogDensity(const linalg::Vec<d64>& x) const = 0;
    }


} // namespace sample

} // namespace calculus