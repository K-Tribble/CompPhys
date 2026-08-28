#pragma once

#include "calculus/target_distributions/target_distribution.hpp"
#include <random>

namespace calculus {

namespace sample {

    class Proposal : public TargetDistribution {
        public:
            // Draw one exact sample x ~ g, usign supplied generator
            virtual linalg::Vec<d64> sample(std::mt19937& gen) const = 0;
    }

} // namespace sample

} // namespace calculus