#pragma once

#include "calculus/target_distributions/target_distribution.hpp"
#include <random>

namespace calculus {

namespace sample {

    class Proposal : public TargetDistribution {
        public:
            // Draw one exact sample x ~ g, usign supplied generator
            // Uses templated random bit generator for repeatability
            template <std::uniform_random_bit_generator Generator>
            virtual linalg::Vec<d64> sample(Generator& gen) const = 0;

            // Convenience overload that just uses an mt19937 bit generator
            virtual linalg::Vec<d64> sample(std::mt19937& gen) const = 0;
    };

} // namespace sample

} // namespace calculus