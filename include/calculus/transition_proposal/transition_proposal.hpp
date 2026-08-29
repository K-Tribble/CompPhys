#pragma once

#include "calculus/target_distributions/target_distribution.hpp"

namespace calculus {

namespace sample {

    class TransitionProposal {
        public:
            virtual linalg::Vec<d64> sample(const linalg::Vec<d64>& current, std::mt19937& gen) const = 0;
            virtual d64 logDensity(const linalg::Vec<d64>& from, const linalg::Vec<d64>& to) const = 0;
    };
} // namespace sample

} // namespace calculus
