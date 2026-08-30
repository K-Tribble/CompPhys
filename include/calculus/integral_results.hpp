#pragma once

#include "types.hpp"
#include "calculus/accumulators.hpp"

namespace calculus {

namespace integrate {

    struct IntegralResult {
        bool converged;
        d64 finalError;
        d64 value;
        u32 numIter;
    };
        
} // namespace integral result

namespace sample {

    struct ImportanceSampleResult {
        bool converged;
        d64 finalError;
        d64 value;                // self-normalized estimate of E_P[f]
        u32 numIter;
        d64 effectiveSampleSize;  // diagnostic to compare against numIter
    };

    struct MCMCResult {
        bool converged;
        d64 finalError;
        d64 value;
        u32 numIter;
        d64 acceptanceRate;
        d64 effectiveSampleSize;
        ESSMethod method;
    }; 

} // namespace sample

} // namespace calculus