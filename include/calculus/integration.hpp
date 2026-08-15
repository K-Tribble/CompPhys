#include <cmath>
#include <vector>
#include "types.hpp"
#include "constants.hpp"

namespace calculus {

namespace integrate {

    struct IntegralResult {
        bool converged;
        d64 finalError;
        d64 value;
        u32 numIter;
    };

    template <typename F>
    inline d64 trapezoidal(F&& func, const d64 a, const d64 b, d64 stopCondition = kIterStopCondition, u32 maxIter = 1000) {
        d64 h = b - a;

        d64 fa = func(a);
        d64 fb = func(b);

        d64 I1 = h * 0.5 * (fa + fb);
        d64 Ij = I1;

        u32 numIter = 0;
        u32 j = 1; // first iteration in loop is to evaluate I2
        d64 err = 0.0;
        bool converged = false;

        while (numIter <= maxIter) {
            h /= 2;

            d64 I_jp1 = 0.5 * Ij;

            for (u32 i = 1; i <= (1u << (j - 1)); ++i) {
                I_jp1 += 0.5 * h * func(a + (2 * i - 1) * h);
            }

            err = std::fabs(I_jp1 - Ij) / std::max(std::fabs(Ij), 1.0); // use absolute error in case I_j is close to zero.
            Ij = I_jp1;
            if (err < stopCondition) {
                converged = true;
                break;
            } else if (numIter == maxiter) {
                converged = false;
            }

            ++numIter;
        }

        IntegralResult res;
        res.converged = converged;
        res.numIter = numIter;
        res.finalError = err;
        res.value = Ij;

        return res;
    }

    template <typename F>
    inline d64 simpsons(F&& func, const d64 a, const d64 b, d64 stopCondition = kIterStopCondition, u32 maxIter = 1000) {
        // Tj is the jth value of the integral as evaluated by the trapzoidal method. 
        // Sj is the jth value of the integral as evaluated by simpsons method, which can be calcualted with succesive Tj
        d64 h = b - a;

        d64 fa = func(a);
        d64 fb = func(b);

        d64 T1 = h * 0.5 * (fa + fb);
        d64 Tj = T1;

        u32 numIter = 0;
        u32 j = 1; // first iteration in loop is to evaluate I2
        d64 err = 0.0;
        bool converged = false;
        d64 T_jm1;
        d64 Sj; 

        while (numIter <= maxIter) {
            h /= 2;

            d64 T_jp1 = 0.5 * Tj;

            for (u32 i = 1; i <= (1u << (j - 1)); ++i) {
                T_jp1 += 0.5 * h * func(a + (2 * i - 1) * h);
            }

            Sj = 4.0 * T_jp1 / 3.0 - Tj / 3.0;
            
            if (numIter >= 1) {
                d64 S_prev = (4.0 * Tj - T_jm1) / 3.0;
                err = std::fabs((4.0 * T_jp1 / 3.0 - 5.0 * Tj / 3.0 + T_jm1 / 3.0) / std::max(std::fabs(S_prev), 1.0)); // prevent division by something close to zero
                if (err < stopCondition) {
                    converged = true;
                    break;
                } else if (numIter == maxIter) {
                    converged = false;
                }
            }

            T_jm1 = Tj;
            Tj = T_jp1;
            ++numIter;
        }

        IntegralResult res;
        res.converged = converged;
        res.numIter = numIter;
        res.finalError = err;
        res.value = Sj;

        return res;
    }

} // namespace integrate
    
} // namespace calculus