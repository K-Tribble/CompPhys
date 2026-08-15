#include <cmath>
#include <vector>
#include <span>
#include "types.hpp"
#include "constants.hpp"

namespace calculus {

namespace differentiate {
    enum class DiffScheme {
        Forward, Backward, Central
    };

    template <typename F>
    inline d64 firstDerivAt(F&& func, d64 x, d64 h = 0.01, DiffScheme scheme = DiffScheme::Central) {
        switch (scheme) {
            case DiffScheme::Central:
                return (func(x + h) - func(x - h)) / (2 * h);
                break;

            case DiffScheme::Forward:
                return (func(x + h) - func(x)) / h;
                break;
                
            case DiffScheme::Backward:
                return (func(x) - func(x - h)) / h;
                break;
        }
    }

    // second derivative uses central difference scheme
    template <typename F>
    inline d64 secondDerivAt(F&& func, d64 x, d64 h = 0.01) {
        return (func(x + h) - 2 * func(x) + func(x - h)) / (h * h);
    }

    // First order can use any scheme, but to return a vector of derivatives the same size as x a forward/backward
    // derivative evaluation is done at either end if needed. So when using a central scheme, the derivative at 
    // the first point is calculated with forward difference, and at the last point with backward difference
    std::vector<d64> differentiate(std::span<const d64> x, std::span<const d64> y, DiffScheme scheme = DiffScheme::Central);

    std::vector<d64> secondDeriv(std::span<const d64> x, std::span<const d64> y); // second derivative uses central difference scheme

} // namespace differentiate
    
} // namespace calculus