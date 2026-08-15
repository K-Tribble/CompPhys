#include "calculus/differentiation.hpp"
#include "constants.hpp"
#include <stdexcept>

namespace calculus {

namespace differentiate {
    
    std::vector<d64> differentiate(std::span<const d64> x, std::span<const d64> y, DiffScheme scheme) {
        if (x.size() != y.size()) {
            throw std::invalid_argument("x and y must have the same size");
        }

        const u32 n = x.size();

        if (n < 2) {
            throw std::invalid_argument("x and y must containt at least 2 points");
        }
        
        std::vector<d64> firstDeriv;
        firstDeriv.reserve(n);

        switch (scheme) {
            case DiffScheme::Central:
                {
                    if (n < 3) {
                        throw std::invalid_argument("central difference requires at least 3 points");
                }
                
                    const d64 h = std::fabs(x[1] - x[0]);
                    const d64 deriv0 = (y[1] - y[0]) / h;
                    firstDeriv.emplace_back(deriv0);
                    for (u32 i = 1; i < n - 1; ++i) {
                        if (std::fabs((x[i] - x[i - 1]) - h) > kDefaultAbsTol || 
                            std::fabs((x[i + 1] - x[i]) - h) > kDefaultAbsTol) {
                            throw std::invalid_argument("x vales must be evenly spaced to use central difference scheme");
                        }
                        const d64 deriv = (y[i + 1] - y[i - 1]) / (2 * h);
                        firstDeriv.emplace_back(deriv);
                    }
                    const d64 derivFinal = (y[n - 1] - y[n - 2]) / h;
                    firstDeriv.emplace_back(derivFinal);
                }
                break;

            case DiffScheme::Forward:
                {
                    for (u32 i = 0; i < n - 1; ++i) {
                        d64 h = x[i + 1] - x[i];
                        d64 deriv = (y[i + 1] - y[i]) / h;
                        firstDeriv.emplace_back(deriv);
                    }
                    d64 derivFinal = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]);
                    firstDeriv.emplace_back(derivFinal);
                }
                break;
            case DiffScheme::Backward:
                {
                    const d64 deriv0 = (y[1] - y[0]) / (x[1] - x[0]);
                    firstDeriv.emplace_back(deriv0);
                    for (u32 i = 1; i < n; ++i) {
                        d64 h = x[i] - x[i - 1];
                        d64 deriv = (y[i] - y[i - 1]) / h;
                        firstDeriv.emplace_back(deriv);
                }
                }
                break;
        }

        return firstDeriv;
    }

    std::vector<d64> secondDeriv(std::span<const d64> x, std::span<const d64> y) {
        if (x.size() != y.size()) {
            throw std::invalid_argument("x and y must have the same size");
        }

        if (x.size() < 3) {
            throw std::invalid_argument("second derivative requires at least three points");
        }

        d64 h = std::fabs(x[1] - x[0]);
        std::vector<d64> secondDeriv;
        secondDeriv.reserve(x.size() - 2);

        for (u32 i = 1; i < x.size() - 1; ++i) {
            d64 forwardh = std::fabs(x[i + 1] - x[i]);
            d64 backh = std::fabs(x[i] - x[i - 1]);
            if (std::fabs(forwardh - h) > kDefaultAbsTol ||
                std::fabs(backh - h) > kDefaultAbsTol) {
                throw std::invalid_argument("x vales must be evenly spaced to use central difference scheme");
            }
            d64 sDVal = (y[i + 1] - 2 * y[i] + y[i - 1]) / (h * h);
            secondDeriv.emplace_back(sDVal);
        }

        return secondDeriv;
    }

} // namespace differentiate

} // namespace calculus
