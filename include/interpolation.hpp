#pragma once
 
#include "linalg/matrix.hpp"
#include "linalg/vec.hpp"
#include "types.hpp"
#include <span>
#include <cmath>

// x_known must be in strictly increasing order, will throw elsewise
std::vector<d64> lerp(std::span<const d64> xInterp, std::span<const d64> x, std::span<const d64> fKnown);

// returns a lambda that will calcualte the lagrange polynomial value of a certain data set at a value x
inline auto lagrangePolynomial() {
    auto lPoly = [](d64 x, std::span<const d64> xKnown, std::span<const d64> fKnown) {
        if (xKnown.size() != fKnown.size()) {
            throw std::invalid_argument("arrays of known x and function values must be the same size");
        }
        d64 sum = 0;
        for (u32 i = 0; i < xKnown.size(); ++i) {
            d64 prod = 1;
            for (u32 j = 0; j < xKnown.size(); ++j) {
                if (j == i) continue;
                prod *= (x - xKnown[j]) / (xKnown[i] - xKnown[j]);
            }
            sum += prod * fKnown[i];
        }

        return sum;
    };

    return lPoly;
}
// Takes a span of doubles to interpolate using lagrange polynomials
std::vector<d64> lagrangePolynomial(std::span<const d64> xInterp, std::span<const d64> x, std::span<const d64> fKnown);

struct CubicSplineResult {
    // x values passed to function.
    std::vector<d64> xVals; 
    /*
    Shape (4, xVals.size() - 1). Column i stores coefficients for the polynomial between xi and xi+1.
    First element in each column is for the x^3 term, then it goes down the powers of x in order.
    These coefficietns are for the cubic polynomial shifted by xi, which is:
    Pi(x) = coeffs(0, i) * (x - xi)^3 + coeffs(1, i) * (x - xi)^2 + coeffs(2, i) * (x - xi) + coeffs(3, i)
    for i = 0 -> xVals.size() - 1
    */ 
    linalg::Matrix coeffs;
    CubicSplineResult(std::span<const d64> x, linalg::Matrix coeffs_) : xVals(x.begin(), x.end()), coeffs(std::move(coeffs_)) {};

    d64 CS(d64 x);
};

// Return a cubic spline result. The boundary conditions are by default for the natural spline.
// The BCs specify the value of the second derivative at the boundaries
// x must be strictly increasing, although it isnt enforced, if it isn't then the result is meaningless
CubicSplineResult cubicSpline(std::span<const d64> x, std::span<const d64> yKnown, d64 leftBC = 0.0, d64 rightBC = 0.0);

