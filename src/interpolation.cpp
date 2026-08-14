#include "interpolation.hpp"
#include "linalg_solve.hpp"

std::vector<d64> lerp(std::span<const d64> xInterp, std::span<const d64> xKnown, std::span<const d64> fKnown) {
    if (xKnown.size() != fKnown.size()) {
        throw std::invalid_argument("arrays of known x and function values must be the same size");
    }

    if (xKnown.size() < 2) {
        throw std::invalid_argument("at least two known points are required for interpolation");
    }

    for (u32 i = 0; i < xKnown.size() - 1; ++i) {
        if (xKnown[i + 1] <= xKnown[i]) {
            throw std::invalid_argument("known x values must be strictly increasing");
        }
    }

    std::vector<d64> fInterp;
    fInterp.reserve(xInterp.size());

    for (u32 j = 0; j < xInterp.size(); ++j) {
        d64 x = xInterp[j];

        if (x < xKnown[0] || x > xKnown[xKnown.size() - 1]) {
            throw std::
            invalid_argument("given x value must be within range of data points");
        }
        for (u32 i = 0; i < xKnown.size() - 1; ++i) {
            if (x <= xKnown[i + 1]) {
                d64 fInterp_i = ((xKnown[i + 1] - x) * fKnown[i] + (x - xKnown[i]) * fKnown[i + 1]) / (xKnown[i + 1] - xKnown[i]);
                fInterp.emplace_back(fInterp_i);
                break;
            }
        }
    }

    return fInterp;
}

std::vector<d64> lagrangePolynomial(std::span<const d64> xInterp, std::span<const d64> xKnown, std::span<const d64> fKnown) {
    if (xKnown.size() != fKnown.size()) {
        throw std::invalid_argument("arrays of known x and function values must be the same size");
    }

    if (xKnown.empty()) {
        throw std::invalid_argument("at least one known point is required for Lagrange interpolation");
    }

    for (u32 i = 0; i < xKnown.size(); ++i) {
        for (u32 j = i + 1; j < xKnown.size(); ++j) {
            if (xKnown[i] == xKnown[j]) {
                throw std::invalid_argument("known x values must be unique");
            }
        }
    }

    std::vector<d64> fInterp;
    fInterp.reserve(xInterp.size());

    auto lPoly = lagrangePolynomial();

    for (const d64& x : xInterp) {
        fInterp.emplace_back(lPoly(x, xKnown, fKnown));
    }

    return fInterp;
}

d64 CubicSplineResult::CS(d64 x) {
    if (x < xVals[0] || x > xVals[xVals.size() - 1]) {
        throw std::
        invalid_argument("given x value must be within range of data points");
    }
    for (u32 i = 0; i < xVals.size() - 1; ++i) {
        if (x <= xVals[i + 1]) {
            d64 xi = xVals[i];
            return coeffs(0, i) * pow(x - xi, 3) + coeffs(1, i) * pow(x - xi, 2) + coeffs(2, i) * (x - xi) + coeffs(3, i);
        }
    }
    return coeffs(3, xVals.size() - 2); // unreachable given the check above. Here to avoid warning.
}

CubicSplineResult cubicSpline(std::span<const d64> x, std::span<const d64> yKnown, d64 leftBC, d64 rightBC) {
    if (x.size() != yKnown.size()) {
        throw std::invalid_argument("arrays of known x and y values must be the same size");
    }

    if (x.size() < 2) {
        throw std::invalid_argument("at least two points are required for cubic spline interpolation");
    }

    for (u32 i = 0; i < x.size() - 1; ++i) {
        if (x[i + 1] <= x[i]) {
            throw std::invalid_argument("x values must be strictly increasing");
        }
    }

    u32 n = x.size();

    linalg::Matrix A(n, n);
    linalg::Vec rhs(n);

    // Set boundary conditions
    rhs(n - 1) = rightBC;
    rhs(0) = leftBC;

    // Set matrix coefficients for boundary conditions
    A(0, 0) = 1;
    A(n - 1, n - 1) = 1;

    for (u32 i = 1; i < n - 1; ++i) {
        rhs(i) = (yKnown[i + 1] - yKnown[i]) / (x[i + 1] - x[i]) - (yKnown[i] - yKnown[i - 1]) / (x[i] - x[i - 1]);

        A(i, i - 1) = (x[i] - x[i - 1]) / 6;
        A(i, i) = (x[i + 1] - x[i - 1]) / 3;
        A(i, i + 1) = (x[i + 1] - x[i]) / 6;
    }

    linalg::Vec secondDerivs = linalg::solve::lu(A, rhs); // solve for second derivatives by lu decomposition

    linalg::Matrix coeffs(4, n - 1);

    for (u32 i = 0; i < n - 1; ++i) {
        d64 hi = x[i + 1] - x[i];
        coeffs(0, i) = (secondDerivs(i + 1) - secondDerivs(i)) / (6 * hi); // x^3 coefficient
        coeffs(1, i) = secondDerivs(i) / 2; // x^2 coefficient
        coeffs(2, i) = (yKnown[i + 1] - yKnown[i]) / hi - hi * (secondDerivs(i + 1) + 2 * secondDerivs(i))/ 6; // x coefficient
        coeffs(3, i) = yKnown[i]; // constant term
    }

    return CubicSplineResult(x, coeffs);
}