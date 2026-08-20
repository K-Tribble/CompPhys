#pragma once

#include <concepts>
#include <complex>
#include <type_traits>
#include <cassert>

template <typename T>
struct is_complex: std::false_type {};

template <std::floating_point U>
struct is_complex<std::complex<U>> : std::true_type {};

template <typename T>
inline constexpr bool is_complex_v = is_complex<T>::value;

// constraint
template <typename T>
concept Scalar = std::floating_point<T> || is_complex_v<T>;

// maps T to its underlying real/magnitude type: double -> double , complex<double> -> double 
template <Scalar T>
struct scalar_traits {
    using real_type = T;
};

template <std::floating_point U>
struct scalar_traits<std::complex<U>> {
    using real_type = U;
};

template <Scalar T>
using RealType = typename scalar_traits<T>::real_type;

// conjugate that returns T itself, not std::complex<T> for real inputs
template <Scalar T>
constexpr T conjugate(const T& x) {
    if constexpr (is_complex_v<T>) {
        return std::conj(x);
    } else {
        return x;
    }
}

// Converts a scalar of type T into a scalar of type U, only in the safe directions:
// Allowed conversions:
// real -> real, precision conversions like float -> double
// real -> complex, gives a complex number with zero imaginary part
// complex -> complex, precision conversion of underlying type
// complex -> real is no supported because that would truncate the imaginary part.
// std::real should be used for an explicit conversion.
template <Scalar U, Scalar T>
constexpr U scalar_cast(const T& x) {
    if constexpr (std::is_same_v<T, U>) {
        return x;
    } else if constexpr (is_complex_v<U>) {
        return U(x); // real-> complex or complex->complex 
    } else {
        static_assert(!is_complex_v<T>, "scalar_cast: cannot implicitely convert complex to real; use std::real(x) explicitly");
        return static_cast<U>(x);
    }
}