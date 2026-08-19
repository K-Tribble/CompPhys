#pragma once

#include <concepts>
#include <complex>
#include <type_traits>

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