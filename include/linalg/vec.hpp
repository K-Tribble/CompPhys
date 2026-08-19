#pragma once
#include "linalg/linalg_common.hpp"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <random>
#include <span>
#include "types.hpp"
#include "constants.hpp"
#include "scalar.hpp"

namespace linalg {

template <Scalar T>
class Matrix;

template <Scalar T>
class Vec {
public:
    explicit Vec(u32 n, T init = T{}) : data_(n, init) {}
    explicit Vec(std::vector<T> vals) : data_(std::move(vals)) {}
    Vec(std::initializer_list<T> init) : data_(init) {}
    Vec() = default;

    static Vec<T> zeros(u32 n) {return Vec(n, 0.0);}
    static Vec<T> ones(u32 n) {return Vec(n, 1.0);}
    static Vec<T> basis(u32 n, u32 i); // e_i one-hot

    std::span<T> span() {return data_;}

    std::span<const T> span() const {return data_;}

    // templated random vector instantation that accepts any distribution
    // for a complex valued vector the real and imaginary parts of each entry
    // will be sampled from the distribution
    template <typename Distribution>
    static Vec<T> random(u32 n, Distribution dist){
        static std::random_device rd;
        static std::mt19937 gen(rd());

        if constexpr(is_complex_v<T>) {
            std::vector<T> data(n);
            for (T& z : data) {
                T val{dist(gen), dist(gen)};
                z = val;
            }
            return Vec(data);
        } else {
            std::vector<T> data(n);
            for (T& x : data) {
                x = dist(gen);
            }
            return Vec(data);
        }
    }
    // default random method which uses a uniform real distribution from -1.0 to 1.0
    static Vec random(u32 n);
    T& operator()(u32 i) {return data_.at(i);}
    T operator()(u32 i) const {return data_.at(i);}

    u32 size() const {return data_.size();}

    Vec<T> operator+(const Vec<T>& other) const;
    Vec<T>& operator+=(const Vec<T>& other);
    Vec<T> operator-(const Vec<T>& other) const;
    Vec<T>& operator-=(const Vec<T>& other);
    Vec<T> operator*(T s) const;
    Vec<T>& operator*=(T s);
    Vec<T> operator/(T s) const;
    Vec<T>& operator/=(T s);

    bool operator==(const Vec<T>& other) const {return data_ == other.data_;}
    bool isApprox(const Vec<T>& other, RealType<T> bsTol = kDefaultAbsTol, RealType<T> relTol = kDefaultRelTol) const;
    bool isZero(RealType<T> absTol = kDefaultAbsTol) const;

    Vec<T> hadamard(const Vec<T>& other) const;
    Vec<T>& hadamardInPlace(const Vec<T>& other);

    // This is the standard bilinear dot product, 
    // computes the sum of the element wise multiplication
    T dot(const Vec<T>& other) const;
    // This is the inner product where the first vectors elements are conjugated
    // eg a.inner(b) returns sum(conjugate(a) * b)
    T inner(const Vec<T>& v) const;
    Matrix<T> outer(const Vec<T>& v) const; 
    RealType<T> normSquared() const;
    RealType<T> norm() const;
    RealType<T> lnorm(RealType<T> l) const;
    Vec<T> normalized() const;
    Vec<T>& normalize();

    Vec<T> conj() const;
    Vec<T>& conj_inplace();

    Vec<T> cross(const Vec<T>& other) const; // requires size() == 3

    T sum() const;
    RealType<T> max() const;
    RealType<T> min() const;

private:
    std::vector<T> data_;
};

template <Scalar T>
std::ostream& operator<<(std::ostream&, const Vec<T>&);

} // namespace linalg

#include "vec.tpp"
