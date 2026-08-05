#pragma once
#include "linalg_common.hpp"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace linalg {

class Vec {
public:
    explicit Vec(u32 n, d64 init = 0.0) : data_(n, init) {}
    explicit Vec(std::vector<d64> vals) : data_(std::move(vals)) {}
    Vec(std::initializer_list<d64> init) : data_(init) {}

    static Vec zeros(u32 n) {return Vec(n, 0.0);}
    static Vec ones(u32 n) {return Vec(n, 1.0);}
    static Vec basis(u32 n, u32 i); // e_i one-hot
    d64& operator()(u32 i) {return data_.at(i);}
    d64 operator()(u32 i) const {return data_.at(i);}

    u32 size() const {return data_.size();}

    Vec operator+(const Vec& other) const;
    Vec& operator+=(const Vec& other);
    Vec operator-(const Vec& other) const;
    Vec& operator-=(const Vec& other);
    Vec operator*(d64 s) const;
    Vec& operator*=(d64 s);
    Vec operator/(d64 s) const;
    Vec& operator/=(d64 s);

    bool operator==(const Vec& other) const {return data_ == other.data_;}
    bool isApprox(const Vec& other, d64 absTol = kDefaultAbsTol, d64 relTol = kDefaultRelTol) const;
    bool isZero(d64 absTol = kDefaultAbsTol) const;

    Vec hadamard(const Vec& other) const;
    Vec& hadamardInPlace(const Vec& other);

    d64 dot(const Vec& other) const;
    d64 normSquared() const;
    d64 norm() const;
    Vec normalized() const;
    Vec& normalize();

    Vec cross(const Vec& other) const; // requires size() == 3

    d64 sum() const;
    d64 max() const;
    d64 min() const;

private:
    std::vector<d64> data_;
};

std::ostream& operator<<(std::ostream&, const Vec&);

} // namespace linalg
