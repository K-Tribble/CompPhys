#include "linalg/vec.hpp"
#include "linalg/linalg_common.hpp"
#include "linalg/matrix.hpp"
#include <iostream>
#include <random>

namespace linalg {

Vec Vec::basis(u32 n, u32 i) {
    if (i >= n) {
        throw std::invalid_argument("Index of out range");
    }
    Vec v(n, 0.0);
    v(i) = 1;

    return v;
}

Vec Vec::random(u32 n) {
    return Vec::random(n, std::uniform_real_distribution<d64>(-1.0, 1.0));
}

Vec Vec::operator+(const Vec& other) const {
    return Vec(detail::elementWise(data_, other.data_, std::plus<d64>()));
}

Vec& Vec::operator+=(const Vec& other) {
    detail::elementWiseInPlace(data_, other.data_, std::plus<d64>());
    return *this;
}

Vec Vec::operator-(const Vec& other) const {
    return Vec(detail::elementWise(data_, other.data_, std::minus<d64>()));
}

Vec& Vec::operator-=(const Vec& other) {
    detail::elementWiseInPlace(data_, other.data_, std::minus<d64>());
    return *this;
}

Vec Vec::operator*(d64 s) const {
    Vec result(*this);

    detail::scaleInPlace(result.data_, s);

    return result;
}

Vec& Vec::operator*=(d64 s) {
    detail::scaleInPlace(data_, s);

    return *this;
}

Vec Vec::operator/(d64 s) const {
    Vec result(*this);

    detail::scaleInPlace(result.data_, 1.0 / s);

    return result;
}

Vec& Vec::operator/=(d64 s) {
    detail::scaleInPlace(data_, 1.0 / s);

    return *this;
}

bool Vec::isApprox(const Vec& other, d64 absTol, d64 relTol) const {
    return detail::approxEqual(data_, other.data_, absTol, relTol);
}

bool Vec::isZero(d64 absTol) const {
    return detail::isZero(data_, absTol);
}

Vec Vec::hadamard(const Vec& other) const {
    if (data_.size() != other.data_.size()) {
        throw std::invalid_argument("shape mismatch");
    }

    return Vec(detail::elementWise(data_, other.data_, std::multiplies<d64>()));
}

Vec& Vec::hadamardInPlace(const Vec& other) {
    if (data_.size() != other.data_.size()) {
        throw std::invalid_argument("shape mismatch");
    }

    detail::elementWiseInPlace(data_, other.data_, std::multiplies<d64>());

    return *this;
}

d64 Vec::sum() const { 
    return detail::sumElements(data_); 
}
d64 Vec::max() const { 
    return detail::maxElement(data_); 
}
d64 Vec::min() const { 
    return detail::minElement(data_); 
}

d64 Vec::dot(const Vec& other) const {
    if (data_.size() != other.data_.size()) {
        throw std::invalid_argument("shape mismatch");
    }
    d64 s = 0.0;
    for (u32 i = 0; i < data_.size(); ++i) {
        s += data_[i] * other.data_[i];
    }
    return s;
}

Matrix Vec::outer(Vec& v) const {
    u32 n = size();
    u32 m = v.size();
    Matrix result(n, m);

    for (u32 i = 0; i < n; ++i) {
        for (u32 j = 0; j < m; ++j) {
            result(i, j) = data_[i] * v.data_[j];
        }
    }

    return result;
}

d64 Vec::normSquared() const {
    return dot(*this); 
}
d64 Vec::norm() const {
    return std::sqrt(normSquared()); 
}

d64 Vec::lnorm(d64 l) const {
    if (std::isinf(l)) {
        d64 max = 0;
        for (d64 x : data_) {
            if (std::fabs(x) > max) {
                max = std::fabs(x);
            }
        }
        return max;
    }

    d64 sum = 0.0;
    for (d64 x : data_) {
        sum += std::pow(std::abs(x), l);
    }

    return std::pow(sum, 1.0 / l);
}

Vec Vec::normalized() const {
    d64 n = norm();
    if (n <= kDefaultAbsTol) throw std::invalid_argument("cannot normalize zero vector");
    return (*this) / n;
}

Vec& Vec::normalize() {
    d64 n = norm();
    if (n <= kDefaultAbsTol) throw std::invalid_argument("cannot normalize zero vector");
    *this /= n;
    return *this;
}

Vec Vec::cross(const Vec& other) const {
    if (size() != 3 || other.size() != 3)
        throw std::invalid_argument("cross product requires 3D vectors");
    return Vec({
        data_[1]*other.data_[2] - data_[2]*other.data_[1],
        data_[2]*other.data_[0] - data_[0]*other.data_[2],
        data_[0]*other.data_[1] - data_[1]*other.data_[0]
    });
}

std::ostream& operator<<(std::ostream& os, const Vec& v) {
    for (u32 i = 0; i < v.size(); ++i) {
        if (i == 0) {
            os << "(" << v(i) << ", ";
        } else if (i == v.size() - 1) {
            os << v(i) << ")" << std::endl;
        } else {
            os << v(i) << ", ";
        }
    }

    return os;
}

} // namespace linalg