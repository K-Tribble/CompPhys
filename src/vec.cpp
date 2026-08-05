#include "vec.hpp"
#include "linalg_common.hpp"

namespace linalg {

Vec Vec::basis(u32 n, u32 i) {
    Vec v(n);
    v(i) = 1;

    return v;
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

    Vec result(*this);

    return Vec(detail::elementWise(data_, other.data_, std::multiplies<d64>()));
}

Vec& Vec::hadamardInPlace(const Vec& other) {
    if (data_.size() != other.data_.size()) {
        throw std::invalid_argument("shape mismatch");
    }

    detail::elementWise(data_, other.data_, std::multiplies<d64>());

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

d64 Vec::dot(const Vec& o) const {
    if (data_.size() != o.data_.size()) throw std::invalid_argument("shape mismatch");
    d64 s = 0.0;
    for (u32 i = 0; i < data_.size(); ++i) s += data_[i] * o.data_[i];
    return s;
}

d64 Vec::normSquared() const {
    return dot(*this); 
}
d64 Vec::norm() const {
    return std::sqrt(normSquared()); 
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

} // namespace linalg