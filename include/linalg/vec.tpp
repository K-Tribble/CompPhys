#include "linalg/vec.hpp"
#include "linalg/linalg_common.hpp"
#include "linalg/matrix.hpp"
#include <iostream>
#include <random>

namespace linalg {

template <Scalar T>
Vec<T> Vec<T>::basis(u32 n, u32 i) {
    if (i >= n) {
        throw std::invalid_argument("Index of out range");
    }
    Vec v(n, 0.0);
    v(i) = 1;

    return v;
}

template <Scalar T>
template <Scalar U>
Vec<U> Vec<T>::cast() const {
    std::vector<U> out;
    out.reserve(data_.size());
    for (const T& v : data_) {
        out.emplace_back(scalar_cast<U>(v));
    }

    return Vec<U>(std::move(out));
}

template <Scalar T>
Vec<T> Vec<T>::random(u32 n) {
    return Vec<T>::random(n, std::uniform_real_distribution<RealType<T>>(-1.0, 1.0));
}

template <Scalar T>
Vec<T> Vec<T>::operator+(const Vec& other) const {
    return Vec(detail::elementWise(data_, other.data_, std::plus<T>()));
}

template <Scalar T>
Vec<T>& Vec<T>::operator+=(const Vec& other) {
    detail::elementWiseInPlace(data_, other.data_, std::plus<T>());
    return *this;
}

template <Scalar T>
Vec<T> Vec<T>::operator-(const Vec<T>& other) const {
    return Vec<T>(detail::elementWise(data_, other.data_, std::minus<T>()));
}

template <Scalar T>
Vec<T>& Vec<T>::operator-=(const Vec<T>& other) {
    detail::elementWiseInPlace(data_, other.data_, std::minus<T>());
    return *this;
}

template <Scalar T>
Vec<T> Vec<T>::operator*(T s) const {
    Vec<T> result(*this);

    detail::scaleInPlace(result.data_, s);

    return result;
}

template <Scalar T>
Vec<T>& Vec<T>::operator*=(T s) {
    detail::scaleInPlace(data_, s);

    return *this;
}

template <Scalar T>
Vec<T> Vec<T>::operator/(T s) const {
    Vec<T> result(*this);

    detail::scaleInPlace(result.data_, 1.0 / s);

    return result;
}

template <Scalar T>
Vec<T>& Vec<T>::operator/=(T s) {
    detail::scaleInPlace(data_, 1.0 / s);

    return *this;
}

template <Scalar T>
bool Vec<T>::isApprox(const Vec<T>& other, RealType<T> absTol, RealType<T> relTol) const {
    return detail::approxEqual(data_, other.data_, absTol, relTol);
}

template <Scalar T>
bool Vec<T>::isZero(RealType<T> absTol) const {
    return detail::isZero(data_, absTol);
}

template <Scalar T>
Vec<T> Vec<T>::hadamard(const Vec<T>& other) const {
    if (data_.size() != other.data_.size()) {
        throw std::invalid_argument("shape mismatch");
    }

    return Vec(detail::elementWise(data_, other.data_, std::multiplies<T>()));
}

template <Scalar T>
Vec<T>& Vec<T>::hadamardInPlace(const Vec<T>& other) {
    if (data_.size() != other.data_.size()) {
        throw std::invalid_argument("shape mismatch");
    }

    detail::elementWiseInPlace(data_, other.data_, std::multiplies<T>());

    return *this;
}

template <Scalar T>
T Vec<T>::sum() const { 
    return detail::sumElements(data_); 
}

template <Scalar T>
T Vec<T>::max() const { 
    return detail::maxElement(data_); 
}
template <Scalar T>
T Vec<T>::min() const { 
    return detail::minElement(data_); 
}

template <Scalar T>
T Vec<T>::dot(const Vec<T>& other) const {
    if (data_.size() != other.data_.size()) {
        throw std::invalid_argument("shape mismatch");
    }
    T s = 0.0;
    for (u32 i = 0; i < data_.size(); ++i) {
        s += data_[i] * other.data_[i];
    }
    return s;
}

template <Scalar T>
T Vec<T>::inner(const Vec<T>& v) const {
    if (data_.size() != v.data_.size()) {
        throw std::invalid_argument("shape mismatch");
    }

    T s = 0.0;
    for (u32 i = 0; i < data_.size(); ++i) {
        s += conjugate(data_[i]) * v.data_[i];
    }

    return s;
}

template <Scalar T>
Matrix<T> Vec<T>::outer(const Vec<T>& v) const {
    u32 n = size();
    u32 m = v.size();
    Matrix<T> result(n, m);

    for (u32 i = 0; i < n; ++i) {
        for (u32 j = 0; j < m; ++j) {
            result(i, j) = data_[i] * v.data_[j];
        }
    }

    return result;
}

template <Scalar T>
RealType<T> Vec<T>::normSquared() const {
    return std::real(inner(*this)); 
}
template <Scalar T>
RealType<T> Vec<T>::norm() const {
    return std::sqrt(normSquared()); 
}

template <Scalar T>
RealType<T> Vec<T>::lnorm(RealType<T> l) const {
    if (std::isinf(l)) { // return maximum absolute value
        RealType<T> max = 0;
        for (T x : data_) {
            if (std::abs(x) > max) {
                max = std::abs(x);
            }
        }
        return max;
    }

    RealType<T> sum = 0.0;
    for (T x : data_) {
        sum += std::pow(std::abs(x), l);
    }

    return std::pow(sum, 1.0 / l);
}

template <Scalar T>
Vec<T> Vec<T>::normalized() const {
    RealType<T> n = norm();
    if (n <= kDefaultAbsTol) throw std::invalid_argument("cannot normalize zero vector");
    return (*this) / n;
}

template <Scalar T>
Vec<T>& Vec<T>::normalize() {
    RealType<T> n = norm();
    if (n <= kDefaultAbsTol) throw std::invalid_argument("cannot normalize zero vector");
    *this /= n;
    return *this;
}

template <Scalar T>
Vec<T> Vec<T>::conj() const {
    Vec<T> res(*this);

    for (u32 i = 0; i < data_.size(); ++i) {
        res(i) = conjugate(data_[i]);
    }

    return res;
}

template <Scalar T>
Vec<T>& Vec<T>::conj_inplace() {
    for (T& x : data_) {
        x = conjugate(x);
    }

    return *this;
}

template <Scalar T>
Vec<T> Vec<T>::cross(const Vec<T>& other) const {
    if (size() != 3 || other.size() != 3)
        throw std::invalid_argument("cross product requires 3D vectors");
    return Vec<T>({
        data_[1]*other.data_[2] - data_[2]*other.data_[1],
        data_[2]*other.data_[0] - data_[0]*other.data_[2],
        data_[0]*other.data_[1] - data_[1]*other.data_[0]
    });
}

template <Scalar T>
std::ostream& operator<<(std::ostream& os, const Vec<T>& v) {
    if constexpr (is_complex_v<T>) {
        for (u32 i = 0; i < v.size(); ++i) {
            if (i == 0) {
                os << "(" << v(i).real() << "+" << v(i).imag() << "i, ";
            } else if (i == v.size() - 1) {
                os << v(i).real() << "+" << v(i).imag() << "i)" << std::endl;
            } else {
                os << v(i).real() << "+" << v(i).imag() << "i, ";
            }
        }
    } else {
        for (u32 i = 0; i < v.size(); ++i) {
            if (i == 0) {
                os << "(" << v(i) << ", ";
            } else if (i == v.size() - 1) {
                os << v(i) << ")" << std::endl;
            } else {
                os << v(i) << ", ";
            }
        }
    }
    

    return os;
}

} // namespace linalg