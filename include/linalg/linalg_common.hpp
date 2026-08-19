#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>
#include <stdexcept>
#include "types.hpp"
#include "constants.hpp"
#include "scalar.hpp"


namespace linalg {

    namespace detail {
        template <Scalar T, typename F>
        inline std::vector<T> elementWise(const std::vector<T>& a, const std::vector<T>& b, F f) {
            if (a.size() != b.size()) throw std::invalid_argument("shape mismatch");
            std::vector<T> result(a.size());
            for (u32 i = 0; i < a.size(); ++i) result[i] = f(a[i], b[i]);
            return result;
        }

        template <Scalar T, typename F>
        inline void elementWiseInPlace(std::vector<T>& a, const std::vector<T>& b, F f) {
            if (a.size() != b.size()) throw std::invalid_argument("shape mismatch");
            for (u32 i = 0; i < a.size(); ++i) a[i] = f(a[i], b[i]);
        }

        template <Scalar T>
        inline void scaleInPlace(std::vector<T>& a, T s) {
            for (auto& v : a) {
                v *= s;
            }
        }

        template <Scalar T>
        inline bool approxEqual(const std::vector<T>& a, const std::vector<T>& b, RealType<T> absTol, RealType<T> relTol) {
            if (a.size() != b.size()) return false;
            for (u32 i = 0; i < a.size(); ++i) {
                RealType<T> diff = std::abs(a[i] - b[i]);
                RealType<T> largest = std::max(std::abs(a[i]), std::abs(b[i]));
                if (diff > std::max(absTol, relTol * largest)) return false;
            }
            return true;
        }

        template <Scalar T>
        inline bool isZero(const std::vector<T>& a, RealType<T> absTol) {
            return std::all_of(a.begin(), a.end(), [absTol](T v){return std::abs(v) <= absTol;});
        }

        template <Scalar T>
        inline T sumElements(const std::vector<T>& a) {
            T s = T{};
            for (auto v : a) {
                s += v;
            }
            return s;
        }

        // returns element with max absolute value
        template <Scalar T>
        inline RealType<T> maxElement(const std::vector<T>& a) {
            std::vector<RealType<T>> mags;
            mags.reserve(a.size());
            for (T val : a) {mags.emplace_back(std::abs(val));}
            return *std::max_element(mags.begin(), mags.end());
        }

        // returns element with min absolute value
        template <Scalar T>
        inline RealType<T> minElement(const std::vector<T>& a) {
            std::vector<RealType<T>> mags;
            mags.reserve(a.size());
            for (T val : a) {mags.emplace_back(std::abs(val));}
            return *std::min_element(mags.begin(), mags.end());
        }
    } // namespace detail
} // namespace linalg