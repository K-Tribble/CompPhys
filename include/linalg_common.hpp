#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>
#include <stdexcept>
#include "types.hpp"
#include "constants.hpp"


namespace linalg {

    namespace detail {
        inline std::vector<d64> elementWise(const std::vector<d64>& a, const std::vector<d64>& b, const std::function<d64(d64, d64)>& f) {
            if (a.size() != b.size()) throw std::invalid_argument("shape mismatch");
            std::vector<d64> result(a.size());
            for (u32 i = 0; i < a.size(); ++i) result[i] = f(a[i], b[i]);
            return result;
        }

        inline void elementWiseInPlace(std::vector<d64>& a, const std::vector<d64>& b, const std::function<d64(d64,d64)>& f) {
            if (a.size() != b.size()) throw std::invalid_argument("shape mismatch");
            for (u32 i = 0; i < a.size(); ++i) a[i] = f(a[i], b[i]);
        }

        inline void scaleInPlace(std::vector<d64>& a, d64 s) {
            for (auto& v : a) {
                v *= s;
            }
        }

        inline bool approxEqual(const std::vector<d64>& a, const std::vector<d64>& b, d64 absTol, d64 relTol) {
            if (a.size() != b.size()) return false;
            for (u32 i = 0; i < a.size(); ++i) {
                d64 diff = std::fabs(a[i] - b[i]);
                d64 largest = std::max(std::fabs(a[i]), std::fabs(b[i]));
                if (diff > std::max(absTol, relTol * largest)) return false;
            }
            return true;
        }

        inline bool isZero(const std::vector<d64>& a, d64 absTol) {
            return std::all_of(a.begin(), a.end(), [absTol](d64 v){return std::fabs(v) <= absTol;});
        }

        inline d64 sumElements(const std::vector<d64>& a) {
            d64 s = 0;
            for (auto v : a) {
                s += v;
            }
            return s;
        }

        inline d64 maxElement(const std::vector<d64>& a) {
            return *std::max_element(a.begin(), a.end());
        }
        inline d64 minElement(const std::vector<d64>& a) {
            return *std::min_element(a.begin(), a.end());
        }
    } // namespace detail
} // namespace linalg