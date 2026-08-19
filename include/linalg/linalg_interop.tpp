#include "linalg/linalg_interop.hpp"
#include "linalg/matrix.hpp"
#include "linalg/vec.hpp"

namespace linalg {

template <Scalar T>
Vec<T> operator*(const Matrix<T>& m, const Vec<T>& v) {
    if (m.cols() != v.size()) {
        throw std::invalid_argument("shape mismatch");
    }
    
    Vec<T> product(m.rows());

    std::vector<Vec<T>> rows = m.getRows();

    for (u32 i = 0; i < rows.size(); ++i) {
        product(i) = rows[i].dot(v);
    }

    return product;
}

template <Scalar T>
Vec<T> operator*(const Vec<T>& v, const Matrix<T>& m) {
    if (v.size() != m.rows()) {
        throw std::invalid_argument("shape mismatch");
    }

    Vec<T> product(m.cols());

    std::vector<Vec<T>> cols = m.getCols();

    for (u32 j = 0; j < cols.size(); ++j) {
        product(j) = v.dot(cols[j]);
    }

    return product;
}

} // namespace linalg
