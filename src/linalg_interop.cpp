#include "linalg_interop.hpp"
#include "matrix.hpp"
#include "vec.hpp"

namespace linalg {

Vec operator*(const Matrix& m, const Vec& v) {
    if (m.cols() != v.size()) {
        throw std::invalid_argument("shape mismatch");
    }
    
    Vec product(v.size());

    std::vector<Vec> rows = m.getRows();

    for (u32 i = 0; i < rows.size(); ++i) {
        product(i) = rows[i].dot(v);
    }

    return product;
}

Vec operator*(const Vec& v, const Matrix& m) {
    if (v.size() != m.rows()) {
        throw std::invalid_argument("shape mismatch");
    }

    Vec product(v.size());

    std::vector<Vec> cols = m.getCols();

    for (u32 j = 0; j < cols.size(); ++j) {
        product(j) = v.dot(cols[j]);
    }

    return product;
}

} // namespace linalg
