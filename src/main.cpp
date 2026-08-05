#include <iostream>
#include "matrix.hpp"

int main() {
    Matrix m{{1, 2, 1}, {-2, 3.1, 1}, {1, -1, 5}};
    Matrix m_inverse = m.inverse();
    Matrix I = Matrix::identity(3);
    Matrix prod = m * m_inverse;

    std::cout << m.determinant() << std::endl;
    std::cout << "m:\n" << m << "m inverse:\n" << m_inverse << "product:\n" << prod;

    bool works = prod.isApprox(I);

    if (works) {
        std::cout << "Matrix inversion works";
    } else {
        std::cout << "Something doesnt work";
    }

    return 0;
}