#include <iostream>
#include "matrix.hpp"

int main() {
    Matrix m{{1, 2, 1}, {-2, 0, 1}, {1, -1, 0}};
    Matrix m_inverse = m.inverse();
    Matrix I = Matrix::identity(3);
    Matrix prod = m * m_inverse;

    std::cout << m.determinant() << std::endl;
    // m.print();
    // m_inverse.print();
    prod.print();

    bool works = prod.isApprox(I);

    if (works) {
        std::cout << "Matrix inversion works";
    } else {
        std::cout << "Something doesnt work";
    }

    return 0;
}