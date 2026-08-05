#include <iostream>
#include "matrix.hpp"
#include <vec.hpp>
#include "linalg_common.hpp"
#include "linalg_interop.hpp"

using namespace linalg;

int main() {
    Matrix m{{1, 2, 1}, {-2, 3.1, 1}, {1, -1, 5}};
    Matrix m_inverse = m.inverse();
    Matrix I = Matrix::identity(3);
    Matrix prod = m * m_inverse;

    std::cout << m.determinant() << std::endl;
    std::cout << "m:\n" << m << "m inverse:\n" << m_inverse << "product:\n" << prod;

    bool works = prod.isApprox(I);

    if (works) {
        std::cout << "Matrix inversion works" << std::endl;
    } else {
        std::cout << "Something doesnt work"  << std::endl;
    }

    Matrix a{{1, -5}, {-2, 3}};
    Matrix a_inverse = a.inverse();
    Vec v{1, 2};

    std::cout << a.cols() << std::endl;
    std::cout << v.size() << std::endl;

    std::cout << "v: " << v << "a:\n" << a << "a inverse:\n" << a_inverse;
    Vec av_prod = a * v;
    std::cout << "a * v:\n" << av_prod;

    Vec vAgain = a_inverse * av_prod;
    std::cout << "a inverse * a * v:\n" << vAgain;

    bool alsoWorks = vAgain.isApprox(v);

    if (alsoWorks) {
        std::cout << "Matrix vector multiplication works"  << std::endl;
    } else {
        std::cout << "Matrix vector multiplication doesn't work"  << std::endl;
    }

    return 0;
}