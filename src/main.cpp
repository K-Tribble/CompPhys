#include <iostream>
#include <chrono>
#include <utility>
#include <random>
#include "matrix.hpp"
#include "vec.hpp"
#include "linalg_common.hpp"
#include "linalg_interop.hpp"

using namespace linalg;

template <typename Func, typename... Args>
auto timeFunction(Func&& func, Args&&... args) {
    auto start = std::chrono::high_resolution_clock::now();

    if constexpr (std::is_void_v<std::invoke_result_t<Func, Args...>>) {
        std::forward<Func>(func)(std::forward<Args>(args)...);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return std::pair{duration, std::monostate{}};
    } else {
        auto result = std::forward<Func>(func)(std::forward<Args>(args)...);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return std::pair{duration, result};
    }
}

void compareInversion(const Matrix& matrix) {
    auto [gjDuration, gjInv] = timeFunction([&](){ return matrix.inverse(); });

    auto [cofactorDuration, cofactorInv] = timeFunction([&](){ return matrix.cofactorInversion(); });

    Matrix I = Matrix::identity(matrix.rows());

    std::cout << "det(matrix) = " << matrix.determinant() << std::endl;
    std::cout << "matrix shape = (" << matrix.shape()[0] << ", " << matrix.shape()[1] << ")" << std::endl;

    if (I.isApprox(matrix * gjInv)) {
        std::cout << "Gauss Jordan Inversion works" << std::endl;
    } else {
        std::cout << "Gauss Jordan Inversion doesn't work" << std::endl << matrix * gjInv;
    }

    if (I.isApprox(matrix * cofactorInv)) {
        std::cout << "Cofactor Inversion works" << std::endl;
    } else {
        std::cout << "Cofactor Inversion doesn't work" << std::endl << matrix * cofactorInv;
    }

    std::cout << "Inverses approximately equal: " << gjInv.isApprox(cofactorInv) << std::endl;

    std::cout << "inverse() took " << gjDuration.count() << " microseconds\n";

    std::cout << "cofactorInversion() took " << cofactorDuration.count() << " microseconds\n\n";
}

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

    Matrix b = {{1, -2, 8, 7, 3}, {-5, 7.4, -8, 9, 2}, {-1, -2.3, 4, 2, 1}, {-4, -7, -6, 4, 1}, {9, -7.4, 2, -4.2, 5}};

    compareInversion(b);

    Matrix c = {
        {1, -2, 8, 7, 3, -4, 6.2, 9},
        {-5, 7.4, -8, 9, 2, 3, -1.5, 6},
        {-1, -2.3, 4, 2, 1, 8, -7, 5.4},
        {-4, -7, -6, 4, 1, -3.2, 9, 2},
        {9, -7.4, 2, -4.2, 5, 6, -8.1, 3},
        {2.5, 6, -3, 8.7, -1, 4, 7, -5},
        {-8, 1.2, 5, -6, 3.4, -9, 2, 7},
        {4, -5.6, 9, 1, -7, 2.8, -3, 6}
    };  

    compareInversion(c);

    Matrix e{{1, 2, 1}, {-2, 3.1, 1}, {1, -1, 5}};

    LUResult res = e.LUDecomp();
    Matrix L = res.L;
    Matrix U = res.U;
    Matrix P = res.P;
    u32 num_swaps = res.numSwaps;
    Matrix LU = L * U;

    std::cout << "Determinant from e = " << e.determinant() << "\n" 
        << "Determinant from U = " << U.diagProduct() * pow(-1, num_swaps) << std::endl;
    std::cout << "e:\n" << e << "LU product:\n" << LU;
    bool luWorks = LU.isApprox(P * e);
    std::cout << "LU decomposition works: " << luWorks;
}