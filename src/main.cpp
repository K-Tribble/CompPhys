#include <iostream>
#include <chrono>
#include <utility>
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

    auto [gjDuration, bGJInv] = timeFunction([&](){return b.inverse();});
    auto [cofactorDuration, bCInv] = timeFunction([&](){return b.cofactorInversion();});

    Matrix I5 = Matrix::identity(5);
    std::cout << "det(b) = " << b.determinant() << std::endl;
    if (I5.isApprox(b * bGJInv)) {
        std::cout << "Gauss Jordan Inversion works" << std::endl;
    } else {
        std::cout << "Gauss Jordan Inversion doesn't work" << std::endl << b * bGJInv;
    }
    if (I5.isApprox(b * bCInv)) {
        std::cout << "Cofactor Inversion works" << std::endl;
    } else {
        std::cout << "Cofactor Inversion doesn't work" << std::endl << b * bCInv;
    }
    std::cout << bGJInv.isApprox(bCInv) << std::endl;

    std::cout << "inverse() took " << gjDuration.count() << " microseconds\n";
    std::cout << "cofactorInversion() took " << cofactorDuration.count() << " microseconds\n";


    return 0;
}