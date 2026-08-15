#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
#include "linalg/matrix.hpp"
#include "linalg/vec.hpp"
#include "linalg/linalg_interop.hpp"

using namespace linalg;

static void checkEigenDecomposition(const Matrix& A) {
    EigenResult res = A.symmetricEigenQR();
    u32 n = A.rows();
    for (u32 k = 0; k < n; ++k) {
        Vec v = res.eigenvectors.getCol(k);
        REQUIRE((A * v).isApprox(v * res.eigenvalues[k], 1e-7, 1e-7));
    }
    REQUIRE((res.eigenvectors.transpose() * res.eigenvectors).isApprox(Matrix::identity(n), 1e-7, 1e-7));
    Matrix D = Matrix::diagonal(res.eigenvalues);
    REQUIRE((res.eigenvectors * D * res.eigenvectors.transpose()).isApprox(A, 1e-7, 1e-7));
    d64 sumEig = 0;
    for (auto v : res.eigenvalues) sumEig += v;
    REQUIRE(std::fabs(A.trace() - sumEig) < 1e-6);
}

TEST_CASE("1x1") { checkEigenDecomposition(Matrix{{7}}); }
TEST_CASE("2x2 simple") { checkEigenDecomposition(Matrix{{2,1},{1,2}}); }
TEST_CASE("3x3 distinct eigenvalues") { checkEigenDecomposition(Matrix{{4,1,2},{1,3,0},{2,0,5}}); }
TEST_CASE("repeated eigenvalue") { checkEigenDecomposition(Matrix{{2,0,0},{0,2,0},{0,0,3}}); }
// TEST_CASE("5x5 random symmetric") { /* seeded random SPD-ish build, same check */ }
TEST_CASE("rejects non-symmetric") {Matrix A{{1, 2}, {3, 4}}; REQUIRE_THROWS_AS(A.symmetricEigenQR(), std::invalid_argument);}

TEST_CASE("rejects non-square") {Matrix A{{1, 2, 3}, {4, 5, 6}}; REQUIRE_THROWS_AS(A.symmetricEigenQR(), std::invalid_argument);}