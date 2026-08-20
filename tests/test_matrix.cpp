// Catch2 (v3) test suite for linalg::Matrix
//
// Build notes:
//   - Requires Catch2 v3 (link against Catch2::Catch2WithMain) plus the
//     linalg library sources (matrix.cpp, vec.cpp, linalg_interop.cpp).
//   - Include paths must expose both "linalg/<header>.hpp" and the
//     top-level "types.hpp" / "constants.hpp" used by those headers.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "linalg/matrix.hpp"
#include "linalg/vec.hpp"
#include "linalg/linalg_interop.hpp"

using namespace linalg;

TEST_CASE("Matrix construction", "[matrix][construction]") {
    SECTION("fill constructor") {
        Matrix<d64> m(2, 3, 7.0);
        REQUIRE(m.rows() == 2);
        REQUIRE(m.cols() == 3);
        for (u32 i = 0; i < 2; ++i) {
            for (u32 j = 0; j < 3; ++j) {
                REQUIRE(m(i, j) == 7.0);
            }
        }
    }

    SECTION("flat row-major data constructor") {
        Matrix<d64> m(2, 2, std::vector<d64>{1.0, 2.0, 3.0, 4.0});
        REQUIRE(m(0, 0) == 1.0);
        REQUIRE(m(0, 1) == 2.0);
        REQUIRE(m(1, 0) == 3.0);
        REQUIRE(m(1, 1) == 4.0);
    }

    SECTION("nested initializer-list constructor") {
        Matrix<d64> m{{1.0, 2.0}, {3.0, 4.0}};
        REQUIRE(m.rows() == 2);
        REQUIRE(m.cols() == 2);
        REQUIRE(m(1, 0) == 3.0);
    }

    SECTION("ragged initializer list throws") {
        auto build = []() { return Matrix<d64>{{1.0, 2.0}, {3.0}}; };
        REQUIRE_THROWS_AS(build(), std::invalid_argument);
    }

    SECTION("zeros/ones/identity/diagonal factories") {
        REQUIRE(Matrix<d64>::zeros(2, 2).isZero());
        REQUIRE(Matrix<d64>::ones(2, 2) == Matrix<d64>({{1.0, 1.0}, {1.0, 1.0}}));

        Matrix<d64> I = Matrix<d64>::identity(3);
        REQUIRE(I(0, 0) == 1.0);
        REQUIRE(I(0, 1) == 0.0);
        REQUIRE(I.trace() == 3.0);

        Matrix D = Matrix<d64>::diagonal({1.0, 2.0, 3.0});
        REQUIRE(D.getDiag() == std::vector<d64>{1.0, 2.0, 3.0});
        REQUIRE(D(0, 1) == 0.0);
    }
}

TEST_CASE("Matrix element access", "[matrix][access]") {
    Matrix<d64> m{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};  // 2x3

    SECTION("element, row and column access") {
        REQUIRE(m(0, 2) == 3.0);
        m(0, 2) = 9.0;
        REQUIRE(m(0, 2) == 9.0);
        REQUIRE(m(1) == Vec<d64>{4.0, 5.0, 6.0});
        REQUIRE(m.getCol(0) == Vec<d64>{1.0, 4.0});
        REQUIRE(m.shape() == std::array<u32, 2>{2, 3});
    }

    SECTION("out-of-range access throws") {
        REQUIRE_THROWS_AS(m(5, 0), std::out_of_range);
        REQUIRE_THROWS_AS(m(0, 5), std::out_of_range);
    }

    SECTION("getRows/getCols") {
        auto rows = m.getRows();
        REQUIRE(rows.size() == 2);
        REQUIRE(rows[0] == Vec<d64>{1.0, 2.0, 3.0});

        auto cols = m.getCols();
        REQUIRE(cols.size() == 3);
        REQUIRE(cols[2] == Vec<d64>{3.0, 6.0});
    }
}

TEST_CASE("Matrix swapRows and swapCols", "[matrix][swap]") {
    Matrix<d64> m{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};  // 3x2

    SECTION("swapRows correctness and no-op on equal indices") {
        Matrix<d64> a = m;
        a.swapRows(0, 2);
        REQUIRE(a == Matrix<d64>({{5.0, 6.0}, {3.0, 4.0}, {1.0, 2.0}}));

        Matrix<d64> b = m;
        b.swapRows(1, 1);
        REQUIRE(b == m);
    }

    SECTION("swapCols correctness") {
        Matrix<d64> a = m;
        a.swapCols(0, 1);
        REQUIRE(a == Matrix<d64>({{2.0, 1.0}, {4.0, 3.0}, {6.0, 5.0}}));
    }

    SECTION("throws on out-of-range indices") {
        REQUIRE_THROWS_AS(m.swapRows(0, 10), std::out_of_range);
        REQUIRE_THROWS_AS(m.swapCols(0, 10), std::out_of_range);
    }
}

TEST_CASE("Matrix arithmetic operators", "[matrix][arithmetic]") {
    Matrix<d64> A{{1.0, 2.0}, {3.0, 4.0}};
    Matrix<d64> B{{5.0, 6.0}, {7.0, 8.0}};

    SECTION("matrix multiplication") {
        REQUIRE(A * B == Matrix<d64>({{19.0, 22.0}, {43.0, 50.0}}));
    }

    SECTION("matmul dimension mismatch throws") {
        Matrix<d64> D(3, 3, 0.0);
        REQUIRE_THROWS_AS(A * D, std::invalid_argument);
    }

    SECTION("scalar multiply/divide") {
        REQUIRE(A * 2.0 == Matrix<d64>({{2.0, 4.0}, {6.0, 8.0}}));
        REQUIRE((B / 2.0).isApprox(Matrix<d64>({{2.5, 3.0}, {3.5, 4.0}}), 1e-12, 0.0));
    }

    SECTION("elementwise add/subtract") {
        REQUIRE(A + B == Matrix<d64>({{6.0, 8.0}, {10.0, 12.0}}));
        REQUIRE(B - A == Matrix<d64>({{4.0, 4.0}, {4.0, 4.0}}));
    }

    SECTION("shape mismatch throws for add/subtract") {
        Matrix<d64> E(3, 3, 0.0);
        REQUIRE_THROWS_AS(A + E, std::invalid_argument);
        REQUIRE_THROWS_AS(A - E, std::invalid_argument);
    }

    SECTION("in-place operators mutate correctly") {
        Matrix<d64> a = A; a += B;
        REQUIRE(a == Matrix<d64>({{6.0, 8.0}, {10.0, 12.0}}));

        Matrix<d64> b = B; b -= A;
        REQUIRE(b == Matrix<d64>({{4.0, 4.0}, {4.0, 4.0}}));

        Matrix<d64> c = A; c *= 2.0;
        REQUIRE(c == Matrix<d64>({{2.0, 4.0}, {6.0, 8.0}}));
    }
}

TEST_CASE("Matrix hadamard product", "[matrix][hadamard]") {
    Matrix<d64> A{{1.0, 2.0}, {3.0, 4.0}};
    Matrix<d64> B{{5.0, 6.0}, {7.0, 8.0}};

    SECTION("correctness") {
        REQUIRE(A.hadamard(B) == Matrix<d64>({{5.0, 12.0}, {21.0, 32.0}}));

        Matrix<d64> a = A;
        a.hadamardInPlace(B);
        REQUIRE(a == Matrix<d64>({{5.0, 12.0}, {21.0, 32.0}}));
    }

    SECTION("shape mismatch throws") {
        Matrix<d64> C(3, 3, 0.0);
        REQUIRE_THROWS_AS(A.hadamard(C), std::invalid_argument);
    }
}

TEST_CASE("Matrix comparisons", "[matrix][compare]") {
    SECTION("equality and approx equality") {
        Matrix<d64> A{{1.0, 2.0}, {3.0, 4.0}};
        Matrix<d64> Aclose{{1.0, 2.0}, {3.0, 4.0 + 1e-7}};
        REQUIRE_FALSE(A == Aclose);
        REQUIRE(A.isApprox(Aclose, 1e-6, 0.0));
        REQUIRE_FALSE(A.isApprox(Aclose, 1e-9, 0.0));
    }

    SECTION("isZero") {
        Matrix<d64> Z(2, 2, 0.0);
        REQUIRE(Z.isZero());

        Matrix<d64> NZ{{0.0, 1e-3}, {0.0, 0.0}};
        REQUIRE_FALSE(NZ.isZero(1e-6));
        REQUIRE(NZ.isZero(1e-2));
    }

    SECTION("isSymmetric") {
        Matrix<d64> Sym{{2.0, 1.0}, {1.0, 3.0}};
        REQUIRE(Sym.isSymmetric());

        Matrix<d64> NonSym{{2.0, 1.0}, {0.0, 3.0}};
        REQUIRE_FALSE(NonSym.isSymmetric());

        Matrix<d64> Rect(2, 3, 0.0);
        REQUIRE_FALSE(Rect.isSymmetric());
    }

    SECTION("absDiff") {
        Matrix<d64> A{{1.0, -2.0}, {3.0, 4.0}};
        Matrix B{{4.0, 2.0}, {0.0, 4.0}};
        REQUIRE(A.absDiff(B) == Matrix<d64>({{3.0, 4.0}, {3.0, 0.0}}));

        Matrix<d64> C(3, 3, 0.0);
        REQUIRE_THROWS_AS(A.absDiff(C), std::invalid_argument);
    }
}

TEST_CASE("Matrix transpose", "[matrix][transpose]") {
    Matrix<d64> A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};  // 2x3
    Matrix<d64> At = A.transpose();
    REQUIRE(At.rows() == 3);
    REQUIRE(At.cols() == 2);
    REQUIRE(At == Matrix<d64>({{1.0, 4.0}, {2.0, 5.0}, {3.0, 6.0}}));
}

TEST_CASE("Matrix determinant", "[matrix][determinant]") {
    SECTION("1x1") {
        Matrix<d64> A(1, 1, std::vector<d64>{5.0});
        REQUIRE(A.determinant() == 5.0);
    }

    SECTION("2x2 exact formula") {
        Matrix<d64> B{{3.0, 8.0}, {4.0, 6.0}};
        REQUIRE(B.determinant() == -14.0);  // 3*6 - 8*4
    }

    SECTION("3x3 via elimination") {
        Matrix<d64> C{{6.0, 1.0, 1.0}, {4.0, -2.0, 5.0}, {2.0, 8.0, 7.0}};
        REQUIRE(std::fabs(C.determinant() - (-306.0)) < 1e-9);
    }

    SECTION("singular 2x2 has zero determinant") {
        Matrix<d64> S{{1.0, 2.0}, {2.0, 4.0}};
        REQUIRE(S.determinant() == 0.0);
    }

    SECTION("non-square throws") {
        Matrix<d64> R(2, 3, 0.0);
        REQUIRE_THROWS_AS(R.determinant(), std::invalid_argument);
    }
}

TEST_CASE("Matrix minors and cofactors", "[matrix][cofactor]") {
    Matrix<d64> A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 10.0}};

    SECTION("getMinor removes the correct row/col") {
        REQUIRE(A.getMinor(0, 0) == Matrix<d64>({{5.0, 6.0}, {8.0, 10.0}}));
    }

    SECTION("getMinor throws when matrix is too small or index is out of range") {
        Matrix<d64> tiny(1, 1, std::vector<d64>{5.0});
        REQUIRE_THROWS_AS(tiny.getMinor(0, 0), std::invalid_argument);
        REQUIRE_THROWS_AS(A.getMinor(5, 0), std::out_of_range);
    }

    SECTION("cofactor matrix applies the checkerboard sign") {
        Matrix<d64> cof = A.getCofactorMatrix();
        REQUIRE(cof(0, 0) == 2.0);   // +det[[5,6],[8,10]]  = 50-48
        REQUIRE(cof(0, 1) == 2.0);   // -det[[4,6],[7,10]]  = -(40-42)
    }
}

TEST_CASE("Matrix LU decomposition", "[matrix][lu]") {
    SECTION("P*A = L*U, with unit-lower L and upper-triangular U") {
        Matrix<d64> A{{0.0, 1.0}, {1.0, 1.0}};  // A(0,0)==0, forces a pivot
        LUResult<d64> f = A.LUDecomp();

        Matrix<d64> PA = f.P * A;
        Matrix<d64> LU = f.L * f.U;
        REQUIRE(PA.isApprox(LU, 1e-9, 0.0));
        REQUIRE(std::fabs(f.L(0, 0) - 1.0) < 1e-9);
        REQUIRE(std::fabs(f.L(1, 1) - 1.0) < 1e-9);
        REQUIRE(std::fabs(f.U(1, 0)) < 1e-9);
        REQUIRE(f.numSwaps == 1);
    }

    SECTION("non-square throws") {
        Matrix<d64> R(2, 3, 0.0);
        REQUIRE_THROWS_AS(R.LUDecomp(), std::invalid_argument);
    }

    SECTION("a matrix with a zero pivot column throws") {
        Matrix<d64> S{{0.0, 1.0}, {0.0, 1.0}};  // entire first column is zero
        REQUIRE_THROWS_AS(S.LUDecomp(), std::runtime_error);
    }
}

TEST_CASE("Matrix QR decomposition", "[matrix][qr]") {
    Matrix<d64> A{{4.0, 1.0}, {3.0, 2.0}};

    SECTION("Q is orthogonal, R is upper triangular, Q*R reconstructs A") {
        QRResult<d64> qr = A.QRDecomp();

        Matrix<d64> QtQ = qr.Q.transpose() * qr.Q;
        REQUIRE(QtQ.isApprox(Matrix<d64>::identity(2), 1e-9, 0.0));
        REQUIRE(std::fabs(qr.R(1, 0)) < 1e-9);

        Matrix<d64> recon = qr.Q * qr.R;
        REQUIRE(recon.isApprox(A, 1e-9, 0.0));
    }

    SECTION("non-square throws") {
        Matrix<d64> R(2, 3, 0.0);
        REQUIRE_THROWS_AS(R.QRDecomp(), std::invalid_argument);
    }
}

TEST_CASE("Matrix symmetric eigen decomposition", "[matrix][eigen]") {
    SECTION("recovers the correct eigenvalues") {
        Matrix<d64> A{{2.0, 1.0}, {1.0, 2.0}};  // eigenvalues 1 and 3
        EigenResult<d64> res = A.hermitianEigenQR();

        std::vector<d64> vals = res.eigenvalues;
        REQUIRE(vals.size() == 2);
        std::sort(vals.begin(), vals.end());
        REQUIRE(std::fabs(vals[0] - 1.0) < 1e-6);
        REQUIRE(std::fabs(vals[1] - 3.0) < 1e-6);
    }

    SECTION("throws for a non-symmetric matrix") {
        Matrix<d64> NonSym{{1.0, 2.0}, {0.0, 1.0}};
        REQUIRE_THROWS_AS(NonSym.hermitianEigenQR(), std::invalid_argument);
    }

    SECTION("throws for a non-square matrix") {
        Matrix<d64> Rect(2, 3, 0.0);
        REQUIRE_THROWS_AS(Rect.hermitianEigenQR(), std::invalid_argument);
    }
}

// more tests to test eigen decomposition of matrix
static void checkEigenDecomposition(const Matrix<d64>& A) {
    EigenResult<d64> res = A.hermitianEigenQR();
    u32 n = A.rows();
    for (u32 k = 0; k < n; ++k) {
        Vec<d64> v = res.eigenvectors.getCol(k);
        REQUIRE((A * v).isApprox(v * res.eigenvalues[k], 1e-7, 1e-7));
    }
    REQUIRE((res.eigenvectors.transpose() * res.eigenvectors).isApprox(Matrix<d64>::identity(n), 1e-7, 1e-7));
    Matrix<d64> D = Matrix<d64>::diagonal(res.eigenvalues);
    REQUIRE((res.eigenvectors * D * res.eigenvectors.transpose()).isApprox(A, 1e-7, 1e-7));
    d64 sumEig = 0;
    for (auto v : res.eigenvalues) sumEig += v;
    REQUIRE(std::fabs(A.trace() - sumEig) < 1e-6);
}

TEST_CASE("Eigen decomp: 1x1") { checkEigenDecomposition(Matrix<d64>{{7}}); }
TEST_CASE("Eigen decomp: 2x2 simple") { checkEigenDecomposition(Matrix<d64>{{2,1},{1,2}}); }
TEST_CASE("Eigen decomp: 3x3 distinct eigenvalues") { checkEigenDecomposition(Matrix<d64>{{4,1,2},{1,3,0},{2,0,5}}); }
TEST_CASE("Eigen decomp: repeated eigenvalue") { checkEigenDecomposition(Matrix<d64>{{2,0,0},{0,2,0},{0,0,3}}); }

TEST_CASE("Matrix trace, diagProduct and getDiag", "[matrix][diag]") {
    Matrix<d64> A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

    SECTION("correctness") {
        REQUIRE(A.trace() == 15.0);
        REQUIRE(A.diagProduct() == 45.0);
        REQUIRE(A.getDiag() == std::vector<d64>{1.0, 5.0, 9.0});
    }

    SECTION("throws for a non-square matrix") {
        Matrix R(2, 3, 0.0);
        REQUIRE_THROWS_AS(R.trace(), std::invalid_argument);
        REQUIRE_THROWS_AS(R.diagProduct(), std::invalid_argument);
        REQUIRE_THROWS_AS(R.getDiag(), std::invalid_argument);
    }
}

TEST_CASE("Matrix getLower and getUpper", "[matrix][triangular]") {
    Matrix<d64> A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

    SECTION("correctness: strictly-triangular parts, diagonal excluded") {
        REQUIRE(A.getLower() == Matrix<d64>({{0.0, 0.0, 0.0}, {4.0, 0.0, 0.0}, {7.0, 8.0, 0.0}}));
        REQUIRE(A.getUpper() == Matrix<d64>({{0.0, 2.0, 3.0}, {0.0, 0.0, 6.0}, {0.0, 0.0, 0.0}}));
    }

    SECTION("throws for a non-square matrix") {
        Matrix<d64> R(2, 3, 0.0);
        REQUIRE_THROWS_AS(R.getLower(), std::invalid_argument);
        REQUIRE_THROWS_AS(R.getUpper(), std::invalid_argument);
    }
}

TEST_CASE("Matrix inverse", "[matrix][inverse]") {
    Matrix<d64> A{{4.0, 7.0}, {2.0, 6.0}};  // det = 10
    Matrix<d64> expectedInv{{0.6, -0.7}, {-0.2, 0.4}};

    SECTION("inverse() is correct and A * A^-1 == I") {
        Matrix<d64> inv = A.inverse();
        REQUIRE(inv.isApprox(expectedInv, 1e-9, 0.0));
        REQUIRE((A * inv).isApprox(Matrix<d64>::identity(2), 1e-9, 0.0));
    }

    SECTION("cofactorInversion() agrees with inverse()") {
        REQUIRE(A.cofactorInversion().isApprox(expectedInv, 1e-9, 0.0));
    }

    SECTION("throws for a singular matrix") {
        Matrix<d64> Singular{{1.0, 2.0}, {2.0, 4.0}};
        REQUIRE_THROWS_AS(Singular.inverse(), std::runtime_error);
        REQUIRE_THROWS_AS(Singular.cofactorInversion(), std::invalid_argument);
    }

    SECTION("throws for a non-square matrix") {
        Matrix<d64> Rect(2, 3, 0.0);
        REQUIRE_THROWS_AS(Rect.inverse(), std::invalid_argument);
        REQUIRE_THROWS_AS(Rect.cofactorInversion(), std::invalid_argument);
    }
}

TEST_CASE("Matrix eigenpair power-iteration methods", "[matrix][eigenpair]") {
    // Symmetric, well-separated eigenvalues: 1 and 3.
    Matrix<d64> A{{2.0, 1.0}, {1.0, 2.0}};

    SECTION("largestEigenPair finds the dominant eigenvalue") {
        auto [val, vec] = A.largestEigenPair(100);
        REQUIRE(std::fabs(val - 3.0) < 1e-6);
        Vec<d64> residual = (A * vec) - (vec * val);
        REQUIRE(residual.lnorm(2) < 1e-6);
    }

    SECTION("smallestEigenPair finds the smallest-magnitude eigenvalue") {
        auto [val, vec] = A.smallestEigenPair(100);
        REQUIRE(std::fabs(val - 1.0) < 1e-6);
        Vec<d64> residual = (A * vec) - (vec * val);
        REQUIRE(residual.lnorm(2) < 1e-6);
    }

    SECTION("eigenPairClosestTo finds the nearest eigenvalue") {
        auto [val, vec] = A.eigenPairClosestTo(1.8, 100);  // closer to 1 than to 3
        REQUIRE(std::fabs(val - 1.0) < 1e-6);
        Vec<d64> residual = (A * vec) - (vec * val);
        REQUIRE(residual.lnorm(2) < 1e-6);
    }

    SECTION("throws for a non-square matrix") {
        Matrix<d64> Rect(2, 3, 0.0);
        REQUIRE_THROWS_AS(Rect.largestEigenPair(), std::invalid_argument);
        REQUIRE_THROWS_AS(Rect.smallestEigenPair(), std::invalid_argument);
        REQUIRE_THROWS_AS(Rect.eigenPairClosestTo(1.0), std::invalid_argument);
    }
}

TEST_CASE("Matrix reductions", "[matrix][reduce]") {
    Matrix A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};  // 2x3

    SECTION("sumElements") {
        REQUIRE(A.sumElements() == 21.0);
    }

    SECTION("sum(axis=0) sums columns into a row vector") {
        Matrix<d64> colSums = A.sum(0);
        REQUIRE(colSums.rows() == 1);
        REQUIRE(colSums.cols() == 3);
        REQUIRE(colSums == Matrix<d64>({{5.0, 7.0, 9.0}}));
    }

    SECTION("sum(axis=1) sums rows into a column vector") {
        Matrix<d64> rowSums = A.sum(1);
        REQUIRE(rowSums.rows() == 2);
        REQUIRE(rowSums.cols() == 1);
        REQUIRE(rowSums == Matrix<d64>({{6.0}, {15.0}}));
    }

    SECTION("max and min") {
        Matrix<d64> B{{1.0, -7.0, 3.0}, {4.0, 5.0, -2.0}};
        REQUIRE(B.max() == -7.0);
        REQUIRE(B.min() == 1.0);
    }
}

TEST_CASE("Matrix slicing", "[matrix][slice]") {
    Matrix<d64> A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

    SECTION("sliceByRows correctness") {
        REQUIRE(A.sliceByRows(1, 3) == Matrix<d64>({{4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}}));
    }

    SECTION("sliceByCols correctness") {
        REQUIRE(A.sliceByCols(0, 2) == Matrix<d64>({{1.0, 2.0}, {4.0, 5.0}, {7.0, 8.0}}));
    }

    SECTION("throws when start >= finish") {
        REQUIRE_THROWS_AS(A.sliceByRows(2, 2), std::invalid_argument);
    }

    SECTION("throws when finish exceeds bounds") {
        REQUIRE_THROWS_AS(A.sliceByRows(0, 10), std::invalid_argument);
        REQUIRE_THROWS_AS(A.sliceByCols(0, 10), std::invalid_argument);
    }
}

TEST_CASE("Matrix matmulInto", "[matrix][matmulInto]") {
    Matrix<d64> A{{1.0, 2.0}, {3.0, 4.0}};
    Matrix<d64> B{{5.0, 6.0}, {7.0, 8.0}};

    SECTION("writes the correct product into a pre-sized output") {
        Matrix<d64> out(2, 2, 0.0);
        A.matmulInto(B, out);
        REQUIRE(out == Matrix<d64>({{19.0, 22.0}, {43.0, 50.0}}));
    }

    SECTION("throws on inner-dimension mismatch") {
        Matrix<d64> Bad(3, 3, 0.0);
        Matrix<d64> out(2, 3, 0.0);
        REQUIRE_THROWS_AS(A.matmulInto(Bad, out), std::invalid_argument);
    }

    SECTION("throws when the output shape doesn't match") {
        Matrix<d64> wrongOut(3, 3, 0.0);
        REQUIRE_THROWS_AS(A.matmulInto(B, wrongOut), std::invalid_argument);
    }
}