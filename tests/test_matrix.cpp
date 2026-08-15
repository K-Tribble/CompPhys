#include <catch2/catch_test_macros.hpp>
#include "linalg/matrix.hpp"
#include "linalg/vec.hpp"

using namespace linalg;

TEST_CASE("Matrix elementwise arithmetic is correct", "[matrix]") {
    Matrix M1{{1,2},{3,4}}, M2{{10,20},{30,40}};
    SECTION("operator+") { REQUIRE((M1+M2).isApprox(Matrix{{11,22},{33,44}})); }
    SECTION("hadamard")  { REQUIRE(M1.hadamard(M2).isApprox(Matrix{{10,40},{90,160}})); }
}
TEST_CASE("Matrix single-arg row accessor returns values, not indices", "[matrix]") {
    Matrix M{{5,6,7},{8,9,10}};
    REQUIRE(M(0).isApprox(Vec({5,6,7})));
}
TEST_CASE("Matrix::QRDecomp survives an already-zero reflection column", "[matrix][qr]") {
    Matrix T{{5,1,2},{0,3,0},{0,0,4}}; // column already zero below the pivot
    QRResult qr;
    REQUIRE_NOTHROW(qr = T.QRDecomp());
    REQUIRE((qr.Q * qr.R).isApprox(T, 1e-9, 1e-9));
}