// Catch2 (v3) test suite for linalg functions dealing with vector matrix interoperations
//
// Build notes:
//   - Requires Catch2 v3 (link against Catch2::Catch2WithMain) plus the
//     linalg library sources (matrix.cpp, vec.cpp, linalg_interop.cpp).
//   - Include paths must expose both "linalg/<header>.hpp" and the
//     top-level "types.hpp" / "constants.hpp" used by those headers.

#include <catch2/catch_test_macros.hpp>
#include "linalg/matrix.hpp"
#include "linalg/vec.hpp"
#include "linalg/linalg_interop.hpp"

using namespace linalg;

TEST_CASE("Matrix * Vec sizes correctly for non-square matrices", "[interop]") {
    Matrix<d64> R{{1,2,3},{4,5,6}}; // 2x3
    Vec<d64> result = R * Vec<d64>({1,1,1});
    REQUIRE(result.size() == 2);
    REQUIRE(result.isApprox(Vec<d64>({6, 15})));
}
TEST_CASE("Vec * Matrix sizes correctly for non-square matrices", "[interop]") {
    Matrix<d64> R{{1,2,3},{4,5,6}};
    Vec<d64> result = Vec<d64>({1,1}) * R;
    REQUIRE(result.size() == 3);
    REQUIRE(result.isApprox(Vec<d64>({5, 7, 9})));
}