#include <catch2/catch_test_macros.hpp>
#include "linalg/matrix.hpp"
#include "linalg/vec.hpp"
#include "linalg/linalg_interop.hpp"

using namespace linalg;

TEST_CASE("Matrix * Vec sizes correctly for non-square matrices", "[interop]") {
    Matrix R{{1,2,3},{4,5,6}}; // 2x3
    Vec result = R * Vec({1,1,1});
    REQUIRE(result.size() == 2);
    REQUIRE(result.isApprox(Vec({6, 15})));
}
TEST_CASE("Vec * Matrix sizes correctly for non-square matrices", "[interop]") {
    Matrix R{{1,2,3},{4,5,6}};
    Vec result = Vec({1,1}) * R;
    REQUIRE(result.size() == 3);
    REQUIRE(result.isApprox(Vec({5, 7, 9})));
}