#include <catch2/catch_test_macros.hpp>
#include "linalg/vec.hpp"
#include <sstream>
#include <string>

using namespace linalg;

TEST_CASE("Vec elementwise arithmetic is correct", "[vec]") {
    Vec p({10, 20, 30}), q({1, 2, 3});
    SECTION("operator+") { REQUIRE((p + q).isApprox(Vec({11, 22, 33}))); }
    SECTION("operator-") { REQUIRE((p - q).isApprox(Vec({9, 18, 27}))); }
    SECTION("hadamard")  { REQUIRE(p.hadamard(q).isApprox(Vec({10, 40, 90}))); }
}
TEST_CASE("Vec::hadamardInPlace actually mutates the caller", "[vec]") {
    linalg::Vec h1({2,3,4}), h2({2,3,4});
    h1.hadamardInPlace(h2);
    REQUIRE(h1.isApprox(Vec({4, 9, 16})));
}
TEST_CASE("Vec::operator<< writes to the passed ostream, not std::cout", "[vec]") {
    std::ostringstream oss;
    oss << Vec({10, 20, 30});
    REQUIRE(oss.str().find("10") != std::string::npos);
}
TEST_CASE("Vec::basis handles n == 0 without unsigned underflow", "[vec]") {
    REQUIRE_THROWS(Vec::basis(0, 0));
}