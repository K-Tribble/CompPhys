// Catch2 (v3) test suite for linalg::Vec
//
// Build notes:
//   - Requires Catch2 v3 (link against Catch2::Catch2WithMain) plus the
//     linalg library sources (vec.cpp, matrix.cpp).
//   - Include paths must expose both "linalg/<header>.hpp" and the
//     top-level "types.hpp" / "constants.hpp" used by those headers.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "linalg/vec.hpp"
#include "linalg/matrix.hpp"

using namespace linalg;

TEST_CASE("Vec construction and element access", "[vec][construction]") {
    SECTION("fill constructor") {
        Vec<d64> a(3, 2.0);
        REQUIRE(a.size() == 3);
        REQUIRE(a == Vec<d64>{2.0, 2.0, 2.0});
    }

    SECTION("vector and initializer-list constructors agree") {
        Vec<d64> b(std::vector<d64>{1.0, 2.0, 3.0});
        Vec<d64> c{1.0, 2.0, 3.0};
        REQUIRE(b == c);
    }

    SECTION("default constructor is empty") {
        Vec<d64> v;
        REQUIRE(v.size() == 0);
    }

    SECTION("zeros/ones/basis factories") {
        REQUIRE(Vec<d64>::zeros(3).isZero());
        REQUIRE(Vec<d64>::ones(3) == Vec<d64>{1.0, 1.0, 1.0});
        REQUIRE(Vec<d64>::basis(3, 1) == Vec<d64>{0.0, 1.0, 0.0});
    }

    SECTION("basis throws for an out-of-range index") {
        REQUIRE_THROWS_AS(Vec<d64>::basis(3, 5), std::invalid_argument);
    }

    SECTION("element access throws out of range") {
        Vec<d64> v{1.0, 2.0};
        REQUIRE_THROWS_AS(v(5), std::out_of_range);
    }
}

TEST_CASE("Vec arithmetic operators", "[vec][arithmetic]") {
    Vec<d64> a{1.0, 2.0, 3.0};
    Vec<d64> b{4.0, 5.0, 6.0};

    SECTION("elementwise add/subtract") {
        REQUIRE(a + b == Vec<d64>{5.0, 7.0, 9.0});
        REQUIRE(a - b == Vec<d64>{-3.0, -3.0, -3.0});
    }

    SECTION("scalar multiply/divide") {
        REQUIRE(a * 2.0 == Vec<d64>{2.0, 4.0, 6.0});
        REQUIRE((b / 2.0).isApprox(Vec<d64>{2.0, 2.5, 3.0}, 1e-12, 0.0));
    }

    SECTION("in-place operators mutate correctly") {
        Vec<d64> x = a; x += b;
        REQUIRE(x == Vec<d64>{5.0, 7.0, 9.0});

        Vec<d64> y = a; y -= b;
        REQUIRE(y == Vec<d64>{-3.0, -3.0, -3.0});

        Vec<d64> z = a; z *= 2.0;
        REQUIRE(z == Vec<d64>{2.0, 4.0, 6.0});

        Vec<d64> w = b; w /= 2.0;
        REQUIRE(w.isApprox(Vec<d64>{2.0, 2.5, 3.0}, 1e-12, 0.0));
    }

    SECTION("shape mismatch throws for add/subtract") {
        Vec<d64> g{1.0, 2.0};
        REQUIRE_THROWS_AS(a + g, std::invalid_argument);
        REQUIRE_THROWS_AS(a - g, std::invalid_argument);
    }
}
TEST_CASE("Vec equality and approx comparisons", "[vec][compare]") {
    Vec<d64> p{1.0, 2.0, 3.0};
    Vec<d64> q{1.0, 2.0, 3.0};
    Vec<d64> r{1.0, 2.0, 3.0000001};  // differs by 1e-7 in the last component

    SECTION("exact equality") {
        REQUIRE(p == q);
        REQUIRE_FALSE(p == r);
    }

    SECTION("isApprox respects the given tolerance") {
        REQUIRE(p.isApprox(r, 1e-6, 0.0));
        REQUIRE_FALSE(p.isApprox(r, 1e-9, 0.0));
    }

    SECTION("isZero") {
        Vec<d64> z{0.0, 0.0, 0.0};
        REQUIRE(z.isZero());

        Vec<d64> nz{0.0, 1e-3, 0.0};
        REQUIRE_FALSE(nz.isZero(1e-6));
        REQUIRE(nz.isZero(1e-2));
    }
}

TEST_CASE("Vec dot, hadamard and outer products", "[vec][products]") {
    Vec<d64> a{1.0, 2.0, 3.0};
    Vec<d64> b{4.0, -5.0, 6.0};

    SECTION("dot product") {
        REQUIRE(a.dot(b) == 12.0);  // 1*4 + 2*-5 + 3*6
    }

    SECTION("hadamard product") {
        REQUIRE(a.hadamard(b) == Vec<d64>{4.0, -10.0, 18.0});

        Vec<d64> c = a;
        c.hadamardInPlace(b);
        REQUIRE(c == Vec<d64>{4.0, -10.0, 18.0});
    }

    SECTION("dot and hadamard throw on shape mismatch") {
        Vec<d64> bad{1.0, 2.0};
        REQUIRE_THROWS_AS(a.dot(bad), std::invalid_argument);
        REQUIRE_THROWS_AS(a.hadamard(bad), std::invalid_argument);
    }

    SECTION("outer product produces the correct rectangular matrix") {
        Vec<d64> u{1.0, 2.0};
        Vec<d64> v{3.0, 4.0, 5.0};
        Matrix M = u.outer(v);
        REQUIRE(M.rows() == 2);
        REQUIRE(M.cols() == 3);
        REQUIRE(M(0, 0) == 3.0);
        REQUIRE(M(0, 2) == 5.0);
        REQUIRE(M(1, 0) == 6.0);
        REQUIRE(M(1, 2) == 10.0);
    }
}

TEST_CASE("Vec norms and normalization", "[vec][norm]") {
    Vec<d64> v{3.0, 4.0};

    SECTION("norm and lnorm") {
        REQUIRE(std::fabs(v.norm() - 5.0) < 1e-12);
        REQUIRE(std::fabs(v.lnorm(2) - 5.0) < 1e-12);
        REQUIRE(std::fabs(v.lnorm(1) - 7.0) < 1e-12);
        REQUIRE(std::fabs(v.lnorm(std::numeric_limits<d64>::infinity()) - 4.0) < 1e-12);
    }

    SECTION("normalized/normalize produce a unit vector in the same direction") {
        Vec<d64> n = v.normalized();
        REQUIRE(n.isApprox(Vec<d64>{0.6, 0.8}, 1e-9, 0.0));

        Vec<d64> v2 = v;
        v2.normalize();
        REQUIRE(v2.isApprox(Vec<d64>{0.6, 0.8}, 1e-9, 0.0));
    }

    SECTION("normalizing a zero vector throws") {
        Vec<d64> z{0.0, 0.0};
        REQUIRE_THROWS_AS(z.normalized(), std::invalid_argument);
        REQUIRE_THROWS_AS(z.normalize(), std::invalid_argument);
    }
}

TEST_CASE("Vec cross product", "[vec][cross]") {
    Vec<d64> e1{1.0, 0.0, 0.0};
    Vec<d64> e2{0.0, 1.0, 0.0};

    SECTION("standard basis vectors") {
        REQUIRE(e1.cross(e2) == Vec<d64>{0.0, 0.0, 1.0});
    }

    SECTION("throws for non-3D vectors") {
        Vec<d64> bad{1.0, 2.0};
        REQUIRE_THROWS_AS(e1.cross(bad), std::invalid_argument);
    }
}

TEST_CASE("Vec reductions", "[vec][reduce]") {
    Vec<d64> v{1.0, -5.0, 3.0};
    REQUIRE(v.sum() == -1.0);
    REQUIRE(v.max() == -5.0);
    REQUIRE(v.min() == 1.0);
}

TEST_CASE("Vec::random produces values within [-1, 1]", "[vec][random]") {
    Vec<d64> r = Vec<d64>::random(10);
    REQUIRE(r.size() == 10);

    bool allInRange = true;
    for (u32 i = 0; i < r.size(); ++i) {
        if (r(i) < -1.0 || r(i) > 1.0) {
            allInRange = false;
        }
    }
    REQUIRE(allInRange);
}