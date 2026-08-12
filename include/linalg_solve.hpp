#pragma once

#include "matrix.hpp"
#include "vec.hpp"
#include "linalg_interop.hpp"
#include <functional>
#include "types.hpp"
#include "constants.hpp"

namespace linalg {

namespace solve {
    // enum that defines the type of error computed between two vectors
    enum class errorType {  
        Fractional, Residual    
    };  

    enum class SplitType {  
        Lower, Upper    
    };  

    struct IterStoppingCondition {  
        d64 stopCondition = kIterStopCondition;  
        d64 lnorm_ord = 2;  
        errorType errType = errorType::Fractional;  
    };  

    struct IterResult { 
        d64 lnorm_ord;  
        errorType errType;  
        bool success;   
        Vec x_final;    
        Vec finalResidualVector;    
        d64 finalFractionalResErr;  
        u32 numIter;    
    };  

    
    // Solves Ax=b for when A is lower triangular by forward substitution
    Vec forwardSub(const Matrix& lt, const Vec& rhs);
    // Solves Ax=b for when A is uperr triangular by backward substitution
    Vec backSub(const Matrix& up, const Vec& rhs);
    
    Vec lu(const LUResult& f, const Vec& b);
    std::vector<Vec> lu(const LUResult& f, const std::vector<Vec>& bs);
    
    Vec lu(const Matrix& A, const Vec& b);
    
    std::vector<Vec> lu(const Matrix& A, const std::vector<Vec>& bs);
    
    // type alias to define a function evaluate the update step in an iterative method
    using solveFn = std::function<Vec(const Matrix&, const Vec&)>;
    IterResult runSplitIteration(const Matrix& A, const Vec& b, const Matrix& B, const Matrix& S, 
        const solveFn& solve, IterStoppingCondition sc, const u32 maxIter);
    // Iterative solver with jacobi method
    IterResult jacobi(const Matrix& A, const Vec& b, IterStoppingCondition sc, const u32 maxIter);
    // Iterative solver with successive over relaxation method
    // the paramter w is the relaxation parameter defined as 1/a
    IterResult sor(const Matrix& A, const Vec& b, const d64 w, IterStoppingCondition sc, const u32 maxIter, SplitType split = SplitType::Lower);
    // Iterative solver with gauss seidel method
    IterResult gaussSeidel(const Matrix& A, const Vec& b, IterStoppingCondition sc, const u32 maxIter, SplitType split = SplitType::Lower);

} // namespace solve
    
} // namespace linalg