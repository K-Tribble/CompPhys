#pragma once

#include "linalg/matrix.hpp"
#include "linalg/vec.hpp"
#include "linalg/linalg_interop.hpp"
#include <functional>
#include "types.hpp"
#include "constants.hpp"
#include "scalar.hpp"

namespace linalg {

namespace solve {
    // enum that defines the type of error computed between two vectors
    enum class errorType {  
        Fractional, Residual    
    };  

    enum class SplitType {  
        Lower, Upper    
    };  

    template <Scalar T>
    struct IterStoppingCondition {  
        RealType<T> stopCondition = kIterStopCondition;  
        RealType<T> lnorm_ord = 2;  
        errorType errType = errorType::Fractional;  
    };  

    template <Scalar T>
    struct IterResult { 
        RealType<T> lnorm_ord;  
        errorType errType;  
        bool success;   
        Vec<T> x_final;    
        Vec<T> finalResidualVector;    
        RealType<T> finalFractionalResErr;  
        u32 numIter;    
    };  

    
    // Solves Ax=b for when A is lower triangular by forward substitution
    template <Scalar T>
    Vec<T> forwardSub(const Matrix<T>& lt, const Vec<T>& rhs);

    // Solves Ax=b for when A is uperr triangular by backward substitution
    template <Scalar T>
    Vec<T> backSub(const Matrix<T>& up, const Vec<T>& rhs);
    
    // Solve Ax=b for a given LU result. Where A has been broken down into PA=LU
    template <Scalar T>
    Vec<T> lu(const LUResult<T>& f, const Vec<T>& b);
    // Solve Ax=b for a given LU result. Where A has been broken down into PA=LU
    // Does this for several matrices b that are stored in an std::vector
    template <Scalar T>
    std::vector<Vec<T>> lu(const LUResult<T>& f, const std::vector<Vec<T>>& bs);
    
    // Solve Ax=b for a given A matrix and b vector.
    template <Scalar T>
    Vec<T> lu(const Matrix<T>& A, const Vec<T>& b);
    
    // Solve Ax=b for a given A matrix, and several b vectors stored in an std::vector
    template <Scalar T>
    std::vector<Vec<T>> lu(const Matrix<T>& A, const std::vector<Vec<T>>& bs);
    
    // type alias to define a function evaluate the update step in an iterative method
    template <Scalar T>
    using solveFn = std::function<Vec<T>(const Matrix<T>&, const Vec<T>&)>;
    template <Scalar T>
    IterResult<T> runSplitIteration(const Matrix<T>& A, const Vec<T>& b, const Matrix<T>& B, const Matrix<T>& S, 
        const solveFn<T>& solve, IterStoppingCondition<T> sc, const u32 maxIter);
    // Iterative solver with jacobi method
    template <Scalar T>
    IterResult<T> jacobi(const Matrix<T>& A, const Vec<T>& b, IterStoppingCondition<T> sc, const u32 maxIter);
    // Iterative solver with successive over relaxation method
    // the paramter w is the relaxation parameter defined as 1/a
    template <Scalar T>
    IterResult<T> sor(const Matrix<T>& A, const Vec<T>& b, const RealType<T> w, IterStoppingCondition<T> sc, const u32 maxIter, SplitType split = SplitType::Lower);
    // Iterative solver with gauss seidel method
    template <Scalar T>
    IterResult<T> gaussSeidel(const Matrix<T>& A, const Vec<T>& b, IterStoppingCondition<T> sc, const u32 maxIter, SplitType split = SplitType::Lower);

} // namespace solve
    
} // namespace linalg

#include "linalg/linalg_solve.tpp"