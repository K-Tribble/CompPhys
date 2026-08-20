# CompPhys - Computational Physics Library

This project contains implementations developed while studying the Computational Physics course at Imperial College London.

## Features

### Linear Algebra (`linalg/`)

**Vector Operations** (`linalg/vec.hpp`)
- Dense vector class with standard operations
- Templated class to accept any floating point, or complex floating point data type
- Element access and span support
- Factory methods: `zeros()`, `ones()`, `basis()`, `random()`
- Arithmetic and algebraic operations
- Support for custom random distributions

**Matrix Operations** (`linalg/matrix.hpp`)
- Dense matrix class supporting various operations
- Templated class to accept any floating point, or complex floating point data type
- Creation methods: `zeros()`, `ones()`, `identity()`, `diagonal()`
- Subscript operators for element and row/column access
- Matrix arithmetic (addition, multiplication, etc.)
- Determinant and trace calculations
- Matrix decompositions:
  - LU decomposition
  - QR decomposition
  - Eigenvalue decomposition
- Matrix inversions:
  - Gauss-Jordan elimination
  - Cofactor method
- Matrix utilities: shape queries, approximate equality checking

**Linear Solvers** (`linalg/linalg_solve.hpp`)

*Direct Methods:*
- LU decomposition solver for general matrices
- Forward and backward substitution for triangular systems
- Single and multiple right-hand side support

*Iterative Methods:*
- **Jacobi method** - simple iterative solver with consistent convergence properties
- **Gauss-Seidel method** - improved convergence over Jacobi through sequential updates
- **Successive Over-Relaxation (SOR)** - accelerated iterative method with customizable relaxation parameter
- Configurable convergence criteria:
  - Fractional error or residual norm monitoring
  - Custom tolerance settings (default: 1e-14)
  - Flexible iteration limits
  - Configurable norm orders (L1, L2, etc.)

### Calculus (`calculus/`)

**Numerical Differentiation** (`calculus/differentiation.hpp`)
- First derivative computation with multiple schemes:
  - **Forward difference** - O(h) accuracy, for lower precision or boundary points
  - **Backward difference** - O(h) accuracy, for right boundary evaluation
  - **Central difference** - O(h²) accuracy, highest precision with symmetric stencil
- Second derivative computation using central difference (O(h²) accuracy)
- Vectorized differentiation supporting arrays and spans
- Automatic scheme selection for boundary handling
- Customizable step size parameter

**Numerical Integration** (`calculus/integration.hpp`)
- Trapezoidal and Simpsons rule with adaptive refinement
- Automatic convergence detection based on Richardson extrapolation
- Configurable stopping conditions and iteration limits
- Convergence tracking with error estimation
- Result structure containing:
  - Integrated value
  - Convergence status
  - Final error estimate
  - Number of iterations performed

### Nonlinear Solvers (`nonlin_solve.hpp`)

**Root Finding Methods**
- **Bisection method** - robust bracketing method for 1D root finding
  - Guaranteed convergence for continuous functions with sign change
  - Linear convergence
  - Separate function and variable tolerance parameters
  - Iteration limit control (default: 1000)
  - Detailed result structure with convergence information
- **Newton-Raphson method** - open method for 1D root finding
  - Requires the analytic derivative of the function
  - Quadratic convergence
  - Separate function and variable tolerance parameters
  - Iteration limit control (default: 1000)
  - Detailed result structure with convergence information
- **Secant method** - open method for 1D root finding
  - Also quadratic convergence
  - Numerically calculates the derivative of the function using finite difference methods
  - Separate function and variable tolerance parameters
  - Iteration limit control (default: 1000)
  - Detailed result structure with convergence information

### Interpolation (`interpolation.hpp`)

**Interpolation Methods**
- **Linear Interpolation (Lerp)** - piecewise linear approximation between known points
- **Lagrange Polynomial Interpolation** - exact polynomial fitting through data points
  - Single point evaluation or batch interpolation
  - Efficient O(n²) Lagrange basis computation
- **Cubic Spline Interpolation** - smooth curves with C² continuity
  - Efficient polynomial coefficient storage (4×(n-1) matrix)
  - Localized interpolation for performance
  - Supports natural spline boundary conditions

### Utilities

**Type Definitions** (`types.hpp`)
- `d64` - (double)
- `u32` - (std::size_t)

**Constants** (`constants.hpp`)
- `kDefaultAbsTol = 1e-10` - default absolute tolerance for general use
- `kDefaultRelTol = 1e-10` - default relative tolerance for general use
- `kIterStopCondition = 1e-14` - default iteration stopping condition
- `kSingularPivotTol = 100.0` - pivot tolerance for singular matrix detection (machine epsilon multiplier)

## Project Structure

```
CompPhys/
├── include/                   # Header files (API definitions)
│   ├── constants.hpp          # Global constants and tolerances
│   ├── types.hpp              # Type aliases
│   ├── scalar.hpp             # Define Scalar concept 
│   ├── nonlin_solve.hpp       # Nonlinear root finding
│   ├── interpolation.hpp      # Interpolation methods
│   ├── calculus/
│   │   ├── differentiation.hpp  # Numerical differentiation
│   │   └── integration.hpp      # Numerical integration
│   └── linalg/
│       ├── matrix.hpp              # Matrix class and operations
│       ├── matrix.tpp              # Matrix class mplementations
│       ├── vec.hpp                 # Vector class and operations
│       ├── vec.tpp                 # Vector implementations
│       ├── linalg_common.hpp       # Shared linear algebra utilities
│       ├── linalg_interop.hpp      # Interoperability utilities
│       ├── linalg_interop.tpp      # Interoperability utilities implementations
│       └── linalg_solve.hpp        # Solver algorithms
│       └── linalg_solve.tpp        # Solver algorithms implementations
├── src/                       # Implementation files
│   ├── main.cpp               # Main executable and examples
│   ├── interpolation.cpp
│   ├── calculus/
│   │   └── differentiation.cpp
├── tests/                     # Comprehensive unit tests
│   ├── test_vec.cpp           # Vector operations tests
│   ├── test_matrix.cpp        # Matrix operations tests
│   ├── test_linalg_solve.cpp  # Linear solver tests
│   ├── test_differentiation.cpp
│   ├── test_integration.cpp
│   ├── test_interop.cpp
│   └── test_nonlin.cpp        # Nonlinear solver tests
├── build/                     # Compiled objects and executables
├── Makefile                   # Build configuration
└── README.md                  # This file
```

## Building and Running

### Requirements
- C++20 compatible compiler (g++ 10.0 or later recommended)
- Standard C++ library with C++20 support

### Compilation
```bash
# Build all source files
make

# Run main executable
./build/main

# Run tests (if configured in Makefile)
make test
```

### Compiler Flags
- `-std=c++20` - C++20 standard
- `-O3` - Optimization level 3 for production performance
- `-Wall -Wextra` - All warnings enabled for code quality
- `-Iinclude` - Include directory specification

## Usage Examples

### Linear Algebra

#### Creating Vectors and Matrices
```cpp
#include "linalg/vec.hpp"
#include "linalg/matrix.hpp"

// Create vectors
linalg::Vec<double> v1(5, 0.0);                    // 5-element vector initialized to 0
linalg::Vec<double> v2 = {1.0, 2.0, 3.0};          // Initialize from list
linalg::Vec<double> v3 = linalg::Vec<double>::ones(5);     // Vector of ones
linalg::Vec<double> v4 = linalg::Vec<double>::random(5);   // Random vector [-1, 1]

// Create matrices
linalg::Matrix<double> A(3, 3, 1.0);                    // 3x3 matrix initialized to 1
linalg::Matrix<double> I = linalg::Matrix<double>::identity(3); // 3x3 identity matrix
linalg::Matrix<double> D = linalg::Matrix<double>::diagonal({1, 2, 3}); // Diagonal matrix

// Access elements
double val = A(0, 1);  // Get element at row 0, column 1
A(0, 1) = 5.0;         // Set element
linalg::Vec row = A(0); // Get row 0 as vector
linalg::Vec col = A.getCol(1); // Get column 1
```

#### Solving Linear Systems
```cpp
#include "linalg/linalg_solve.hpp"

linalg::Matrix<double> A = {...};  // Coefficient matrix
linalg::Vec<double> b = {...};      // Right-hand side

// Direct solver using LU decomposition
linalg::Vec<double> x = linalg::solve::lu(A, b);

// Iterative solvers with custom stopping conditions
linalg::solve::IterStoppingCondition<double> sc;
sc.stopCondition = 1e-12;
sc.errType = linalg::solve::errorType::Residual;

auto result = linalg::solve::jacobi(A, b, sc, 1000);
if (result.success) {
    std::cout << "Solution found in " << result.numIter << " iterations" << std::endl;
    linalg::Vec solution = result.x_final;
}

auto result2 = linalg::solve::gaussSeidel(A, b, sc, 1000);
auto result3 = linalg::solve::sor(A, b, 1.5, sc, 1000); // w=1.5 relaxation parameter
```

### Calculus

#### Numerical Differentiation
```cpp
#include "calculus/differentiation.hpp"

auto f = [](double x) { return x * x; };

// Compute first derivative at x=2 with different schemes
double df_central = calculus::differentiate::firstDerivAt(
    f, 2.0, 0.01, 
    calculus::differentiate::DiffScheme::Central);

double df_forward = calculus::differentiate::firstDerivAt(
    f, 2.0, 0.01,
    calculus::differentiate::DiffScheme::Forward);

// Compute second derivative
double d2f = calculus::differentiate::secondDerivAt(f, 2.0, 0.01);

// Differentiate arrays of data points
std::vector<d64> x = {1.0, 1.01, 1.02, 1.03, 1.04};
std::vector<d64> y = {1.0, 1.0201, 1.0404, 1.0609, 1.0816};  // y = x²
auto derivatives = calculus::differentiate::differentiate(x, y);  // Should be ~2x
auto second_deriv = calculus::differentiate::secondDeriv(x, y);   // Should be ~2
```

#### Numerical Integration
```cpp
#include "calculus/integration.hpp"

auto f = [](double x) { return std::sin(x); };

// Integrate using trapezoidal rule with adaptive refinement
auto trapzresult = calculus::integrate::trapezoidal(f, 0.0, M_PI, 1e-10, 1000);
if (trapzResult.converged) {
    std::cout << "Integral = " << trapzResult.value << std::endl;
    std::cout << "Error: " << trapzResult.finalError << std::endl;
    std::cout << "Iterations: " << trapzResult.numIter << std::endl;
} else {
    std::cout << "Failed to converge" << std::endl;
}

// Integrate using simpsons rule with adaptive refinement
auto simpsonsResult = calculus::integrate::simpsons(f, 0.0, M_PI, 1e-10, 1000);
if (simpsonsResult.converged) {
    std::cout << "Integral = " << simpsonsResult.value << std::endl;
    std::cout << "Error: " << simpsonsResult.finalError << std::endl;
    std::cout << "Iterations: " << simpsonsResult.numIter << std::endl;
} else {
    std::cout << "Failed to converge" << std::endl;
}
```

### Nonlinear Solvers

#### Root Finding
```cpp
#include "nonlin_solve.hpp"

auto f = [](double x) { return x * x - 2.0; };

// Find root using bisection method (finds sqrt(2))
auto bisectionResult = nonlin::bisection(f, 1.0, 2.0, 1e-10, 1e-10, 1000);
if (bisectionResult.converged && bisectionResult.foundRoot) {
    std::cout << "Root found at x = " << bisectionResult.root << std::endl;
    std::cout << "f(x) = " << bisectionResult.function_val << std::endl;
    std::cout << "Iterations: " << bisectionResult.numIter << std::endl;
}
// Find root using newton-raphson method
auto fprime = [](double x) { return 2.0 * x};
auto newtonResult = nonlin::newton(f, fprime, 2.0);
if (newtonResult.converged && newtonResult.foundRoot) {
    std::cout << "Root found at x = " << newtonResult.root << std::endl;
    std::cout << "f(x) = " << newtonResult.function_val << std::endl;
    std::cout << "Iterations: " << newtonResult.numIter << std::endl;
}
find root using secant method
auto secantResult = nonlin::secant(f, 3.0, 2.0, 1e-10, 1e-10, 1000);
if (secantResult.converged && secantResult.foundRoot) {
    std::cout << "Root found at x = " << secantResult.root << std::endl;
    std::cout << "f(x) = " << secantResult.function_val << std::endl;
    std::cout << "Iterations: " << secantResult.numIter << std::endl;
}
```

### Interpolation

#### Lagrange Polynomial Interpolation
```cpp
#include "interpolation.hpp"

std::vector<d64> x_known = {0.0, 1.0, 2.0};
std::vector<d64> f_known = {0.0, 1.0, 4.0};  // f(x) = x²
std::vector<d64> x_interp = {0.5, 1.5, 2.5};

// Interpolate using Lagrange polynomials
auto y = lagrangePolynomial(x_interp, x_known, f_known);
// y ≈ {0.25, 2.25, 6.25}
```

#### Cubic Spline Interpolation
```cpp
std::vector<d64> x_known = {0.0, 1.0, 2.0, 3.0};
std::vector<d64> f_known = {0.0, 1.0, 4.0, 9.0};

// Create cubic spline (natural spline by default)
CubicSplineResult spline = cubicSpline(x_known, f_known);

// Evaluate at new points
double y1 = spline.CS(0.5);   // Smooth interpolation
double y2 = spline.CS(1.5);
double y3 = spline.CS(2.5);
```

#### Linear Interpolation
```cpp
std::vector<d64> x = {0.0, 1.0, 2.0};
std::vector<d64> y = {0.0, 2.0, 4.0};
std::vector<d64> x_new = {0.25, 0.75, 1.5};

auto y_interp = lerp(x_new, x, y);
// Fast piecewise linear interpolation
```

## Testing

The project includes comprehensive unit tests covering:
- Vector operations (creation, arithmetic, norms)
- Matrix operations (creation, arithmetic, decompositions)
- Linear system solvers (direct and iterative methods)
- Numerical differentiation (all schemes)
- Numerical integration (convergence behavior)
- Nonlinear root finding (every method on different types of functions)
- Interoperability between vector and matrix types

Tests are located in the `tests/` directory and validate correctness of implementations.

## Numerical Methods Overview

### Linear System Solving
- **Direct Methods**: Suitable for small to medium systems, exact (up to machine precision)
- **Iterative Methods**: Better for large sparse systems, may converge slowly or diverge
  - Jacobi: Simple, requires diagonal dominance for convergence
  - Gauss-Seidel: Better convergence than Jacobi, sequential update scheme
  - SOR: Accelerated convergence with relaxation parameter tuning

### Matrix Decompositions
- **LU Factorization** - decomposes A into lower and upper triangular matrices; used for solving systems and computing determinants
- **QR Factorization** - decomposes into orthogonal and upper triangular matrices;
- **Eigenvalue Decomposition** - finds eigenvalues and eigenvectors;

### Differentiation Schemes
- **Central Difference** (O(h²)) - highest accuracy, requires function at symmetric points
- **Forward Difference** (O(h)) - useful for causality-constrained problems
- **Backward Difference** (O(h)) - alternative to forward difference

### Integration Methods
- **Trapezoidal Rule** - simple linear approximation between points
  - Adaptive refinement automatically increases accuracy
  - Has O(h^2) accuracy
- **Simpsons Rule** - quadratic interpolation between points
  - Uses Richardson extrapolation on trapezoidal rule to gain O(h^4) error
  - Adaptive refinement automatically increases accuracy

### Root Finding
- **Bisection** - guaranteed convergence for continuous functions with sign change
  - Linear convergence rate (O(log(1/ε)) iterations for tolerance ε)
  - Very robust, no derivative required
- **Newton-Raphson** - implemented as newton
  - Improved quadratic convergence
  - Requires an analytic derivative of the function
- **Secant** - does not require a derivative
  - Implements the Newton-Raphson iteration method without an analytic derivative
  - Uses finite difference method to approximate derivative
  - Requires two initial guesses

### Interpolation
- **Linear Interpolation** - returns a point along a straight line between the two neighboring points
  - Good for densely sampled data
  - Discontinous derivaties at each point
- **Lagrange Polynomials** - exact interpolation through all data points
  - Runge's phenomenon risk with many points
  - Useful for smooth functions with few data points
- **Cubic Splines** - C² continuous piecewise cubic polynomials
  - Avoids oscillations of high-degree polynomials
  - Good for smooth approximations of data

## Performance Considerations

- Compiled with `-O3` optimization for production performance
- Matrix and vector operations use `std::span` and move semantics for memory efficiency
- Iterative solvers support early convergence termination to avoid unnecessary computation
- Trapezoidal rule uses adaptive refinement for efficient accuracy
- Cubic spline uses compact coefficient representation (minimal memory footprint)
- Forward substitution and back substitution are O(n²) algorithms optimized for speed

## C++ Standards

This library requires **C++20** features including:
- Concepts (for template constraints)
- Ranges and spans for efficient data handling
- Modern template features and SFINAE
- Constexpr support for compile-time evaluation
- Designated initializers for structured data

## License

MIT