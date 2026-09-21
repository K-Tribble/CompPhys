# CompPhys - Computational Physics Library

This project contains numerical methods and Monte Carlo implementations developed while studying Computational Physics at Imperial College London.

The codebase covers both classic numerical-analysis routines (linear algebra, differentiation, optimization, interpolation, integration) and the newer Markov chain Monte Carlo infrastructure used for lattice and path-integral simulations. The modern MCMC stack lives primarily under `include/mcmc/` and is exercised by the examples in `src/ising_demo.cpp` and `src/anharmonic_pimc.cpp`.

## Features

### Linear Algebra (`linalg/`)

**Vector Operations** (`linalg/vec.hpp`)
- Dense vector class with standard operations
- Templated class to accept any floating point, or complex floating point data type
- Element access and span support
- Factory methods: `zeros()`, `ones()`, `basis()`, `random()`
- Arithmetic and algebraic operations
- Support for custom random distributions
- Includes methods for begin and end pointers to have support for range concepts

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
- Supports parallelized gradient computation of a function of multiple variables
- Supports parallelized hessian computation of a function of multiple variables
- Both above are parallelized with OMP

**Optimization** ('calculus/optimize.hpp')
- Includes two robust iterative methods to minimize a function
 - **Gradient Descent with adaptive stepsize** - linear convergence
    - Calculates the minimum of a function with gradient descent
    - Uses parallelized gradient method
    - Stepsize is calculated at every iteration
    - Step size is calculated by a backtracking line search, such that the function decreases enough to meet the Armijo condition
    - This guarantees stability by preventing overshooting and stalling
 - **Newtons Method with Levenberg-Marquardt regularization** - quadratic convergence
    - Calculates the minimum of a function with Newtons method using LV regularization to guarantee stability when the hessian is not positive definite
    - Uses parallelized hessian method

**Numerical Integration** (`calculus/integration.hpp`)
- Trapezoidal and Simpsons rule with adaptive refinement
- Monte Carlo integration over axis-aligned hyperrectangles
- Automatic convergence detection based on Richardson extrapolation
- Configurable stopping conditions and iteration limits
- Convergence tracking with error estimation
- Result structure containing:
  - Integrated value
  - Convergence status
  - Final error estimate
  - Number of iterations performed

**Sampling and Monte Carlo Methods** (`calculus/integration.hpp`)
- Self-normalized importance sampling with effective sample size diagnostics
- Metropolis-Hastings sampling with general transition proposals
- Isotropic Gaussian Metropolis-Hastings convenience function
- Metropolis-adjusted Langevin algorithm (MALA)
- Hamiltonian Monte Carlo (HMC) with configurable mass matrix and leapfrog steps
- Reproducible overloads accepting a caller-owned random-number generator
- Convenience overloads with thread-local engines seeded from `std::random_device`

**Sampling Infrastructure** (`calculus/accumulators.hpp` and related headers)
- Welford online mean and variance accumulation
- Numerically stable log-weight accumulation for importance sampling
- Sokal and Geyer integrated autocorrelation-time estimates
- Target-distribution interface for unnormalized log densities
- Analytic and finite-difference differentiable target interfaces
- Gaussian target, Gaussian proposal, Gaussian transition, and MALA transition implementations

**Integration Results** (`calculus/integral_results.hpp`)
- `IntegralResult` for deterministic and Monte Carlo integration
- `ImportanceSampleResult` with estimate, error, convergence, and effective sample size
- `MCMCResult` with estimate, error, acceptance rate, effective sample size, and ESS method

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
- **Brent's method** - bracketed hybrid method combining bisection, secant, and inverse quadratic interpolation
  - Retains bisection's robustness while usually converging faster
  - Requires a continuous function and an initial sign-changing bracket

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
- `kMonteCarloStopCondition = 1e-4` - default Monte Carlo stopping condition
- `kSingularPivotTol = 100.0` - pivot tolerance for singular matrix detection (machine epsilon multiplier)

## Project Structure

```
CompPhys/
├── include/                              # public headers and numerical kernels
│   ├── calculus/
│   │   ├── accumulators.hpp             # online statistics and ESS estimators
│   │   ├── differentiation.hpp          # numerical differentiation
│   │   ├── integral_results.hpp         # integration and sampling result containers
│   │   ├── integration.hpp              # older MCMC convenience routines and MC integration
│   │   ├── optimize.hpp                 # optimization routines
│   │   ├── proposal/
│   │   │   ├── proposal.hpp
│   │   │   └── gaussian_proposal.hpp
│   │   ├── target_distributions/
│   │   │   ├── target_distribution.hpp
│   │   │   ├── differentiable_target.hpp
│   │   │   ├── gaussian_target.hpp
│   │   │   └── finite_difference_target.hpp
│   │   └── transition_proposal/
│   │       ├── transition_proposal.hpp
│   │       ├── gaussian_transition.hpp
│   │       └── mala_transition.hpp
│   ├── linalg/
│   │   ├── matrix.hpp
│   │   ├── matrix.tpp
│   │   ├── vec.hpp
│   │   ├── vec.tpp
│   │   ├── linalg_common.hpp
│   │   ├── linalg_interop.hpp
│   │   ├── linalg_interop.tpp
│   │   ├── linalg_solve.hpp
│   │   └── linalg_solve.tpp
│   ├── mcmc/
│   │   ├── chain_stats.hpp             # autocorrelation and ESS diagnostics
│   │   ├── continuous.hpp              # MALA/HMC stepper wrappers around the target interface
│   │   ├── driver.hpp                  # generic multi-chain driver and RunConfig/RunResult
│   │   ├── io.hpp                     # output helpers and serialization
│   │   ├── stepper.hpp                 # ChainStepper and Observables concepts
│   │   ├── models/
│   │   │   └── ising2d.hpp             # 2D Ising model state and energy utilities
│   │   └── steppers/
│   │       └── ising_stepper.hpp       # single-spin-flip Metropolis stepper
│   ├── constants.hpp
│   ├── interpolation.hpp
│   ├── nonlin_solve.hpp
│   ├── scalar.hpp
│   ├── types.hpp
│   └── ...
├── src/                                  # executable examples and project-specific demos
│   ├── main.cpp                         # general smoke-test / demonstration driver
│   ├── ising_demo.cpp                   # modern multi-chain Ising Metropolis example
│   ├── anharmonic_pimc.cpp              # path-integral Monte Carlo with modern mcmc infrastructure
│   ├── anharmonic_oscillator_basic_test.cpp  # legacy prototype using initial integration.hpp MCMC API
│   ├── interpolation.cpp
│   └── calculus/
│       └── differentiation.cpp
├── tests/                                # Catch2 unit tests
│   ├── test_vec.cpp
│   ├── test_matrix.cpp
│   ├── test_linalg_solve.cpp
│   ├── test_differentiation.cpp
│   ├── test_integration.cpp
│   ├── test_interop.cpp
│   ├── test_nonlin.cpp
│   ├── test_optimize.cpp
│   ├── test_ising.cpp
│   └── test_continuous.cpp
├── analysis/                             # analysis scripts for output data
│   ├── analyze_anharmonic.py
│   └── analyze_ising.py
├── data/                                 # example output artifacts
├── ising_data/                           # generated Ising scan outputs
├── runs/                                 # run directories / collected output
├── build/                                # generated binaries and object files
├── Makefile
├── README.md
├── todo.txt
├── .gitignore
└── .github/workflows/
    └── ci.yml
```

The local `build/`, `.vscode/`, and `todo.txt` paths are intentionally omitted
because they are ignored by Git.

## Building and Running

### Requirements
- C++20 compatible compiler (g++ 10.0 or later recommended)
- Standard C++ library with C++20 support

### Dependencies
- Catch2 for the test executable
- OpenMP (`libomp` on macOS; compiler OpenMP support on Linux)

On macOS, the Makefile expects Homebrew packages named `catch2` and `libomp`.

### Compilation
```bash
# Build the core library / smoke-test executable
make

# Run the general example binary
./build/main

# Build and run the full Catch2 suite
make test

# Modern MCMC examples
make ising
./build/ising --L 16 --direction cooling

make pimc
./build/anharmonic_pimc
```

The current project focus is on the MCMC examples in `src/ising_demo.cpp` and `src/anharmonic_pimc.cpp`. Older standalone targets such as `make anharmonic` and the legacy prototype in `src/anharmonic_oscillator_basic_test.cpp` are retained for historical comparison, but they are not the main active API.

Other Make targets are `run`, `clean`, `rebuild`, and `count`.

### Compiler Flags
- `-std=c++20` - C++20 standard
- `-O2` - Optimization level used by the Makefile
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
auto trapzResult = calculus::integrate::trapezoidal(f, 0.0, M_PI, 1e-10, 1000);
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

#### Monte Carlo Integration
```cpp
#include "calculus/integration.hpp"

auto f = [](const linalg::Vec<d64>& x) { return x(0) * x(0); };
std::vector<d64> left = {0.0};
std::vector<d64> right = {1.0};
std::mt19937 gen(1234);  // Caller-owned engine for reproducible samples

auto result = calculus::integrate::mc(f, left, right, gen, 1e-4, 10000);
```

### Sampling and MCMC Infrastructure

The modern MCMC infrastructure is built around a generic driver and stepper model.
The central abstractions are:

- `mcmc::RunConfig` and `mcmc::RunResult` in `include/mcmc/driver.hpp`
- `mcmc::ChainStepper` and `mcmc::Observables` in `include/mcmc/stepper.hpp`
- `MALAStepper` and `HMCStepper` in `include/mcmc/continuous.hpp`
- model-specific steppers such as `IsingMetropolis` in `include/mcmc/steppers/ising_stepper.hpp`

These components let you run multiple chains, burn-in, thinning, autocorrelation diagnostics, and pooled ESS/R-hat reporting through one unified interface.

#### Example: Ising model with the modern driver

The current 2D Ising example in `src/ising_demo.cpp` uses a production-style `run(factory, observables, cfg, harvest)` pattern:

```cpp
RunConfig cfg;
cfg.sweeps = 20000;
cfg.burnIn = 1000;
cfg.thin = 4;
cfg.numChains = 4;
cfg.seed = 20260905ull;

auto factory = [&](u32 c, std::mt19937& g) {
    Ising2D model(L, 1.0, 0.0);
    model.randomise(g);
    return IsingMetropolis(std::move(model), beta);
};

const auto res = mcmc::run(factory, IsingObservables{}, cfg, harvest);
```

This is the active MCMC infrastructure for lattice systems: each chain is configured independently, run with burn-in and thinning, and then merged with pooled chain diagnostics. The Ising demo also scans temperatures and writes output arrays and metadata for later analysis.

#### Example: anharmonic path-integral Monte Carlo with MALA/HMC

The path-integral example in `src/anharmonic_pimc.cpp` uses the same driver but with a custom differentiable target for the quantum anharmonic oscillator. The target defines a discretized Euclidean action and its gradient, and the run then uses a stepper factory that creates either a `MALAStepper` or `HMCStepper` for each chain.

```cpp
class AnharmonicPath : public calculus::sample::DifferentiableTarget {
public:
    d64 logDensity(const linalg::Vec<d64>& x) const override;
    linalg::Vec<d64> gradLogDensity(const linalg::Vec<d64>& x) const override;
};

auto factory = [&](u32 c, std::mt19937& g) {
    auto x0 = initialisePath(...);
    return mcmc::MALAStepper(target, std::move(x0), h);
};

const RunConfig cfg{.sweeps = 5000, .burnIn = 2000, .thin = 2, .numChains = 4};
const auto res = mcmc::run(factory, PIMCObservables{p, true}, cfg);
```

This is the modern path-integral Monte Carlo workflow in the project: a differentiable target, multi-chain driver, and diagnostics suitable for thermalized quantum systems.

#### Legacy note

`src/anharmonic_oscillator_basic_test.cpp` is an older test that uses the initial MCMC convenience functions from `include/calculus/integration.hpp` (`calculus::sample::mala` and `calculus::sample::hmc`). That prototype is useful for historical comparison, but the current, more general infrastructure is the newer `mcmc::run`/`ChainStepper` design used by `src/ising_demo.cpp` and `src/anharmonic_pimc.cpp`.

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
auto fprime = [](double x) { return 2.0 * x;};
auto newtonResult = nonlin::newton(f, fprime, 2.0);
if (newtonResult.converged && newtonResult.foundRoot) {
    std::cout << "Root found at x = " << newtonResult.root << std::endl;
    std::cout << "f(x) = " << newtonResult.function_val << std::endl;
    std::cout << "Iterations: " << newtonResult.numIter << std::endl;
}
// find root using secant method
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
- Numerical optimization (gradient descent and Newton's method)
- Monte Carlo integration and sampling infrastructure
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
- **Backward Difference** (O(h)) - alternative to forward 
- **Gradient** - estimates the gradient of a function
- **Hessian** - estimates the hessian of a function

### Optimization
- **Gradient descent** O(h) - minimizes a function, uses backtracking line search for stability
- **Newtons Method** O(h²) - minimizes a function, uses Levenberg-Marquadt regularization for stability


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
- **Brents Method** - Combines bisection, secant, and inverse quadratic interpolation.
- It maintains a bracket [a,b] where f(a) and f(b) have opposite signs, so a root is guaranteed to lie inside (assuming continuity).
- It usually takes a faster secant/interpolation step, but if that step looks unreliable, it falls back to bisection.
- Advantages: much faster than pure bisection in typical cases, while retaining bisection's robustness.
- Requirements: f must be continuous over the bracket and f(a)f(b)<0.
- Key idea: use interpolation when it is safe; otherwise, shrink the bracket with bisection.

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