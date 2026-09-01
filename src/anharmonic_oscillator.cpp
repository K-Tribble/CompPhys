#include <iostream>
#include <random>
#include <cmath>
#include "calculus/integration.hpp"
#include "calculus/target_distributions/differentiable_target.hpp"
#include "linalg/vec.hpp"
#include "types.hpp"

class QuantumAnharmonicOscillator : public calculus::sample::DifferentiableTarget {
public:
    QuantumAnharmonicOscillator(u32 N, d64 m, d64 omega, d64 lambda, d64 beta)
        : N_(N), m_(m), omega_(omega), lambda_(lambda), dtau_(beta / N) {}

    u32 dim() const override {
        return N_;
    }

    // Returns the negative Euclidean action (equivalent to log unnormalized density)
    // Must not mutate shared/global state due to potential OpenMP parallelization
    d64 logDensity(const linalg::Vec<d64>& x) const override {
        d64 action = 0.0;
        for (u32 i = 0; i < N_; ++i) {
            u32 next = (i + 1) % N_;

            d64 kinetic = (m_ / (2.0 * dtau_)) * std::pow(x(next) - x(i), 2);
            d64 potential = dtau_ * (0.5 * m_ * omega_ * omega_ * x(i) * x(i) + lambda_ * std::pow(x(i), 4));
            
            action += kinetic + potential;
        }
        return -action; 
    }

    // Gradient of the log density, required for MALA
    linalg::Vec<d64> gradLogDensity(const linalg::Vec<d64>& x) const override {
        linalg::Vec<d64> grad(N_);
        
        for (u32 i = 0; i < N_; ++i) {
            u32 next = (i + 1) % N_;
            u32 prev = (i + N_ - 1) % N_;
            
            d64 d_kinetic = -(m_ / dtau_) * (x(next) - 2.0 * x(i) + x(prev));
            d64 d_potential = dtau_ * (m_ * omega_ * omega_ * x(i) + 4.0 * lambda_ * std::pow(x(i), 3));
            
            // Gradient of log density is the negative gradient of the action
            grad(i) = -(d_kinetic + d_potential); 
        }
        return grad;
    }

private:
    u32 N_;
    d64 m_, omega_, lambda_, dtau_;
};

void run_pimc_simulation_mala() {
    // 1. Physical Parameters
    u32 N = 100;         // Number of imaginary time slices (beads)
    d64 m = 1.0;         // Mass
    d64 omega = 1.0;     // Harmonic frequency
    d64 lambda = 0.1;    // Anharmonic coupling constant
    d64 beta = 5.0;      // Inverse temperature (1/kT)
    
    QuantumAnharmonicOscillator target(N, m, omega, lambda, beta);

    // 2. The Ideal Initialization (Cold Start)
    // Initial sample dimension must match target distribution dimension.
    linalg::Vec<d64> initial(N);
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<d64> thermal_noise(0.0, 0.01);
    
    for (u32 i = 0; i < N; ++i) {
        initial(i) = thermal_noise(gen); // Collapsed polymer with tiny noise
    }
    std::cout << target.dim() << std::endl;

    // 3. Define the Observable (Integrand)
    // Calculates the mean squared position for the current MCMC sample
    auto observable_x2 = [](const linalg::Vec<d64>& x) {
        d64 sum_sq = 0.0;
        for (u32 i = 0; i < x.size(); ++i) {
            sum_sq += x(i) * x(i);
        }
        return sum_sq / x.size();
    };

    // 4. Run the Integrator
    d64 h = 1.14e-2;     // Transition proposal variance - must be positive for MALA
    u32 maxN = 5000000 ;  // Total MCMC iterations
    
    // Call the convenience overload of MALA which owns its own engine
    calculus::sample::MCMCResult res = calculus::sample::mala(
        target, 
        observable_x2, 
        initial, 
        h, 
        maxN
    );

    // 5. Access Results
    std::cout << "MALA Result:\n";
    std::cout << "Expectation Value <x^2>: " << res.value << "\n";
    std::cout << "Acceptance Rate:         " << res.acceptanceRate << "\n";
    std::cout << "Effective Sample Size:   " << res.effectiveSampleSize << "\n";
    std::cout << "Monte Carlo Error:       " << res.finalError << "\n";
}

void run_pimc_simulation_hmc() {
    // 1. Physical Parameters
    u32 N = 100;         
    d64 m = 1.0;         
    d64 omega = 1.0;     
    d64 lambda = 0.1;    
    d64 beta = 5.0;      
    
    QuantumAnharmonicOscillator target(N, m, omega, lambda, beta);

    // 2. Initialization
    // The initial sample dimension must match the target distribution dimension.
    linalg::Vec<d64> initial(N);
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<d64> thermal_noise(0.0, 0.01);
    
    for (u32 i = 0; i < N; ++i) {
        initial(i) = thermal_noise(gen); 
    }

    // 3. Define the Observable
    auto observable_x2 = [](const linalg::Vec<d64>& x) {
        d64 sum_sq = 0.0;
        for (u32 i = 0; i < x.size(); ++i) {
            sum_sq += x(i) * x(i);
        }
        return sum_sq / x.size();
    };

    // 4. Construct the Mass Matrix
    // The mass matrix must be square and match the dimension of the initial position.
    linalg::Matrix<d64> mass_matrix = linalg::Matrix<d64>::identity(N); 

    // 5. Run the Integrator
    // The step size must be positive, and the number of steps must be greater than zero
    d64 stepSize = 0.08;  // Leapfrog step size (epsilon)
    u32 numSteps = 15;    // Number of leapfrog steps (L)
    u32 maxN = 50000;     // Total MCMC iterations
    
    // Call the convenience overload of HMC which owns its own random engine
    calculus::sample::MCMCResult res = calculus::sample::hmc(
        target, 
        observable_x2, 
        initial, 
        mass_matrix,
        stepSize,
        numSteps,
        maxN
    );

    // 6. Access Results
    std::cout << "HMC Results:\n";
    std::cout << "Expectation Value <x^2>: " << res.value << "\n";
    std::cout << "Acceptance Rate:         " << res.acceptanceRate << "\n";
    std::cout << "Effective Sample Size:   " << res.effectiveSampleSize << "\n";
    std::cout << "Monte Carlo Error:       " << res.finalError << "\n";
}

int main() {
    run_pimc_simulation_mala();
    run_pimc_simulation_hmc();
    return 0;
}