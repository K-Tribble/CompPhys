#pragma once

#include <concepts>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "types.hpp"

namespace mcmc {
    // The stepper owns the chain state and can advance it by one sweep
    // a sweep is the unit everything else is measured in, such as burn in, thinning 
    // tau_int etc. Has to mean roughly same amount of work across algorithms
    // Convention: one sweep is approximately one attempted update per DOF
    template <typename S>
    concept ChainStepper = requires(S& s, const S& cs, std::mt19937& gen) {
        typename S::State;

        // advance chain by one step
        {s.sweep(gen)} -> std::same_as<void>;

        // current config
        {cs.state()} -> std::convertible_to<const typename S::State&>;

        // Fraction of accepted proposals
        {cs.acceptedFraction()} -> std::convertible_to<d64>;

        {s.resetCounters()} -> std::same_as<void>;
    };

    template <typename O, typename State>
    concept Observables = requires(const O& o, const State& s, std::span<d64> out) {
        {o.names()} -> std::convertible_to<std::vector<std::string>>;

        {o.eval(s, out)} -> std::same_as<void>;
    };
} // namespace mcmc