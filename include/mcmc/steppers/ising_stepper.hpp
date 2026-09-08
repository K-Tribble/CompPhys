#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
#include <cassert>

#include "types.hpp"
#include "mcmc/models/ising2d.hpp"

namespace mcmc {
// Single spin flip Metropolis, one sweep is N attempted flips at 
// random sites.

class IsingMetropolis {
public: 
    using State = Ising2D;

    IsingMetropolis(Ising2D model, d64 beta) 
        : model_(std::move(model)), beta_(beta), site_(0, model_.size() - 1) {
        if (beta <= 0.0) {
            throw std::invalid_argument("IsingMetropolis: beta must be positive");
        }
        buildTable();
    }

    void sweep(std::mt19937& gen) {
        const u32 n = model_.size();
        for (u32 k = 0; k < n; ++k) {
            const u32 i = site_(gen);
            const int h = model_.localField(i);
            const int s = model_.spins()[i];
            
            // calculating acceptance probability uses the table to look up 
            // a value since de can take oly 10 possible values.
            // If dE <= 0 for a flip then w = 1.0, else it equals std::exp(-beta_ * dE)
            const d64 w = table_[index(h, s)];
            if (w >= 1.0 || uniform_(gen) < w) {
                model_.flip(i);
                ++accepted_;
            }

            ++attempted_;
        }
    }

    const State& state() const {return model_;}
    d64 acceptedFraction() const {
        return attempted_ ? static_cast<d64>(accepted_) / static_cast<d64>(attempted_)
                          : std::numeric_limits<d64>::quiet_NaN();
    }

    void resetCounters() {
        accepted_ = 0;
        attempted_ = 0;
    }

    d64 beta() const {return beta_;}

private:
    static u32 index(int h, int s) {
        const u32 i = static_cast<u32>(((h + 4) / 2) * 2 + (s + 1) / 2);
        assert(i < 10);
        return i;
    }

    void buildTable() {
        for (int h = -4; h <= 4; h+= 2) {
            for (int s = -1; s <= 1; s += 2) {
                const d64 dE = 2.0 * static_cast<d64>(s) * (model_.J() * static_cast<d64>(h) + model_.field());
                table_[index(h, s)] = (dE <= 0.0) ? 1.0 : std::exp(-beta_ * dE);
            }
        }
    }

    Ising2D model_;
    d64 beta_;
    std::array<d64, 10> table_{};
    std::uniform_int_distribution<u32> site_;
    std::uniform_real_distribution<d64> uniform_{0.0, 1.0};
    std::uint64_t accepted_ = 0;
    std::uint64_t attempted_ = 0;
};

} // namespace mcmc