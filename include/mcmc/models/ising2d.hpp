#pragma once
 
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
 
#include "types.hpp"

namespace mcmc {
// 2D Ising model on LxL square lattice wiwth periodic boundaries
// H = -J * sum_<i,j> s_i * s_j - B * sum_i s_i, s_i = +-1

class Ising2D {
public:
    using Spin = std::int8_t;

    explicit Ising2D(u32 L, d64 J = 1.0, d64 field = 0.0)
        : L_(L), N_(L * L), J_(J), field_(field), spins_(L * L, Spin{1}) {
        if (L < 2) {
            throw std::invalid_argument("Ising2D: L must be at least 2");
        }

        // precomputed neighbor table, order R, L, D, U.

        nbr_.resize(4 * N_);
        for (u32 y = 0; y < L_; ++y) {
            for (u32 x = 0; x < L_; ++x) {
                const u32 i = y * L_ + x;
                nbr_[4 * i + 0] = y * L_ + ((x + 1) % L_);
                nbr_[4 * i + 1] = y * L_ + ((x + L_ - 1) % L_);
                nbr_[4 * i + 2] = ((y + 1) % L_) * L_ + x;
                nbr_[4 * i + 3] = ((y + L_ - 1) % L_) * L_ + x;
            }
        }

        refresh();
    }

    u32 L() const {return L_;}
    u32 size() const {return N_;}
    d64 J() const {return J_;}
    d64 field() const {return field_;}

    const std::vector<Spin>& spins() const {return spins_;}
    const u32* neighbors(u32 i) const {return &nbr_[4 * i];}

    d64 energy() const {return energy_;}
    d64 energyPerSite() const {return energy_ / static_cast<d64>(N_);}
    std::int64_t magnetization() const {return magnetization_;}
    d64 magnetizationPerSite() const {
        return static_cast<d64>(magnetization_) / static_cast<d64>(N_);
    }

    // Sum of four neighboring spins
    int localField(u32 i) const {
        const u32* n = neighbors(i);
        return spins_[n[0]] + spins_[n[1]] + spins_[n[2]] + spins_[n[3]];
    }

    // Change in energy for flipping site i
    d64 deltaEnergyOnFlip(u32 i) const {
        return 2.0 * static_cast<d64>(spins_[i]) * (J_ * static_cast<d64>(localField(i)) + field_);
    }

    void flip(u32 i) {
        energy_ += deltaEnergyOnFlip(i);
        magnetization_ -= 2 * static_cast<std::int64_t>(spins_[i]);
        spins_[i] = static_cast<Spin>(-spins_[i]);
    }

    void setAllUp() {
        std::fill(spins_.begin(), spins_.end(), Spin{1});
        refresh();
    }

    // Load a whole configuration
    void setSpins(std::span<const Spin> s) {
        if (s.size() != N_) {
            throw std::invalid_argument("Ising2D::setSpins: wrong configuration size");
        }
        std::copy(s.begin(), s.end(), spins_.begin());
        refresh();
    }

    void randomise(std::mt19937& gen) {
        std::bernoulli_distribution coin(0.5);
        for (Spin& s : spins_) {
            s = coin(gen) ? Spin{1} : Spin{-1};
        }
        refresh();
    }

    d64 computeEnergy() const {
        d64 bonds = 0.0;
        std::int64_t m = 0;
        for (u32 i = 0; i < N_; ++i) {
            const u32* n = neighbors(i);
            // right and down only to counts bonds once
            bonds += static_cast<d64>(spins_[i]) 
                * (static_cast<d64>(spins_[n[0]]) + static_cast<d64>(spins_[n[2]]));
            m += spins_[i];
        }
        return - J_ * bonds - field_ * static_cast<d64>(m);
    }

    std::int64_t computeMagnetization() const {
        std::int64_t m = 0;
        for (const Spin& s : spins_) {
            m += s;
        }
        return m;
    }

    void refresh() {
        magnetization_ = computeMagnetization();
        energy_ = computeEnergy();
    }


private:
    u32 L_, N_;
    d64 J_, field_;
    std::vector<Spin> spins_;
    std::vector<u32> nbr_;
    d64 energy_ = 0.0;
    std::int64_t magnetization_ = 0;
};

// Default observable set, only calcualte energy per spin - e, 
// magnetization per spin - m, and |m|, |m| is computd because it
// is the finite order parameter. Also compute m^2 and m^4
struct IsingObservables {
    std::vector<std::string> names() const {
        return {"e", "m", "abs_m", "m2", "m4"};
    }

    void eval(const Ising2D& s, std::span<d64> out) const {
        const d64 e = s.energyPerSite();
        const d64 m = s.magnetizationPerSite();
        out[0] = e;
        out[1] = m;
        out[2] = std::abs(m);
        out[3] = m * m;
        out[4] = m * m * m * m;
    }
};

} // namespace mcmc