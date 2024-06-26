#pragma once

#include <memory>
#include <vector>
#include "random.h"

class MonteCarlo
{
public:
    MonteCarlo(std::unique_ptr<class Random> rng, bool slater);
    virtual ~MonteCarlo() = default;

    virtual bool Step(
        double stepsize,
        class WaveFunction &wavefunction,
        std::vector<std::unique_ptr<class Particle>> &particles
    ) = 0;

protected:
    std::unique_ptr<class Random> m_rng;
    bool m_slater;
};