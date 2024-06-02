#pragma once

#include <memory>
#include "montecarlo.h"
#include "WaveFunctions.h"

class MetropolisHastings : public MonteCarlo
{
public:
    MetropolisHastings(std::unique_ptr<class Random> rng, bool slater);
    bool Step(
        double stepsize,
        class WaveFunction &wavefunction,
        std::vector<std::unique_ptr<class Particle>> &particles
    );
};