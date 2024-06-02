#include <memory>
#include "montecarlo.h"

MonteCarlo::MonteCarlo(std::unique_ptr<class Random> rng, bool slater)
{
    m_rng = std::move(rng);
    m_slater = slater;
}