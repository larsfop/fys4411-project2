
#include "metropolishastings.h"
#include "Particle.h"

#include <memory>
#include <iostream>
#include <iomanip>
using namespace std;

MetropolisHastings::MetropolisHastings(std::unique_ptr<class Random> rng, bool slater) : MonteCarlo(std::move(rng), slater) {}

bool MetropolisHastings::Step(
    double stepsize,
    class WaveFunction &wavefunction,
    std::vector<std::unique_ptr<class Particle>> &particles
)
{
    double sqrt_dt = sqrt(stepsize);
    int numberofdimensions = particles[0]->getNumberofDimensions();
    int numberofparticles = particles.size();
    int index = m_rng->NextInt(numberofparticles-1);
    double D = 0.5;

    arma::vec qforce = wavefunction.QuantumForce(particles, index);
    arma::vec params = wavefunction.getParameters(); // params = {alpha, beta}
    arma::vec beta_z = {1, 1, params(1)};

    // Sample new position
    arma::vec pos = particles[index]->getPosition();
    arma::vec step(numberofdimensions);
    for (int i = 0; i < numberofdimensions; i++)
    {
        step(i) = sqrt_dt*m_rng->NextGaussian() + qforce(i)*stepsize*D;
    }

    // update inverse Slater determinants for new position
    if (m_slater)
    {
        double det = wavefunction.w(particles, index, step);
        wavefunction.UpdateInverseSlater(particles, index, det, step);
    }

    // Compute the quantum force
    arma::vec qforcenew = wavefunction.QuantumForce(particles, index, step);
    // qforce.print();
    // qforcenew.print();
    double greens = 0;
    for (int i = 0; i < numberofdimensions; i++)
    {
        greens += 0.5*(qforce(i) + qforcenew(i))*(D*stepsize*0.5*\
        (qforce(i) - qforcenew(i)) - (pos(i) + step(i)) + pos(i));
    }

    if (m_slater)
    {
        wavefunction.KeepOldInverseSlater();
    }

    // Slater determinant ratio
    double R = wavefunction.w(particles, index, step);

    // Compute Jastrow ratio
    bool Jastrow = wavefunction.Jastrow();
    double J = Jastrow ? 0 : 1;
    if (Jastrow) 
    {
        for (int j = 0; j < index; j++)
        {
            arma::vec pos_j = particles[j]->getPosition();
            double a = wavefunction.spinParallelFactor(index, j, numberofparticles/2);
            double r_old = arma::norm(pos - pos_j);
            double r_new = arma::norm(pos + step - pos_j);
            J += a*r_new/(1 + params(1)*r_old);
        }
        for (int j = index+1; j < numberofparticles; j++)
        {
            arma::vec pos_j = particles[j]->getPosition();
            double a = wavefunction.spinParallelFactor(index, j, numberofparticles/2);
            double r_old = arma::norm(pos - pos_j);
            double r_new = arma::norm(pos + step - pos_j);
            J += a*r_new/(1 + params(1)*r_old);
        }
        J = exp(J);
    }

    double random = m_rng->NextDouble();
    if(random <= R*R*J*J*exp(greens))
    {
        if (m_slater)
        {
            wavefunction.UpdateInverseSlater(particles, index, R, step);
            wavefunction.ChangeOldInverseSlater();
        }

        particles.at(index)->ChangePosition(step);
        return true;
    }
    else
    {
        if (m_slater)
        {
            wavefunction.KeepOldInverseSlater();
        }

        return false;
    }
}