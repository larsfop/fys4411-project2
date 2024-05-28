
#include "simplegaussianSlater.h"

#include <iostream>
#include <iomanip>
using namespace std;

SimpleGaussianSlater::SimpleGaussianSlater(
    const double alpha,
    double beta,
    int N
)
{
    m_alpha = alpha;
    m_beta = beta;
    m_beta_z = {1.0, 1.0, beta};
    m_parameters = {alpha, beta};

    m_D_up.zeros(N/2, N/2);
    m_D_down.zeros(N/2, N/2);
}

double SimpleGaussianSlater::Wavefunction(std::vector<std::unique_ptr<class Particle>> &particles)
{
    int numberofparticles = particles.size();
    int numberofdimensions = particles[0]->getNumberofDimensions();
    double r2 = 0;
    for (int i = 0; i < numberofparticles; i++)
    {
        arma::vec pos = particles[i]->getPosition();
        for (int j = 0; j < numberofdimensions; j++)
        {
            r2 += pos(j)*pos(j)*m_beta_z(j);
        }
    }
    return std::exp(-m_alpha*r2);
}

double SimpleGaussianSlater::EvalWavefunction(std::vector<std::unique_ptr<class Particle>> &particles, int i, int j)
{
    arma::vec pos = particles[i]->getPosition();
    double r2 = arma::sum(arma::square(pos));

    return Hermite_poly(j, pos)*exp(-m_alpha*m_omega*r2/2);
}

void SimpleGaussianSlater::FillSlaterDeterminants(std::vector<std::unique_ptr<class Particle>> &particles)
{
    int N = particles.size();
    for (int i = 0; i < N/2; i++)
    {
        for (int j = 0; j < N/2; j++)
        {
            // Fill each Slater determinant with its wavefunction value
            m_D_up(i,j) = EvalWavefunction(particles, i, j);
            m_D_down(i,j) = EvalWavefunction(particles, i+N/2, j);
        }
    }
    // Compute the Slater determinant
    D_sum = arma::det(m_D_up)*arma::det(m_D_down);
}

arma::vec SimpleGaussianSlater::SingleDerivative(
    std::vector<std::unique_ptr<class Particle>> &particles,
    int index,
    int n
)
{
    int numberofparticles = particles.size();
    int numberofdimensions = particles[0]->getNumberofDimensions();

    arma::vec pos = particles[index]->getPosition();

    arma::vec dphi(2);
    switch (n)
    {
        case 0:
            return -m_alpha*m_omega*pos;
        case 1:
            for (int i = 0; i < numberofdimensions; i++)
            {
                dphi[i] = m_omega*(1/pos[i] - m_alpha*m_omega*pos[i]);
            }
            return dphi;
        case 2:
            for (int i = 0; i < numberofdimensions; i++)
            {
                dphi[i] = 2*m_omega*pos[i]/(m_omega*pos[i]*pos[i] - 1) - m_alpha*m_omega*pos[i];
            }
            return dphi;
    }
}

double SimpleGaussianSlater::DoubleDerivative(
    std::vector<std::unique_ptr<class Particle>> &particles,
    int index,
    int n
)
{
    int numberofparticles = particles.size();
    int numberofdimensions = particles[0]->getNumberofDimensions();

    arma::vec pos = particles[index]->getPosition();
    double r2 = arma::sum(arma::square(pos));

    double alpha2 = m_alpha*m_alpha;
    double omega2 = m_omega*m_omega;
    switch(n)
    {
        case 0:
            return alpha2*omega2*r2 - 2*m_alpha*m_omega;
        case 1:
            return -2*m_alpha*m_omega - m_alpha*m_omega*pos[0]*(sqrt(m_omega) 
                    - m_alpha*m_omega*pos[0]) - alpha2*omega2*pos[1]*pos[1];
        case 2:
            return -2*m_alpha*m_omega - m_alpha*m_omega*pos[1]*(sqrt(m_omega) 
                    - m_alpha*m_omega*pos[1]) - alpha2*omega2*pos[0]*pos[0];
    }
}

double SimpleGaussianSlater::LocalEnergy(std::vector<std::unique_ptr<class Particle>> &particles)
{
    int numberofparticles = particles.size();
    int numberofdimensions = particles[0]->getNumberofDimensions();

    double kinetic = 0;
    double potential = 0;
    for (int i = 0; i < numberofparticles; i++)
    {
        int j = (i < numberofparticles/2) ? 0 : 3;
        for (int j = 0; j < 3; j++)
        {
            kinetic += DoubleDerivative(particles, i, j);
        }

        arma::vec pos = particles[i]->getPosition();
        for (int j; j < numberofdimensions; j++)
        {
            potential += pos(j)*pos(j);
        }
    }

    // E_L for two fermions
    // potential: = 0.5*omega^2*r^2
    return 0.5*(-kinetic + m_omega*m_omega*potential);
}

arma::vec SimpleGaussianSlater::QuantumForce(
    std::vector<std::unique_ptr<class Particle>> &particles,
    const int index
)
{
    int numberofdimensions = particles[0]->getNumberofDimensions();
    arma::vec pos = particles[index]->getPosition();
    arma::vec qforce(numberofdimensions);
    for (int i = 0; i < numberofdimensions; i++)
    {
        qforce(i) = pos(i);
    }
    return -4*m_alpha*qforce;
}

arma::vec SimpleGaussianSlater::QuantumForce(
    std::vector<std::unique_ptr<class Particle>> &particles,
    const int index,
    const arma::vec Step
)
{   
    int numberofdimensions = particles[0]->getNumberofDimensions();
    arma::vec pos = particles[index]->getPosition();
    arma::vec qforce(numberofdimensions);
    for (int i = 0; i < numberofdimensions; i++)
    {
        qforce(i) = pos(i) + Step(i);
    }
    return -4*m_alpha*qforce; 
}

double SimpleGaussianSlater::w(std::vector<std::unique_ptr<class Particle>> &particles,
    const int index, 
    const arma::vec step
)
{
    int numberofdimensions = particles[0]->getNumberofDimensions();
    arma::vec pos = particles[index]->getPosition();
    double dr2 = 0;
    for (int i = 0; i < numberofdimensions; i++)
    {
        dr2 += (2*pos(i) + step(i))*step(i);
    }

    arma::vec pos1 = particles[0]->getPosition();
    arma::vec pos2 = particles[1]->getPosition();

    double r12_old = arma::norm(pos1 - pos2);
    double r12_new = arma::norm(pos + step - particles[(index+1)%2]->getPosition());

    return std::exp(-2*m_alpha*dr2)*std::exp(2*(r12_new/(1 + m_beta*r12_new) - r12_old/(1 + m_beta*r12_old))); //*omega
}

// Take the derivative of the the wavefunction as a function of the parameters alpha, beta
arma::vec SimpleGaussianSlater::dPsidParam(std::vector<std::unique_ptr<class Particle>> &particles)
{
    int numberofdimensions = particles[0]->getNumberofDimensions();
    int numberofparticles = particles.size();

    arma::vec derivative(2);
    arma::vec r2(3);
    for (int i = 0; i < numberofparticles; i++)
    {
        arma::vec pos = particles[i]->getPosition();
        for (int j = 0; j < numberofdimensions; j++)
        {
            r2(j) += pos(j)*pos(j);
        }
    }
    derivative(0) = -(r2(0) + r2(1) + m_beta*r2(2));
    derivative(1) = -m_alpha*r2(2);
    return derivative; // ex. Psi[alpha]/Psi -> Psi[alpha] = dPsi/dalpha
}

void SimpleGaussianSlater::ChangeParameters(const double alpha, const double beta)
{
    m_alpha = alpha,
    m_beta = beta;
    m_parameters = {alpha, beta};
}

double SimpleGaussianSlater::Hermite_poly(int n, arma::vec pos)
{
    switch(n)
    {
        case 0:
            return 1;
        case 1:
            return sqrt(m_omega)*pos[0];
        case 2:
            return sqrt(m_omega)*pos[1];
        case 3:
            return m_omega*pos[0]*pos[1];
        case 4:
            return m_omega*pos[0]*pos[0] - 1;
        case 5:
            return m_omega*pos[1]*pos[1] - 1;
    }
}




SimpleGaussianNumerical::SimpleGaussianNumerical(double alpha, double beta, double dx, int N) : SimpleGaussianSlater(alpha, beta, N)
{
    m_alpha = alpha;
    m_dx = dx;
    m_beta_z = {1.0, 1.0, beta};
}

double SimpleGaussianNumerical::EvaluateSingleParticle(class Particle particle)
{
    int numnerofdimensions = particle.getNumberofDimensions();
    arma::vec pos = particle.getPosition();
    double r2 = 0;
    for (int i = 0; i < numnerofdimensions; i++)
    {
        r2 += pos(i)*pos(i)*m_beta_z(i);
    }
    return std::exp(-m_alpha*r2);
}

double SimpleGaussianNumerical::EvaluateSingleParticle(class Particle particle, double step, double step_index)
{
    int numnerofdimensions = particle.getNumberofDimensions();
    arma::vec pos = particle.getPosition();
    pos(step_index) += step;
    double r2 = 0;
    for (int i = 0; i < numnerofdimensions; i++)
    {
        r2 += pos(i)*pos(i)*m_beta_z(i);
    }
    return std::exp(-m_alpha*r2);
}

double SimpleGaussianNumerical::DoubleDerivative(std::vector<std::unique_ptr<class Particle>> &particles)
{
    int numberofparticles = particles.size();
    int numberofdimensions = particles[0]->getNumberofDimensions();
    double dersum = 0;
    double g, gdx_p, gdx_m;
    arma::vec pos, der;

    for (int i = 0; i < numberofparticles; i++)
    {
        Particle &particle = *particles[i];
        g = EvaluateSingleParticle(particle);
        for (int j = 0; j < numberofdimensions; j++)
        {
            gdx_p = EvaluateSingleParticle(particle, m_dx, j);
            gdx_m = EvaluateSingleParticle(particle, -2*m_dx, j);
            dersum += (gdx_p - 2*g + gdx_m)/(m_dx*m_dx);
        }
        dersum /= g;
    }

    return dersum;
}