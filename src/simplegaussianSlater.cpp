
#include "simplegaussianSlater.h"

#include <iostream>
#include <iomanip>
using namespace std;

SimpleGaussianSlater::SimpleGaussianSlater(
    const double alpha,
    double beta,
    double omega,
    int N
)
{
    m_alpha = alpha;
    m_beta = beta;
    m_omega = omega;
    m_beta_z = {1.0, 1.0, beta};
    m_parameters = {alpha, beta};

    m_D_up.zeros(N/2, N/2);
    m_D_down.zeros(N/2, N/2);
    m_DI_up.zeros(N/2, N/2);
    m_DI_down.zeros(N/2, N/2);
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

double SimpleGaussianSlater::EvalWavefunction(
    std::vector<std::unique_ptr<class Particle>> &particles, 
    int i, 
    int j)
{
    arma::vec pos = particles[i]->getPosition();
    double r2 = arma::sum(arma::square(pos));

    return Hermite_poly(j, pos)*exp(-m_alpha*m_omega*r2/2.);
}

double SimpleGaussianSlater::EvalWavefunction(
    std::vector<std::unique_ptr<class Particle>> &particles,
    int i, 
    int j,
    arma::vec step)
{
    arma::vec pos = particles[i]->getPosition() + step;
    double r2 = arma::sum(arma::square(pos));

    return Hermite_poly(j, pos)*exp(-m_alpha*m_omega*r2/2.);
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
    m_D_sum = arma::det(m_D_up)*arma::det(m_D_down);

    m_DI_up = arma::inv(m_D_down);
    m_DI_down = arma::inv(m_D_down);
}

arma::vec SimpleGaussianSlater::SingleDerivative(
    std::vector<std::unique_ptr<class Particle>> &particles,
    int i,
    int j
)
{
    int numberofparticles = particles.size();
    int numberofdimensions = particles[0]->getNumberofDimensions();

    arma::vec pos = particles[i]->getPosition();
    double r2 = arma::sum(arma::square(pos));
    double e = exp(-m_alpha*m_omega*r2);

    arma::vec dphi(numberofdimensions);
    switch (j)
    {
        case 0:
            return -m_alpha*m_omega*pos*e;
        case 1:
            dphi(0) = sqrt(m_omega) - m_alpha*m_omega*pos(0);
            dphi(1) = -m_alpha*m_omega*pos(1);
            return dphi*e;
        case 2:
            dphi(0) = -m_alpha*m_omega*pos(0);
            dphi(1) = sqrt(m_omega) - m_alpha*m_omega*pos(1);
            return dphi*e;
    }
}

arma::vec SimpleGaussianSlater::SingleDerivative(
    std::vector<std::unique_ptr<class Particle>> &particles,
    int i,
    int j,
    arma::vec step
)
{
    int numberofparticles = particles.size();
    int numberofdimensions = particles[0]->getNumberofDimensions();

    arma::vec pos = particles[i]->getPosition() + step;
    double r2 = arma::sum(arma::square(pos));
    double e = exp(-m_alpha*m_omega*r2);

    arma::vec dphi(numberofdimensions);
    switch (j)
    {
        case 0:
            return -m_alpha*m_omega*pos*e;
        case 1:
            dphi(0) = sqrt(m_omega) - m_alpha*m_omega*pos(0);
            dphi(1) = -m_alpha*m_omega*pos(1);
            return dphi*e;
        case 2:
            dphi(0) = -m_alpha*m_omega*pos(0);
            dphi(1) = sqrt(m_omega) - m_alpha*m_omega*pos(1);
            return dphi*e;
    }
}

double SimpleGaussianSlater::DoubleDerivative(
    std::vector<std::unique_ptr<class Particle>> &particles,
    int i,
    int j
)
{
    int numberofparticles = particles.size();
    int numberofdimensions = particles[0]->getNumberofDimensions();

    arma::vec pos = particles[i]->getPosition();
    double r2 = arma::sum(arma::square(pos));
    double e = exp(-m_alpha*m_omega*r2/2);

    double alpha2 = m_alpha*m_alpha;
    double omega2 = m_omega*m_omega;
    switch(j)
    {
        case 0:
            return (alpha2*omega2*r2 - 2*m_alpha*m_omega)*e;
        case 1:
            return (-2*m_alpha*m_omega - m_alpha*m_omega*pos(0)*(sqrt(m_omega) 
                    - m_alpha*m_omega*pos(0)) - alpha2*omega2*pos(1)*pos(1))*e;
        case 2:
            return (-2*m_alpha*m_omega - m_alpha*m_omega*pos[1]*(sqrt(m_omega) 
                    - m_alpha*m_omega*pos(1)) - alpha2*omega2*pos(0)*pos(0))*e;
    }
}

double SimpleGaussianSlater::LocalEnergy(std::vector<std::unique_ptr<class Particle>> &particles)
{
    int numberofparticles = particles.size();
    int numberofdimensions = particles[0]->getNumberofDimensions();
    int N2 = numberofparticles/2;

    double kinetic = 0;
    double potential = 0;
    for (int i = 0; i < numberofparticles; i++)
    {
        if (i < numberofparticles/2) // Spin up
        {
            for (int j = 0; j < N2; j++)
            {
                kinetic += DoubleDerivative(particles, i, j)*m_DI_up(j, i);
            }       
        }
        else // Spin down
        {
            for (int j = 0; j < N2; j++)
            {
                kinetic += DoubleDerivative(particles, i, j)*m_DI_down(j, i-N2);
            }
        }

        arma::vec pos = particles[i]->getPosition();
        for (int j = 0; j < numberofdimensions; j++)
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
    int numberofparticles = particles.size();
    int N2 = numberofparticles/2;
    arma::vec qforce(numberofdimensions);
    if (index < N2)
    {
        for (int j = 0; j < N2; j++)
        {
            qforce = SingleDerivative(particles, index, j)*m_DI_up(j, index);
        }
    }
    else
    {
        for (int j = 0; j < N2; j++)
        {
            qforce = SingleDerivative(particles, index, j)*m_DI_down(j, index - N2);
        }
    }
    return 2*qforce;
}

arma::vec SimpleGaussianSlater::QuantumForce(
    std::vector<std::unique_ptr<class Particle>> &particles,
    const int index,
    const arma::vec Step
)
{   
    int numberofdimensions = particles[0]->getNumberofDimensions();
    int numberofparticles = particles.size();
    int N2 = numberofparticles/2;
    arma::vec qforce(numberofdimensions);
    if (index < N2)
    {
        for (int j = 0; j < N2; j++)
        {
            qforce = SingleDerivative(particles, index, j, Step)*m_DI_up(j, index);
        }
    }
    else
    {
        for (int j = 0; j < N2; j++)
        {
            qforce = SingleDerivative(particles, index, j, Step)*m_DI_down(j, index - N2);
        }
    }
    return 2*qforce;
}

double SimpleGaussianSlater::w(std::vector<std::unique_ptr<class Particle>> &particles,
    const int index,
    const arma::vec step
)
{
    int numberofdimensions = particles[0]->getNumberofDimensions();
    int numberofparticles = particles.size();
    int N2 = numberofparticles/2;

    double R = 0;
    //for (int i = 0; i < numberofparticles; i++)
    //{
        if (index < N2)
        {
            for (int j = 0; j < N2; j++)
            {
                R += EvalWavefunction(particles, index, j, step)*m_DI_up(j, index);
            }
        }
        else
        {
            for (int j = 0; j < N2; j++)
            {
                R += EvalWavefunction(particles, index, j, step)*m_DI_down(j, index - N2);
            }
        }
    //}
    
    return R;
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
            return sqrt(m_omega)*pos(0);
        case 2:
            return sqrt(m_omega)*pos(1);
        case 3:
            return m_omega*pos(0)*pos(1);
        case 4:
            return m_omega*pos(0)*pos(0) - 1;
        case 5:
            return m_omega*pos(1)*pos(1) - 1;
    }
}

void SimpleGaussianSlater::UpdateInverseSlater(
    std::vector<std::unique_ptr<class Particle>> &particles,
    int k,
    double R,
    arma::vec step
)
{
    int numberofparticles = particles.size();
    int N2 = numberofparticles/2;

    if (k < N2)
    {
        for (int i = 0; i < N2; i++)
        {
            for (int j = 0; j < N2; j++)
            {
                if (j == i)
                {
                    m_DI_up(k,j) = m_DI_up(k,i)/R;
                    for (int l = 0; l < N2; l++)
                    {
                        m_DI_up(k,j) *= m_D_up(i,l)*m_DI_up(l,j);
                    }
                }
                else
                {
                    double tmp = 0;
                    for (int l = 0; l < N2; l++)
                    {
                        tmp += EvalWavefunction(particles, i, l, step)*m_DI_up(l,j);
                    }
                    m_DI_up(k,j) -= m_DI_up(k,i)/R*tmp;
                }
            }
        }

    }
    else
    {
        for (int i = 0; i < N2; i++)
        {
            for (int j = 0; j < N2; j++)
            {
                if (j == i)
                {
                    m_DI_down(k-N2,j) = m_DI_down(k-N2,i)/R;
                    for (int l = 0; l < N2; l++)
                    {
                        m_DI_down(k-N2,j) *= m_D_down(i,l)*m_DI_down(l,j);
                    }
                }
                else
                {
                    double tmp = 0;
                    for (int l = 0; l < N2; l++)
                    {
                        tmp += EvalWavefunction(particles, i, l, step)*m_DI_down(l,j);
                    }
                    m_DI_down(k-N2,j) -= m_DI_down(k-N2,i)/R*tmp;
                }
            }
        }
    }
}




SimpleGaussianSlaterNumerical::SimpleGaussianSlaterNumerical(double alpha, double beta, double omega, double dx, int N) : SimpleGaussianSlater(alpha, beta, omega, N)
{
    m_alpha = alpha;
    m_dx = dx;
    m_beta_z = {1.0, 1.0, beta};
}

double SimpleGaussianSlaterNumerical::EvaluateSingleParticle(class Particle particle)
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

double SimpleGaussianSlaterNumerical::EvaluateSingleParticle(class Particle particle, double step, double step_index)
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

double SimpleGaussianSlaterNumerical::DoubleDerivative(std::vector<std::unique_ptr<class Particle>> &particles)
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