
#include "simplegaussian.h"

#include <iostream>
#include <iomanip>
using namespace std;

SimpleGaussian::SimpleGaussian(
    const double alpha,
    double beta,
    double omega
)
{
    m_alpha = alpha;
    m_beta = beta;
    m_omega = omega;
    m_beta_z = {1.0, 1.0, beta};
    m_parameters = {alpha, beta};
}

double SimpleGaussian::Wavefunction(std::vector<std::unique_ptr<class Particle>> &particles)
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

double SimpleGaussian::DoubleDerivative(
    std::vector<std::unique_ptr<class Particle>> &particles
)
{
    int numberofdimensions = particles[0]->getNumberofDimensions();
    int numberofparticles = particles.size();

    // constant term
    double constant = numberofdimensions*numberofparticles*m_alpha*m_omega;

    // for two fermions
    arma::vec pos1 = particles[0]->getPosition();
    arma::vec pos2 = particles[1]->getPosition();

    // double r1_2 = arma::norm(pos1);
    // double r2_2 = arma::norm(pos2);
    double r1_2 = 0;
    double r2_2 = 0;
    for (int i = 0; i < numberofdimensions; i++)
    {
        r1_2 += pos1(i)*pos1(i);
        r2_2 += pos2(i)*pos2(i);
    }

    double tmp = 0;
    for (int i = 0; i < numberofdimensions; i++)
    {
        tmp += pos1(i)*pos2(i);
    }

    double d2psi = m_omega*m_omega*m_alpha*m_alpha*(r1_2 + r2_2); 

    return d2psi - constant; // return d2psi/psi
}

double SimpleGaussian::LocalEnergy(std::vector<std::unique_ptr<class Particle>> &particles)
{
    int numberofparticles = particles.size();
    int numberofdimensions = particles[0]->getNumberofDimensions();
    double kinetic = DoubleDerivative(particles);

    double potential = 0;
    for (int i = 0; i < numberofparticles; i++)
    {
        arma::vec pos = particles[i]->getPosition();
        for (int j = 0; j < numberofdimensions; j++)
        {
            potential += pos(j)*pos(j);
        }
    }

    // E_L for two fermions
    // potential: = 0.5*omega^2*r^2
    // kinetic: 2D: -4*alpha*omega + alpha^2*omega^2(r1^2 + r2^2 + 2*x1*x2 + 2*y1*y2)
    //          3D: -6*alpha*omega + alpha^2*omega^2(r1^2 + r2^2 + 2*x1*x2 + 2*y1*y2 + 2*z1*z2)
    return 0.5*(-kinetic + potential); // omega = 1
}

arma::vec SimpleGaussian::QuantumForce(
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

arma::vec SimpleGaussian::QuantumForce(
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

double SimpleGaussian::w(std::vector<std::unique_ptr<class Particle>> &particles,
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
arma::vec SimpleGaussian::dPsidParam(std::vector<std::unique_ptr<class Particle>> &particles)
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

void SimpleGaussian::ChangeParameters(const double alpha, const double beta)
{
    m_alpha = alpha,
    m_beta = beta;
    m_parameters = {alpha, beta};
}

double SimpleGaussian::Hermite_poly(int n, arma::vec pos)
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

void SimpleGaussian::FillSlaterDeterminants(std::vector<std::unique_ptr<class Particle>> &particles)
{
    std::printf("Used FillSlaterDeterminants, do not, wrong class");
}

void SimpleGaussian::UpdateInverseSlater(
    std::vector<std::unique_ptr<class Particle>> &particles,
    int k,
    double R,
    arma::vec step
)
{
    std::printf("Used FillSlaterDeterminants, do not, wrong class");
}




SimpleGaussianNumerical::SimpleGaussianNumerical(double alpha, double beta, double omega, double dx) : SimpleGaussian(alpha, omega, beta)
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