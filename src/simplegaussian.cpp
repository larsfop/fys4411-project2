
#include "simplegaussian.h"

#include <iostream>
#include <iomanip>
using namespace std;

SimpleGaussian::SimpleGaussian(
    const double alpha,
    double beta,
    double omega,
    bool Jastrow,
    bool Interaction
)
{
    m_alpha = alpha;
    m_beta = beta;
    m_omega = omega;
    m_parameters = {alpha, beta};

    m_Jastrow = Jastrow;
    m_Interaction = Interaction;
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

    if (m_Jastrow)
    {
        double r = 0;
        double r12 = arma::norm(pos1 - pos2);
        for (int i = 0; i < 2; i++)
        {
            r += (pos1(i) + pos2(i))*(pos1(i) - pos2(i));
        }
        d2psi -= 2*m_alpha*m_omega*r/r12/std::pow(1 + m_beta*r12, 2);

        d2psi += 1/std::pow(1 + m_beta*r12, 4) + 2/std::pow(1 + m_beta*r12, 2) - 2*m_beta/std::pow(1 + m_beta*r12, 3);
    }



    return d2psi - constant; // return d2psi/psi
}

double SimpleGaussian::LocalEnergy(std::vector<std::unique_ptr<class Particle>> &particles)
{
    int numberofparticles = particles.size();
    int numberofdimensions = particles[0]->getNumberofDimensions();
    double kinetic =  DoubleDerivative(particles);

    double potential = 0;
    for (int i = 0; i < numberofparticles; i++)
    {
        arma::vec pos = particles[i]->getPosition();
        //cout << exp(m_alpha*m_omega*(pos(0)*pos(0) + pos(1)*pos(1)/2)) << endl;;
        for (int j = 0; j < numberofdimensions; j++)
        {
            potential += pos(j)*pos(j);
        }
    }

    arma::vec pos1 = particles[0]->getPosition();
    arma::vec pos2 = particles[1]->getPosition();

    double r12 = arma::norm(pos1 - pos2);

    // E_L for two fermions
    // potential: = 0.5*omega^2*r^2
    // kinetic: 2D: -4*alpha*omega + alpha^2*omega^2(r1^2 + r2^2 + 2*x1*x2 + 2*y1*y2)
    //          3D: -6*alpha*omega + alpha^2*omega^2(r1^2 + r2^2 + 2*x1*x2 + 2*y1*y2 + 2*z1*z2)

    m_kinetic = kinetic;
    m_potential = !m_Interaction ? 0.5*m_omega*m_omega*potential : 0.5*m_omega*m_omega*potential + 1.0/r12;

    return -0.5*m_kinetic + m_potential;
}

arma::vec SimpleGaussian::QuantumForce(
    std::vector<std::unique_ptr<class Particle>> &particles,
    const int index
)
{
    int numberofparticles = particles.size();
    int numberofdimensions = particles[0]->getNumberofDimensions();
    arma::vec qforce(numberofdimensions);
    arma::vec pos = particles[index]->getPosition();
    qforce = pos;
    // arma::vec pos2 = particles[1]->getPosition();
    // for (int j = 0; j < numberofdimensions; j++)
    // {
    //     qforce(j) += pos1(j) + pos2(j);
    // }
    qforce *= -m_alpha*m_omega;

    arma::vec Jastrow(numberofdimensions);
    if (m_Jastrow)
    {
        for (int i = 0; i < numberofparticles; i++)
        {
            if (i != index)
            {
                arma::vec pos_i = particles[i]->getPosition();
                double r12 = arma::norm(pos - pos_i);
                Jastrow += (pos - pos_i)/r12*1.0/3/std::pow(1 + m_beta*r12, 2);
            }
        }
        qforce += Jastrow;
    }

    return 2*qforce;
}

arma::vec SimpleGaussian::QuantumForce(
    std::vector<std::unique_ptr<class Particle>> &particles,
    const int index,
    const arma::vec Step
)
{   
    int numberofparticles = particles.size();
    int numberofdimensions = particles[0]->getNumberofDimensions();
    arma::vec qforce(numberofdimensions);
    arma::vec pos = particles[index]->getPosition() + Step;
    qforce = pos;
    // arma::vec pos2 = particles[1]->getPosition();
    // for (int j = 0; j < numberofdimensions; j++)
    // {
    //     qforce(j) += pos1(j) + pos2(j);
    // }
    qforce *= -m_alpha*m_omega;

    arma::vec Jastrow(numberofdimensions);
    if (m_Jastrow)
    {
        for (int i = 0; i < numberofparticles; i++)
        {
            if (i != index)
            {
                arma::vec pos_i = particles[i]->getPosition();
                double r12 = arma::norm(pos - pos_i);
                Jastrow += (pos - pos_i)/r12*1.0/3/std::pow(1 + m_beta*r12, 2);
            }
        }
        qforce += Jastrow;
    }

    return 2*qforce; 
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

    // cout << exp(-m_alpha*dr2) << endl;;
    return std::exp(-m_alpha*m_omega*dr2/2);//*std::exp(2*(r12_new/(1 + m_beta*r12_new) - r12_old/(1 + m_beta*r12_old))); //*omega
}

// Take the derivative of the the wavefunction as a function of the parameters alpha, beta
arma::vec SimpleGaussian::dPsidParam(std::vector<std::unique_ptr<class Particle>> &particles)
{
    int numberofdimensions = particles[0]->getNumberofDimensions();
    int numberofparticles = particles.size();

    arma::vec pos1 = particles[0]->getPosition();
    arma::vec pos2 = particles[1]->getPosition();
    arma::vec derivative(2);
    double r2 = 0;
    for (int i = 0; i < numberofparticles; i++)
    {
        arma::vec pos = particles[i]->getPosition();
        for (int j = 0; j < numberofdimensions; j++)
        {
            r2 += pos(j)*pos(j);
        }
    }
    double r12 = arma::norm(pos1 - pos2);

    derivative(0) = -m_omega*(r2)/2;
    derivative(1) = -r12*r12/std::pow(1 + m_beta*r12, 2);

    // pos1.print();
    // pos2.print();
    // cout << r2 << " ; " << r12 << endl;
    // derivative.print();
    // cout << m_alpha << " ; " << m_beta << endl;
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
        default:
            cout << "Index out of bounds, func: " << __PRETTY_FUNCTION__ << endl;
            exit(0);
    }
}

void SimpleGaussian::FillSlaterDeterminants(std::vector<std::unique_ptr<class Particle>> &particles)
{
    std::printf("Used FillSlaterDeterminants, do not, wrong class");
    exit(0);
}

void SimpleGaussian::UpdateInverseSlater(
    std::vector<std::unique_ptr<class Particle>> &particles,
    int k,
    double R,
    arma::vec step
)
{
    std::printf("Used FillSlaterDeterminants, do not, wrong class");
    exit(0);
}

void SimpleGaussian::CheckSlater(std::vector<std::unique_ptr<class Particle>> &particles)
{
    exit(0);
}

double SimpleGaussian::spinParallelFactor(int i, int j, int N2)
{
    double a;

    if (i < N2 && j < N2 || i >= N2 && j >= N2) 
    {
        a = 1;
    }        
    else
    {
        a = 1.0/3;
    }

    return a;
}

double SimpleGaussian::getKinetic()
{
    return m_kinetic;
}

double SimpleGaussian::getPotential()
{
    return m_potential;
}



SimpleGaussianNumerical::SimpleGaussianNumerical(double alpha, double beta, double omega, double dx, bool Jastrow, bool Interaction) : SimpleGaussian(alpha, omega, beta, Jastrow, Interaction)
{
    m_alpha = alpha;
    m_dx = dx;
    m_omega = omega;
    m_parameters = {alpha, beta};
}

double SimpleGaussianNumerical::EvaluateSingleParticle(class Particle particle)
{
    int numberofdimensions = particle.getNumberofDimensions();
    arma::vec pos = particle.getPosition();
    double r2 = 0;
    for (int i = 0; i < numberofdimensions; i++)
    {
        r2 += pos(i)*pos(i);
    }
    return std::exp(-m_alpha*m_omega*r2/2);
}

double SimpleGaussianNumerical::EvaluateSingleParticle(class Particle particle, double step, double step_index)
{
    int numberofdimensions = particle.getNumberofDimensions();
    arma::vec pos = particle.getPosition();
    pos(step_index) += step;
    double r2 = 0;
    for (int i = 0; i < numberofdimensions; i++)
    {
        r2 += pos(i)*pos(i);
    }
    return std::exp(-m_alpha*m_omega*r2/2);
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