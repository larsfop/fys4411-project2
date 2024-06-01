
#include "simplegaussianSlater.h"

#include <iostream>
#include <iomanip>
#include <cmath>
using namespace std;

SimpleGaussianSlater::SimpleGaussianSlater(
    const double alpha,
    double beta,
    double omega,
    int N,
    bool Jastrow
)
{
    m_alpha = alpha;
    m_beta = beta;
    m_omega = omega;
    m_parameters = {alpha, beta};
    m_Jastrow = Jastrow;

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
    int numberofdimensions = particles[0]->getNumberofDimensions();
    int N2 = particles.size()/2;
    arma::vec pos = particles[i]->getPosition();
    double r2 = 0;
    for (int i = 0; i < numberofdimensions; i++)
    {
        r2 += pos(i)*pos(i);
    }
    double wf = Hermite_poly(j, pos)*exp(-m_alpha*m_omega*r2/2);

    // bool spin = i < N2 ? true : false;
    // arma::mat &Slater = spin ? m_D_up : m_D_down;
    // int index = spin ? i : i - N2;
    // Slater(index,j) = wf;

    return wf;
}

double SimpleGaussianSlater::EvalWavefunction(
    std::vector<std::unique_ptr<class Particle>> &particles,
    int i, 
    int j,
    arma::vec step)
{
    int numberofdimensions = particles[0]->getNumberofDimensions();
    int N2 = particles.size()/2;
    arma::vec pos = particles[i]->getPosition() + step;
    double r2 = 0;
    for (int i = 0; i < numberofdimensions; i++)
    {
        r2 += pos(i)*pos(i);
    }
    double wf = Hermite_poly(j, pos)*exp(-m_alpha*m_omega*r2/2);

    // bool spin = i < N2 ? true : false;
    // arma::mat &Slater = spin ? m_D_up : m_D_down;
    // int index = spin ? i : i - N2;
    // Slater(index,j) = wf;

    return wf;
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

    m_DI_up = arma::inv(m_D_up);
    m_DI_down = arma::inv(m_D_down);

    // (m_D_up*m_DI_up).print();
    // (m_D_down*m_DI_down).print();
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
    double r2 = 0;
    for (int i = 0; i < numberofdimensions; i++)
    {
        r2 += pos(i)*pos(i);
    }    
    double e = exp(-m_alpha*m_omega*r2/2);

    arma::vec dphi(numberofdimensions);
    switch (j)
    {
        case 0:
            return -m_alpha*m_omega*pos*e;
        case 1:
            dphi(0) = sqrt(m_omega)*e;
            dphi -= m_alpha*sqrt(m_omega)*m_omega*pos*e;
            return dphi;
        case 2:
            dphi(1) = sqrt(m_omega)*e;
            dphi -= m_alpha*sqrt(m_omega)*m_omega*pos*e;
            return dphi;
        case 3:
            return (m_omega - m_alpha*m_omega*m_omega*pos(0)*pos(1)*pos)*e;
        case 4:
            dphi(0) = 2*m_omega*pos(0)*e;
            dphi -= m_alpha*m_omega*pos*(m_omega*pos(0)*pos(0) - 1)*e;
            return dphi;
        case 5:
            dphi(1) = 2*m_omega*pos(1)*e;
            dphi -= m_alpha*m_omega*pos*(m_omega*pos(1)*pos(1) - 1)*e;
            return dphi;
        default:
            cout << "Index out of bounds, func: " << __PRETTY_FUNCTION__ << endl;
            exit(0);
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
    double r2 = 0;
    for (int i = 0; i < numberofdimensions; i++)
    {
        r2 += pos(i)*pos(i);
    }    
    double e = exp(-m_alpha*m_omega*r2/2);

    arma::vec dphi(numberofdimensions);
    switch (j)
    {
        case 0:
            return -m_alpha*m_omega*pos*e;
        case 1:
            dphi(0) = sqrt(m_omega)*e;
            dphi -= m_alpha*sqrt(m_omega)*m_omega*(pos)*e;
            return dphi;
        case 2:
            dphi(1) = sqrt(m_omega)*e;
            dphi -= m_alpha*sqrt(m_omega)*m_omega*pos*e;
            return dphi;
        case 3:
            return (m_omega - m_alpha*m_omega*m_omega*pos(0)*pos(1)*pos)*e;
        case 4:
            dphi(0) = 2*m_omega*pos(0)*e;
            dphi -= m_alpha*m_omega*pos*(m_omega*pos(0)*pos(0) - 1)*e;
            return dphi;
        case 5:
            dphi(1) = 2*m_omega*pos(1)*e;
            dphi -= m_alpha*m_omega*pos*(m_omega*pos(1)*pos(1) - 1)*e;
            return dphi;
        default:
            cout << "Index out of bounds, func: " << __PRETTY_FUNCTION__ << endl;
            exit(0);
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
    double r2 = 0;
    for (int i = 0; i < numberofdimensions; i++)
    {
        r2 += pos(i)*pos(i);
    }
    double e = exp(-m_alpha*m_omega*r2/2);

    double alpha2 = m_alpha*m_alpha;
    double omega2 = m_omega*m_omega;
    switch(j)
    {
        case 0:
            return (alpha2*omega2*r2 - 2*m_alpha*m_omega)*e;
        case 1:
            return (-4*m_alpha*sqrt(m_omega)*m_omega*pos(0)
                    + alpha2*omega2*sqrt(m_omega)*pos(0)*r2)*e;
        case 2:
            return (-4*m_alpha*sqrt(m_omega)*m_omega*pos(1)
                    + alpha2*omega2*sqrt(m_omega)*pos(1)*r2)*e;
        case 3:
            return m_alpha*omega2*pos(0)*pos(1)*(m_alpha*m_omega*r2 - 6)*e;
        case 4:
            return (2*m_omega*(1 + m_alpha) - 6*m_alpha*omega2*pos(0)*pos(0)
                    + alpha2*omega2*(m_omega*pos(0)*pos(0)*r2 - r2))*e;
        case 5:
            return (2*m_omega*(1 + m_alpha) - 6*m_alpha*omega2*pos(1)*pos(1)
                    + alpha2*omega2*(m_omega*pos(1)*pos(1)*r2 - r2))*e;
        default:
            cout << "Index out of bounds, func: " << __PRETTY_FUNCTION__ << endl;
            exit(0);
    }
}

double SimpleGaussianSlater::LocalEnergy(std::vector<std::unique_ptr<class Particle>> &particles)
{
    int numberofparticles = particles.size();
    int numberofdimensions = particles[0]->getNumberofDimensions();
    int N2 = numberofparticles/2;

    double kinetic = 0;
    double potential = 0;
    double pot_int = 0;
    // m_DI_down.print();
    // m_DI_up.print();
    for (int i = 0; i < numberofparticles; i++)
    {
        if (i < N2) // Spin up
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

        if (m_Jastrow)
        {
            // Jastro laplacian
            kinetic += DoubleDerivativeJastrow(particles, i);

            // Interference
            arma::vec D_int(numberofdimensions);
            if (i < N2)
            {
                for (int j = 0; j < N2; j++)
                {
                    D_int += SingleDerivative(particles, i, j)*m_DI_up(j, i);
                }
            }
            else
            {
                for (int j = 0; j < N2; j++)
                {
                    D_int += SingleDerivative(particles, i, j)*m_DI_down(j, i-N2);
                }
            }

            arma::vec J_int = SingleDerivativeJastrow(particles, i);
            kinetic += 2*arma::dot(J_int, D_int);
        }    

        // Potential
        arma::vec pos = particles[i]->getPosition();
        for (int j = 0; j < numberofdimensions; j++)
        {
            potential += pos(j)*pos(j);
        }

        // Particle interaction
        for (int j = i+1; j < numberofparticles; j++)
        {
            arma::vec pos_j = particles[j]->getPosition();
            double r_ij = arma::norm(pos - pos_j);
            pot_int += 1.0/r_ij;
        }

    }

    // E_L for two fermions
    // potential: = 0.5*omega^2*r^2
    return 0.5*(-kinetic + m_omega*m_omega*potential);// + pot_int;
}

arma::vec SimpleGaussianSlater::QuantumForce(
    std::vector<std::unique_ptr<class Particle>> &particles,
    const int index
)
{
    int numberofdimensions = particles[0]->getNumberofDimensions();
    int numberofparticles = particles.size();
    int N2 = numberofparticles/2;

    bool spin = index < N2 ? true : false;
    arma::mat Slater = spin ? m_DI_up : m_DI_down;
    int i = spin ? index : index - N2;

    arma::vec qforce(numberofdimensions);
    for (int j = 0; j < N2; j++)
    {
        qforce = SingleDerivative(particles, index, j)*Slater(j, i);
    }

    if (m_Jastrow)
    {
        qforce += SingleDerivativeJastrow(particles, index);
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

    bool spin = index < N2 ? true : false;
    arma::mat Slater = spin ? m_DI_up : m_DI_down;
    int i = spin ? index : index - N2;

    arma::vec qforce(numberofdimensions);
    for (int j = 0; j < N2; j++)
    {
        qforce = SingleDerivative(particles, index, j, Step)*Slater(j, i);
    }

    if (m_Jastrow)
    {
        qforce += SingleDerivativeJastrow(particles, index);
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

    bool spin = index < N2 ? true : false;
    arma::mat Slater = spin ? m_DI_up : m_DI_down;
    int i = spin ? index : index - N2;

    double R = 0;
    //for (int i = 0; i < numberofparticles; i++)
    //{
    for (int j = 0; j < N2; j++)
    {
        R += EvalWavefunction(particles, index, j, step)*Slater(j, i);
    }
    //}

    // cout << R << endl;
    return R;
}

// Take the derivative of the the wavefunction as a function of the parameters alpha, beta
arma::vec SimpleGaussianSlater::dPsidParam(std::vector<std::unique_ptr<class Particle>> &particles)
{
    int numberofdimensions = particles[0]->getNumberofDimensions();
    int numberofparticles = particles.size();
    int N2 = numberofparticles/2;

    arma::vec derivative(2);
    double der = 0;
    // arma::vec r2(3);
    double r2 = 0;
    for (int i = 0; i < numberofparticles; i++)
    {
        arma::vec pos = particles[i]->getPosition();
        for (int j = 0; j < numberofdimensions; j++)
        {
            r2 += pos(j)*pos(j);
        }        
        
        arma::mat Slater = (i < N2) ? m_DI_up : m_DI_down;
        for (int j = 0; j < N2; j++)
        {
            der -= m_omega/2*r2*EvalWavefunction(particles, i, j)*Slater(j,i%N2);
        }
        

    }
    derivative(0) = der;
    //derivative(1) = -m_alpha*r2(2);
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
        default:
            cout << "Index out of bounds, func: " << __PRETTY_FUNCTION__ << endl;
            exit(0);
    }
}

void SimpleGaussianSlater::UpdateInverseSlater(
    std::vector<std::unique_ptr<class Particle>> &particles,
    int index,
    double R,
    arma::vec step
)
{
    int numberofparticles = particles.size();
    int N2 = numberofparticles/2;

    // Check for spin up (true) or spin down (false)
    bool spin = index < N2 ? true : false;
    arma::mat &Slater = spin ? m_DI_up : m_DI_down;
    int i = spin ? index : index - N2;

    arma::mat Slaternew = Slater;
    for (int k = 0; k < N2; k++)
    {
        for (int j = 0; j < N2; j++)
        {
            if (j != i)
            {
                double sum = 0;
                for (int l = 0; l < N2; l++)
                {
                    sum += EvalWavefunction(particles, index, l, step)*Slater(l,j);
                }
                Slaternew(k,j) = Slater(k,j) - Slater(k,i)/R*sum;
            }
            else
            {
                double sum = 0;
                for (int l = 0; l < N2; l++)
                {
                    sum += EvalWavefunction(particles, index, l)*Slater(l,j);
                }
                Slaternew(k,j) = Slater(k,i)/R*sum;
            }
        }
    }
    Slater = Slaternew;

    // (m_D_up*m_DI_up).print();
    // (m_D_down*m_DI_down).print();
}

arma::vec SimpleGaussianSlater::SingleDerivativeJastrow(
    std::vector<std::unique_ptr<class Particle>> &particles, 
    int k
    )
{
    int numberofparticles = particles.size();
    int numberofdimensions = particles[0]->getNumberofDimensions();
    int N2 = numberofparticles/2;

    arma::vec pos_k = particles[k]->getPosition();
    arma::vec Jastrow(numberofdimensions);
    for (int j = 0; j < k; j++)
    {
        arma::vec pos_j = particles[j]->getPosition();
        double r_kj = arma::norm(pos_k - pos_j);

        double a = spinParallelFactor(k, j, N2);

        Jastrow += (pos_k - pos_j)/r_kj * a/std::pow(1 + m_beta*r_kj, 2);
    }
    for (int j = k+1; j < numberofparticles; j++)
    {
        arma::vec pos_j = particles[j]->getPosition();
        double r_kj = arma::norm(pos_k - pos_j);

        double a = spinParallelFactor(k, j, N2);

        Jastrow += (pos_k - pos_j)/r_kj * a/std::pow(1 + m_beta*r_kj, 2);
    }

    return Jastrow;
}

double SimpleGaussianSlater::DoubleDerivativeJastrow(
    std::vector<std::unique_ptr<class Particle>> &particles, 
    int k
    )
{
    int numberofparticles = particles.size();
    int numberofdimensions = particles[0]->getNumberofDimensions();
    int N2 = numberofparticles/2;

    arma::vec pos_k = particles[k]->getPosition();
    double Jastrow = 0;

    for (int j = 0; j < k; j++)
    {
        arma::vec pos_j = particles[j]->getPosition();
        double r_kj = arma::norm(pos_k - pos_j);
        double aj = spinParallelFactor(k, j, N2);
        Jastrow += 2*aj/(r_kj*std::pow(1 + m_beta*r_kj, 2) - 2*aj*m_beta/std::pow(1 + m_beta*r_kj, 3));
    }
    for (int j = k+1; j < numberofparticles; j++)
    {
        arma::vec pos_j = particles[j]->getPosition();
        double r_kj = arma::norm(pos_k - pos_j);
        double aj = spinParallelFactor(k, j, N2);
        Jastrow += 2*aj/(r_kj*std::pow(1 + m_beta*r_kj, 2) - 2*aj*m_beta/std::pow(1 + m_beta*r_kj, 3));
    }


    for (int j = 0; j < k; j++)
    {
        arma::vec pos_j = particles[j]->getPosition();
        double r_kj = arma::norm(pos_k - pos_j);
        double aj = spinParallelFactor(k, j, N2);
        for (int i = 0; i < k; i++)
        {
            arma::vec pos_i = particles[i]->getPosition();
            double r_ki = arma::norm(pos_k - pos_i);
            double ai = spinParallelFactor(k, i, N2);

            double r = 0;
            for (int l = 0; l < numberofdimensions; l++)
            {
                r += (pos_k(l) - pos_j(l))*(pos_k(l) - pos_i(l));
            }

            Jastrow += r/(r_ki*r_kj)*aj*ai/std::pow(1 + m_beta*r_ki, 4);
        }
        for (int i = k+1; i < numberofparticles; i++)
        {
            arma::vec pos_i = particles[i]->getPosition();
            double r_ki = arma::norm(pos_k - pos_i);
            double ai = spinParallelFactor(k, i, N2);

            double r = 0;
            for (int l = 0; l < numberofdimensions; l++)
            {
                r += (pos_k(l) - pos_j(l))*(pos_k(l) - pos_i(l));
            }

            Jastrow += r/(r_ki*r_kj)*aj*ai/std::pow(1 + m_beta*r_ki, 4);        
        }
    }

    for (int j = k+1; j < numberofparticles; j++)
    {
        arma::vec pos_j = particles[j]->getPosition();
        double r_kj = arma::norm(pos_k - pos_j);
        double aj = spinParallelFactor(k, j, N2);
        for (int i = 0; i < k; i++)
        {
            arma::vec pos_i = particles[i]->getPosition();
            double r_ki = arma::norm(pos_k - pos_i);
            double ai = spinParallelFactor(k, i, N2);

            double r = 0;
            for (int l = 0; l < numberofdimensions; l++)
            {
                r += (pos_k(l) - pos_j(l))*(pos_k(l) - pos_i(l));
            }

            Jastrow += r/(r_ki*r_kj)*aj*ai/std::pow(1 + m_beta*r_ki, 4);        
        }
        for (int i = k+1; i < numberofparticles; i++)
        {
            arma::vec pos_i = particles[i]->getPosition();
            double r_ki = arma::norm(pos_k - pos_i);
            double ai = spinParallelFactor(k, i, N2);

            double r = 0;
            for (int l = 0; l < numberofdimensions; l++)
            {
                r += (pos_k(l) - pos_j(l))*(pos_k(l) - pos_i(l));
            }

            Jastrow += r/(r_ki*r_kj)*aj*ai/std::pow(1 + m_beta*r_ki, 4);        
        }
    }

    return Jastrow;
}

double SimpleGaussianSlater::spinParallelFactor(int i, int j, int N2)
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

void SimpleGaussianSlater::CheckSlater(std::vector<std::unique_ptr<class Particle>> &particles)
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

    (m_D_up*m_DI_up).print();
    (m_D_down*m_DI_down).print();
}




SimpleGaussianSlaterNumerical::SimpleGaussianSlaterNumerical(double alpha, double beta, double omega, double dx, int N, bool Jastrow) : SimpleGaussianSlater(alpha, beta, omega, N, Jastrow)
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