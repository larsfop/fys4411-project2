#pragma once

#include <memory>
#include "WaveFunctions.h"
#include "Particle.h"

class SimpleGaussian : public WaveFunction
{
public:
    SimpleGaussian(const double alpha, double beta, double omega, bool Jastrow, bool m_Interaction);
    double Wavefunction(std::vector<std::unique_ptr<class Particle>> &particles);
    double DoubleDerivative(std::vector<std::unique_ptr<class Particle>> &particles);
    double LocalEnergy(std::vector<std::unique_ptr<class Particle>> &particles);
    arma::vec QuantumForce(std::vector<std::unique_ptr<class Particle>> &particles, const int index);
    double w(std::vector<std::unique_ptr<class Particle>> &particles, const int index, const arma::vec step);
    arma::vec getParameters() {return m_parameters; };
    arma::vec dPsidParam(std::vector<std::unique_ptr<class Particle>> &particles);
    void ChangeParameters(const double alpha, const double beta);
    arma::vec QuantumForce(
        std::vector<std::unique_ptr<class Particle>> &particles,
        const int index,
        const arma::vec Step
    );
    double geta() {return 0; };
    double Hermite_poly(int n, arma::vec pos);
    void FillSlaterDeterminants(std::vector<std::unique_ptr<class Particle>> &particles);
    void UpdateInverseSlater(std::vector<std::unique_ptr<class Particle>> &particles,
    int k,
    double R,
    arma::vec step
    );
    void CheckSlater(std::vector<std::unique_ptr<class Particle>> &particles);
    bool Jastrow() {return m_Jastrow; };
    double spinParallelFactor(int i, int j, int N2);
    double getKinetic();
    double getPotential();
    void KeepOldInverseSlater() {exit(0); };
    void ChangeOldInverseSlater() {exit(0); };
    
private:
    double m_alpha;
    double m_beta;
    double m_omega;
    arma::vec m_beta_z;
    arma::vec m_parameters;
    bool m_Jastrow;
    bool m_Interaction;

    double m_kinetic;
    double m_potential;
};


class SimpleGaussianNumerical : public SimpleGaussian
{
public:
    SimpleGaussianNumerical(double alpha, double beta, double omega, double dx, bool Jastrow, bool m_Interaction);
    double DoubleDerivative(std::vector<std::unique_ptr<class Particle>> &particles);
    double EvaluateSingleParticle(class Particle particle);
    double EvaluateSingleParticle(class Particle particle, double step, double step_index);

private:
    double m_dx;
    double m_alpha;
    double m_omega;
    arma::vec m_beta_z;
    arma::vec m_parameters;
};