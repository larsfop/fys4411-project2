#pragma once

#include <memory>
#include "WaveFunctions.h"
#include "../Particle.h"

class SimpleGaussianSlater : public WaveFunction
{
public:
    SimpleGaussianSlater(
        const double alpha, 
        double beta, 
        int N);
    double Wavefunction(std::vector<std::unique_ptr<class Particle>> &particles);
    double EvalWavefunction(std::vector<std::unique_ptr<class Particle>> &particles, int i, int j);
    double EvalWavefunction(std::vector<std::unique_ptr<class Particle>> &particles, int i, int j, arma::vec step);
    void FillSlaterDeterminants(std::vector<std::unique_ptr<class Particle>> &particles);
    arma::vec SingleDerivative(std::vector<std::unique_ptr<class Particle>> &particles, int i, int j);
    arma::vec SingleDerivative(std::vector<std::unique_ptr<class Particle>> &particles, int i, int j, arma::vec step);
    double DoubleDerivative(std::vector<std::unique_ptr<class Particle>> &particles, int i, int j);
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
    void UpdateInverseSlater(std::vector<std::unique_ptr<class Particle>> &particles, int k, double R, arma::vec step);

private:
    double m_alpha;
    double m_beta;
    double m_omega;
    arma::vec m_beta_z;
    arma::vec m_parameters;

    arma::mat m_D_up;
    arma::mat m_D_down;
    arma::mat m_DI_down;
    arma::mat m_DI_up;
    double m_D_sum;
};


class SimpleGaussianSlaterNumerical : public SimpleGaussianSlater
{
public:
    SimpleGaussianSlaterNumerical(double alpha, double beta, double dx, int N);
    double DoubleDerivative(std::vector<std::unique_ptr<class Particle>> &particles);
    double EvaluateSingleParticle(class Particle particle);
    double EvaluateSingleParticle(class Particle particle, double step, double step_index);

private:
    double m_dx;
    double m_alpha;
    arma::vec m_beta_z;
};