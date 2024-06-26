#pragma once

#include <vector>
#include <armadillo>

class WaveFunction
{
private:
    
public:
    virtual ~WaveFunction() = default;

    virtual double Wavefunction(std::vector<std::unique_ptr<class Particle>> &particles) = 0;
    virtual void FillSlaterDeterminants(std::vector<std::unique_ptr<class Particle>> &particles) = 0;
    virtual double LocalEnergy(std::vector<std::unique_ptr<class Particle>> &particles) = 0;
    virtual arma::vec QuantumForce(std::vector<std::unique_ptr<class Particle>> &particles, const int index) = 0;
    virtual double w(std::vector<std::unique_ptr<class Particle>> &particles, const int index, const arma::vec step) = 0;
    virtual arma::vec dPsidParam(std::vector<std::unique_ptr<class Particle>> &particles) = 0;
    virtual arma::vec getParameters() = 0;
    virtual void ChangeParameters(const double alpha, const double beta) = 0;
    virtual arma::vec QuantumForce(
        std::vector<std::unique_ptr<class Particle>> &particles,
        const int index,
        const arma::vec Step
    ) = 0;
    virtual double geta() = 0;
    virtual double spinParallelFactor(int i, int j, int N2) = 0;
    virtual double Hermite_poly(int n, arma::vec pos) = 0;
    virtual void UpdateInverseSlater(std::vector<std::unique_ptr<class Particle>> &particles, int index, double R, arma::vec step) = 0;
    virtual void CheckSlater(std::vector<std::unique_ptr<class Particle>> &particles) = 0;
    virtual bool Jastrow() = 0;
    virtual double getKinetic() = 0;
    virtual double getPotential() = 0;
    virtual void KeepOldInverseSlater() = 0;
    virtual void ChangeOldInverseSlater() = 0;
};

