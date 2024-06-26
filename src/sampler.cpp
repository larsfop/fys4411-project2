
#include <memory>
#include "sampler.h"
#include "system.h"
#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <numeric>
#include <sstream>
#include <iomanip>

using std::cout;
using std::endl;

using std::setw;
using std::setprecision;
using std::fixed;
using std::scientific;

Sampler::Sampler(
    int numberofparticles,
    int numberofdimensions,
    double steplength,
    int numberofMetropolisSteps
)
{
    m_stepnumber = 0;
    m_numberofMetropolisSteps = numberofMetropolisSteps;
    m_numberofparticles = numberofparticles;
    m_numberofdimensions = numberofdimensions;
    m_Energy = 0;
    m_Energy2 = 0;
    m_DeltaEnergy = 0;
    m_steplength = steplength;
    m_numberofacceptedsteps = 0;

    m_Kinetic = 0;
    m_Potential = 0;

    m_DeltaPsi = arma::vec(2);
    m_PsiEnergyDerivative = arma::vec(2);
    m_params = arma::vec(2);
}

Sampler::Sampler(std::vector<std::unique_ptr<class Sampler>> &samplers, std::string Filename)
{
    m_numberofthreads = samplers.size();

    m_numberofMetropolisSteps = samplers[0]->m_numberofMetropolisSteps;
    m_numberofparticles = samplers[0]->m_numberofparticles;
    m_numberofdimensions = samplers[0]->m_numberofdimensions;
    m_steplength = samplers[0]->m_steplength;

    m_Energy = 0;
    m_Energy2 = 0;
    m_numberofacceptedsteps = 0;
    m_stepnumber = 0;

    m_Kinetic = 0;
    m_Potential = 0;

    m_Filename = samplers[0]->m_Filename;
    int N_params = samplers[0]->m_params.n_elem;
    m_DeltaPsi = arma::vec(N_params);
    m_PsiEnergyDerivative = arma::vec(N_params);
    m_params = arma::vec(N_params);
    for (auto &sampler : samplers)
    {
        m_Energy += sampler->m_Energy;
        m_Energy2 += sampler->m_Energy2;
        //m_variance += sampler->m_variance;

        m_Kinetic += sampler->m_Kinetic;
        m_Potential += sampler->m_Potential;
        
        m_DeltaPsi += sampler->m_DeltaPsi;
        m_PsiEnergyDerivative += sampler->m_PsiEnergyDerivative;
        //m_EnergyDerivative += sampler->m_EnergyDerivative;

        m_stepnumber += sampler->m_stepnumber;
        m_numberofacceptedsteps += sampler->m_numberofacceptedsteps; 

        for (int i = 0; i < N_params; i++)
        {
            m_params(i) += sampler->m_params(i);
        }
        m_hist.insert(m_hist.end(), sampler->m_hist.begin(), sampler->m_hist.end());
    }

    m_Energy /= m_numberofthreads;
    m_Energy2 /= m_numberofthreads;
    m_DeltaPsi /= m_numberofthreads;
    m_PsiEnergyDerivative /= m_numberofthreads;

    m_variance = m_Energy2 - m_Energy*m_Energy;
    m_EnergyDerivative = 2*(m_PsiEnergyDerivative - m_DeltaPsi*m_Energy);

    m_stepnumber /= m_numberofthreads;
    m_numberofacceptedsteps /= m_numberofthreads;

    m_params /= m_numberofthreads;

    m_Filename = Filename;
}

void Sampler::Sample(bool acceptedstep, class System *system)
{
    double localenergy = system->ComputeLocalEnergy();
    m_Energy += localenergy;
    m_Energy2 += localenergy*localenergy;
    // printf("E_L = %f ; E = %f ; E^2 = %f ; sqrt = %f\n", localenergy, m_Energy, m_Energy2, sqrt(m_Energy2));
    m_stepnumber++;
    m_numberofacceptedsteps += acceptedstep;

    m_Kinetic += system->getKinetic();
    m_Potential += system->getPotential();

    arma::vec dparams = system->ComputeDerivatives();
    m_DeltaPsi += dparams;
    m_PsiEnergyDerivative += dparams*localenergy;

    m_params = system->getParameters();
}

void Sampler::SampleHist(class System *system)
{
    arma::vec pos = system->getPosition(0);
    for (int i = 1; i < m_numberofparticles; i++)
    {
        arma::vec posi = system->getPosition(i);
        m_hist.push_back(arma::norm(pos - posi));
    }
}

void Sampler::ComputeDerivatives()
{
    double Energy = m_Energy/m_numberofMetropolisSteps;
    arma::vec DeltaPsi = m_DeltaPsi/m_numberofMetropolisSteps;
    arma::vec PsiEnergyDerivative = m_PsiEnergyDerivative/m_numberofMetropolisSteps;

    m_EnergyDerivative = 2*(PsiEnergyDerivative - DeltaPsi*Energy);
}

void Sampler::ComputeAverages()
{
    m_Energy /= m_numberofMetropolisSteps;
    m_Energy2 /= m_numberofMetropolisSteps;
    m_variance = m_Energy2 - m_Energy*m_Energy;

    m_Kinetic /= 1.0*m_numberofMetropolisSteps;
    m_Potential /= 1.0*m_numberofMetropolisSteps;

    //m_stepnumber /= m_numberofthreads;
    //m_numberofacceptedsteps /= m_numberofthreads;
}

void Sampler::printOutput(System &system)
{
    auto params = system.getParameters();


    cout << endl;
    cout << "  -- System info -- " << endl;
    cout << " Number of particles  : " << m_numberofparticles << endl;
    cout << " Number of dimensions : " << m_numberofdimensions << endl;
    cout << " Number of Metropolis steps run : 2^" << std::log2(m_numberofMetropolisSteps) << endl;
    cout << " Step length used : " << m_steplength << endl;
    cout << " Ratio of accepted steps: " << ((double) m_numberofacceptedsteps) / ((double) m_numberofMetropolisSteps) << endl;
    cout << endl;
    cout << "  -- Wave function parameters -- " << endl;
    cout << " Number of parameters : " << params.n_elem << endl;
    for (unsigned int i=0; i < params.n_elem; i++) {
        cout << " Parameter " << i+1 << " : " << params(i) << endl;
    }
    cout << endl;
    cout << "  -- Results -- " << endl;
    cout << " Energy : " << m_Energy << endl;
    cout << " Variance : " << m_variance << endl;
    cout << endl;
}

void Sampler::printOutput()
{
    cout << endl;
    cout << "  -- System info -- " << endl;
    cout << " Number of threads  : " << m_numberofthreads << endl;
    cout << " Number of particles  : " << m_numberofparticles << endl;
    cout << " Number of dimensions : " << m_numberofdimensions << endl;
    cout << " Number of Metropolis steps run :" << m_numberofMetropolisSteps << " (2^" << std::log2(m_numberofMetropolisSteps) << ")" << endl;
    cout << " Step length used : " << m_steplength << endl;
    cout << " Ratio of accepted steps: " << ((double) m_numberofacceptedsteps) / ((double) m_numberofMetropolisSteps) << endl;
    cout << endl;
    cout << "  -- Wave function parameters -- " << endl;
    cout << " Number of parameters : " << m_params.n_elem << endl;
    for (unsigned int i=0; i < m_params.n_elem; i++) {
        cout << " Parameter " << i+1 << " : " << m_params(i) << endl;
    }
    cout << endl;
    cout << "  -- Results -- " << endl;
    cout << " Energy : " << m_Energy << endl;
    cout << " Energy^2 : " << m_Energy2 << endl;
    cout << " Variance : " << m_variance << endl;
    cout << " Kinetic : " << m_Kinetic << endl;
    cout << " Potential : " << m_Potential << endl;
    cout << endl;
}

void Sampler::CreateFile()
{
    int width = 20;
    std::ofstream ofile(m_Filename, std::ofstream::trunc);
    ofile << setw(width-8) << "alpha"
            << setw(width) << "EnergyDerivative"
            << setw(width) << "Energy"
            << endl;
    ofile.close();
}

void Sampler::WritetoFile()
{
    int width = 16;

    std::ofstream ofile(m_Filename, std::ofstream::app);

    // ofile << setprecision(6);
    ofile << setw(width-6) << m_numberofMetropolisSteps
            << setw(width) << m_numberofacceptedsteps
            << setw(width) << m_numberofdimensions
            << setw(width) << m_numberofparticles
            << setw(width) << m_steplength
            << setw(width) << m_params(0)
            << setw(width) << m_EnergyDerivative(0)
            << setw(width) << m_params(1)
            << setw(width) << m_EnergyDerivative(1)
            << setw(width) << m_Energy
            << setw(width) << m_variance
            << setw(width) << m_time.count()
            << setw(width) << omp_get_thread_num()
            << endl;
    ofile.close();
}

void Sampler::WriteEnergiestoFile(System &system, int iteration)
{
    int width = 16;
    std::ofstream ofile("Outputs/Energies.dat", std::ofstream::app);
    ofile.fixed;
    ofile.precision(15);
    ofile << m_Energy/iteration << endl;
    ofile.close();
}

void Sampler::setParameters(double alpha, double beta)
{
    m_params = {alpha, beta};
}

void Sampler::SaveHist(bool Jastrow)
{
    std::string Filename = Jastrow ? "IW" : "SG";
    std::ofstream output_file("Outputs/histogram_"+Filename+".dat");

    std::ostream_iterator<double> output_iterator(output_file, "\n");
    std::copy(std::begin(m_hist), std::end(m_hist), output_iterator);
}
