# fys4411-project2   -    Variational Monte Carlo for simulating electrons in their lowest quantum states
Project 2 for Computational physics 2 - FYS4411

A continuation of project 1. This project uses VMC with Slater determinants to simulate and compute the expectation value for the lowest state energy for a N electron system. We will use 2, 6 and 12 electons in two different classes. The first class "harmonicoscillator" simulates the analytical and numerical derivations for the two electron system. While the class "Slater" uses Slater determinants instead, but have not been finished with a few bugs with the importance sampling and gradient descent. 

The rest of the code works more or less the same as project 1. The few tweaks are made for the Slater determinants. A few improvements have been made, now there is not a different class for interacting vs not interactinc wavefunctions, while there is an addition of a Slater option in the config for deciding which class you want to use and the Jastrow option allows you to simulate with to without the Jastrow factor.

### Compiling the project

compile the project with g++ using this command
```bash
g++ -O3 main.cpp src/*.cpp -I include -o main -larmadillo -fopenmp
```
Then run the the file 
```bash
./main kwargs
```
The kwargs can be found in the config file

The program was run with mostly the same options in the config. Most of the options changes was either changing what wavefunction to use or number of particles.
