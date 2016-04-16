#include <fstream>
#include <cmath>
#include <iostream>
#include <string>
#include <complex>
#include <cstring>
#include <vector>
#include "cmaxent.h"

using namespace std;

extern"C"
{
void __maxent_stoch_mod_MOD_maxent_stoch(double* /*xqmc*/, double* /*xtau*/, double* /*cov*/, int* datapoints,
					 double* /*xmom1*/, double (*xker)(const double&, double&, double&),
double (*Back_trans_Aom)(double&,double&,double&), double* /*beta_1*/, double*/*alpha_tot*/, int*/*n_alpha*/, const int* /*Ngamma*/,
const double* /*omega_start*/, const double* /*omega_end*/, const int* /*omega_points*/, int* /*Nsweeps*/, const int* /*nbins*/, const int* /*nwarmup*/, const char *const /*File_root*/, const char *const /*dump_Aom*/, const char *const /*Max_stoch_log*/,
const char *const /*energies*/, const char *const /*best_fit*/, const char *const/*dump*/, int* /*L_cov*/);
}


double Back_trans_Aom_fermionic(double& Aom, double& om, double& beta)
{
return Aom;
}

double xker_symmetric(const double& tau, double& om, double& beta)
{
return 0.5*std::cosh(om*(0.5*beta - tau))/std::cosh(0.5*beta*om);//the symmetric kernel for transformation from tau to omega
}

double xker(const double& tau, double& om, double& beta)//the plain kernel
{
  return 0.5*std::exp(om*(0.5*beta - tau))/std::cosh(0.5*beta*om);
}


int main(int argc, char** argv)
{
  std::ifstream g(argv[1]);
  int nsweeps = 50; int nbins = 100; int nwarmup = 50;
  std::string fr("Aom"); string dA("dump_Aom"); string ml("max_stoch_log"); string energies("energies"); string bf("best_fit"); string dump("dump");
  std::vector<double> xqmc;
  std::vector<double> xtau;
  std::vector<double> sigma;
  uint n_alpha = 14;
  double xmom = 1.0;
  uint ndis = 500;
  int ngamma = 400;
  double omst = -6.0;
  double omend = 6.0;
    double* alpha_tot = new double[n_alpha];
    double R = 1.2;
    double alpha_st = 1.0;
    for(uint l = 0; l < n_alpha; ++l)
       alpha_tot[l] = alpha_st*std::pow(R, l);
  while (g)
  {
    double temp;
    g>>temp;
    xtau.push_back(temp);
    g>>temp;
    xqmc.push_back(temp);
    g>>temp;
    temp *= temp;
    sigma.push_back(temp);
  }
  int sub;
  int len = xqmc.size() - 1;//quick hack...
  double beta = len*(xtau[1] - xtau[0]);
  //set to !=0 to compile cmaxent
#if 1

  cmaxent(& xqmc[0],& xtau[0], len, xmom, xker, Back_trans_Aom_fermionic, beta, alpha_tot, n_alpha, ngamma, omst, omend, ndis, nsweeps, nbins, nwarmup,
	   fr, dA, ml, energies, bf, dump, NULL,& sigma[0]);
  sub = 11;
#else
  int usecov = 1;
double* cov = new double[len*len];
memset(cov, 0, len*len*sizeof(double));
for(uint nt = 0; nt < len; ++nt)
cov[nt*len + nt] = sigma[nt];
__maxent_stoch_mod_MOD_maxent_stoch(& xqmc[0], & xtau[0], cov, &len, &xmom, xker, Back_trans_Aom_fermionic, &beta,
				    alpha_tot, &n_alpha, &ngamma, &omst, &omend, &ndis, &nsweeps, &nbins, &nwarmup, fr.c_str(), dA.c_str(), ml.c_str(), energies.c_str(), bf.c_str(), dump.c_str(), &usecov);
delete [] cov;
sub = 10;
#endif
  delete [] alpha_tot;
   std::ifstream aps((string("Aom") + string("_ps_") + to_string(n_alpha - sub /*C vs. Fortran*/)).c_str());
   std::ofstream green("green");

    if(!aps)
    {
      std::cout<<"Error while opening file Aom_ps"<<std::endl;
      return -1;
    }
    double* xom = new double[ndis];
    double* aom = new double[ndis];
    double x, x1, x2;
    for(uint k = 0; k < ndis; ++k)
    {
      aps >> xom[k];
      aps >> aom[k];
      aps >> x;
      aps >> x1;
      aps >> x2;
    }
    const double dom = xom[1] - xom[0];
    const double delta = 4*dom;
    for(uint nw = 0; nw < ndis; ++nw)
    {
        std::complex<double> z(0, 0);
        double om = xom[nw];
        for(uint nwp = 0; nwp < ndis; ++nwp)
        {
    	    double omp = xom[nwp];
    	    z += std::complex<double>(aom[nwp], 0.0)/std::complex<double>(-om - omp, delta);//Kernel for Rubtsov Green's function
        }
        z *= -dom;//the minus sign, is to compensate for the lack of a minus sign in the definition of the kernel, above
        green<<xom[nw]<<" "<<real(z)<<" "<<imag(z)/M_PI<<std::endl;//this normalization is only useful for the spectral function.
    }
    delete [] aom;
    delete [] xom;
  return 0;
}
