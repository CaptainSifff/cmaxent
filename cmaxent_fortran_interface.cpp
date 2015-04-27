#include "cmaxent.h"
#include <iostream>


/**
 * @param xqmc the tau/i-omega resolved monte carlo data
 * @param xtau the tau/i-omega grid
 * @param len the number of grid points
 * @param xmom1 the first moment
 * @param xker the kernel to use
 * @param backtrans the back-transformation
 * @param omega_points the number of points in the omega spectrum
 * @param nsweeps
 * @param nbins
 * @param nwarmup the number of configurations that get discarded from the beginning
 * @param u the matrix of eigenvectors or a NULL-pointer if the covariance matrix shall be assumed diagonal
 * @param sigma eigenvalues of the covariance matrix or errors
 * */
void cmaxent_fortran( double* xqmc, double* xtau, int32_t len, double xmom1, double ( *xker ) ( const double&, double&, double& ),
		      double ( *backtrans ) ( double&, double&, double& ), double beta, double* alpha_tot, int32_t n_alpha, int32_t ngamma, double omega_start, double omega_end,
		      int32_t omega_points, int32_t nsweeps, int32_t nbins, int32_t nwarmup,/* double* u,*/ double* sigma) asm("cmaxent_fortran");

void cmaxent_fortran( double* xqmc, double* xtau, int32_t len, double xmom1, double ( *xker ) ( const double&, double&, double& ),
		      double ( *backtrans ) ( double&, double&, double& ), double beta, double* alpha_tot, int32_t n_alpha, int32_t ngamma, double omega_start, double omega_end,
		      int32_t omega_points, int32_t nsweeps, int32_t nbins, int32_t nwarmup,/* double* u,*/ double* sigma)
{
  std::string fr("Aom"); std::string dA("dump_Aom"); std::string ml("max_stoch_log"); std::string energies("energies"); std::string bf("best_fit"); std::string dump("dump");
  cmaxent(xqmc, xtau, len, xmom1, xker, backtrans, beta, alpha_tot, n_alpha, ngamma, omega_start, omega_end, omega_points, nsweeps, nbins, nwarmup,
	  fr, dA, ml, energies, bf, dump, /*u*/NULL, sigma);
}