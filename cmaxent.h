#include <string>
/**
 * @param xqmc the tau/i-omega resolved monte carlo data
 * @param xtau the tau/i-omega grid
 * @param len the number of grid points
 * @param xmom1 the first moment
 * @param xker the kernel to use
 * @param backtrans the back-transformation
 * @param omega_points the number of points in the omega spectrum
 * @param nsweeps The number of updates/sweeps we do between two sweeps
 * @param nbins The number of bins that we collect
 * @param nwarmup the number of configurations that get discarded from the beginning
 * @param u the matrix of eigenvectors or a NULL-pointer if the covariance matrix shall be assumed diagonal
 * @param sigma eigenvalues of the covariance matrix which is equal to the variance squared.
 * */
void cmaxent ( double *const __restrict__ xqmc, const double *const xtau, const int len,
               double xmom1, double ( *xker ) ( const double&, double&, double& ), double ( *backtrans ) ( double&, double&, double& ),
               double beta, double *const alpha_tot, const int n_alpha, const uint ngamma, const double omega_start, const double omega_end,
               const uint omega_points, int nsweeps, uint nbins, uint nwarmup,
               const std::string file_root, const std::string dump_Aom, std::string max_stoch_log, std::string energies, std::string best_fit, std::string dump,
               const double *const __restrict__ u, double *const __restrict__ sigma );
