#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
using namespace std;
// ================== PARAMETERS ==================
const int N = 500; // matrix size
const double omega = 1.8; // relaxation parameter
const double tol = 1e-6; // convergence tolerance
const int max_iter = 50000; // max iterations
// ================== SOR SEQUENTIAL ==================
int sorSequential(vector<vector<double>>& A, vector<double>& b, vector<double>& x)
{
	int n = A.size();
	int iterations = 0;
	for (int iter = 0; iter < max_iter; iter++)
	{
		double error = 0.0;
		for (int i = 0; i < n; i++)
		{
			double sigma = 0.0;
			for (int j = 0; j < n; j++)
			{
				if (j != i)
					sigma += A[i][j] * x[j];
			}
			double new_x = (1 - omega) * x[i] +
				(omega / A[i][i]) * (b[i] - sigma);
			error += fabs(new_x - x[i]);
			x[i] = new_x;
		}
		if (error < tol)
			return iter + 1;
	}
	return max_iter;
}
// SOR PARALLEL
int sorParallel(vector<vector<double>>& A, vector<double>& b, vector<double>& x, int threads)
{
	omp_set_num_threads(threads);
	int n = A.size();
	for (int iter = 0; iter < max_iter; iter++)
	{
		double error = 0.0, sigma = 0.0;
		for (int i = 0; i < n; i++)
		{
			sigma = 0.0;
#pragma omp parallel for reduction(+:sigma) 
			for (int j = 0; j < n; j++)
			{
				if (j != i)
					sigma += A[i][j] * x[j]; 
			}
			double new_x = (1 - omega) * x[i] +
				(omega / A[i][i]) * (b[i] - sigma);
			error += fabs(new_x - x[i]);
			x[i] = new_x;
		}
		if (error < tol)
			return iter + 1;
	}
	return max_iter;
}
// MAIN
int main()
{
	cout << "Successive Over-Relaxation (SOR) Method\n";
	cout << "Matrix size: " << N << " x " << N << endl;
	cout << "Using 16 threads for parallel version\n\n";
	
	// Build A (diagonally dominant matrix) and b
	vector<vector<double>> A(N, vector<double>(N, 0));
	vector<double> b(N, 1.0);
	vector<double> x_seq(N, 0.0);
	vector<double> x_par(N, 0.0);
	
	// Make matrix A strongly diagonally dominant
	for (int i = 0; i < N; i++)
	{
		A[i][i] = 4.0;
		if (i > 0) A[i][i - 1] = -1.0;
		if (i < N - 1) A[i][i + 1] = -1.0;
	}
	
	// SEQUENTIAL
	double t1 = omp_get_wtime();
	int seq_iter = sorSequential(A, b, x_seq);
	double t2 = omp_get_wtime();
	double seq_time = (t2 - t1) * 1000.0;
	
	cout << "Sequential iterations: " << seq_iter << endl;
	
	cout << "Sequential time: " << seq_time << " ms\n\n";
	
	// PARALLEL (16 threads)
	int threads = 16;
	double t3 = omp_get_wtime();
	int par_iter = sorParallel(A, b, x_par, threads);
	double t4 = omp_get_wtime();
	double par_time = (t4 - t3) * 1000.0;
	
	cout << "Parallel iterations: " << par_iter << endl;
	
	cout << "Parallel time: " << par_time << " ms\n\n";
	
	// SPEED-UP & EFFICIENCY
	double speedup = seq_time / par_time;
	double efficiency = speedup / threads;
	
	cout << "========== PERFORMANCE ==========\n";
	cout << "Speed-Up: " << speedup << endl;
	cout << "Efficiency: " << efficiency << endl;
	cout << "=================================\n";
	
	return 0;
}
