#include<iostream> 
#include<vector>
#include<algorithm>
#include<random>
#include<numeric>
#include<limits.h>
#include<cmath>
#include<chrono> 
#include<string>
#include<fstream>
#include<cstdlib>
#include<array>
#include<Eigen/Sparse>

using Eigen::SparseMatrix;
using Eigen::VectorXd;
using Eigen::ConjugateGradient;
using Eigen::Triplet;

using namespace std;

constexpr double lambda_constant = 0.8;
constexpr double activity_constant = 0.05;
constexpr double PI = 3.14159265358979323846;

constexpr int timesteps = 2000;
constexpr double delta_t = 0.5;
// constexpr double learning_rate = 0.01;
constexpr int num_iters = 50;

constexpr double D = 1;
constexpr double polarization_ini_mag = 0.1;
constexpr double density_mean  = 1.07;
constexpr double rho_c = 1;

constexpr double K = 1;
constexpr double C = 1.2;
constexpr double E = 1.2;
constexpr double B = 1;
constexpr double A = 1;

int get_periodic_index(int i, int size) {
    return (i % size + size) % size; 
}

void shift_2d_array(vector<vector<double>>& arr) {

    const int Ny = arr.size();    
    const int Nx = arr[0].size(); 
    vector<vector<double>> temp_arr = arr;
    int dy = 4; 
    int dx = -4; 
    for (int i = 0; i < Ny; ++i) {
        for (int j = 0; j < Nx; ++j) {
            int source_i = i - dy;
            int source_j = j - dx;
            int wrapped_i = (source_i % Ny + Ny) % Ny;
            int wrapped_j = (source_j % Nx + Nx) % Nx;
            temp_arr[i][j] = arr[wrapped_i][wrapped_j];
        }
    }
    arr = temp_arr;
}

void shift_3d_array(vector<vector<vector<double>>>& arr) {
    if (arr.empty() || arr[0].empty() || arr[0][0].empty()) {
        cerr << "Error: 3D Array is empty or malformed." << endl;
        return;
    }

    const int Nz = arr.size();     
    const int Ny = arr[0].size();  
    const int Nx = arr[0][0].size(); 

    vector<vector<vector<double>>> temp_arr = arr;

    int dy = 4; 
    int dx = -4; 
        
    for (int i = 0; i < Nz; ++i) { 
        for (int j = 0; j < Ny; ++j) { 
            for (int k = 0; k < Nx; ++k) { 
                
                int source_i = i - dy;
                int source_j = j - dx;

  
                int wrapped_i = (source_i % Nz + Nz) % Nz;
                int wrapped_j = (source_j % Ny + Ny) % Ny;

                temp_arr[i][j][k] = arr[wrapped_i][wrapped_j][k];
            }
        }
    }
    arr = temp_arr;
}

void readvectors(string folder_path, vector<vector<double>> &particle_density, vector<vector<vector<double>>> &tau, int iter, int Ny, int Nx){
    ifstream inFile(folder_path+"particle_density_"+to_string(iter)+".bin", std::ios::binary);
    for (int i = 0; i < Ny; ++i) {
        inFile.read(reinterpret_cast<char*>(particle_density[i].data()), Nx * sizeof(double));
    }
    inFile.close();

    inFile.open(folder_path+"tau_"+to_string(iter)+".bin", std::ios::binary);

    for (int i = 0; i < Ny; ++i) {
        for (int j = 0; j < Nx; ++j) {
            inFile.read(reinterpret_cast<char*>(tau[i][j].data()), 2 * sizeof(double));
        }
    }
    inFile.close();

}

void save_activity_field(vector<vector<vector<double>>> &activity_field, string folder_path, int iter){
    string activity_field_path = "activity_field_final"+to_string(iter)+".bin";

    ofstream outFile(folder_path+activity_field_path, ios::binary);
    outFile.open(folder_path+activity_field_path, ios::binary);
    for (const auto &plane : activity_field) {
        for (const auto &row : plane) {
            outFile.write(reinterpret_cast<const char *>(row.data()), row.size() * sizeof(double));
        }
    }
    outFile.close();
}

void savevectors(vector<vector<double>> &particle_density, vector<vector<vector<double>>> &polarization_field, vector<vector<vector<double>>> &tau,string folder_path, int iter){
    string particle_density_path = "particle_density_"+to_string(iter)+".bin";
    string polarization_field_path = "polarization_field_"+to_string(iter)+".bin";
    string tau_path = "tau_"+to_string(iter)+".bin";
    ofstream outFile(folder_path+particle_density_path, ios::binary);

    for (const auto &row : particle_density) {
            outFile.write(reinterpret_cast<const char *>(row.data()), row.size() * sizeof(double));
    }
    outFile.close();

    outFile.open(folder_path+polarization_field_path, ios::binary);
    for (const auto &plane : polarization_field) {
        for (const auto &row : plane) {
            outFile.write(reinterpret_cast<const char *>(row.data()), row.size() * sizeof(double));
        }
    }
    outFile.close();

    
    outFile.open(folder_path+tau_path, ios::binary);
    for (const auto &plane : tau) {
        for (const auto &row : plane) {
            outFile.write(reinterpret_cast<const char *>(row.data()), row.size() * sizeof(double));
        }
    }
    outFile.close();
}

void initializeGaussian(vector<vector<double>> &particle_density,int N, double sigma, double A = 1.0) {
    int cx = N / 20;
    int cy = N / 20;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int dx = i - cx;
            int dy = j - cy;
            double r2 = dx*dx + dy*dy;
            particle_density[i][j] = 0;
            particle_density[i][j] += A * exp(-r2 / (2.0 * sigma * sigma));
            particle_density[i][j] += A * exp(-r2 / (2.0 * sigma * sigma));
        }
    }
}


void initialize_grid(vector<vector<double>> &grid, double mean, int Nx, int Ny){
    static mt19937 g(time(nullptr));  
    double mean_dist = 0;
    double stddev = 0.2;
    normal_distribution<double> dist(mean_dist, stddev);
  for(int i=0;i<Ny;i++){
    for(int j=0;j<Nx;j++){
        grid[i][j] = mean;
    }
  }
  return;
}

void initialize_3Dgrid(vector<vector<vector<double>>> &grid, int Nx, int Ny){
  random_device rd;  
  mt19937 g(time(nullptr));   
  double lower_bound = 0.0;
  double upper_bound = 2*PI;
  uniform_real_distribution<double> dist(lower_bound, upper_bound);   
  for(int i=0;i<Ny;i++){
    for(int j=0;j<Nx;j++){
      double angle = dist(g);
      grid[i][j][0] = polarization_ini_mag*cos(angle);
      grid[i][j][1] = polarization_ini_mag*sin(angle);
    }
  }
  return;
}

void update_activity(vector<vector<double>> &activity_field, double rho_c, int Nx, int Ny){
  for(int i=0;i<Ny;i++){
    for(int j=0;j<Nx;j++){
        activity_field[i][j] = activity_constant;
    }
  }
  return;
}

void tau_calculate(vector<vector<vector<double>>> &tau, vector<vector<vector<double>>> &polarization_field, vector<vector<double>> &particle_density,  int Nx, int Ny){
  for(int i=0;i<Ny;i++){
    for(int j=0;j<Nx;j++){
      for(int k=0;k<2;k++){
          tau[i][j][k] = particle_density[i][j]*polarization_field[i][j][k];
      }
    }
  }
  return;
}

void divergence(vector<vector<vector<double>>> &grid, vector<vector<double>> &return_div, int Nx, int Ny, double spacing){
  for(int i=0;i<Ny;i++){
      for(int j=0;j<Nx;j++){
            return_div[i][j] = 0;
            return_div[i][j]+= (grid[i][get_periodic_index(j+1,Nx)][0] - grid[i][get_periodic_index(j-1,Nx)][0])/(2*spacing);
            return_div[i][j]+= (grid[get_periodic_index(i+1,Ny)][j][1] - grid[get_periodic_index(i-1,Ny)][j][1])/(2*spacing);
        }
      }
    return;
  }

void gradient(vector<vector<double>> &grid, vector<vector<vector<double>>> &return_grad, int Nx, int Ny, double spacing){
    for(int i=0;i<Ny;i++){
      for(int j=0;j<Nx;j++){
            return_grad[i][j][0] = (grid[i][get_periodic_index(j+1, Nx)] - grid[i][get_periodic_index(j-1, Nx)]) / (2 * spacing);
            return_grad[i][j][1] = (grid[get_periodic_index(i+1, Ny)][j] - grid[get_periodic_index(i-1, Ny)][j]) / (2 * spacing);
        }
      }
    return;
}

void laplacian(vector<vector<vector<double>>> &vec_3D, vector<vector<vector<double>>> &vec_3D_return, int Nx, int Ny, double spacing){
    for(int i=0;i<Ny;i++){
      for(int j=0;j<Nx;j++){
        for(int k=0;k<2;k++){
            vec_3D_return[i][j][k] = (vec_3D[get_periodic_index(i+1,Ny)][j][k]+vec_3D[get_periodic_index(i-1,Ny)][j][k] + vec_3D[i][get_periodic_index(j-1,Nx)][k]+vec_3D[i][get_periodic_index(j+1,Nx)][k] - 4*vec_3D[i][j][k])/(spacing*spacing);
        }
      }
    }
}

void laplacian_2D(vector<vector<double>> &vec_2D, vector<vector<double>> &vec_2D_return, int Nx, int Ny, double spacing){
    for(int i=0;i<Ny;i++){
      for(int j=0;j<Nx;j++){
        vec_2D_return[i][j] = (vec_2D[get_periodic_index(i+1,Ny)][j]+vec_2D[get_periodic_index(i-1,Ny)][j] + vec_2D[i][get_periodic_index(j-1,Nx)]+vec_2D[i][get_periodic_index(j+1,Nx)] - 4*vec_2D[i][j])/(spacing*spacing);
      }
    }
  }

void calculate_omegtau(vector<vector<vector<double>>> &tau,vector<vector<double>> &activity_field, vector<vector<vector<double>>> &omeg_tau,int Nx, int Ny){
    for(int i=0;i<Ny;i++){
      for(int j=0;j<Nx;j++){
        for(int k=0;k<2;k++){
          omeg_tau[i][j][k] = tau[i][j][k]*activity_field[i][j];
        }
      }
    }
}

void lambda_terms(vector<vector<vector<double>>> &vec_3D_return, vector<vector<vector<double>>> &tau,int Nx, int Ny, double spacing){
    vector<vector<double>> del_tau_xx(Ny, vector<double>(Nx, -1));
    vector<vector<double>> del_tau_yx(Ny, vector<double>(Nx, -1));
    vector<vector<double>> del_tau_xy(Ny, vector<double>(Nx, -1));
    vector<vector<double>> del_tau_yy(Ny, vector<double>(Nx, -1));
    for(int i=0;i<Ny;i++){
      for(int j=0;j<Nx;j++){
        del_tau_yy[i][j] = (tau[get_periodic_index(i+1,Ny)][j][1]-tau[get_periodic_index(i-1,Ny)][j][1])/(2*spacing);
        del_tau_yx[i][j] = (tau[i][get_periodic_index(j+1,Nx)][1]-tau[i][get_periodic_index(j-1,Nx)][1])/(2*spacing);
        del_tau_xx[i][j] = (tau[i][get_periodic_index(j+1,Nx)][0]-tau[i][get_periodic_index(j-1,Nx)][0])/(2*spacing);
        del_tau_xy[i][j] = (tau[get_periodic_index(i+1,Ny)][j][0]-tau[get_periodic_index(i-1,Ny)][j][0])/(2*spacing);
      }
    }
    for(int i=0;i<Ny;i++){
      for(int j=0;j<Nx;j++){
        vec_3D_return[i][j][0] = tau[i][j][0]*(del_tau_xx[i][j]+del_tau_yy[i][j]) + tau[i][j][1]*(del_tau_yx[i][j]-del_tau_xy[i][j]);
        vec_3D_return[i][j][1] = tau[i][j][0]*(del_tau_xy[i][j]-del_tau_yx[i][j]) + tau[i][j][1]*(del_tau_yy[i][j]+del_tau_xx[i][j]);
      }
    }
    return;
}

void update_rho(vector<vector<double>> &particle_density_t,vector<vector<double>> &particle_density_t1, vector<vector<vector<double>>> &omeg_tau, int Nx, int Ny, double spacing){

  vector<vector<vector<double>>> calc_3D(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
  gradient(particle_density_t,calc_3D,Nx,Ny,spacing);
  for(int i=0;i<Ny;i++){
    for(int j=0;j<Nx;j++){
      for(int k=0;k<2;k++){
        calc_3D[i][j][k] = omeg_tau[i][j][k]-D*calc_3D[i][j][k];
      }
    }
  }
  vector<vector<double>> calc_2D(Ny, vector<double>(Nx, 0));
  divergence(calc_3D, calc_2D,Nx,Ny,spacing);
  for(int i=0;i<Ny;i++){
    for(int j=0;j<Nx;j++){
      particle_density_t1[i][j] = -calc_2D[i][j]*delta_t + particle_density_t[i][j];
    }
  }
}

void update_rho_backward(vector<vector<double>> &particle_density_t,vector<vector<double>> &particle_density_t1, vector<vector<vector<double>>> &omeg_tau, int Nx, int Ny, double spacing)
{
    auto idx = [&](int i, int j) { return get_periodic_index(i,Ny) * Nx + get_periodic_index(j,Nx);};

    const int N = Nx * Ny;
    SparseMatrix<double> A(N, N);
    vector<Triplet<double>> triplets;

    double coeff_center = 1.0 + 4.0 * delta_t * D / (spacing * spacing);
    double coeff_neighbor = -delta_t * D / (spacing * spacing);

    for (int i = 0; i < Ny; ++i) {
        for (int j = 0; j < Nx; ++j) {
            int k = idx(i, j);
            triplets.emplace_back(k, k, coeff_center);
            triplets.emplace_back(k, idx(i - 1, j), coeff_neighbor);
            triplets.emplace_back(k, idx(i + 1, j), coeff_neighbor);
            triplets.emplace_back(k, idx(i, j - 1), coeff_neighbor);
            triplets.emplace_back(k, idx(i, j + 1), coeff_neighbor);
        }
    }
    A.setFromTriplets(triplets.begin(), triplets.end());

  
    vector<vector<double>> div_omeg_tau(Ny, vector<double>(Nx, 0));
    divergence(omeg_tau, div_omeg_tau,Nx,Ny,spacing);

    VectorXd rhs(N);
    for (int i = 0; i < Ny; ++i){
        for (int j = 0; j < Nx; ++j){
            rhs[idx(i, j)] = particle_density_t[i][j] - delta_t * div_omeg_tau[i][j];
        }
      }

    ConjugateGradient<SparseMatrix<double>, Eigen::Lower | Eigen::Upper> solver;
    solver.compute(A);

    if (solver.info() != Eigen::Success) {
        std::cerr << "Matrix factorization failed!" << std::endl;
        return;
    }

    VectorXd x = solver.solve(rhs);

    if (solver.info() != Eigen::Success) {
        std::cerr << "Solver failed to converge!" << std::endl;
        return;
    }

    for (int i = 0; i < Ny; ++i)
        for (int j = 0; j < Nx; ++j)
            particle_density_t1[i][j] = x[idx(i, j)];
}


void update_tau(vector<vector<vector<double>>> &lambda_terms, vector<vector<vector<double>>> &tau_t,vector<vector<vector<double>>> &tau_t1, vector<vector<double>> &particle_density_t, vector<vector<double>> &activity_field, int Nx, int Ny, double spacing){
  vector<vector<vector<double>>> omeg_rho_3d(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
  vector<vector<vector<double>>> laplacian_grid(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
  vector<vector<double>> omeg_rho_2d(Ny, vector<double>(Nx, -1));
  for(int i=0;i<Ny;i++){
    for(int j=0;j<Nx;j++){
      omeg_rho_2d[i][j] = activity_field[i][j]*particle_density_t[i][j];
    }
  }
  gradient(omeg_rho_2d,omeg_rho_3d,Nx,Ny,spacing);
  laplacian(tau_t,laplacian_grid,Nx,Ny,spacing);
  for(int i=0;i<Ny;i++){
    for(int j=0;j<Nx;j++){
      for(int k=0;k<2;k++){
        double to_add = 0;
        to_add = -(1-particle_density_t[i][j]/rho_c+(1+particle_density_t[i][j]/rho_c)*((tau_t[i][j][0]*tau_t[i][j][0]+tau_t[i][j][1]*tau_t[i][j][1]))/pow(particle_density_t[i][j],2))*tau_t[i][j][k];
        to_add-=omeg_rho_3d[i][j][k];
        to_add+=K*laplacian_grid[i][j][k];
        to_add+=lambda_constant*lambda_terms[i][j][k];
        tau_t1[i][j][k] = to_add*delta_t + tau_t[i][j][k];
      }
    }
  }
  return;
}

void update_tau_backward(vector<vector<vector<double>>> &lambda_terms, vector<vector<vector<double>>> &tau_t,vector<vector<vector<double>>> &tau_t1, vector<vector<double>> &particle_density_t, vector<vector<double>> &activity_field, int Nx, int Ny, double spacing){

  auto idx = [&](int i, int j) { return get_periodic_index(i,Ny) * Nx + get_periodic_index(j,Nx);};
  
  int N = Nx * Ny;
  
  vector<vector<vector<double>>> omeg_rho_3d(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
  vector<vector<double>> omeg_rho_2d(Ny, vector<double>(Nx, -1));
  
  for(int i=0; i<Ny; i++){
    for(int j=0; j<Nx; j++){
      omeg_rho_2d[i][j] = activity_field[i][j]*particle_density_t[i][j];
    }
  }
  gradient(omeg_rho_2d, omeg_rho_3d, Nx, Ny, spacing);
  

  double h2 = spacing * spacing;
  double diag_coef = 1.0 + 4.0 * delta_t * K / h2;
  double off_coef = -delta_t * K / h2;
  
  vector<Triplet<double>> triplets;
  triplets.reserve(5 * N);
  

  for (int i = 0; i < Ny; ++i) {
      for (int j = 0; j < Nx; ++j) {
          int k = idx(i, j);
          triplets.emplace_back(k, k, diag_coef);
          triplets.emplace_back(k, idx(i - 1, j), off_coef);
          triplets.emplace_back(k, idx(i + 1, j), off_coef);
          triplets.emplace_back(k, idx(i, j - 1), off_coef);
          triplets.emplace_back(k, idx(i, j + 1), off_coef);
      }
  }

  SparseMatrix<double> A(N, N);
  A.setFromTriplets(triplets.begin(), triplets.end());
  
  ConjugateGradient<SparseMatrix<double>, Eigen::Lower | Eigen::Upper> solver;
  solver.compute(A);
  
  if(solver.info() != Eigen::Success){
    cerr << "Matrix decomposition failed in update_tau!" << endl;
    return;
  }
  
  for(int k=0; k<2; k++){
    VectorXd rhs(N);
    for(int i=0; i<Ny; i++){
      for(int j=0; j<Nx; j++){
        int idx = i*Nx + j;
        double tau_mag_sq = tau_t[i][j][0]*tau_t[i][j][0] + 
                           tau_t[i][j][1]*tau_t[i][j][1];
        
        double nonlinear_coef = -(1.0 - particle_density_t[i][j]/rho_c + 
                                  (1.0 + particle_density_t[i][j]/rho_c) * 
                                  tau_mag_sq / pow(particle_density_t[i][j], 2));
        
        double nonlinear_term = nonlinear_coef * tau_t[i][j][k];
        
        rhs[idx] = tau_t[i][j][k] + delta_t * (
          nonlinear_term - 
          omeg_rho_3d[i][j][k] + 
          lambda_constant * lambda_terms[i][j][k]
        );
      }
    }
    
    VectorXd tau_new_vec = solver.solve(rhs);
    
    if(solver.info() != Eigen::Success){
      cerr << "Solving failed in update_tau for component " << k << endl;
      cerr << "Iterations: " << solver.iterations() << endl;
      cerr << "Error: " << solver.error() << endl;
      return;
    }
    
    for(int i=0; i<Ny; i++){
      for(int j=0; j<Nx; j++){
        int idx = i*Nx + j;
        tau_t1[i][j][k] = tau_new_vec[idx];
      }
    }
  }
}


void integrate( vector<vector<double>> &particle_density_t,vector<vector<double>> &particle_density_t1, vector<vector<double>> &activity_field, vector<vector<vector<double>>> &tau_t, vector<vector<vector<double>>> &tau_t1, vector<vector<vector<double>>> &omeg_tau,vector<vector<vector<double>>> &lambda_terms, int Nx, int Ny, double spacing){

  // update_rho_backward(particle_density_t,particle_density_t1,omeg_tau,Nx,Ny,spacing);
  update_rho(particle_density_t,particle_density_t1,omeg_tau,Nx,Ny,spacing);
  // update_tau_backward(lambda_terms,tau_t,tau_t1,particle_density_t,activity_field,Nx,Ny,spacing);
  update_tau(lambda_terms,tau_t,tau_t1,particle_density_t,activity_field,Nx,Ny,spacing);

}

void update_eta(vector<vector<double>> &rho,vector<vector<double>> &rho_target,vector<vector<double>> &eta_old, vector<vector<double>> &eta_new,vector<vector<double>> &activity_field,vector<vector<vector<double>>> &nu,
  vector<vector<vector<double>>> &tau, int Nx, int Ny, double spacing){
  vector<vector<double>> result(Ny, vector<double>(Nx, -1));
  vector<vector<double>> eta_lap(Ny, vector<double>(Nx, -1));
  laplacian_2D(eta_old, eta_lap, Nx,Ny,spacing);
  vector<vector<double>> nu_div(Ny, vector<double>(Nx, -1));
  divergence(nu,nu_div,Nx,Ny,spacing);
  for(int i=0;i<Ny;i++){
    for(int j=0;j<Nx;j++){
      result[i][j] = C*(rho[i][j] - rho_target[i][j]);
      result[i][j]-= eta_lap[i][j];
      result[i][j]-= activity_field[i][j]*activity_field[i][j]*nu_div[i][j];
      result[i][j]+= ((-1/rho_c)+(-2/pow(rho[i][j],3)-1/(pow(rho[i][j],2)*rho_c))*((tau[i][j][0]*tau[i][j][0]+tau[i][j][1]*tau[i][j][1])))*(nu[i][j][0]*tau[i][j][0] + nu[i][j][1]*tau[i][j][1]);
      eta_new[i][j] = eta_old[i][j]-delta_t*result[i][j];
    } 
  }
}

void update_eta_implicit(vector<vector<double>> &rho,vector<vector<double>> &rho_target,vector<vector<double>> &eta_old, vector<vector<double>> &eta_new,vector<vector<double>> &activity_field,vector<vector<vector<double>>> &nu,
  vector<vector<vector<double>>> &tau, int Nx, int Ny, double spacing){


  auto idx = [&](int i, int j) { return get_periodic_index(i,Ny) * Nx + get_periodic_index(j,Nx);};
  const int N = Nx * Ny;
  SparseMatrix<double> A(N, N);
  vector<Triplet<double>> triplets;
  double coeff_center = 1.0 + 4.0 * delta_t * D / (spacing * spacing);
  double coeff_neighbor = -delta_t * D / (spacing * spacing);

  for (int i = 0; i < Ny; ++i) {
    for (int j = 0; j < Nx; ++j) {
            int k = idx(i, j);
            triplets.emplace_back(k, k, coeff_center);
            triplets.emplace_back(k, idx(i - 1, j), coeff_neighbor);
            triplets.emplace_back(k, idx(i + 1, j), coeff_neighbor);
            triplets.emplace_back(k, idx(i, j - 1), coeff_neighbor);
            triplets.emplace_back(k, idx(i, j + 1), coeff_neighbor);
        }
    }
    A.setFromTriplets(triplets.begin(), triplets.end());


  // vector<vector<double>> eta_lap(Ny, vector<double>(Nx, -1));
  // laplacian_2D(eta_old, eta_lap, Nx,Ny,spacing);
  vector<vector<double>> nu_div(Ny, vector<double>(Nx, -1));
  divergence(nu,nu_div,Nx,Ny,spacing);
  VectorXd rhs(N);

  for (int i = 0; i < Ny; ++i){
    for (int j = 0; j < Nx; ++j){
        rhs[idx(i, j)] = +C*(rho[i][j] - rho_target[i][j]);
        rhs[idx(i, j)] -= activity_field[i][j]*activity_field[i][j]*nu_div[i][j];
        rhs[idx(i, j)] += ((-1/rho_c)+(-2/pow(rho[i][j],3)-1/(pow(rho[i][j],2)*rho_c))*((tau[i][j][0]*tau[i][j][0]+tau[i][j][1]*tau[i][j][1])))*(nu[i][j][0]*tau[i][j][0] + nu[i][j][1]*tau[i][j][1]);
        rhs[idx(i,j)]*= -delta_t;
        rhs[idx(i,j)]+=eta_old[i][j];
      }
  }
  ConjugateGradient<SparseMatrix<double>, Eigen::Lower | Eigen::Upper> solver;
  solver.compute(A);

  if (solver.info() != Eigen::Success) {
      std::cerr << "Matrix factorization failed!" << std::endl;
      return;
  }

  VectorXd x = solver.solve(rhs);

  if (solver.info() != Eigen::Success) {
      std::cerr << "Solver failed to converge!" << std::endl;
      return;
  }

  for (int i = 0; i < Ny; ++i){
    for (int j = 0; j < Nx; ++j){
        eta_new[i][j] = x[idx(i, j)];
    }
  }
}

void lambda_terms_bck(vector<vector<vector<double>>> &nu,vector<vector<vector<double>>> &vec3D_return,vector<vector<vector<double>>> &tau,int Nx, int Ny, double spacing){
  vector<vector<double>> del_tau_yy(Ny, vector<double>(Nx, -1));
  vector<vector<double>> del_nu_xx(Ny, vector<double>(Nx, -1));
  vector<vector<double>> del_nu_xy(Ny, vector<double>(Nx, -1));
  vector<vector<double>> del_nu_yy(Ny, vector<double>(Nx, -1));
  vector<vector<double>> del_tau_yx(Ny, vector<double>(Nx, -1));
  vector<vector<double>> del_tau_xx(Ny, vector<double>(Nx, -1));
  vector<vector<double>> del_nu_yx(Ny, vector<double>(Nx, -1));
  vector<vector<double>> del_tau_xy(Ny, vector<double>(Nx, -1));
  for(int i=0;i<Ny;i++){
    for(int j=0;j<Nx;j++){
        del_tau_yy[i][j] = (tau[get_periodic_index(i+1,Ny)][j][1]-tau[get_periodic_index(i-1,Ny)][j][1])/(2*spacing);
        del_tau_yx[i][j] = (tau[i][get_periodic_index(j+1,Nx)][1]-tau[i][get_periodic_index(j-1,Nx)][1])/(2*spacing);
        del_tau_xx[i][j] = (tau[i][get_periodic_index(j+1,Nx)][0]-tau[i][get_periodic_index(j-1,Nx)][0])/(2*spacing);
        del_tau_xy[i][j] = (tau[get_periodic_index(i+1,Ny)][j][0]-tau[get_periodic_index(i-1,Ny)][j][0])/(2*spacing);
        del_nu_xx[i][j] = (nu[i][get_periodic_index(j+1,Nx)][0]-nu[i][get_periodic_index(j-1,Nx)][0])/(2*spacing);
        del_nu_yy[i][j] = (nu[get_periodic_index(i+1,Ny)][j][1]-nu[get_periodic_index(i-1,Ny)][j][1])/(2*spacing);
        del_nu_xy[i][j] = (nu[get_periodic_index(i+1,Ny)][j][0]-nu[get_periodic_index(i-1,Ny)][j][0])/(2*spacing);
        del_nu_yx[i][j] = (nu[i][get_periodic_index(j+1,Nx)][1]-nu[i][get_periodic_index(j-1,Nx)][1])/(2*spacing);
    }
  }
  for(int i=0;i<Ny;i++){
    for(int j=0;j<Nx;j++){
      vec3D_return[i][j][0] = -tau[i][j][0]*del_nu_yy[i][j]+2*nu[i][j][0]*del_tau_yy[i][j]-tau[i][j][0]*del_nu_xx[i][j] - tau[i][j][1]*del_nu_yx[i][j]+tau[i][j][1]*del_nu_xy[i][j]-2*nu[i][j][1]*del_tau_yx[i][j];
      vec3D_return[i][j][1] = -tau[i][j][1]*del_nu_xx[i][j]-tau[i][j][1]*del_nu_yy[i][j]+2*nu[i][j][1]*del_tau_xx[i][j]-2*nu[i][j][0]*del_tau_xy[i][j]-tau[i][j][0]*del_nu_xy[i][j]+tau[i][j][0]*del_nu_yx[i][j];
    }
  }
  return;
}

double cost_function(vector<vector<vector<double>>> &tau,vector<vector<vector<double>>> &tau_target, vector<vector<double>> &activity_field,vector<vector<double>> &activity_field_baseline,vector<vector<double>> &rho,vector<vector<double>> &rho_target, int Nx, int Ny, double spacing){
  double cost = 0;
  vector<vector<double>> act_field_sq(Ny, vector<double>(Nx, -1));
  for(int i=0;i<Ny;i++){
    for(int j=0;j<Nx;j++){
      act_field_sq[i][j] = activity_field[i][j]*activity_field[i][j];
    }
  }
  vector<vector<vector<double>>> act_field_sq_grad(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
  gradient(act_field_sq,act_field_sq_grad,Nx,Ny,spacing);
  for(int i=0;i<Ny;i++){
    for(int j=0;j<Nx;j++){
      cost+=(pow(pow(activity_field[i][j],2)-pow(activity_field_baseline[i][j],2),2))*A/2;
      cost+=(pow(rho_target[i][j]-rho[i][j],2))*C/2;
      cost+=(pow(tau_target[i][j][0]-tau[i][j][0],2)+pow(tau_target[i][j][1]-tau[i][j][1],2))*D/2;
      cost+=(act_field_sq_grad[i][j][0]*act_field_sq_grad[i][j][0] + act_field_sq_grad[i][j][1]*act_field_sq_grad[i][j][1])*B/2;
    }
  }
  return cost;
}

void update_nu(vector<vector<vector<double>>> &tau,vector<vector<double>> &activity_field,vector<vector<vector<double>>> &tau_target, vector<vector<vector<double>>> &nu_old, vector<vector<vector<double>>> &nu_new,vector<vector<double>> &eta, vector<vector<double>> &rho, int Nx, int Ny, double spacing){
  vector<vector<vector<double>>> result(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
  vector<vector<vector<double>>> nu_lap(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
  laplacian(nu_old,nu_lap,Nx,Ny,spacing);
  vector<vector<vector<double>>> eta_grad(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
  gradient(eta,eta_grad,Nx,Ny,spacing);
  vector<vector<vector<double>>> lambda_terms_vector(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
  lambda_terms_bck(nu_old,lambda_terms_vector,tau,Nx,Ny,spacing);
  for(int i=0;i<Ny;i++){
    for(int j=0;j<Nx;j++){
      for(int k=0;k<2;k++){
          result[i][j][k] = E*(tau[i][j][k]-tau_target[i][j][k]);
          result[i][j][k]+= (1-rho[i][j]/rho_c+(1+rho[i][j]/rho_c)*((tau[i][j][0]*tau[i][j][0]+tau[i][j][1]*tau[i][j][1]))/pow(rho[i][j],2))*nu_old[i][j][k];
          result[i][j][k]+= 2*((1+rho[i][j]/rho_c)/pow(rho[i][j],2))*tau[i][j][k]*(tau[i][j][0]*nu_old[i][j][0]+tau[i][j][1]*nu_old[i][j][1]);
          result[i][j][k]-=lambda_constant*(lambda_terms_vector[i][j][k]);
          result[i][j][k]-=nu_lap[i][j][k];
          result[i][j][k]-=pow(activity_field[i][j],2)*(eta_grad[i][j][k]);
          nu_new[i][j][k] = nu_old[i][j][k]-delta_t*result[i][j][k];
      }
    }
  }
  return;
}

void update_nu_implicit(vector<vector<vector<double>>> &tau,vector<vector<double>> &activity_field,vector<vector<vector<double>>> &tau_target, vector<vector<vector<double>>> &nu_old, vector<vector<vector<double>>> &nu_new,vector<vector<double>> &eta, vector<vector<double>> &rho, int Nx, int Ny, double spacing){

  auto idx = [&](int i, int j) { return get_periodic_index(i,Ny) * Nx + get_periodic_index(j,Nx);};
  const int N = Nx * Ny;
  SparseMatrix<double> A(N, N);
  vector<Triplet<double>> triplets;
  double coeff_center = 1.0 + 4.0 * delta_t / (spacing * spacing);
  double coeff_neighbor = -delta_t  / (spacing * spacing);

  for (int i = 0; i < Ny; ++i) {
    for (int j = 0; j < Nx; ++j) {
            int k = idx(i, j);
            triplets.emplace_back(k, k, coeff_center);
            triplets.emplace_back(k, idx(i - 1, j), coeff_neighbor);
            triplets.emplace_back(k, idx(i + 1, j), coeff_neighbor);
            triplets.emplace_back(k, idx(i, j - 1), coeff_neighbor);
            triplets.emplace_back(k, idx(i, j + 1), coeff_neighbor);
        }
    }
  A.setFromTriplets(triplets.begin(), triplets.end());
  
  ConjugateGradient<SparseMatrix<double>, Eigen::Lower | Eigen::Upper> solver;
  solver.compute(A);


  // vector<vector<vector<double>>> nu_lap(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
  // laplacian(nu_old,nu_lap,Nx,Ny,spacing);
  vector<vector<vector<double>>> eta_grad(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
  gradient(eta,eta_grad,Nx,Ny,spacing);
  vector<vector<vector<double>>> lambda_terms_vector(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
  lambda_terms_bck(nu_old,lambda_terms_vector,tau,Nx,Ny,spacing);

  for(int k=0; k<2; k++){
    VectorXd rhs(N);
    for (int i = 0; i < Ny; ++i){
      for (int j = 0; j < Nx; ++j){
          rhs[idx(i, j)] = +E*(tau[i][j][k]-tau_target[i][j][k]);
          rhs[idx(i, j)] += (1-rho[i][j]/rho_c+(1+rho[i][j]/rho_c)*((tau[i][j][0]*tau[i][j][0]+tau[i][j][1]*tau[i][j][1]))/pow(rho[i][j],2))*nu_old[i][j][k];
          rhs[idx(i, j)] += 2*((1+rho[i][j]/rho_c)/pow(rho[i][j],2))*tau[i][j][k]*(tau[i][j][0]*nu_old[i][j][0]+tau[i][j][1]*nu_old[i][j][1]);
          rhs[idx(i, j)]-=lambda_constant*(lambda_terms_vector[i][j][k]);
          rhs[idx(i, j)]-=pow(activity_field[i][j],2)*(eta_grad[i][j][k]);
          rhs[idx(i,j)]*= -delta_t;
          rhs[idx(i,j)]+=nu_old[i][j][k];
        }
    }
    VectorXd nu_new_vec = solver.solve(rhs);
    
    if(solver.info() != Eigen::Success){
      cerr << "Solving failed in update_nu for component " << k << endl;
      cerr << "Error: " << solver.error() << endl;
      return;
    }
    
    for(int i=0; i<Ny; i++){
      for(int j=0; j<Nx; j++){
        int grid_idx = i*Nx + j;
        nu_new[i][j][k] = nu_new_vec[grid_idx];
      }
    }
    
  }
  return;
}

void gradient_weight(vector<vector<double>> &gradients,vector<vector<double>> &activity_field, vector<vector<double>> &activity_field_baseline,vector<vector<double>> &rho, vector<vector<double>> &eta, vector<vector<vector<double>>> &tau, vector<vector<vector<double>>> &nu, int Nx, int Ny, double spacing){

  vector<vector<vector<double>>> eta_grad(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
  vector<vector<vector<double>>> rho_grad(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
  vector<vector<double>> activity_field_laplacian(Ny, vector<double>(Nx, -1));
  laplacian_2D(activity_field,activity_field_laplacian,Nx,Ny,spacing);
  gradient(rho,rho_grad,Nx,Ny,spacing);
  gradient(eta, eta_grad,Nx, Ny, spacing);
  double weight = 0;
  for(int i=0;i<Ny;i++){
    for(int j=0;j<Nx;j++){
      gradients[i][j] = 2*A*activity_field[i][j]*(activity_field[i][j] - activity_field_baseline[i][j])-2*B*activity_field[i][j]*activity_field_laplacian[i][j]-2*activity_field[i][j]*(tau[i][j][0]*eta_grad[i][j][0]+tau[i][j][1]*eta_grad[i][j][1])-2*activity_field[i][j]*(nu[i][j][0]*rho_grad[i][j][0]+nu[i][j][1]*rho_grad[i][j][1]);
    }
  }

}




int main(){
  int Nx = 60;
  int Ny = 60;
  double spacing = 2;
  static mt19937 g(time(nullptr));  
  double mean_dist = 0;
  double stddev = 0.001;
  normal_distribution<double> dist(mean_dist, stddev);

  vector<vector<vector<double>>> activity_field_times(timesteps,vector<vector<double>>(Ny, vector<double>(Nx, -1)));
  vector<vector<double>> activity_field_baseline(Ny, vector<double>(Nx, -1));
  update_activity(activity_field_baseline,rho_c,Nx,Ny);
  vector<vector<double>> activity_field(Ny, vector<double>(Nx, -1));
  for(int i=0;i<Ny;i++){
    for(int j=0;j<Nx;j++){
        activity_field[i][j] = activity_field_baseline[i][j];
      }
    }
  for(int t=0;t<timesteps;t++){
      activity_field_times[t] = activity_field;
  }

  string folder_path = "C:\\PhD\\Work\\Trial2_Control\\";
  string command = "mkdir "+folder_path;
  try{
      int result = std::system(command.c_str());
  }
  catch (exception &e){
      cout << e.what() << "\n";
      throw e;
  }

  double learning_rate = 0.01;

  for(int iter = 0;iter<num_iters;iter++){
    if(iter%4==1){
      learning_rate/=2;
    }
    cout << "Iteration Number: " << iter << '\n';
    vector<vector<double>> particle_density_t(Ny, vector<double>(Nx, -1));
    vector<vector<vector<double>>> particle_density_times(timesteps,vector<vector<double>>(Ny, vector<double>(Nx, -1)));
  
    // initialize_grid(particle_density_t,density_mean,Nx,Ny);
    vector<vector<vector<double>>> polarization_field(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
    // initialize_3Dgrid(polarization_field,Nx,Ny);

    vector<vector<vector<double>>> tau_t(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
    vector<vector<vector<vector<double>>>> tau_times(timesteps,vector<vector<vector<double>>>(Ny,vector<vector<double>>(Nx, vector<double>(2, -1))));
    // tau_calculate(tau_t,polarization_field,particle_density_t,Nx,Ny);
    readvectors("C:\\PhD\\Work\\Aster1\\",particle_density_t,tau_t,19999,Ny,Nx);

    double rho_eps = 1e-6;
    for(int i=0;i<Ny;i++){
      for(int j=0;j<Nx;j++){
        double r = max(particle_density_t[i][j], rho_eps);
        for(int k=0;k<2;k++){
          polarization_field[i][j][k] = tau_t[i][j][k]/r;
        }
      }
    }
    vector<vector<vector<double>>> omeg_tau(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
    // calculate_omegtau(tau_t,activity_field_times[0],omeg_tau,Nx,Ny);
    vector<vector<vector<double>>> lambda_etc(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
    // lambda_terms(lambda_etc,tau_t,Nx,Ny,spacing);

    vector<vector<double>> particle_density_t1(Ny, vector<double>(Nx, -1));
    vector<vector<vector<double>>> tau_t1(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));

    for(int t=0;t<timesteps;t++){

      particle_density_times[t] = particle_density_t;
      tau_times[t] = tau_t;

      calculate_omegtau(tau_t,activity_field_times[t],omeg_tau,Nx,Ny);
      lambda_terms(lambda_etc,tau_t,Nx,Ny,spacing);

      if(iter==num_iters-1 && t%10==0){
          savevectors(particle_density_t,polarization_field,tau_t,folder_path,t);
      }

      if(t%1000==0 || t==timesteps-1){
        double sum = 0;
        if(t%10==0){
          for(int i=0;i<Ny;i++){
            for(int j=0;j<Nx;j++){
              sum+=particle_density_t[i][j];
            }
          }
          cout << sum << endl;
        }
        
        savevectors(particle_density_t,polarization_field,tau_t,folder_path,t);
      }
      
      integrate(particle_density_t,particle_density_t1,activity_field_times[t],tau_t,tau_t1,omeg_tau,lambda_etc,Nx,Ny,spacing);

      for(int i=0;i<Ny;i++){
        for(int j=0;j<Nx;j++){
          particle_density_t[i][j] = particle_density_t1[i][j];
        }
      }

      for(int i=0;i<Ny;i++){
        for(int j=0;j<Nx;j++){
          for(int k=0;k<2;k++){
            tau_t[i][j][k] = tau_t1[i][j][k];
          }
        }
      }
      double rho_eps = 1e-6;
      for(int i=0;i<Ny;i++){
        for(int j=0;j<Nx;j++){
          double r = max(particle_density_t[i][j], rho_eps);
          for(int k=0;k<2;k++){
            polarization_field[i][j][k] = tau_t[i][j][k]/r;
          }
        }
      }
    }

    cout << "Forward propagation completed" << '\n';

    vector<vector<vector<double>>> nu_t(Ny,vector<vector<double>>(Nx, vector<double>(2, 0)));
    vector<vector<vector<double>>> nu_t1(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
    vector<vector<double>> eta_t(Ny, vector<double>(Nx, 0));
    vector<vector<double>> eta_t1(Ny, vector<double>(Nx, -1));
    vector<vector<double>> rho_target(Ny, vector<double>(Nx, -1));
    vector<vector<vector<double>>> tau_target(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
    readvectors("C:\\PhD\\Work\\Aster1\\", rho_target, tau_target, 19999,Ny,Nx);
    shift_2d_array(rho_target);
    shift_3d_array(tau_target);
    savevectors(rho_target,tau_target,tau_target,"C:\\PhD\\Work\\Aster1\\", 101);

    vector<vector<double>> gradients(Ny, vector<double>(Nx));


    for(int t=timesteps-1;t>=1;t--){
      lambda_terms_bck(nu_t,lambda_etc,tau_times[t],Nx,Ny,spacing);
      update_eta(particle_density_times[t],rho_target, eta_t,eta_t1,activity_field_times[t] ,nu_t,tau_times[t],Nx,Ny,spacing);
      update_nu(tau_times[t],activity_field_times[t],tau_target, nu_t,nu_t1,eta_t,particle_density_times[t],Nx,Ny,spacing);

      if(t%500==0 || t==timesteps-1){
        double sum = 0;
        if(t%10==0){
          for(int i=0;i<Ny;i++){
            for(int j=0;j<Nx;j++){
              sum+=eta_t[i][j];
            }
          }
          cout << sum << endl;
        }
      }
      for(int i=0;i<Ny;i++){
        for(int j=0;j<Nx;j++){
          for(int k=0;k<2;k++){
            nu_t[i][j][k] = nu_t1[i][j][k];
          }
        }
      }
      for(int i=0;i<Ny;i++){
        for(int j=0;j<Nx;j++){
          eta_t[i][j] = eta_t1[i][j];
        }
      }
      gradient_weight(gradients,activity_field_times[t] ,activity_field_baseline,particle_density_times[t],eta_t, tau_times[t],nu_t,Nx,Ny,spacing);
      // if(t%100==0){
      //   double sum_grads = 0;
      //   for(int i=0;i<Ny;i++){
      //     for(int j=0;j<Nx;j++){
      //       sum_grads += gradients[i][j];
      //     }
      //   }
      //   cout << sum_grads/(60*60) << '\n';
      // }
      for(int i=0;i<Ny;i++){
        for(int j=0;j<Nx;j++){
          activity_field_times[t][i][j]-= learning_rate*gradients[i][j];
        }
      }
      if(t==1){
        for(int i=0;i<3;i++){
          for(int j=0;j<3;j++){
            cout << activity_field_times[t][i][j] << ' ';
          }
        }
        cout << '\n';
      }

      if(iter==num_iters-1){
        save_activity_field(activity_field_times,folder_path,iter);
      }

    }
  }
    
  return 0;

}