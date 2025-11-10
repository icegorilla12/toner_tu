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

using namespace std;

constexpr int Nx = 50;
constexpr int Ny = 50;
constexpr double spacing = 1;
constexpr int timesteps = 200;
constexpr double PI = 3.14159265358979323846;
constexpr double C = 100;
constexpr double A = 1;
constexpr double B = 1;
constexpr double D = 1;
constexpr double E = 1;
constexpr double activity_constant = 1;
constexpr double delta_t = 0.01;
constexpr double polarization_ini_mag = 0.001;



int get_periodic_index(int i, int size) {
    return (i % size + size) % size; 
}

void update_activity(vector<vector<vector<double>>> &activity_field, int size){
  static mt19937 g(time(nullptr)); 
  // static mt19937 g(42);  
  double mean_dist = 0;
  double stddev = 0.01;
  normal_distribution<double> dist(mean_dist, stddev);
  for(int k=0;k<size;k++){
    for(int i=0;i<Ny;i++){
      for(int j=0;j<Nx;j++){
          activity_field[k][i][j] = activity_constant + dist(g);
          // activity_field[k][i][j] = activity_constant;
      }
    }
  }
  return;
}

void shift_2d_array(vector<vector<double>>& arr, int shiftx, int shifty) {

    const int Ny = arr.size();    
    const int Nx = arr[0].size(); 
    vector<vector<double>> temp_arr = arr;
    int dy = shifty; 
    int dx = shiftx; 
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

void shift_3d_array(vector<vector<vector<double>>>& arr, int shiftx, int shifty) {
    if (arr.empty() || arr[0].empty() || arr[0][0].empty()) {
        cerr << "Error: 3D Array is empty or malformed." << endl;
        return;
    }

    const int Nz = arr.size();     
    const int Ny = arr[0].size();  
    const int Nx = arr[0][0].size(); 

    vector<vector<vector<double>>> temp_arr = arr;

    int dy = shifty; 
    int dx = shiftx; 
        
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


void savevectors(vector<vector<double>> &particle_density, vector<vector<vector<double>>> &polarization_field, string folder_path, int idx){
    string particle_density_path = "particle_density_"+to_string(idx)+".bin";
    string polarization_field_path = "polarization_field_"+to_string(idx)+".bin";
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
}

void readvectors(string folder_path, vector<vector<double>> &particle_density, vector<vector<double>> &activity_field, vector<vector<vector<double>>> &tau, int idx){
    ifstream inFile(folder_path+"particle_density_"+to_string(idx)+".bin", std::ios::binary);
    for (int i = 0; i < Ny; ++i) {
        inFile.read(reinterpret_cast<char*>(particle_density[i].data()), Nx * sizeof(double));
    }
    inFile.close();

    inFile.open(folder_path+"polarization_field_"+to_string(idx)+".bin", std::ios::binary);

    for (int i = 0; i < Ny; ++i) {
        for (int j = 0; j < Nx; ++j) {
            inFile.read(reinterpret_cast<char*>(tau[i][j].data()), 2 * sizeof(double));
        }
    }
    inFile.close();

    inFile.open(folder_path+"activity_field_"+to_string(0)+".bin", std::ios::binary);

    for (int i = 0; i < Ny; ++i) {
        inFile.read(reinterpret_cast<char*>(activity_field[i].data()), Nx * sizeof(double));
    }
    inFile.close();

}

void save_activity_field(vector<vector<vector<double>>> &activity_field, string folder_path, int iter){
    string activity_field_path = "activity_field_final_"+to_string(iter)+".bin";

    ofstream outFile(folder_path+activity_field_path, ios::binary);
    for (const auto &plane : activity_field) {
        for (const auto &row : plane) {
            outFile.write(reinterpret_cast<const char *>(row.data()), row.size() * sizeof(double));
        }
    }
    outFile.close();
}


void divergence(vector<vector<vector<double>>> &grid, vector<vector<double>> &return_div){
  for(int i=0;i<Ny;i++){
      for(int j=0;j<Nx;j++){
            return_div[i][j] = 0;
            return_div[i][j]+= (grid[i][get_periodic_index(j+1,Nx)][0] - grid[i][get_periodic_index(j-1,Nx)][0])/(2*spacing);
            return_div[i][j]+= (grid[get_periodic_index(i+1,Ny)][j][1] - grid[get_periodic_index(i-1,Ny)][j][1])/(2*spacing);
        }
      }
    return;
  }

void gradient(vector<vector<double>> &grid, vector<vector<vector<double>>> &return_grad){
    for(int i=0;i<Ny;i++){
      for(int j=0;j<Nx;j++){
            return_grad[i][j][0] = (grid[i][get_periodic_index(j+1, Nx)] - grid[i][get_periodic_index(j-1, Nx)]) / (2 * spacing);
            return_grad[i][j][1] = (grid[get_periodic_index(i+1, Ny)][j] - grid[get_periodic_index(i-1, Ny)][j]) / (2 * spacing);
        }
      }
    return;
}

void laplacian(vector<vector<vector<double>>> &vec_3D, vector<vector<vector<double>>> &vec_3D_return){
    for(int i=0;i<Ny;i++){
      for(int j=0;j<Nx;j++){
        for(int k=0;k<2;k++){
            vec_3D_return[i][j][k] = (vec_3D[get_periodic_index(i+1,Ny)][j][k]+vec_3D[get_periodic_index(i-1,Ny)][j][k] + vec_3D[i][get_periodic_index(j-1,Nx)][k]+vec_3D[i][get_periodic_index(j+1,Nx)][k] - 4*vec_3D[i][j][k])/(spacing*spacing);
        }
      }
    }
}

void initialize_grid(vector<vector<double>> &grid, double mean){
    static mt19937 g(time(nullptr));  
    // static mt19937 g(42); 
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

void initialize_3Dgrid(vector<vector<vector<double>>> &grid){
  random_device rd;  
  mt19937 g(time(nullptr));   
  // mt19937 g(42); 
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


void laplacian_2D(vector<vector<double>> &vec_2D, vector<vector<double>> &vec_2D_return){
  for(int i=0;i<Ny;i++){
    for(int j=0;j<Nx;j++){
      vec_2D_return[i][j] = (vec_2D[get_periodic_index(i+1,Ny)][j]+vec_2D[get_periodic_index(i-1,Ny)][j] + vec_2D[i][get_periodic_index(j-1,Nx)]+vec_2D[i][get_periodic_index(j+1,Nx)] - 4*vec_2D[i][j])/(spacing*spacing);
    }
  }
}

void update_density(vector<vector<double>> &rho_old, vector<vector<double>> &rho_new,vector<vector<vector<double>>> &polarization_field, vector<vector<double>> &activity_field){
  vector<vector<double>> result(Ny, vector<double>(Nx, -1));
  vector<vector<vector<double>>> wmT(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
  laplacian_2D(rho_old, result);
  for(int i=0;i<Ny;i++){
    for(int j=0;j<Nx;j++){
      for(int k=0;k<2;k++){
        wmT[i][j][k] = rho_old[i][j]*polarization_field[i][j][k]*activity_field[i][j];
      }
    }
  }
  vector<vector<double>> wmTdiv(Ny, vector<double>(Nx, -1));
  divergence(wmT,wmTdiv);
  for(int i=0;i<Ny;i++){
    for(int j=0;j<Nx;j++){
      result[i][j]-=wmTdiv[i][j];
    }
  }
  for(int i=0;i<Ny;i++){
    for(int j=0;j<Nx;j++){
      rho_new[i][j] = rho_old[i][j] + delta_t*result[i][j];
    }
  }
}

void non_linear(vector<vector<double>> &rho, vector<vector<vector<double>>> &polarization_field, vector<vector<vector<double>>> &result, vector<vector<double>> &activity_field ){
    
    vector<vector<vector<double>>> polarization_field_mod(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
    for(int i=0;i<Ny;i++){
      for(int j=0;j<Nx;j++){
        polarization_field_mod[i][j][0] = polarization_field[i][j][0]*activity_field[i][j];
        polarization_field_mod[i][j][1] = polarization_field[i][j][1]*activity_field[i][j];
      }
    }
  
    vector<vector<double>> del_t_xx(Ny, vector<double>(Nx, -1));
    vector<vector<double>> del_t_yx(Ny, vector<double>(Nx, -1));
    vector<vector<double>> del_t_xy(Ny, vector<double>(Nx, -1));
    vector<vector<double>> del_t_yy(Ny, vector<double>(Nx, -1));
    for(int i=0;i<Ny;i++){
      for(int j=0;j<Nx;j++){
        del_t_yy[i][j] = (polarization_field_mod[get_periodic_index(i+1,Ny)][j][1]-polarization_field_mod[get_periodic_index(i-1,Ny)][j][1])/(2*spacing);
        del_t_yx[i][j] = (polarization_field_mod[i][get_periodic_index(j+1,Nx)][1]-polarization_field_mod[i][get_periodic_index(j-1,Nx)][1])/(2*spacing);
        del_t_xx[i][j] = (polarization_field_mod[i][get_periodic_index(j+1,Nx)][0]-polarization_field_mod[i][get_periodic_index(j-1,Nx)][0])/(2*spacing);
        del_t_xy[i][j] = (polarization_field_mod[get_periodic_index(i+1,Ny)][j][0]-polarization_field_mod[get_periodic_index(i-1,Ny)][j][0])/(2*spacing);
      }
    }
    vector<vector<vector<double>>> rho_grad(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
    gradient(rho, rho_grad);
    for(int i=0;i<Ny;i++){
      for(int j=0;j<Nx;j++){
        result[i][j][0] = rho_grad[i][j][0]*del_t_xx[i][j] + rho_grad[i][j][1]*del_t_xy[i][j];
        result[i][j][1] = rho_grad[i][j][0]*del_t_yx[i][j] + rho_grad[i][j][1]*del_t_yy[i][j];
      }
    }
    return;

}

void update_T(vector<vector<double>> &rho, vector<vector<vector<double>>> &polarization_field_old, vector<vector<vector<double>>> &polarization_field_new, vector<vector<double>> &activity_field){

  vector<vector<vector<double>>> nonlinearterms(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
  non_linear(rho,polarization_field_old, nonlinearterms, activity_field);
  vector<vector<vector<double>>> result(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
  vector<vector<vector<double>>> polarization_field_mod(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
  for(int i=0;i<Ny;i++){
      for(int j=0;j<Nx;j++){
        polarization_field_mod[i][j][0] = polarization_field_old[i][j][0]*activity_field[i][j];
        polarization_field_mod[i][j][1] = polarization_field_old[i][j][1]*activity_field[i][j];
      }
    }
  laplacian(polarization_field_mod,result);
  // vector<vector<double>> wrho(Ny, vector<double>(Nx, -1));
  // for(int i=0;i<Ny;i++){
  //   for(int j=0;j<Nx;j++){
  //     wrho[i][j] = activity_field[i][j]*rho[i][j];
  //   }
  // }
  // vector<vector<vector<double>>> divwrho(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
  // gradient(wrho, divwrho);
  
  for(int i=0;i<Ny;i++){
    for(int j=0;j<Nx;j++){
      for(int k=0;k<2;k++){
        result[i][j][k] = result[i][j][k]*rho[i][j];
        result[i][j][k] += nonlinearterms[i][j][k];
        result[i][j][k] += polarization_field_old[i][j][k]*C*(1-(polarization_field_old[i][j][0]*polarization_field_old[i][j][0] + polarization_field_old[i][j][1]*polarization_field_old[i][j][1]));
        // result[i][j][k] -= divwrho[i][j][k];
        polarization_field_new[i][j][k] = polarization_field_old[i][j][k] + delta_t*result[i][j][k];
      }
    }
  }
  return;
}

void eta_nonlinear(vector<vector<vector<double>>> &T,vector<vector<vector<double>>> &nu, vector<vector<double>> &activity_field, vector<vector<double>> &result){
    vector<vector<vector<double>>> T_mod(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
    for(int i=0;i<Ny;i++){
      for(int j=0;j<Nx;j++){
        T_mod[i][j][0] = T[i][j][0]*activity_field[i][j];
        T_mod[i][j][1] = T[i][j][1]*activity_field[i][j];
      }
    }
  
    vector<vector<double>> del_t_xx(Ny, vector<double>(Nx, -1));
    vector<vector<double>> del_t_yx(Ny, vector<double>(Nx, -1));
    vector<vector<double>> del_t_xy(Ny, vector<double>(Nx, -1));
    vector<vector<double>> del_t_yy(Ny, vector<double>(Nx, -1));
    for(int i=0;i<Ny;i++){
      for(int j=0;j<Nx;j++){
        del_t_yy[i][j] = (T_mod[get_periodic_index(i+1,Ny)][j][1]-T_mod[get_periodic_index(i-1,Ny)][j][1])/(2*spacing);
        del_t_yx[i][j] = (T_mod[i][get_periodic_index(j+1,Nx)][1]-T_mod[i][get_periodic_index(j-1,Nx)][1])/(2*spacing);
        del_t_xx[i][j] = (T_mod[i][get_periodic_index(j+1,Nx)][0]-T_mod[i][get_periodic_index(j-1,Nx)][0])/(2*spacing);
        del_t_xy[i][j] = (T_mod[get_periodic_index(i+1,Ny)][j][0]-T_mod[get_periodic_index(i-1,Ny)][j][0])/(2*spacing);
      }
    }
    vector<vector<double>> del_nu_xx(Ny, vector<double>(Nx, -1));
    vector<vector<double>> del_nu_yx(Ny, vector<double>(Nx, -1));
    vector<vector<double>> del_nu_xy(Ny, vector<double>(Nx, -1));
    vector<vector<double>> del_nu_yy(Ny, vector<double>(Nx, -1));
    for(int i=0;i<Ny;i++){
      for(int j=0;j<Nx;j++){
        del_nu_yy[i][j] = (nu[get_periodic_index(i+1,Ny)][j][1]-nu[get_periodic_index(i-1,Ny)][j][1])/(2*spacing);
        del_nu_yx[i][j] = (nu[i][get_periodic_index(j+1,Nx)][1]-nu[i][get_periodic_index(j-1,Nx)][1])/(2*spacing);
        del_nu_xx[i][j] = (nu[i][get_periodic_index(j+1,Nx)][0]-nu[i][get_periodic_index(j-1,Nx)][0])/(2*spacing);
        del_nu_xy[i][j] = (nu[get_periodic_index(i+1,Ny)][j][0]-nu[get_periodic_index(i-1,Ny)][j][0])/(2*spacing);
      }
    }

    for(int i=0;i<Ny;i++){
      for(int j=0;j<Nx;j++){
        result[i][j] = del_nu_xx[i][j]*del_t_xx[i][j] + del_nu_xy[i][j]*del_t_xy[i][j] + del_nu_yx[i][j]*del_t_yx[i][j] + del_nu_yy[i][j]*del_t_yy[i][j] ;
      }
    }
    return;

}

void update_m_adjoint(vector<vector<double>> &rho, vector<vector<double>> &rho_target, vector<vector<double>> &eta, vector<vector<double>> &eta_new, vector<vector<vector<double>>> &T,vector<vector<vector<double>>> &nu, vector<vector<double>> &activity_field){

  vector<vector<double>> result(Ny, vector<double>(Nx, -1));
  laplacian_2D(eta,result);
  vector<vector<vector<double>>> grad_eta(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
  gradient(eta, grad_eta);
  vector<vector<double>> non_linear_terms(Ny, vector<double>(Nx, -1));
  eta_nonlinear(T,nu,activity_field,non_linear_terms);

  for(int i=0;i<Ny;i++){
    for(int j=0;j<Nx;j++){
      result[i][j]*=-1;
      result[i][j]+=-1*(activity_field[i][j]*(T[i][j][0]*grad_eta[i][j][0]+T[i][j][1]*grad_eta[i][j][1]));
      result[i][j]+=non_linear_terms[i][j];
      result[i][j]+=A*(rho[i][j]-rho_target[i][j]); 
      eta_new[i][j] = eta[i][j] - delta_t*result[i][j];
    }
  }
  return;
}

void nu_nonlinear(vector<vector<vector<double>>> &T,vector<vector<vector<double>>> &nu, vector<vector<double>> &activity_field, vector<vector<double>> &eta, vector<vector<vector<double>>> &result){

    vector<vector<vector<double>>> grad_eta(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
    gradient(eta, grad_eta);

    vector<vector<double>> del_nu_xx(Ny, vector<double>(Nx, -1));
    vector<vector<double>> del_nu_yx(Ny, vector<double>(Nx, -1));
    vector<vector<double>> del_nu_xy(Ny, vector<double>(Nx, -1));
    vector<vector<double>> del_nu_yy(Ny, vector<double>(Nx, -1));
    for(int i=0;i<Ny;i++){
      for(int j=0;j<Nx;j++){
        del_nu_yy[i][j] = (nu[get_periodic_index(i+1,Ny)][j][1]-nu[get_periodic_index(i-1,Ny)][j][1])/(2*spacing);
        del_nu_yx[i][j] = (nu[i][get_periodic_index(j+1,Nx)][1]-nu[i][get_periodic_index(j-1,Nx)][1])/(2*spacing);
        del_nu_xx[i][j] = (nu[i][get_periodic_index(j+1,Nx)][0]-nu[i][get_periodic_index(j-1,Nx)][0])/(2*spacing);
        del_nu_xy[i][j] = (nu[get_periodic_index(i+1,Ny)][j][0]-nu[get_periodic_index(i-1,Ny)][j][0])/(2*spacing);
      }
    }

    for(int i=0;i<Ny;i++){
      for(int j=0;j<Nx;j++){
        result[i][j][0] = del_nu_xx[i][j]*grad_eta[i][j][0] + del_nu_xy[i][j]*grad_eta[i][j][1];
        result[i][j][1] = del_nu_yx[i][j]*grad_eta[i][j][0]+ del_nu_yy[i][j]*grad_eta[i][j][1]; 
      }
    }
    return;

}

void update_T_adjoint(vector<vector<double>> &rho, vector<vector<double>> &eta,vector<vector<vector<double>>> &T,vector<vector<vector<double>>> &nu_new,vector<vector<vector<double>>> &T_target,vector<vector<vector<double>>> &nu, vector<vector<double>> &activity_field){

  vector<vector<vector<double>>> result(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
  laplacian(nu,result);
  vector<vector<vector<double>>> grad_eta(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
  gradient(eta, grad_eta);
  vector<vector<vector<double>>> non_linear_terms(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
  nu_nonlinear(T,nu,activity_field,eta,non_linear_terms);

  for(int i=0;i<Ny;i++){
    for(int j=0;j<Nx;j++){
      for(int k=0;k<2;k++){
        result[i][j][k] = -rho[i][j]*activity_field[i][j]*result[i][j][k];
        result[i][j][k] -= non_linear_terms[i][j][k]*activity_field[i][j];
        result[i][j][k] -= rho[i][j]*activity_field[i][j]*grad_eta[i][j][k];
        result[i][j][k] += -C*(1-(T[i][j][0]*T[i][j][0] + T[i][j][1]*T[i][j][1]))*nu[i][j][k] + 2*C*(nu[i][j][0]*T[i][j][0] + nu[i][j][1]*T[i][j][1])*T[i][j][k]; 
        result[i][j][k]+=B*(T[i][j][k]-T_target[i][j][k]); 
        nu_new[i][j][k] = nu[i][j][k] - delta_t*(result[i][j][k]);
      }
    }
  }
  
}

void one_backward(vector<vector<double>> &rho, vector<vector<double>> &eta_t, vector<vector<double>> &eta_t1, vector<vector<vector<double>>> &T,vector<vector<double>> &rho_target,vector<vector<vector<double>>> &nu_t,vector<vector<vector<double>>> &nu_t1, vector<vector<vector<double>>> &T_target, vector<vector<double>> &activity_field){

  update_m_adjoint(rho,rho_target, eta_t, eta_t1,T,nu_t, activity_field);
  update_T_adjoint(rho, eta_t,T,nu_t1,T_target, nu_t, activity_field);

  for(int i=0;i<Ny;i++){
      for(int j=0;j<Nx;j++){
        nu_t[i][j][0] = nu_t1[i][j][0];
        nu_t[i][j][1] = nu_t1[i][j][1];
        eta_t[i][j] = eta_t1[i][j];
      }
    }
}


void one_forward(vector<vector<double>> &rho_t,vector<vector<double>> &rho_t1, vector<vector<vector<double>>> &polarization_field_t, vector<vector<vector<double>>> &polarization_field_t1, vector<vector<double>> &activity_field, int t){
  update_density(rho_t,rho_t1,polarization_field_t, activity_field);
  update_T(rho_t, polarization_field_t, polarization_field_t1, activity_field);
}

void forward_run(vector<vector<vector<double>>> &rho_all,vector<vector<vector<vector<double>>>> &T_all,vector<vector<vector<double>>> &activity_field_all, int timesteps){

    for(int t = 0; t< timesteps-1;t++){
      one_forward(rho_all[t],rho_all[t+1],T_all[t], T_all[t+1],activity_field_all[t],t);
    }

}

void gradient_non_linear(vector<vector<double>> &rho,vector<vector<vector<double>>> &T,vector<vector<vector<double>>> &nu, vector<vector<double>> &result){
    vector<vector<double>> del_nu_xx(Ny, vector<double>(Nx, -1));
    vector<vector<double>> del_nu_yx(Ny, vector<double>(Nx, -1));
    vector<vector<double>> del_nu_xy(Ny, vector<double>(Nx, -1));
    vector<vector<double>> del_nu_yy(Ny, vector<double>(Nx, -1));
    for(int i=0;i<Ny;i++){
      for(int j=0;j<Nx;j++){
        del_nu_yy[i][j] = (nu[get_periodic_index(i+1,Ny)][j][1]-nu[get_periodic_index(i-1,Ny)][j][1])/(2*spacing);
        del_nu_yx[i][j] = (nu[i][get_periodic_index(j+1,Nx)][1]-nu[i][get_periodic_index(j-1,Nx)][1])/(2*spacing);
        del_nu_xx[i][j] = (nu[i][get_periodic_index(j+1,Nx)][0]-nu[i][get_periodic_index(j-1,Nx)][0])/(2*spacing);
        del_nu_xy[i][j] = (nu[get_periodic_index(i+1,Ny)][j][0]-nu[get_periodic_index(i-1,Ny)][j][0])/(2*spacing);
      }
    }
    vector<vector<vector<double>>> rho_grad(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
    gradient(rho, rho_grad);
    vector<vector<vector<double>>> nu_laplacian(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
    laplacian(nu, nu_laplacian);
    for(int i=0;i<Ny;i++){
      for(int j=0;j<Nx;j++){
        result[i][j] = 0;
        result[i][j] += T[i][j][0]*(del_nu_xx[i][j]*rho_grad[i][j][0]+ del_nu_xy[i][j]*rho_grad[i][j][1]+rho[i][j]*nu_laplacian[i][j][0]);
        result[i][j] += T[i][j][1]*(del_nu_yx[i][j]*rho_grad[i][j][0]+ del_nu_yy[i][j]*rho_grad[i][j][1]+rho[i][j]*nu_laplacian[i][j][1]);
      }
    }


}

void single_gradient_update(vector<vector<double>> &eta,vector<vector<vector<double>>> &nu,vector<vector<double>> &rho,vector<vector<vector<double>>> &T, vector<vector<double>> &activity_field,vector<vector<double>> &activity_field_baseline, double learning_rate){
  vector<vector<vector<double>>> eta_grad(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
  vector<vector<vector<double>>> rho_grad(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
  vector<vector<double>> activity_field_laplacian(Ny, vector<double>(Nx, -1));
  vector<vector<double>> gradients(Ny, vector<double>(Nx, -1));
  laplacian_2D(activity_field,activity_field_laplacian);
  gradient(eta, eta_grad);
  vector<vector<double>> non_linear_terms(Ny, vector<double>(Nx, -1));
  gradient_non_linear(rho,T,nu,non_linear_terms);
  for(int i=0;i<Ny;i++){  
    for(int j=0;j<Nx;j++){
      gradients[i][j] = 2*D*sqrt(activity_field[i][j])*(activity_field[i][j] - activity_field_baseline[i][j])-2*E*sqrt(activity_field[i][j])*activity_field_laplacian[i][j];
      gradients[i][j] -= 2*sqrt(activity_field[i][j])*(eta_grad[i][j][0]*(rho[i][j]*T[i][j][0]) + eta_grad[i][j][1]*(rho[i][j]*T[i][j][1]));
      gradients[i][j] -= 2*sqrt(activity_field[i][j])*non_linear_terms[i][j];
      activity_field[i][j] -= learning_rate*gradients[i][j];
      if(activity_field[i][j] != activity_field[i][j]){
        cout << "NAN Values in activity" << '\n';
        return;
      }
    }
  }

}

void backward_run(vector<vector<vector<double>>> &rho_all, vector<vector<double>> &eta_t,vector<vector<double>> &activity_field_baseline, vector<vector<double>> &eta_t1, vector<vector<vector<vector<double>>>> &T_all,vector<vector<double>> &rho_target,vector<vector<vector<double>>> &nu_t,vector<vector<vector<double>>> &nu_t1, vector<vector<vector<double>>> &T_target, vector<vector<vector<double>>> &activity_field_all, double learning_rate){

  for(int t = timesteps; t>=1;t--){
    one_backward(rho_all[t-1],eta_t,eta_t1,T_all[t-1],rho_target, nu_t,nu_t1,T_target, activity_field_all[t-1]);
    single_gradient_update(eta_t,nu_t,rho_all[t-1],T_all[t-1],activity_field_all[t-1],activity_field_baseline, learning_rate);
  }
}

void update_gradients(vector<vector<vector<double>>> &rho_all,vector<vector<vector<vector<double>>>> &T_all,vector<vector<vector<double>>> &activity_field_all,vector<vector<double>> &activity_field_baseline, vector<vector<vector<double>>> &T_target,vector<vector<double>> &rho_target, int timesteps, double learning_rate){

  forward_run(rho_all, T_all, activity_field_all, timesteps);

  cout << "Forward Run Completed " << '\n';

  vector<vector<double>> eta_t(Ny, vector<double>(Nx, 0));
  vector<vector<double>> eta_t1(Ny, vector<double>(Nx, -1));
  vector<vector<vector<double>>> nu_t(Ny,vector<vector<double>>(Nx, vector<double>(2, 0)));
  vector<vector<vector<double>>> nu_t1(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));

  backward_run(rho_all, eta_t, activity_field_baseline, eta_t1, T_all, rho_target, nu_t, nu_t1,T_target, activity_field_all, learning_rate);

  cout << "Backward Run Completed " << '\n';

}


int main(){

  string folder_path =  "C:\\PhD\\Work\\KardarMPC\\";
  string command = "mkdir "+folder_path;

  try{
      int result = std::system(command.c_str());
  }
  catch (exception &e){
      cout << e.what() << "\n";
      throw e;
  }

  int size = timesteps;

  vector<vector<vector<double>>> activity_field_all(size,vector<vector<double>>(Ny, vector<double>(Nx, -1)));
  vector<vector<vector<double>>> rho_all(size,vector<vector<double>>(Ny, vector<double>(Nx, -1)));
  vector<vector<vector<vector<double>>>> T_all(size,vector<vector<vector<double>>>(Ny,vector<vector<double>>(Nx, vector<double>(2, -1))));

  cout << "Arrays initialzied" << '\n';

  string folder_path_initial = "C:\\PhD\\Work\\KardarAsterModified4\\";
  readvectors(folder_path_initial, rho_all[0],activity_field_all[0], T_all[0], 199999);
  savevectors(rho_all[0], T_all[0], folder_path,-1);
  // initialize_grid(rho_all[0], 0.1);
  // initialize_3Dgrid(T_all[0]);
  // update_activity(activity_field_all, size);

  for(int s=0;s<size;s++){
    activity_field_all[s] = activity_field_all[0];
  }

  
  vector<vector<double>> activity_field_baseline(Ny, vector<double>(Nx, -1));
  activity_field_baseline = activity_field_all[0];
  vector<vector<double>> rho_target(Ny, vector<double>(Nx, -1));
  vector<vector<vector<double>>> T_target(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));


  readvectors(folder_path_initial, rho_target, activity_field_all[0], T_target, 199999);
  shift_2d_array(rho_target, 1,0);
  shift_3d_array(T_target, 1,0);
  savevectors(rho_target, T_target, folder_path, 101);


  cout << "Initial and target arrays defined" << '\n';

  int iterations = 40;
  double learning_rate = 0.001;
  for(int iter = 0; iter< iterations; iter++){
      if(iter%5==0){
        learning_rate/=2;
      }
      cout << "Iteration: " << iter << '\n';
      update_gradients(rho_all, T_all, activity_field_all, activity_field_baseline, T_target, rho_target, timesteps, learning_rate);

      for(int t=0;t<timesteps;t++){
        if(t%100==0 || t==timesteps-1){
            savevectors(rho_all[t], T_all[t], folder_path, t);
        } 
      }

      save_activity_field(activity_field_all, folder_path, timesteps);
  }


  cout << "Completed" << '\n';

}
