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

constexpr int timesteps = 20000;
constexpr double lambda_constant = 0.3;
constexpr double PI = 3.14159265358979323846;
constexpr double delta_t = 0.5;
constexpr double D = 1;
constexpr double polarization_ini_mag = 0.1;
constexpr double density_mean  = 1.07;
constexpr double rho_c = 1;
constexpr double activity_constant = 0.1;
constexpr double K = 1;

int get_periodic_index(int i, int size) {
    return (i % size + size) % size; 
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

void update_activity(vector<vector<double>> &particle_density,vector<vector<double>> &activity_field, double rho_c, int Nx, int Ny){
  for(int i=0;i<Ny;i++){
    for(int j=0;j<Nx;j++){
        // activity_field[i][j] =particle_density[i][j]-rho_c ;
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

void integrate( vector<vector<double>> &particle_density_t,vector<vector<double>> &particle_density_t1, vector<vector<double>> &activity_field, vector<vector<vector<double>>> &tau_t, vector<vector<vector<double>>> &tau_t1, vector<vector<vector<double>>> &omeg_tau,vector<vector<vector<double>>> &lambda_terms, int Nx, int Ny, double spacing){

  update_rho(particle_density_t,particle_density_t1,omeg_tau,Nx,Ny,spacing);
  update_tau(lambda_terms,tau_t,tau_t1,particle_density_t,activity_field,Nx,Ny,spacing);

}




int main(){
  int Nx = 60;
  int Ny = 60;
  double spacing = 2;

  vector<vector<double>> particle_density_t(Ny, vector<double>(Nx, -1));
  string folder_path = "C:\\PhD\\Work\\Trial1\\";
  string command = "mkdir "+folder_path;
  try{
      int result = std::system(command.c_str());
  }
  catch (exception &e){
      cout << e.what() << "\n";
      throw e;
  }
  initialize_grid(particle_density_t,density_mean,Nx,Ny);
  vector<vector<vector<double>>> polarization_field(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
  initialize_3Dgrid(polarization_field,Nx,Ny);
  vector<vector<double>> activity_field(Ny, vector<double>(Nx, -1));
  update_activity(particle_density_t,activity_field,rho_c,Nx,Ny);
  vector<vector<vector<double>>> tau_t(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
  tau_calculate(tau_t,polarization_field,particle_density_t,Nx,Ny);
  vector<vector<vector<double>>> omeg_tau(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
  calculate_omegtau(tau_t,activity_field,omeg_tau,Nx,Ny);
  vector<vector<vector<double>>> lambda_etc(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
  lambda_terms(lambda_etc,tau_t,Nx,Ny,spacing);

  vector<vector<double>> particle_density_t1(Ny, vector<double>(Nx, -1));
  vector<vector<vector<double>>> tau_t1(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));


  for(int t=0;t<timesteps;t++){

    if(t%500==0 || t==timesteps-1){
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
    
    integrate(particle_density_t,particle_density_t1,activity_field,tau_t,tau_t1,omeg_tau,lambda_etc,Nx,Ny,spacing);

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
    calculate_omegtau(tau_t,activity_field,omeg_tau,Nx,Ny);
    lambda_terms(lambda_etc,tau_t,Nx,Ny,spacing);


  }


}