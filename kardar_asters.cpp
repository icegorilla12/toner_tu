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

constexpr int Nx = 20;
constexpr int Ny = 20;
constexpr double spacing = 1;
constexpr int timesteps = 500000;
constexpr double PI = 3.14159265358979323846;
constexpr double delta_t = 0.01;
constexpr double polarization_ini_mag = 0.001;
constexpr double density_mean  = 0.15;
constexpr double C = 1;



int get_periodic_index(int i, int size) {
    return (i % size + size) % size; 
}

void savevectors(vector<vector<double>> &particle_density, vector<vector<vector<double>>> &polarization_field, string folder_path, int iter){
    string particle_density_path = "particle_density_"+to_string(iter)+".bin";
    string polarization_field_path = "polarization_field_"+to_string(iter)+".bin";
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

void update_density(vector<vector<double>> &rho_old, vector<vector<double>> &rho_new,vector<vector<vector<double>>> &polarization_field){
  vector<vector<double>> result(Ny, vector<double>(Nx, -1));
  vector<vector<vector<double>>> mT(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
  laplacian_2D(rho_old, result);
  for(int i=0;i<Ny;i++){
    for(int j=0;j<Nx;j++){
      for(int k=0;k<2;k++){
        mT[i][j][k] = rho_old[i][j]*polarization_field[i][j][k];
      }
    }
  }
  vector<vector<double>> mTdiv(Ny, vector<double>(Nx, -1));
  divergence(mT,mTdiv);
  for(int i=0;i<Ny;i++){
    for(int j=0;j<Nx;j++){
      result[i][j]-=mTdiv[i][j];
    }
  }
  for(int i=0;i<Ny;i++){
    for(int j=0;j<Nx;j++){
      rho_new[i][j] = rho_old[i][j] + delta_t*result[i][j];
    }
  }
}

void non_linear(vector<vector<double>> &rho, vector<vector<vector<double>>> &polarization_field, vector<vector<vector<double>>> &result){
    vector<vector<double>> del_t_xx(Ny, vector<double>(Nx, -1));
    vector<vector<double>> del_t_yx(Ny, vector<double>(Nx, -1));
    vector<vector<double>> del_t_xy(Ny, vector<double>(Nx, -1));
    vector<vector<double>> del_t_yy(Ny, vector<double>(Nx, -1));
    for(int i=0;i<Ny;i++){
      for(int j=0;j<Nx;j++){
        del_t_yy[i][j] = (polarization_field[get_periodic_index(i+1,Ny)][j][1]-polarization_field[get_periodic_index(i-1,Ny)][j][1])/(2*spacing);
        del_t_yx[i][j] = (polarization_field[i][get_periodic_index(j+1,Nx)][1]-polarization_field[i][get_periodic_index(j-1,Nx)][1])/(2*spacing);
        del_t_xx[i][j] = (polarization_field[i][get_periodic_index(j+1,Nx)][0]-polarization_field[i][get_periodic_index(j-1,Nx)][0])/(2*spacing);
        del_t_xy[i][j] = (polarization_field[get_periodic_index(i+1,Ny)][j][0]-polarization_field[get_periodic_index(i-1,Ny)][j][0])/(2*spacing);
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

void update_T(vector<vector<double>> &rho, vector<vector<vector<double>>> &polarization_field_old, vector<vector<vector<double>>> &polarization_field_new){
  vector<vector<vector<double>>> nonlinearterms(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
  non_linear(rho,polarization_field_old, nonlinearterms);
  vector<vector<vector<double>>> result(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
  laplacian(polarization_field_old,result);
  for(int i=0;i<Ny;i++){
    for(int j=0;j<Nx;j++){
      for(int k=0;k<2;k++){
        result[i][j][k] = result[i][j][k]*rho[i][j];
        result[i][j][k] += nonlinearterms[i][j][k];
        result[i][j][k] += C*polarization_field_old[i][j][k]*(1-(polarization_field_old[i][j][0]*polarization_field_old[i][j][0] + polarization_field_old[i][j][1]*polarization_field_old[i][j][1]));
        polarization_field_new[i][j][k] = polarization_field_old[i][j][k] + delta_t*result[i][j][k];
      }
    }
  }
  return;
}



int main(){

  string folder_path = "C:\\PhD\\Work\\KardarAster\\";
  string command = "mkdir "+folder_path;

  try{
      int result = std::system(command.c_str());
  }
  catch (exception &e){
      cout << e.what() << "\n";
      throw e;
  }

  vector<vector<double>> rho_t(Ny, vector<double>(Nx, -1));
  vector<vector<vector<double>>> polarization_field_t(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
  vector<vector<double>> rho_t1(Ny, vector<double>(Nx, -1));
  vector<vector<vector<double>>> polarization_field_t1(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
  initialize_grid(rho_t,density_mean);
  initialize_3Dgrid(polarization_field_t);
  
  for(int t = 0; t< timesteps;t++){
    if(t%10000==0){
        savevectors(rho_t, polarization_field_t, folder_path, t);
    }
    update_density(rho_t,rho_t1,polarization_field_t);
    update_T(rho_t, polarization_field_t, polarization_field_t1);
    for(int i=0;i<Ny;i++){
      for(int j=0;j<Nx;j++){
        polarization_field_t[i][j][0] = polarization_field_t1[i][j][0];
        polarization_field_t[i][j][1] = polarization_field_t1[i][j][1];
      }
    }
    double sum = 0;
    for(int i=0;i<Ny;i++){
      for(int j=0;j<Nx;j++){
        rho_t[i][j] = rho_t1[i][j];
        sum+= rho_t1[i][j];
      }
    }
    if(t%10000==0){
      cout << sum << endl;
    }
  }
}
