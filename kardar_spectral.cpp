
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
#include <fftw3.h>
#include <complex>

using namespace std;

constexpr int Nx = 50;
constexpr int Ny = 50;
constexpr int Lx = 50;
constexpr int Ly = 50;
constexpr double spacing = 1;
constexpr int timesteps = 100000;

constexpr double delta_t = 0.01;

const double PI = acos(-1.0);
constexpr double C = 100;
constexpr double activity_constant = 0.8;
constexpr double density_mean  = 0.1;
constexpr double polarization_ini_mag = 0.001;


int get_periodic_index(int i, int size) {
    return (i % size + size) % size; 
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


void initializeGaussian(vector<vector<double>> &particle_density,int N, double sigma, double A = 1.0) {

    int cx = N / 2;
    int cy = N / 2;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int dx = (i - cx);
            int dy = (j - cy);
            double r2 = dx*dx + dy*dy;
            particle_density[i][j] = 0;
            particle_density[i][j] += A * exp(-r2 / (2.0 * sigma * sigma));
        }
    }
}

void initializeSinusoid(vector<vector<double>> &particle_density,int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
          double x = j * spacing;
          particle_density[i][j] = sin(2.0 * PI * x / Lx);
        }
    }
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


void update_activity(vector<vector<double>> &activity_field){
  static mt19937 g(time(nullptr));  
  // static mt19937 g(42); 
  double mean_dist = 0;
  double stddev = 0.01;
  normal_distribution<double> dist(mean_dist, stddev);
  for(int i=0;i<Ny;i++){
    for(int j=0;j<Nx;j++){
        activity_field[i][j] = activity_constant + dist(g);
        // activity_field[i][j] = activity_constant;
    }
  }
  return;
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
  // double angle = dist(g);
  for(int i=0;i<Ny;i++){
    for(int j=0;j<Nx;j++){
      double angle = dist(g);
      grid[i][j][0] = polarization_ini_mag*cos(angle);
      grid[i][j][1] = polarization_ini_mag*sin(angle);
    }
  }
  return;
}


void divergence(vector<vector<vector<double>>> &grid, vector<vector<double>> &return_div, int Nx, int Ny, double dx, double dy){

  for(int i=0;i<Ny;i++){
      for(int j=0;j<Nx;j++){
            return_div[i][j]= 0;
            return_div[i][j]+= (grid[i][get_periodic_index(j+1,Nx)][0] - grid[i][get_periodic_index(j-1,Nx)][0])/(2*dx);
            return_div[i][j]+= (grid[get_periodic_index(i+1,Ny)][j][1] - grid[get_periodic_index(i-1,Ny)][j][1])/(2*dy);
        }
      }
    return;
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

class SpectralSolver{
  private:
    int Nx,Ny;
    double Lx,Ly;
    fftw_plan forward_plan_m, backward_plan_m, forward_plan_fx, forward_plan_fy, forward_plan_temp, backward_plan_temp, backward_plan_mx, backward_plan_my;
    double *m_real, *flux_real_x, *flux_real_y, *m_real_x, *m_real_y, *temp_real;
    fftw_complex *m_hat, *flux_hat_x, *flux_hat_y,*m_hat_x,*m_hat_y, *temp_hat;
    vector<double> kx, ky;
    vector<vector<double>> k_squared;
  
  public:
    SpectralSolver(int Nx_, int Ny_, double Lx_, double Ly_) 
        : Nx(Nx_), Ny(Ny_), Lx(Lx_), Ly(Ly_) {
        
        m_real = (double*) fftw_malloc(sizeof(double) * Ny * Nx);
        m_real_x = (double*) fftw_malloc(sizeof(double) * Ny * Nx);
        m_real_y = (double*) fftw_malloc(sizeof(double) * Ny * Nx);
        flux_real_x = (double*) fftw_malloc(sizeof(double) * Ny * Nx);
        flux_real_y = (double*) fftw_malloc(sizeof(double) * Ny * Nx);
        temp_real = (double*) fftw_malloc(sizeof(double) * Ny * Nx);
        m_hat = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Ny * (Nx/2 + 1));
        m_hat_x = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Ny * (Nx/2 + 1));
        m_hat_y = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Ny * (Nx/2 + 1));
        flux_hat_x = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Ny * (Nx/2 + 1));
        flux_hat_y = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Ny * (Nx/2 + 1));
        temp_hat = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Ny * (Nx/2 + 1));
        
        forward_plan_m = fftw_plan_dft_r2c_2d(Ny, Nx, m_real, m_hat, FFTW_ESTIMATE);
        backward_plan_m = fftw_plan_dft_c2r_2d(Ny, Nx, m_hat, m_real, FFTW_ESTIMATE);
        backward_plan_mx = fftw_plan_dft_c2r_2d(Ny, Nx, m_hat_x, m_real_x, FFTW_ESTIMATE);
        backward_plan_my = fftw_plan_dft_c2r_2d(Ny, Nx, m_hat_y, m_real_y, FFTW_ESTIMATE);
        forward_plan_fx = fftw_plan_dft_r2c_2d(Ny, Nx, flux_real_x, flux_hat_x, FFTW_ESTIMATE);
        forward_plan_fy = fftw_plan_dft_r2c_2d(Ny, Nx, flux_real_y, flux_hat_y, FFTW_ESTIMATE);
        forward_plan_temp = fftw_plan_dft_r2c_2d(Ny, Nx, temp_real, temp_hat, FFTW_ESTIMATE);
        backward_plan_temp = fftw_plan_dft_c2r_2d(Ny, Nx, temp_hat, temp_real, FFTW_ESTIMATE);
        
        setup_wavenumbers();
    }
    
    ~SpectralSolver() {
        fftw_destroy_plan(forward_plan_m);
        fftw_destroy_plan(backward_plan_m);
        fftw_destroy_plan(forward_plan_fx);
        fftw_destroy_plan(forward_plan_fy);
        fftw_destroy_plan(forward_plan_temp);
        fftw_destroy_plan(backward_plan_temp);

        fftw_free(m_real);
        fftw_free(m_real_x);
        fftw_free(m_real_y);
        fftw_free(flux_real_x);
        fftw_free(flux_real_y);
        fftw_free(m_hat);
        fftw_free(flux_hat_x);
        fftw_free(flux_hat_y);
        fftw_free(m_hat_x);
        fftw_free(m_hat_y);
    }

    void setup_wavenumbers() {
        kx.resize(Nx/2 +1);
        ky.resize(Ny);
        k_squared.resize(Ny, vector<double>(Nx/2 + 1));
        
        for(int i = 0; i < Ny; i++) {
            if(i <= Ny/2)
                ky[i] = 2.0 * PI * i / Ly;
            else
                ky[i] = 2.0 * PI * (i - Ny) / Ly;
        }
        
        for(int j = 0; j < Nx/2 + 1; j++) {
            kx[j] = 2.0 * PI * j / Lx;
        }
        
        for(int i = 0; i < Ny; i++) {
            for(int j = 0; j < Nx/2 + 1; j++) {
                k_squared[i][j] = kx[j]*kx[j] + ky[i]*ky[i];
            }
        }
      }

      void spectral_gradient(vector<vector<double>> &m){
        for(int i = 0; i < Ny; i++) {
          for(int j = 0; j < Nx; j++) {
                m_real[i*Nx + j] = m[i][j];
            }
        }
        fftw_execute_dft_r2c(forward_plan_m, m_real, m_hat);
        for(int i = 0; i < Ny; i++) {
            for(int j = 0; j < Nx/2 + 1; j++) {
                int idx = i*(Nx/2 + 1) + j;
                double m_hat_real = m_hat[idx][0];
                double m_hat_imag = m_hat[idx][1];
                m_hat[idx][0] = -kx[j]*m_hat_imag;
                m_hat[idx][1] = kx[j]*m_hat_real;
            }
          }

          fftw_execute_dft_c2r(backward_plan_m, m_hat, m_real);
          for(int i = 0; i < Ny; i++) {
            for(int j = 0; j < Nx; j++) {
              cout << m_real[i*Nx + j]*(1.0/(Nx*Ny)) << ' ';
            }
          }
      }

      void update_m_spectral(vector<vector<double>> &m_old, vector<vector<double>> &m_new, vector<vector<double>> &omega, vector<vector<vector<double>>> &T, double delta_t) {

        double dx = Lx / double(Nx);
        double dy = Ly / double(Ny);

        vector<vector<vector<double>>> flux(Ny, vector<vector<double>>(Nx, vector<double>(2,-1)));
        
        for(int i = 0; i < Ny; i++) {
            for(int j = 0; j < Nx; j++) {
                flux[i][j][0] = m_old[i][j] * omega[i][j] * T[i][j][0];
                flux[i][j][1] = m_old[i][j] * omega[i][j] * T[i][j][1];
            }
        }
  
        for(int i = 0; i < Ny; i++) {
            for(int j = 0; j < Nx; j++) {
                m_real[i*Nx + j] = m_old[i][j];
                flux_real_x[i*Nx + j] = flux[i][j][0];
                flux_real_y[i*Nx + j] = flux[i][j][1];
            }
        }
        fftw_execute_dft_r2c(forward_plan_m, m_real, m_hat);
        fftw_execute_dft_r2c(forward_plan_fx, flux_real_x, flux_hat_x);
        fftw_execute_dft_r2c(forward_plan_fy, flux_real_y, flux_hat_y);

        // int cutoff_x = Nx / 3;
        // int cutoff_y = Ny / 3;


        // for (int i = 0; i < Ny; i++) {
        //   int ky_index = (i <= Ny/2) ? i : i - Ny;
        //   for (int j = 0; j < Nx/2 + 1; j++) {
        //       int kx_index = j;

        //       if (abs(kx_index) > cutoff_x || abs(ky_index) > cutoff_y) {
        //           flux_hat_x[i*(Nx/2 + 1) + j][0] = 0.0;
        //           flux_hat_x[i*(Nx/2 + 1) + j][1] = 0.0;
        //           flux_hat_y[i*(Nx/2 + 1) + j][0] = 0.0;
        //           flux_hat_y[i*(Nx/2 + 1) + j][1] = 0.0;
        //        }
        //    }
        // }
        
        
        double norm = 1.0 / (Nx * Ny);  
        
        for(int i = 0; i < Ny; i++) {
            for(int j = 0; j < Nx/2 + 1; j++) {
                int idx = i*(Nx/2 + 1) + j;
                
                double m_real_part = m_hat[idx][0];
                double m_imag_part = m_hat[idx][1];
                
                double flux_x_real_part = flux_hat_x[idx][0];
                double flux_x_imag_part = flux_hat_x[idx][1];
                double flux_y_real_part = flux_hat_y[idx][0];
                double flux_y_imag_part = flux_hat_y[idx][1];
                
                double k2 = k_squared[i][j];
                
                m_hat[idx][0] = m_real_part + delta_t * (-k2 * m_real_part + (kx[j]*flux_x_imag_part +ky[i]*flux_y_imag_part));
                m_hat[idx][1] = m_imag_part + delta_t * (-k2 * m_imag_part - (kx[j]*flux_x_real_part +ky[i]*flux_y_real_part));

                // m_hat[idx][0] = m_real_part + delta_t * (-k2 * m_real_part );
                // m_hat[idx][1] = m_imag_part + delta_t * (-k2 * m_imag_part );
            }
        }

        fftw_execute_dft_c2r(backward_plan_m, m_hat, m_real);
        
        for(int i = 0; i < Ny; i++) {
            for(int j = 0; j < Nx; j++) {
                m_new[i][j] = m_real[i*Nx + j]*norm;
            }
        }
      }

      void update_T_spectral(vector<vector<double>> &m, vector<vector<double>> &omega, vector<vector<vector<double>>> &T, vector<vector<vector<double>>> &T_new, double delta_t) {

          // vector<vector<vector<double>>> nonlinearterms(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
          // non_linear(m,T, nonlinearterms, omega);

          for(int i=0;i<Ny;i++){
            for(int j=0;j<Nx;j++){
              T_new[i][j][0] = 0;
              T_new[i][j][1] = 0;
            }
          }

          double dx = Lx / double(Nx);
          double dy = Ly / double(Ny);

          vector<vector<vector<double>>> flux(Ny, vector<vector<double>>(Nx, vector<double>(2,-1)));
          
          for(int i = 0; i < Ny; i++) {
              for(int j = 0; j < Nx; j++) {
                  flux[i][j][0] = omega[i][j] * T[i][j][0];
                  flux[i][j][1] = omega[i][j] * T[i][j][1];
              }
          }
          // vector<vector<vector<double>>> result(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
          // laplacian(flux,result);
    
            for(int i = 0; i < Ny; i++) {
                for(int j = 0; j < Nx; j++) {
                    m_real[i*Nx + j] = m[i][j];
                }
            }
            fftw_execute_dft_r2c(forward_plan_m, m_real, m_hat);

              for(int i = 0; i < Ny; i++) {
                for(int j = 0; j < Nx/2 + 1; j++) {

                  int idx = i*(Nx/2 + 1) + j;

                  double m_hat_real = m_hat[idx][0];
                  double m_hat_imag = m_hat[idx][1];
                  
                  m_hat_x[idx][0] = -kx[j]*m_hat_imag;
                  m_hat_x[idx][1] = kx[j]*m_hat_real;
                  m_hat_y[idx][0] = -ky[i]*m_hat_imag;
                  m_hat_y[idx][1] = ky[i]*m_hat_real;
                }
              }

            fftw_execute_dft_c2r(backward_plan_mx, m_hat_x, m_real_x);
            fftw_execute_dft_c2r(backward_plan_my, m_hat_y, m_real_y);

            for(int i=0;i<Ny;i++){
              for(int j=0;j<Nx;j++){
                int idx = i*Nx+j;
                m_real_x[idx] = m_real_x[idx]*(1.0/(Nx*Ny));
                m_real_y[idx] = m_real_y[idx]*(1.0/(Nx*Ny));
              }
            }
            
            for(int k=0;k<2;k++){
              for(int i = 0; i < Ny; i++) {
                for(int j = 0; j < Nx; j++) {
                    temp_real[i*Nx + j] = flux[i][j][k];
                  }
                }
              fftw_execute_dft_r2c(forward_plan_temp, temp_real, temp_hat);

            // int cutoff_x = Nx / 3;
            // int cutoff_y = Ny / 3;


            // for (int i = 0; i < Ny; i++) {
            //   int ky_index = (i <= Ny/2) ? i : i - Ny;
            //   for (int j = 0; j < Nx/2 + 1; j++) {
            //       int kx_index = j;
            //       if (abs(kx_index) > cutoff_x || abs(ky_index) > cutoff_y) {
            //           temp_hat[i*(Nx/2 + 1) + j][0] = 0.0;
            //           temp_hat[i*(Nx/2 + 1) + j][1] = 0.0;
            //       }
            //     } 
            //   }

            for(int i = 0; i < Ny; i++) {
                for(int j = 0; j < Nx/2 + 1; j++) {
                    int idx = i*(Nx/2 + 1) + j;
                    
                    double temp_x_real_part = temp_hat[idx][0];
                    double temp_x_imag_part = temp_hat[idx][1];
            
                    temp_hat[idx][0] = -(kx[j]*temp_x_imag_part);
                    temp_hat[idx][1] = (kx[j]*temp_x_real_part);
                }
              }
            
              fftw_execute_dft_c2r(backward_plan_temp, temp_hat, temp_real);
              for(int i=0;i<Ny;i++){
                for(int j=0;j<Nx;j++){
                  int idx = i*Nx+j;
                  temp_real[idx] = temp_real[idx]*(1.0/(Nx*Ny));
                }
              }
              for(int i = 0; i < Ny; i++) {
                  for(int j = 0; j < Nx; j++) {
                      int idx = i*Nx + j;
                      T_new[i][j][k] += m_real_x[idx]*temp_real[idx];
                  }
                }
              for(int i = 0; i < Ny; i++) {
                for(int j = 0; j < Nx; j++) {
                    temp_real[i*Nx + j] = flux[i][j][k];
                  }
                }
              fftw_execute_dft_r2c(forward_plan_temp, temp_real, temp_hat);
                for(int i = 0; i < Ny; i++) {
                  for(int j = 0; j < Nx/2 + 1; j++) {
                      int idx = i*(Nx/2 + 1) + j;

                      double temp_x_real_part = temp_hat[idx][0];
                      double temp_x_imag_part = temp_hat[idx][1];
              
                      temp_hat[idx][0] = -(ky[i]*temp_x_imag_part);
                      temp_hat[idx][1] = (ky[i]*temp_x_real_part);
                  }
                }

              fftw_execute_dft_c2r(backward_plan_temp, temp_hat, temp_real);
              for(int i=0;i<Ny;i++){
                for(int j=0;j<Nx;j++){
                  int idx = i*Nx+j;
                  temp_real[idx] = temp_real[idx]*(1.0/(Nx*Ny));
                }
              }
              for(int i = 0; i < Ny; i++) {
                  for(int j = 0; j < Nx; j++) {
                      int idx = i*Nx + j;
                      T_new[i][j][k] += m_real_y[idx]*temp_real[idx];
                  }
                }
            }

            for(int k=0;k<2;k++){
              for(int i = 0; i < Ny; i++) {
                for(int j = 0; j < Nx; j++) {
                    temp_real[i*Nx + j] = flux[i][j][k];
                  }
                }
              fftw_execute_dft_r2c(forward_plan_temp, temp_real, temp_hat);

            // int cutoff_x = Nx / 3;
            // int cutoff_y = Ny / 3;


            // for (int i = 0; i < Ny; i++) {
            //   int ky_index = (i <= Ny/2) ? i : i - Ny;
            //   for (int j = 0; j < Nx/2 + 1; j++) {
            //       int kx_index = j;
            //       if (abs(kx_index) > cutoff_x || abs(ky_index) > cutoff_y) {
            //           temp_hat[i*(Nx/2 + 1) + j][0] = 0.0;
            //           temp_hat[i*(Nx/2 + 1) + j][1] = 0.0;
            //       }
            //     } 
            //   }

              for(int i = 0; i < Ny; i++) {
                for(int j = 0; j < Nx/2 + 1; j++) {

                  double k2 = k_squared[i][j];
                  int idx = i*(Nx/2+1)+j;

                  temp_hat[idx][0] = -k2*temp_hat[idx][0];
                  temp_hat[idx][1] = -k2*temp_hat[idx][1];
                }
              }
              fftw_execute_dft_c2r(backward_plan_temp, temp_hat, temp_real);
              for(int i=0;i<Ny;i++){
                for(int j=0;j<Nx;j++){
                  int idx = i*Nx+j;
                  temp_real[idx] = temp_real[idx]*(1.0/(Nx*Ny));
              }
            }
              for(int i = 0; i < Ny; i++) {
                for(int j = 0; j < Nx; j++) {
                    int idx = i*Nx + j;
                    T_new[i][j][k] += m[i][j]*temp_real[idx];
                }
              }
            }
            for(int k=0;k<2;k++){
              for(int i = 0; i < Ny; i++) {
                for(int j = 0; j < Nx; j++) {
                  // T_new[i][j][k] += nonlinearterms[i][j][k];
                  // T_new[i][j][k]+= result[i][j][k];
                  T_new[i][j][k] += C*T[i][j][k]*(1 -(T[i][j][0]*T[i][j][0] + T[i][j][1]*T[i][j][1]));
                  T_new[i][j][k]*=delta_t;
                  T_new[i][j][k] += T[i][j][k];
                }
              }
            }
            return;
      }
};

int main(){

  cout << "Start" << '\n';

  string folder_path = "C:\\PhD\\Work\\KardarAsterSpectral2\\";
  string command = "mkdir "+folder_path;

  try{
      int result = std::system(command.c_str());
  }
  catch (exception &e){
      cout << e.what() << "\n";
      throw e;
  }


  SpectralSolver solver(Nx, Ny, Lx, Ly);

  vector<vector<double>> rho_t(Ny, vector<double>(Nx, -1));
  vector<vector<double>> activity_field(Ny, vector<double>(Nx, -1));
  vector<vector<vector<double>>> polarization_field_t(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
  vector<vector<double>> rho_t1(Ny, vector<double>(Nx, -1));
  vector<vector<vector<double>>> polarization_field_t1(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));

  initialize_grid(rho_t,density_mean);
  // initializeSinusoid(rho_t,Nx);
  // initializeGaussian(rho_t,Nx,40);
  initialize_3Dgrid(polarization_field_t);
  update_activity(activity_field);

  // vector<vector<vector<double>>> result(Ny,vector<vector<double>>(Nx, vector<double>(2, -1)));
  // gradient(rho_t, result);
  // for(int i=0;i<Ny;i++){
  //   for(int j=0;j<Nx;j++){
  //     cout << result[i][j][0] << ' ';
  //   }
  // }
  // cout << "First end" << endl;
  // solver.spectral_gradient(rho_t);


  for(int t=0;t<timesteps;t++){

    if(t%10000==0 || t== timesteps-1){
      savevectors(rho_t, polarization_field_t, folder_path, t);
    }

    solver.update_m_spectral(rho_t, rho_t1, activity_field, polarization_field_t, delta_t);
    solver.update_T_spectral(rho_t, activity_field, polarization_field_t, polarization_field_t1, delta_t);

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
      cout << "Timestep: " << t << ' '<< "Sum: " << sum << '\n';
    }
  }

  return 0;

}