
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

const double PI = acos(-1.0);
constexpr double C = 100;

int get_periodic_index(int i, int size) {
    return (i % size + size) % size; 
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

class SpectralSolver{
  private:
    int Nx,Ny;
    double Lx,Ly;
    fftw_plan forward_plan_m, backward_plan_m, forward_plan_fx, forward_plan_fy;
    double *m_real, *flux_real_x, *flux_real_y, *m_real_x, *m_real_y;
    fftw_complex *m_hat, *flux_hat_x, *flux_hat_y,*m_hat_x,*m_hat_y;
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
        m_hat = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Ny * (Nx/2 + 1));
        m_hat_x = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Ny * (Nx/2 + 1));
        m_hat_y = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Ny * (Nx/2 + 1));
        
        forward_plan_m = fftw_plan_dft_r2c_2d(Ny, Nx, m_real, m_hat, FFTW_ESTIMATE);
        backward_plan_m = fftw_plan_dft_c2r_2d(Ny, Nx, m_hat, m_real, FFTW_ESTIMATE);
        forward_plan_fx = fftw_plan_dft_r2c_2d(Ny, Nx, flux_real_x, flux_hat_x, FFTW_ESTIMATE);
        forward_plan_fy = fftw_plan_dft_r2c_2d(Ny, Nx, flux_real_y, flux_hat_y, FFTW_ESTIMATE);
        
        setup_wavenumbers();
    }
    
    ~SpectralSolver() {
        fftw_destroy_plan(forward_plan_m);
        fftw_destroy_plan(backward_plan_m);
        fftw_destroy_plan(forward_plan_fx);
        fftw_destroy_plan(forward_plan_fy);

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

      void update_m_spectral(vector<vector<double>> &m, vector<vector<double>> &omega, vector<vector<vector<double>>> &T, double delta_t) {

        double dx = Lx / double(Nx);
        double dy = Ly / double(Ny);

        vector<vector<vector<double>>> flux(Ny, vector<vector<double>>(Nx, vector<double>(2,-1)));
        
        for(int i = 0; i < Ny; i++) {
            for(int j = 0; j < Nx; j++) {
                flux[i][j][0] = m[i][j] * omega[i][j] * T[i][j][0];
                flux[i][j][1] = m[i][j] * omega[i][j] * T[i][j][1];
            }
        }
  
        for(int i = 0; i < Ny; i++) {
            for(int j = 0; j < Nx; j++) {
                m_real[i*Nx + j] = m[i][j];
                flux_real_x[i*Nx + j] = flux[i][j][0];
                flux_real_y[i*Nx + j] = flux[i][j][1];
            }
        }
        fftw_execute_dft_r2c(forward_plan_m, m_real, m_hat);
        fftw_execute_dft_r2c(forward_plan_fx, flux_real_x, flux_hat_x);
        fftw_execute_dft_r2c(forward_plan_fy, flux_real_y, flux_hat_y);
        
        
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
            }
        }

        fftw_execute_dft_c2r(backward_plan_m, m_hat, m_real);
        
        for(int i = 0; i < Ny; i++) {
            for(int j = 0; j < Nx; j++) {
                m[i][j] = m_real[i*Nx + j]*norm;
            }
        }
      }



      void update_T_spectral(vector<vector<double>> &m, vector<vector<double>> &omega, vector<vector<vector<double>>> &T, vector<vector<vector<double>>> &T_new, double delta_t) {

          double dx = Lx / double(Nx);
          double dy = Ly / double(Ny);

          vector<vector<vector<double>>> flux(Ny, vector<vector<double>>(Nx, vector<double>(2,-1)));
          
          for(int i = 0; i < Ny; i++) {
              for(int j = 0; j < Nx; j++) {
                  flux[i][j][0] = omega[i][j] * T[i][j][0];
                  flux[i][j][1] = omega[i][j] * T[i][j][1];
              }
          }
    
            for(int i = 0; i < Ny; i++) {
                for(int j = 0; j < Nx; j++) {
                    m_real[i*Nx + j] = m[i][j];
                    flux_real_x[i*Nx + j] = flux[i][j][0];
                    flux_real_y[i*Nx + j] = flux[i][j][1];
                }
            }
            fftw_execute_dft_r2c(forward_plan_fx, flux_real_x, flux_hat_x);
            fftw_execute_dft_r2c(forward_plan_fy, flux_real_y, flux_hat_y);
            fftw_execute_dft_r2c(forward_plan_m, m_real, m_hat);

              for(int i = 0; i < Ny; i++) {
                for(int j = 0; j < Nx/2 + 1; j++) {
                  int idx = i*(Nx/2 + 1) + j;
                  m_hat_x[idx][0] = -kx[j]*m_hat_x[idx][1];
                  m_hat_x[idx][1] = kx[j]*m_hat_x[idx][0];
                  m_hat_y[idx][0] = -ky[i]*m_hat_y[idx][1];
                  m_hat_y[idx][1] = ky[i]*m_hat_y[idx][0];
                }
              }

            fftw_execute_dft_c2r(backward_plan_m, m_hat_x, m_real_x);
            fftw_execute_dft_c2r(backward_plan_m, m_hat_y, m_real_y);
            
            
            double norm = 1.0 / (Nx * Ny);  
            
            for(int i = 0; i < Ny; i++) {
                for(int j = 0; j < Nx/2 + 1; j++) {
                    int idx = i*(Nx/2 + 1) + j;

                    double k2 = k_squared[i][j];
                    
                    double flux_x_real_part = flux_hat_x[idx][0];
                    double flux_x_imag_part = flux_hat_x[idx][1];
                    double flux_y_real_part = flux_hat_y[idx][0];
                    double flux_y_imag_part = flux_hat_y[idx][1];
                    
                    flux_hat_x[idx][0] = -(kx[j]*flux_x_imag_part);
                    flux_hat_x[idx][1] = (kx[j]*flux_x_real_part);
                    flux_hat_y[idx][0] = -(ky[i]*flux_x_imag_part);
                    flux_hat_y[idx][1] = (ky[i]*flux_x_real_part);
                }
              }
            
            fftw_execute_dft_c2r(backward_plan_m, flux_hat_x, flux_real_x);
            fftw_execute_dft_c2r(backward_plan_m, flux_hat_y, flux_real_y);
          
          for(int i = 0; i < Ny; i++) {
              for(int j = 0; j < Nx; j++) {
                  int idx = i*Nx + j;
                  T_new[i][j][0] += m_real_x[idx]*flux_real_x[idx] + m_real_y[idx]*flux_real_x[idx];
              }
          }

           for(int i = 0; i < Ny; i++) {
                for(int j = 0; j < Nx/2 + 1; j++) {
                    int idx = i*(Nx/2 + 1) + j;

                    double k2 = k_squared[i][j];
                    
                    double flux_x_real_part = flux_hat_x[idx][0];
                    double flux_x_imag_part = flux_hat_x[idx][1];
                    double flux_y_real_part = flux_hat_y[idx][0];
                    double flux_y_imag_part = flux_hat_y[idx][1];
                    
                    flux_hat_x[idx][0] = -(kx[j]*flux_y_imag_part);
                    flux_hat_x[idx][1] = (kx[j]*flux_y_real_part);
                    flux_hat_y[idx][0] = -(ky[i]*flux_y_imag_part);
                    flux_hat_y[idx][1] = (ky[i]*flux_y_real_part);
                }
              }
            
            fftw_execute_dft_c2r(backward_plan_m, flux_hat_x, flux_real_x);
            fftw_execute_dft_c2r(backward_plan_m, flux_hat_y, flux_real_y);
    }
};