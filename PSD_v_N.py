from numpy import sum, power, array, pi, exp, subtract, divide, argmin, log, mean, sqrt, logspace, ones
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.font_manager import FontProperties 
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from os import listdir
from os.path import isfile, join
from time import time


# constants of the universe
mu_0 = 4 * pi * 10.**-7
hbar = 1.0545718 * 10.**-34
c = 299792458
mu_b = hbar * 2 * pi * 1.39962460 * 10.**6
k_b = 1.38 * 10**-23

# sodium constants
Isat = 6.26 * 10
Gamma = 2 * pi * 9.7946 * 10.**6
f0 = 508.8487162 * 10.**12
k = 2 * pi * f0 / c
m = 22.989769 * 1.672623 * 10**-27
g_f = 0.5
a_s = 2.75 * 10.**-9

# Rb87 constants
#Isat = 1.669 * 10
#Gamma = 2 * pi * 6.066 * 10.**6
#f0 = 384.230484468 * 10.**12
#k = 2 * pi * f0 / c
#m = 86.909 * 1.672623 * 10**-27
#a_s = 98 * 5.29 * 10.**-11

def density(N, T, B_prime, m_F = 1):
    return N * power(g_f * m_F * mu_b * B_prime, 3) / (32 * pi) * power(k_b * T, -3)

def collision_rate(N, T, B_prime, m_F = 1):
    return density(N, T, B_prime, m_F = m_F) * thermal_velocity(T) * 4 * pi * a_s**2

def thermal_velocity(T):
    return sqrt(2 * k_b * T / m)

def PSD(N, T, B_prime, m_F = 1):
    return N * power(g_f * m_F * mu_b * B_prime, 3) / (32 * pi) * power(k_b * T, -3) * power(2 * pi * hbar / sqrt(2 * pi * m * k_b * T), 3)

def delta_PSD(N, delta_N, T, delta_T, B_prime, delta_B_prime, m_F = 1):
    d_rho_d_N = power(g_f * m_F * mu_b, 3) / (32 * pi) * power(k_b, -3) * power(2 * pi * hbar / sqrt(2 * pi * m * k_b),3) * power(B_prime, 3) * power(T, -4.5)
    d_rho_d_T = power(g_f * m_F * mu_b, 3) / (32 * pi) * power(k_b, -3) * power(2 * pi * hbar / sqrt(2 * pi * m * k_b),3) * N * power(B_prime, 3) * power(T, -5.5) * 4.5
    d_rho_d_B_prime = power(g_f * m_F * mu_b, 3) / (32 * pi) * power(k_b, -3) * power(2 * pi * hbar / sqrt(2 * pi * m * k_b),3) * N * power(B_prime, 2) * power(T, -4.5) * 3
    return sqrt( power(d_rho_d_N * delta_N, 2) + power(d_rho_d_T * delta_T, 2) + power(d_rho_d_B_prime * delta_B_prime, 2) )

def fit_efficiency(N, rho):
    N1 = log(N)
    rho1 = log(rho)
    
    def func(x, m, b):
        return m * x + b
    
    popt, pcov = curve_fit(func, N1, rho1)
    return popt

def main():
    fig, (ax1) = plt.subplots(1)
    fig.set_size_inches(10, 8.5, forward=True)
    
    N_fit = logspace(4, 9)
    
    N = array( [400, 300, 151, 27, 3] ) * 10.**6
    T = array( [1500, 423, 271, 91, 34] ) * 10.**-6
    B_prime = array( [239, 239, 239, 239, 239] ) * 100.

    rho = PSD(N, T, B_prime, m_F = 2)
    m, b = fit_efficiency(N, rho)
    
    ax1.scatter(N, rho, color = 'b', label = 'Na F = 2, 7-19')
    #ax1.plot( N_fit, exp( m * log(N_fit) + b ), color = 'b', label = 'Fit F = 2' )
    
    N = array( [400, 19.8, 4, 0.75] ) * 10.**6
    T = array( [350, 43.7, 20.1, 8] ) * 10.**-6
    B_prime = ones(len(N)) * 260.7 * 100

    rho = PSD(N, T, B_prime, m_F = 1)
    n_0 = density(N, T, B_prime, m_F = 1)
    Gamma_el = collision_rate(N, T, B_prime, m_F = 1)
    m, b = fit_efficiency(N, rho)
    
    
    ax1.scatter(N, rho, color = 'cyan', label = 'Na F = 1, plug stabilized 9/19')
    #ax1.plot( N_fit, exp( m * log(N_fit) + b ), color = 'g', label = 'Fit F = 1, no plug' )
    
    #CURRENT NUMBERS IN THE BELOW LIST
    N = array( [1014, 660, 608, 527, 215, 178, 167, 152, 113] ) * 10.**6
    T = array( [634, 362, 332, 286, 178, 173, 166, 158, 154] ) * 10.**-6
    B_prime = ones(len(N)) * 260.7 * 100

    rho = PSD(N, T, B_prime, m_F = 1)
    n_0 = density(N, T, B_prime, m_F = 1)
    Gamma_el = collision_rate(N, T, B_prime, m_F = 1)
    m, b = fit_efficiency(N, rho)
    
    print('The efficiency is ' + str(m))
    print('PSD \t n_0 (10^11) \t Gamma_el')
    for i in range(len(rho)):
        print(rho[i], n_0[i] * 10.**-6 * 10.**-11, Gamma_el[i])
    ax1.scatter(N, rho, color = 'K', label = '9/25 after MOT opt')
    #ax1.plot( N_fit, exp( m * log(N_fit) + b ), color = 'g', label = 'Fit F = 1, no plug' )
    N = array( [750, 475, 200, 25, 19, 11] ) * 10.**6
    rho = array( [3.3 * 10.**-7, 10.**-6, 1.8 * 10.**-5, 7 * 10.**-3, 10.**-2, 2 * 10.**-2] )
    m, b = fit_efficiency(N, rho)
    
    #print(m)
    
    ax1.scatter(N, rho, color = 'orange', label = 'Na F = 2,  MIT')
    #ax1.plot( N_fit, exp( m * log(N_fit) + b ), color = 'yellow', label = 'Fit F = 1, m_F evap' )
    
    N = array( [1300, 600, 400, 200, 110, 80, 50, 17, 9, 0.57] ) * 10.**6
    T = array( [500, 135, 110, 65, 42, 27, 17, 7, 2.7, 0.8] ) * 10.**-6
    B_prime = ones(len(N)) * 302 * 100

    rho = PSD(N, T, B_prime, m_F = 1)
    m, b = fit_efficiency(N, rho)
    
    #print(m)
    
    ax1.scatter(N, rho, color = 'red', label = 'Na F = 1, Brazil')
    
    N = array( [3550, 3210, 2050, 1680, 967, 738, 401, 322, 125] ) * 10.**6
    T = array( [476, 402, 241, 177, 126, 88.7, 49.7, 34.7, 12.2] ) * 10.**-6
    B_prime = ones(len(N)) * 320 * 100

    rho = PSD(N, T, B_prime, m_F = 1)
    m, b = fit_efficiency(N, rho)
    
    #print(m)
    
    ax1.scatter(N, rho, color = 'green', label = 'Na F = 1, GaTech')
    
    font0 = FontProperties()
    
    font0.set_size('20')
    
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    
    ax1.set_ylabel(r'PSD', fontproperties = font0)
    ax1.set_xlabel(r'Atom Number', fontproperties = font0)
    ax1.tick_params('both', width=2, labelsize=16)
    ax1.grid(True)
    ax1.axis( [10.**4, 10.**10, 10.**-8, 10.**2] )
    ax1.legend(loc = 'lower left')
    
    plt.show()
    

if __name__ == "__main__":
    main()