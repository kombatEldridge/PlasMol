import datetime
import numpy as np
import matplotlib.pyplot as plt

# Settings for matplotlib
np.set_printoptions(precision=5, linewidth=200, suppress=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['ytick.major.width'] = 2

def JK(wfn, D):
    pot = wfn.jk.get_veff(wfn.ints_factory, 2.*D)
    Fa = wfn.T + wfn.Vne + pot
    return Fa

def eT(curr_Time):
    alpha = 0.1
    return (1e-4)*(4*alpha*np.pi)**(-1/2) * np.exp(-curr_Time**2/(4*alpha))

def rks_propagate(wfn):
    print("\n\nComputing Induced Dipole now!")
    mu_xx = ind_dipole(0, 0, wfn)
    mu_yy = ind_dipole(1, 1, wfn)
    mu_zz = ind_dipole(2, 2, wfn)
    freq_arr = np.arange(0, 3, 0.002)

    print("\n\nComputing Fourier Transform now!")
    # Fourier Transform for the mu vs time
    fw_xx = FT(mu_xx[0], mu_xx[2], freq_arr)
    ew_xx = FT(mu_xx[1], mu_xx[2], freq_arr)
    fw_yy = FT(mu_yy[0], mu_yy[2], freq_arr)
    ew_yy = FT(mu_yy[1], mu_yy[2], freq_arr)
    fw_zz = FT(mu_zz[0], mu_zz[2], freq_arr)
    ew_zz = FT(mu_zz[1], mu_zz[2], freq_arr)

    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")
    picture_name = f'fourier_{formatted_datetime}.png'

    header = "freq_arr\tew_xx.real\tew_xx.imag\tew_yy.real\ttew_yy.imag\tew_zz.real\tew_zz.iamg"
    combined_array = np.array([freq_arr, ew_xx.real, ew_xx.imag, ew_yy.real, ew_yy.imag, ew_zz.real, ew_zz.imag]).T
    file_name = f'combined_arrays_{formatted_datetime}.txt'
    with open(file_name, 'wb') as f:
        np.savetxt(f, combined_array, delimiter='\t', header=header, comments='', fmt='%s')

    plt.plot(27.21138*freq_arr, (fw_xx/ew_xx).imag, linestyle='-', label='xx')
    plt.plot(27.21138*freq_arr, (fw_yy/ew_yy).imag, linestyle='-', label='yy')
    plt.plot(27.21138*freq_arr, (fw_zz/ew_zz).imag, linestyle='-', label='zz')
    plt.xlabel('Frequency (eV)')
    plt.ylabel('Intensity')
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    ax = plt.subplot(111)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.yticks(fontsize=18, fontweight='bold')
    plt.xticks(fontsize=18, fontweight='bold')
    fig.tight_layout()
    plt.savefig(picture_name)

    return None

def ind_dipole(direction1, direction2, wfn):
    D_ao = wfn.D[0]
    D_ao_init = wfn.D[0]
    D_mo = wfn.C[0].T@wfn.S.T@D_ao@wfn.S@wfn.C[0]
    mu_arr = []
    e_arr = []

    # RK4 method
    for i in range(20000):
        dt = 0.1
        t = i*dt

        Ft_ao = JK(wfn, D_ao) - wfn.mu[direction1]*eT(t)
        Ft_mo = wfn.C[0].T@Ft_ao@wfn.C[0]
        k1 = (-1j*(Ft_mo@D_mo - D_mo@Ft_mo))
        temp_D_mo = D_mo + 0.5*k1*dt
        temp_D_ao = wfn.C[0]@temp_D_mo@wfn.C[0].T

        Ft_ao = JK(wfn, temp_D_ao) - wfn.mu[direction1]*eT((0.5+i)*dt)
        Ft_mo = wfn.C[0].T@Ft_ao@wfn.C[0]
        k2 = (-1j*(Ft_mo@temp_D_mo - temp_D_mo@Ft_mo))
        temp_D_mo = D_mo + 0.5*k2*dt
        temp_D_ao = wfn.C[0]@temp_D_mo@wfn.C[0].T

        k3 = (-1j*(Ft_mo@temp_D_mo - temp_D_mo@Ft_mo))
        temp_D_mo = D_mo + k3*dt
        temp_D_ao = wfn.C[0]@temp_D_mo@wfn.C[0].T

        Ft_ao = JK(wfn, temp_D_ao) - wfn.mu[direction1]*eT((1+i)*dt)
        Ft_mo = wfn.C[0].T@Ft_ao@wfn.C[0]
        k4 = (-1j*(Ft_mo@temp_D_mo - temp_D_mo@Ft_mo))

        D_mo = D_mo + dt*(k1 + 2*k2 + 2*k3 + k4)/6
        D_ao = wfn.C[0]@D_mo@wfn.C[0].T

        mu = np.trace(wfn.mu[direction2]@D_ao) - np.trace(wfn.mu[direction2]@D_ao_init)
        mu_arr.append(mu.real)
        e_arr.append(eT(t))

        if (i%1000) == 0:
            print("\t".join([str(i), str(direction1)]))

    return (mu_arr, e_arr, dt)

def FT(mu_arr, dt, freq_arr):
   gam = 0.25/27.21138
   fw = np.zeros(len(freq_arr), dtype=complex)
   for i, wi in enumerate(freq_arr):
       for tindex in range(len(mu_arr)):
           t = tindex*dt
           fw[i] += mu_arr[tindex] * np.exp(-1j*wi*t - gam*t) * dt
   return fw