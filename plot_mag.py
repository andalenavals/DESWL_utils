#! /usr/bin/env python

from __future__ import print_function
import matplotlib
matplotlib.use('Agg') # Don't use X-server.  Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

def size_residual(m, dT, min_mused, legend=None, color=None):

    import numpy as np
    import matplotlib.pyplot as plt

    min_mag = 13.5
    max_mag = 21
    #min_mused = 15

    # Bin by mag
    mag_bins = np.linspace(min_mag,max_mag,71)
    print('mag_bins = ',mag_bins)

    index = np.digitize(m, mag_bins)
    print('len(index) = ',len(index))
    bin_dT = [dT[index == i].mean() for i in range(1, len(mag_bins))]
    print('bin_dT = ',bin_dT)
    bin_dT_err = [ np.sqrt(dT[index == i].var() / len(dT[index == i]))
                    for i in range(1, len(mag_bins)) ]
    print('bin_dT_err = ',bin_dT_err)

    # Fix up nans
    for i in range(1,len(mag_bins)):
        if i not in index:
            bin_dT[i-1] = 0.
            bin_dT_err[i-1] = 0.
    print('fixed nans')

    print('index = ',index)
    print('bin_dT = ',bin_dT)

    if legend is None:
        legend =  [r'$\delta T$']  

    plt.plot([min_mag,max_mag], [0,0], color='black')
    plt.plot([min_mused,min_mused],[-1,1], color='Grey')
    plt.fill( [min_mag,min_mag,min_mused,min_mused], [-1,1,1,-1], fill=False, hatch='/', color='Grey')
    t_line = plt.errorbar(mag_bins[:-1], bin_dT, yerr=bin_dT_err, color=color, fmt='o')
    plt.legend([t_line], [legend])
    plt.set_ylabel(r'$T_{\rm PSF} - T_{\rm model} ({\rm arcsec}^2)$')
    plt.set_ylim(-0.002,0.012)
    plt.set_xlim(min_mag,max_mag)
    plt.set_xlabel('Magnitude')



def shape1_residual(m, de1,  min_mused,  legend=None, color=None):
    import numpy as np
    import matplotlib.pyplot as plt

    min_mag = 13.5
    max_mag = 21
    #min_mused = 15

    # Bin by mag
    mag_bins = np.linspace(min_mag,max_mag,71)
    print('mag_bins = ',mag_bins)

    index = np.digitize(m, mag_bins)
    print('len(index) = ',len(index))
    bin_de1 = [de1[index == i].mean() for i in range(1, len(mag_bins))]
    print('bin_de1 = ',bin_de1)
    bin_de1_err = [ np.sqrt(de1[index == i].var() / len(de1[index == i]))
                    for i in range(1, len(mag_bins)) ]
    print('bin_de1_err = ',bin_de1_err)

    # Fix up nans
    for i in range(1,len(mag_bins)):
        if i not in index:
            bin_de1[i-1] = 0.
            bin_de1_err[i-1] = 0.
    print('fixed nans')

    print('index = ',index)
    print('bin_de1 = ',bin_de1)
    print('bin_de2 = ',bin_de2)
    print('bin_dT = ',bin_dT)

    if legend is None:
        legend =  [r'$\delta e_1$']

    plt.plot([min_mag,max_mag], [0,0], color='black')
    plt.plot([min_mused,min_mused],[-1,1], color='Grey')
    plt.fill( [min_mag,min_mag,min_mused,min_mused], [-1,1,1,-1], fill=False, hatch='/', color='Grey')
    e1_line = plt.errorbar(mag_bins[:-1], bin_de1, yerr=bin_de1_err, color=color, fmt='o')
    plt.set_ylim(-3.e-4,6.5e-4)
    plt.set_xlim(min_mag,max_mag)
    plt.set_xlabel('Magnitude')



def shape2_residual(m, de2, min_mused,  legend=None, color=None):
    import numpy as np
    import matplotlib.pyplot as plt

    min_mag = 13.5
    max_mag = 21
    #min_mused = 15

    # Bin by mag
    mag_bins = np.linspace(min_mag,max_mag,71)
    print('mag_bins = ',mag_bins)

    index = np.digitize(m, mag_bins)
    print('len(index) = ',len(index))
    bin_de2 = [de2[index == i].mean() for i in range(1, len(mag_bins))]
    print('bin_de2 = ',bin_de2)
    bin_de2_err = [ np.sqrt(de2[index == i].var() / len(de2[index == i]))
                    for i in range(1, len(mag_bins)) ]
    print('bin_de2_err = ',bin_de2_err)


    # Fix up nans
    for i in range(1,len(mag_bins)):
        if i not in index:
            bin_de2[i-1] = 0.
            bin_de2_err[i-1] = 0.
    print('fixed nans')

    print('index = ',index)
    print('bin_de2 = ',bin_de2)

    if legend is None:
        legend =  [r'$\delta e_2$']

    plt.plot([min_mag,max_mag], [0,0], color='black')
    plt.plot([min_mused,min_mused],[-1,1], color='Grey')
    plt.fill( [min_mag,min_mag,min_mused,min_mused], [-1,1,1,-1], fill=False, hatch='/', color='Grey')
    e2_line = plt.errorbar(mag_bins[:-1], bin_de2, yerr=bin_de2_err, color=color, fmt='o')
    plt.set_ylim(-3.e-4,6.5e-4)
    plt.set_xlim(min_mag,max_mag)
    plt.set_xlabel('Magnitude')
    plt.set_ylabel(r'$e_{\rm PSF} - e_{\rm model}$')
    plt.legend([e2_line], [legend2])
   


def bin_by_mag(m, dT, de1, de2, min_mused, bands):

    import numpy as np
    import matplotlib.pyplot as plt

    min_mag = 13.5
    max_mag = 21
    #min_mused = 15

    # Bin by mag
    mag_bins = np.linspace(min_mag,max_mag,71)
    print('mag_bins = ',mag_bins)

    index = np.digitize(m, mag_bins)
    print('len(index) = ',len(index))
    bin_de1 = [de1[index == i].mean() for i in range(1, len(mag_bins))]
    print('bin_de1 = ',bin_de1)
    bin_de2 = [de2[index == i].mean() for i in range(1, len(mag_bins))]
    print('bin_de2 = ',bin_de2)
    bin_dT = [dT[index == i].mean() for i in range(1, len(mag_bins))]
    print('bin_dT = ',bin_dT)
    bin_de1_err = [ np.sqrt(de1[index == i].var() / len(de1[index == i]))
                    for i in range(1, len(mag_bins)) ]
    print('bin_de1_err = ',bin_de1_err)
    bin_de2_err = [ np.sqrt(de2[index == i].var() / len(de2[index == i]))
                    for i in range(1, len(mag_bins)) ]
    print('bin_de2_err = ',bin_de2_err)
    bin_dT_err = [ np.sqrt(dT[index == i].var() / len(dT[index == i]))
                    for i in range(1, len(mag_bins)) ]
    print('bin_dT_err = ',bin_dT_err)

    # Fix up nans
    for i in range(1,len(mag_bins)):
        if i not in index:
            bin_de1[i-1] = 0.
            bin_de2[i-1] = 0.
            bin_dT[i-1] = 0.
            bin_de1_err[i-1] = 0.
            bin_de2_err[i-1] = 0.
            bin_dT_err[i-1] = 0.
    print('fixed nans')

    print('index = ',index)
    print('bin_de1 = ',bin_de1)
    print('bin_de2 = ',bin_de2)
    print('bin_dT = ',bin_dT)

    fig, axes = plt.subplots(2,1, sharex=True)

    ax = axes[0]
    ax.set_ylim(-0.002,0.012)
    ax.plot([min_mag,max_mag], [0,0], color='black')
    ax.plot([min_mused,min_mused],[-1,1], color='Grey')
    ax.fill( [min_mag,min_mag,min_mused,min_mused], [-1,1,1,-1], fill=False, hatch='/', color='Grey')
    t_line = ax.errorbar(mag_bins[:-1], bin_dT, yerr=bin_dT_err, color='green', fmt='o')
    ax.legend([t_line], [r'$\delta T$'])
    ax.set_ylabel(r'$T_{\rm PSF} - T_{\rm model} ({\rm arcsec}^2)$')

    ax = axes[1]
    ax.set_ylim(-3.e-4,6.5e-4)
    ax.plot([min_mag,max_mag], [0,0], color='black')
    ax.plot([min_mused,min_mused],[-1,1], color='Grey')
    ax.fill( [min_mag,min_mag,min_mused,min_mused], [-1,1,1,-1], fill=False, hatch='/', color='Grey')
    e1_line = ax.errorbar(mag_bins[:-1], bin_de1, yerr=bin_de1_err, color='red', fmt='o')
    e2_line = ax.errorbar(mag_bins[:-1], bin_de2, yerr=bin_de2_err, color='blue', fmt='o')
    ax.legend([e1_line, e2_line], [r'$\delta e_1$', r'$\delta e_2$'])
    ax.set_ylabel(r'$e_{\rm PSF} - e_{\rm model}$')

    ax.set_xlim(min_mag,max_mag)
    ax.set_xlabel('Magnitude')

    fig.set_size_inches(7.0,10.0)
    plt.tight_layout()
    plt.savefig('dpsf_mag_' + bands + '.pdf')

    if True:
        cols = np.array((mag_bins[:-1],
                         bin_dT, bin_dT_err,
                         bin_de1, bin_de1_err,
                         bin_de2, bin_de2_err))
        outfile = 'dpsf_mag_' + bands + '.dat'
        np.savetxt(outfile, cols, fmt='%.6e',
                   header='mag_bins  '+
                          'dT  sig_dT '+
                          'de1  sig_de1 '+
                          'de2  sig_de2 ')
        print('wrote',outfile)
