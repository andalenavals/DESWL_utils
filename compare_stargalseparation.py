# STANDARD WAY TO RUN 
# ./plot_rho.py --file=zone29.riz 
import os
import glob
import galsim
import json
import numpy as np
from read_psf_cats import read_data
from plot_mag import size_residual,  shape1_residual,  shape2_residual

plt.style.use('supermongo.mplstyle')

RESERVED = 64

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='Run PSFEx on a set of runs/exposures')

    # Drectory arguments
    parser.add_argument('--catalogpaths',
                        default=['/home2/dfa/sobreira/alsina/ourcatalogs/y3a1-v10-piff02-magcut0-maxmag21-sf1',
                                 '/home2/dfa/sobreira/alsina/ourcatalogs/y3a1-v10-piff02-magcut0-maxmag21-sf2'],
                        help='location of the catalog')
    parser.add_argument('--outpath', default='./',
                        help='location of the output of the files')
    # Exposure inputs
    parser.add_argument('--file', default='',
                        help='list of exposures (in lieu of exps)')
    parser.add_argument('--exps', default='', nargs='+',
                        help='list of exposures to run')

    # Options
    parser.add_argument('--use_reserved', default=False, action='store_const', const=True,
                        help='just use the objects with the RESERVED flag')
    parser.add_argument('--use_psfex', default=False, action='store_const', const=True,
                        help='Use PSFEx rather than Piff model')
    parser.add_argument('--bands', default='grizY', type=str,
                        help='Limit to the given bands')
    parser.add_argument('--frac', default=1., type=float,
                        help='Choose a random fraction of the input stars')

    args = parser.parse_args()
    return args


def get_fields(bands, expsfile,  catpath):

    with open(expsfile) as fin:
        exps = [ line.strip() for line in fin if line[0] != '#' ]
    exps = sorted(exps)

    keys = ['ra', 'dec', 'x', 'y', 'mag', 'obs_e1', 'obs_e2', 'obs_T', 'obs_flag',
            prefix+'_e1', prefix+'_e2', prefix+'_T', prefix+'_flag']
    prefix = 'piff'

    
    work = os.path.expanduser(catpath)
    print('work dir = ',work)
    data, bands, tilings = read_data(exps, work, keys,
                                     limit_bands=bands,
                                     prefix=prefix,
                                     use_reserved=args.use_reserved,
                                     frac=args.frac)


    bands = ['riz']
    this_data = data[np.in1d(data['band'], list(bands))]
    if len(this_data) == 0:
        print('No files with bands ',bands)
        raise Exception ('No files with bands ',bands)
    
    used = this_data[prefix+'_flag'] & ~RESERVED == 0
    
    ra = this_data['ra']
    dec = this_data['dec']
    x = this_data['x']
    y = this_data['y']
    m = this_data['mag']
    print('full mag range = ',np.min(m),np.max(m))
    print('used mag range = ',np.min(m[used]),np.max(m[used]))
    
    e1 = this_data['obs_e1']
    print('mean e1 = ',np.mean(e1))
    e2 = this_data['obs_e2']
    print('mean e2 = ',np.mean(e2))
    T = this_data['obs_T']
    print('mean s = ',np.mean(T))
    pe1 = this_data[prefix+'_e1']
    print('mean pe1 = ',np.mean(pe1))
    pe2 = this_data[prefix+'_e2']
    print('mean pe2 = ',np.mean(pe2))
    pT = this_data[prefix+'_T']
    print('mean pT = ',np.mean(pT))
    
    print('min mag = ',np.min(m))
    print('max mag = ',np.max(m))
    print('mean T (used) = ',np.mean(T[used]))
    print('mean e1 (used) = ',np.mean(e1[used]))
    print('mean e2 (used) = ',np.mean(e2[used]))
    print('mean pT (used) = ',np.mean(pT[used]))
    print('mean pe1 (used) = ',np.mean(pe1[used]))
    print('mean pe2 (used) = ',np.mean(pe2[used]))
    de1 = e1 - pe1
    de2 = e2 - pe2
    dT = T - pT
    print('mean dT (used) = ',np.mean(dT[used]))
    print('mean de1 (used) = ',np.mean(de1[used]))
    print('mean de2 (used) = ',np.mean(de2[used]))
    
    if args.use_psfex:
        min_mused = 0
    else:
        min_mused = np.min(m[used])
    print('min_mused = ',min_mused)

    return m[used], dT[used], de1[used], de2[used], min_mused

def main():
    matplotlib.use('Agg') # needs to be done before import pyplot

    args = parse_args()

   

    outpath = os.path.expanduser(args.outpath)
    try:
        if not os.path.exists(outpath):
            os.makedirs(outpath)
    except OSError:
        if not os.path.exists(outpath): raise

    

    m, dT, de1, de2, min_mused =  getfields(args.bands, args.file, args.catalogpaths[0] )
    m2, dT2, de12, de22, min_mused2 =  getfields(args.bands, args.file, args.catalogpaths[1] )
    
    size_residual(m, dT, de1, de2, min_mused,legend=r'$\delta T$', color='red')
    plt.tight_layout()
    plt.savefig(outpath + '/e1_residuals_mag_' + bands + '.png')

    shape1_residual(m, de1, min_mused, legend='$\delta e_{1}$', color='red')
    plt.set_ylabel(r'$e_{\rm PSF} - e_{\rm model}$')
    plt.legend([e1_line], [legend1])

    shape2_residual(m,  de2, min_muse, legend='$\delta e_{2}$', color='red')
    plt.tight_layout()
    plt.savefig(outpath + '/e2_residuals_mag_' + bands + '.png')


if __name__ == "__main__":
    main()
