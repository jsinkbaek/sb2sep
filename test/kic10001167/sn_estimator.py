import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.coordinates import EarthLocation
from barycorrpy import get_BC_vel, utc_tdb


files_science = [
    'FIBi300035_step011_merge.fits',
    'FIDg060105_step011_merge.fits', 'FIBj040096_step011_merge.fits', 'FIBl010111_step011_merge.fits',
    'FIDh200065_step011_merge.fits', 'FIEh190105_step011_merge.fits', 'FIDh150097_step011_merge.fits',
    'FIBl050130_step011_merge.fits', 'FIEh020096_step012_merge.fits', 'FIBk050060_step011_merge.fits',
    'FIDg070066_step011_merge.fits', 'FIDg080034_step011_merge.fits', 'FIBl080066_step011_merge.fits',
    'FIDg050102_step011_merge.fits', 'FIDh170096_step011_merge.fits', 'FIDh160097_step011_merge.fits',
    'FIBi240077_step011_merge.fits', 'FIBi230039_step011_merge.fits', 'FIBi240074_step011_merge.fits',
    'FIBk230065_step011_merge.fits', 'FIBk060008_step011_merge.fits', 'FIFj100096_step011_merge.fits',
    'FIEh060100_step012_merge.fits', 'FIBj010039_step011_merge.fits', 'FIBk030040_step011_merge.fits',
    'FIBk140072_step011_merge.fits', 'FIDh100076_step011_merge.fits', 'FIBk040034_step011_merge.fits',
    'FIEf140066_step011_merge.fits', 'FIBj150077_step011_merge.fits', 'FIDg160034_step011_merge.fits',
    # NEW SPECTRA BELOW
    'FIGb130102_step011_merge.fits', 'FIGb200113_step011_merge.fits', 'FIGb260120_step011_merge.fits',
    'FIGc030078_step011_merge.fits', 'FIGc110124_step011_merge.fits', 'FIGc170105_step011_merge.fits',
    'FIGc280075_step011_merge.fits', 'FIGc290066_step011_merge.fits', 'FIGc290075_step011_merge.fits',
    'FIGd010114_step011_merge.fits', 'FIGd070138_step011_merge.fits', 'FIGd120038_step011_merge.fits',
    'FIGd260101_step011_merge.fits', 'FIGe040084_step011_merge.fits'
]

use_for_spectral_separation = np.array([
    'FIDg060105_step011_merge.fits', 'FIDh200065_step011_merge.fits',
    'FIEh190105_step011_merge.fits', 'FIDh150097_step011_merge.fits',
    'FIBl050130_step011_merge.fits', 'FIEh020096_step012_merge.fits',
    'FIBk050060_step011_merge.fits', 'FIDg070066_step011_merge.fits',
    'FIDg080034_step011_merge.fits', 'FIBl080066_step011_merge.fits',
    'FIDg050102_step011_merge.fits', 'FIDh170096_step011_merge.fits',
    'FIDh160097_step011_merge.fits', 'FIBi240077_step011_merge.fits',
    'FIBi230039_step011_merge.fits', 'FIBi240074_step011_merge.fits',
    'FIBk060008_step011_merge.fits', 'FIFj100096_step011_merge.fits',
    'FIEh060100_step012_merge.fits', 'FIBk030040_step011_merge.fits',
    'FIBk140072_step011_merge.fits', 'FIDh100076_step011_merge.fits',
    'FIBk040034_step011_merge.fits', 'FIEf140066_step011_merge.fits',
    'FIBj150077_step011_merge.fits'
])

use_for_spectral_separation = np.array(['FIDg060105_step011_merge.fits', 'FIDh200065_step011_merge.fits',
       'FIEh190105_step011_merge.fits', 'FIDh150097_step011_merge.fits',
       'FIBl050130_step011_merge.fits', 'FIEh020096_step012_merge.fits',
       'FIBk050060_step011_merge.fits', 'FIDg070066_step011_merge.fits',
       'FIDg080034_step011_merge.fits', 'FIBl080066_step011_merge.fits',
       'FIDg050102_step011_merge.fits', 'FIDh170096_step011_merge.fits',
       'FIDh160097_step011_merge.fits', 'FIBi240077_step011_merge.fits',
       'FIBi230039_step011_merge.fits', 'FIBi240074_step011_merge.fits',
       'FIBk060008_step011_merge.fits', 'FIFj100096_step011_merge.fits',
       'FIEh060100_step012_merge.fits', 'FIBk030040_step011_merge.fits',
       'FIBk140072_step011_merge.fits', 'FIDh100076_step011_merge.fits',
       'FIBk040034_step011_merge.fits', 'FIEf140066_step011_merge.fits',
       # 'FIBj150077_step011_merge.fits'
                                        ])

files = [x.replace('_step011_merge', '') for x in files_science]
files = [x.replace('_step012_merge', '') for x in files]
files = np.array(files)
ssr = [x.replace('_step011_merge', '') for x in use_for_spectral_separation]
ssr = [x.replace('_step012_merge', '') for x in ssr]
ssr = np.array(ssr)
asort = np.argsort(files)
files = files[asort]
directory = '/media/sinkbaek/NOT_DATA/fiestool/Data/raw/kic10001167/'
files_science = np.array(files_science)[asort]

pixels = np.empty((2102, 2198, len(files)))
exptimes = np.empty((len(files), ))
dates = np.array([])
ras = np.empty((len(files), ))
decs = np.empty((len(files), ))
gains = np.empty((len(files), ))
observers = []
airmasses = np.empty((len(files), ))
middle = 1051
# plt.figure()

for i in range(len(files)):
    file = files[i]
    with fits.open(directory+file[2:6]+'/'+file) as hdul:
        xoverscan = int(hdul[0].header['XOVERSC'])
        yoverscan = int(hdul[0].header['YOVERSC'])
        gains[i] = hdul[1].header['GAIN']
        data = hdul[1].data
        overscan = data[2:data.shape[0]-2, 2:xoverscan-1]
        bias_estimate = np.median(overscan)

        pixels[:, :, i] = data - bias_estimate
        dates = np.append(dates, hdul[0].header['DATE-AVG'])
        exptimes[i] = hdul[0].header['EXPTIME']
        ras[i] = hdul[0].header['OBJRA']*15.0
        decs[i] = hdul[0].header['OBJDEC']
        observers.append(hdul[0].header['OBSERVER'])
        airmasses[i] = hdul[0].header['AIRMASS']

    # plt.plot(pixels[middle, :, i] / np.nanquantile(pixels[middle, :, i], 0.95) - 1.5*i)

# plt.ylim([-1.5*len(files)-1, 2])
# plt.show()

# Group 1 is spectrum 0 to 15 (inclusive)
# Group 2 is spectrum 16 to 26
# Group 3 is spectrum 27 to end
x_group1 = 1599
x_group2 = 1627
x_group3 = 1635
y_group1 = 1100
y_group2 = 1050
y_group3 = 1100
xs = [
    1600, 1600, 1600, 1600, 1600, 1600, 1600,
    1599, 1599, 1599, 1599, 1599, 1599, 1599, 1599, 1599,
    1635, 1635, 1635, 1635, 1635, 1635, 1635, 1635, 1635, 1635,
    1634,
    1624, 1624,
    1625,
    1624, 1624,
    1625, 1625, 1625, 1625,
    1626,
    1625, 1625, 1625, 1625, 1625, 1625,
    1627, 1627
]
xws = [
    11, 11, 11, 11, 11, 11, 11,
    11, 11, 11, 11, 11, 11, 11, 11, 11,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7,
    7, 7,
    7,
    7, 7,
    7, 7, 7, 7,
    7,
    7, 7, 7, 7, 7, 7,
    7, 7
]
ys = [
    1096, 1080, 1099, 1078, 1097, 1085, 1099,
    1097, 1090, 1110, 1100, 1085, 1129, 1080, 1095, 1098,
    1104, 1097, 1102, 1088, 1115, 1123, 1091, 1090, 1096, 1093,
    1128,
    1050, 1054,
    1050,
    1039, 1054,
    1050, 1051, 1051, 1056,
    1053,
    1050, 1050, 1050, 1050, 1032, 1050,
    1050, 1050
]

# bckg_percent_1 = 15     # guesstimated background contribution in percent (from 1 frame) Bl08 ~15-20%. Bl01 ~12%
# bckg_percent_2 = 12     # Dg05 ~9-10%
# bckg_percent_3 = 8
bckg_percent = 15


counts = np.empty((len(files), ))
counts_star = np.empty((len(files), ))
for i in range(len(files)):
    # if i < 16:
    #     x = x_group1
    #     y = y_group1
    #     bckg_percent = bckg_percent_1
    #     xw = 11
    # elif i < 27:
    #     x = x_group2
    #     y = y_group2
    #     bckg_percent = bckg_percent_2
    #     xw = 7
    # else:
    #     x = x_group3
    #     y = y_group3
    #     bckg_percent = bckg_percent_3
    #     xw = 7
    x = xs[i]
    y = ys[i]
    xw = xws[i]
    yw = 7
    vals = pixels[y-(yw-1)//2:y+(yw-1)//2+1, x-(xw-1)//2:x+(xw-1)//2+1, i]
    vals_bckg = np.concatenate((
        pixels[y-(yw-1)//2:y+(yw-1)//2+1, x-(xw-1)//2-4:x-(xw-1)//2-1, i],
        pixels[y-(yw-1)//2:y+(yw-1)//2+1, x+(xw-1)//2+3:x+(xw-1)//2+6, i]
    ))
    median_bck = np.median(vals_bckg)
    median_y = np.nanmedian(vals, axis=0)
    # print(f'{np.mean(median_y) * bckg_percent/100:.0f}\t{median_bck:.0f}')
    # median_y -= median_bck
    # median_y -= np.mean(median_y) * bckg_percent/100
    count_x = np.sum(median_y)
    counts[i] = count_x
    counts_star[i] = np.sum(median_y-median_bck)

sn = np.sqrt(counts*gains)
sn_star = np.sqrt(counts_star*gains)
sn_1sec = np.sqrt(counts_star*gains/exptimes)
e_s = counts_star*gains/exptimes        # electrons per second


# BJD and Barycentric correction
RA, DEC = ras[0], decs[0]
observatory_location = EarthLocation.of_site("Roque de los Muchachos")
observatory_name = "Roque de los Muchachos"
stellar_target = "kic10001167"
times = Time(dates, scale='utc', location=observatory_location)
times.format = 'jd'
times.out_subfmt = 'long'
bc_rv_cor, _, _ = get_BC_vel(
    times, ra=RA, dec=DEC, starname=stellar_target, ephemeris='de432s', obsname=observatory_name
)
bc_rv_cor = bc_rv_cor/1000      # from m/s to km/s
bjdtdb, _, _ = utc_tdb.JDUTC_to_BJDTDB(times, ra=RA, dec=DEC, starname=stellar_target, obsname=observatory_name)


# Load uncertainty estimates
err_thar = np.loadtxt('thar_uncertainties_sorted.dat')
err_telluric = np.loadtxt('telluric_cross_validation_std.dat')[asort]
idx_b = [
    1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
    36,  # 37, 38, 39, 40
    41, 42, 43, 44
]
err_sn_a = np.loadtxt('errs_A_corrected.txt')[asort]
err_sn_b = np.loadtxt('errs_B_corrected.txt')

# Load RVs
rvA = np.loadtxt('/home/sinkbaek/PycharmProjects/Seismic-dEBs/Binary_Analysis/JKTEBOP/kic10001167/kepler_pdcsap_23/rvA_with_jitter.dat')
rvA = rvA[asort, :]
rvB = np.loadtxt('/home/sinkbaek/PycharmProjects/Seismic-dEBs/Binary_Analysis/JKTEBOP/kic10001167/kepler_pdcsap_23/rvB_with_jitter.dat')

idx_b_sorted = []
for i in range(rvB.shape[0]):
    tb = rvB[i, 0]
    tas = rvA[:, 0]
    amin = np.argmin(np.abs(tas-tb))
    idx_b_sorted.append(amin)

# Load telluric correction
telluric = np.loadtxt('tell_rv_tot.txt')[asort]

# Spectral separation mask
ssr_mask = np.isin(files, ssr)
ssr_mask.astype(float)

# SSR weight
ssr_weight = ssr_mask * sn**2
ssr_weight /= np.max(ssr_weight)
ssr_weight = np.sqrt(exptimes) / np.sqrt(np.max(exptimes)) * ssr_mask

# Pretty print
daystr = 'RBJD'
snstr = 'SN@5880Å'
snstarstr = 'SN*'
snsecstr = 'SN(1s)'
wstr = 'W'
bcstr = 'BaryC'
tellstr = 'TellC'
rvastr = 'RV A'
rvbstr = 'RV B'
eastr = 'eA'
ebstr = 'eB'
esnastr = 'eSN,A'
esnbstr = 'eSN,B'
ethstr = 'eTHAR'
etellstr = 'eTELL'
filestr = 'Filestring'
phasestr = 'Phase'
exptstr = 'ExpT'
esstr = 'e-/s'
airmstr = 'Airmass'
obsstr = 'Observer'
print(f'Median electron rate: {np.median(e_s):.4f}')
print(f'Median electron rate 2023: {np.median(e_s[-13:]):.4f}')
print(f'STD electron rate 2023: {np.std(e_s[-13:]):.4f}')

print(f'{daystr:>10}\t{phasestr:>5}\t{snstr:>8}\t{snstarstr:>3}\t{snsecstr:>6}\t{esstr:>5}\t{exptstr:>4}\t{wstr:>5}\t{bcstr:>8}\t{tellstr:>8}\t{rvastr:>8}\t{rvbstr:>8}\t{eastr:>5}\t{ebstr:>5}\t{esnastr:>5}\t{esnbstr:>5}\t{ethstr:>5}\t{etellstr:>5}\t{filestr:>13}\t{airmstr:>7}\t{obsstr:>30}')
t0 = 2455028.0987612916
period = 120.3900601075
phase = np.mod(bjdtdb-t0, period)/period

with open('sn_table.csv', 'w') as outfile:
    outfile.write(
        'RBJD\tSN\tEXPT\tW\tBC\tTC\tRVA\tRVB\tERRA\tERRB\tESNA\tESNB\tETHA\tETELL\tAIRM\n'
    )
    for i in range(len(files)):
        if i in idx_b_sorted:
            idx = np.isin(idx_b_sorted, i)
            rvb, errb, errsnb = f'{rvB[idx, 1][0]:8.3f}', f'{rvB[idx, 2][0]:5.3f}', f'{err_sn_b[idx][0]:5.3f}'
        else:
            rvb, errb, errsnb = '---', '---', '---'
            rvb = f'{rvb:>8}'
            errb = f'{errb:>5}'
            errsnb = f'{errsnb:>5}'
        outfile.write(
            f'{bjdtdb[i]-2450000:10.4f}\t{sn_star[i]:3.0f}\t{exptimes[i]:4.0f}\t{ssr_weight[i]:5.3f}\t{bc_rv_cor[i]:8.3f}\t{-telluric[i]:8.3f}\t{rvA[i, 1]:8.3f}\t{rvb}\t{rvA[i, 2]:5.3f}\t{errb}\t{err_sn_a[i]:5.3f}\t{errsnb}\t{err_thar[i]:5.3f}\t{err_telluric[i]:5.3f}\t{airmasses[i]:7.2f}\n'
        )
        print(
            f'{bjdtdb[i] - 2450000:10.4f}\t{phase[i]:5.3f}\t{sn[i]:8.0f}\t{sn_star[i]:3.0f}\t{sn_1sec[i]:6.4f}\t{e_s[i]:5.3f}\t{exptimes[i]:4.0f}\t{ssr_weight[i]:5.3f}\t{bc_rv_cor[i]:8.3f}\t{-telluric[i]:8.3f}\t{rvA[i, 1]:8.3f}\t{rvb}\t{rvA[i, 2]:5.3f}\t{errb}\t{err_sn_a[i]:5.3f}\t{errsnb}\t{err_thar[i]:5.3f}\t{err_telluric[i]:5.3f}\t{files[i]:.>13}\t{airmasses[i]:7.2f}\t{observers[i]:>30}')