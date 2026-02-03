
import os
import math
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from photutils.detection import DAOStarFinder
from astropy.time import Time
from photutils.psf import fit_fwhm
from scipy.optimize import curve_fit
from scipy.stats import chisquare
from scipy.stats import shapiro
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
from astropy import units as u
import shutil
from astropy.coordinates import Angle
from astropy.table import QTable
from photutils.aperture import CircularAperture, aperture_photometry
# from photutils import CircularAperture
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)
from scipy.optimize import OptimizeWarning
warnings.filterwarnings("ignore", category=OptimizeWarning)
from scipy.stats import multivariate_normal
# size =7
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from photutils.aperture import CircularAperture
import matplotlib.pyplot as plt
import os

# Define a 1D Gaussian model
def gaussian_1d(x, A, mu, sigma, offset):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + offset
def gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y, offset):
    (x, y) = coords
    g = offset + amplitude * np.exp(
        -(((x - x0) ** 2) / (2 * sigma_x ** 2)
          + ((y - y0) ** 2) / (2 * sigma_y ** 2))
    )
    return g.ravel()   # <--- flatten before returning
def fit_star_cutout(star_cutout, sigma_guess=2.0, plot=True):
    ny, nx = star_cutout.shape

    # Pixel grid
    y = np.arange(ny)
    x = np.arange(nx)
    x, y = np.meshgrid(x, y)

    # Initial guesses
    amp_guess = star_cutout.max() - star_cutout.min()
    x0_guess = nx / 2
    y0_guess = ny / 2
    offset_guess = star_cutout.min()
    initial_guess = (amp_guess, x0_guess, y0_guess, sigma_guess, sigma_guess, offset_guess)
    try:
        popt, pcov = curve_fit(
            gaussian_2d,
            (x, y),
            star_cutout.ravel(),
            p0=initial_guess,
            maxfev=5000
        )
        # print("Fit successful!")
        # print("Parameters:", popt)

    except RuntimeError as e:
        # print("Optimal parameters not found:", e)
        # print("Showing the problematic star_cutout...")
        # plt.imshow(star_cutout, origin='lower', cmap='viridis')
        # plt.show()
        pcov = None
        popt = [0, 0, 0, 0, 0, 0]
    # Extract fitted parameters
    amp_fit, x0_fit, y0_fit, sigx_fit, sigy_fit, offset_fit = popt

    # Compute FWHM
    # Compute FWHM
    # if sigx_fit > 3.5 and sigy_fit >3.5:
    #     highsig = "high"
    # else:
    #     highsig = "low"
    fwhm_x = 2.355 * sigx_fit
    fwhm_y = 2.355 * sigy_fit

    # print(f"FWHM_x = {fwhm_x:.2f}, FWHM_y = {fwhm_y:.2f}")
    return fwhm_x,fwhm_y

    # larger = max(fwhm_x, fwhm_y)
    # smaller = min(fwhm_x, fwhm_y)
    # # Avoid dividing by zero
    # d =larger-smaller
    # ratio = larger / smaller if smaller != 0 else float('inf')
    # if ratio <1.1:
    #     star_fit = gaussian_2d((x, y), *popt).reshape(ny, nx)

    #     # Residuals
    #     residuals = star_cutout - star_fit

    #     # Plot
    #     fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    #     ax[0].imshow(star_cutout, origin='lower', cmap='viridis')
    #     ax[0].set_title("Original Star Cutout")

    #     ax[1].imshow(star_fit, origin='lower', cmap='viridis')
    #     ax[1].set_title("Fitted 2D Gaussian")

    #     contour = ax[2].contour(star_cutout, levels=5, origin='lower', cmap='viridis')
    #     ax[2].set_title("2D Gaussian Contour")
    #     ax[2].clabel(contour, inline=False, fontsize=8)
    #     # plt.title(
    #     #     f"σx={sigx_fit:.2f} px, σy={sigy_fit:.2f} px, "
    #     #     f"FWHM≈{iq:.2f} arc, f_d : {round(ratio,3)}")
    #     plt.show()
    # else:
    #     print(ratio)
















    # Avoid division by zero by handling the special case first
    # if sigx_fit == 0 and sigy_fit == 0:
    #     ratio = 0.1
    # else:
    #     # Pick the larger and smaller FWHM to compute a consistent ratio ≥ 1
    #     larger = max(fwhm_x, fwhm_y)
    #     smaller = min(fwhm_x, fwhm_y)
    #     # Avoid dividing by zero
    #     d =larger-smaller
    #     ratio = larger / smaller if smaller != 0 else float('inf')
    #     print("\nsigma diff : ",d)
    #     print("ratio      : ",ratio)
    #     print("Iq         : ",round(iq,3))
    #     print("sigma_x    : ",round(sigx_fit,3))
    #     print("sigma_x    : ",round(sigx_fit,3))
    # print(f"Amplitude = {amp_fit:.2f}")
    # print(f"Centroid  = ({x0_fit:.2f}, {y0_fit:.2f})")
    # print(f"Sigma_x   = {sigx_fit:.2f}, Sigma_y = {sigy_fit:.2f}")

    # print(f"ratio     = {ratio:.2f}")
    # print(f"FWHM      = {(fwhm_x+fwhm_y)/2:.2f}")
    # print(f"IQ        = {iq:.2f}")
    # print(f"Background offset = {offset_fit:.2f}")
    # if fwhm_y >fwhm_x:
    #     ratio = fwhm_y/fwhm_x
    # else:
    #     ratio =fwhm_x/fwhm_y
    # if 1 <= ratio < 1.2 and d<.4 and highsig == "low" :

    # # if plot:
    #     # Generate fitted model
    #     # star_fit = gaussian_2d((x, y), *popt).reshape(ny, nx)

    #     # # Residuals
    #     # residuals = star_cutout - star_fit

    #     # # Plot
    #     # fig, ax = plt.subplots(1, 3, figsize=(15, 7))
    #     # ax[0].imshow(star_cutout, origin='lower', cmap='viridis')
    #     # ax[0].set_title("Original Star Cutout")

    #     # ax[1].imshow(star_fit, origin='lower', cmap='viridis')
    #     # ax[1].set_title("Fitted 2D Gaussian")

    #     # im = ax[2].imshow(residuals, origin='lower', cmap='bwr')
    #     # ax[2].set_title("Residuals")
    #     # plt.colorbar(im, ax=ax[2], fraction=0.046)

    #     # plt.title(
    #     #     f"σx={sigx_fit:.2f}, σy={sigy_fit:.2f}, "
    #     #     f"FWHM≈{iq:.2f}, f_d : {round(ratio,3)}"
    #     # )

    #     # plt.tight_layout()
    #     # plt.savefig(f"/home/dataarchive/Documents/doastarfinder/good/{str(round(iq,3))}.jpg")
    #     # # plt.show()
    #     # plt.close()
    #     status = "GOOD"
    #     print(status)
    # else:
    #     star_fit = gaussian_2d((x, y), *popt).reshape(ny, nx)

    #     # Residuals
    #     residuals = star_cutout - star_fit

    #     # Plot
    #     fig, ax = plt.subplots(1, 3, figsize=(15, 7))
    #     ax[0].imshow(star_cutout, origin='lower', cmap='viridis')
    #     ax[0].set_title("Original Star Cutout")

    #     ax[1].imshow(star_fit, origin='lower', cmap='viridis')
    #     ax[1].set_title("Fitted 2D Gaussian")

    #     im = ax[2].imshow(residuals, origin='lower', cmap='bwr')
    #     ax[2].set_title("Residuals")
    #     plt.colorbar(im, ax=ax[2], fraction=0.046)

    #     plt.title(
    #         f"σx={sigx_fit:.2f} px, σy={sigy_fit:.2f} px, "
    #         f"FWHM≈{iq:.2f} arc, f_d : {round(ratio,3)}"
    #     )

    #     plt.tight_layout()

    #     os.makedirs(f"/home/dataarchive/Documents/doastarfinder/bad/{year}",exist_ok=True)
    #     plt.savefig(f"/home/dataarchive/Documents/doastarfinder/bad/{year}/iq_{str(round(iq,3))}.jpg")
    #     # plt.show()
    #     plt.close()
    #     status= "BAD"
    #     print(status)

    # return popt,status



def save_bad_star_image(image, median, fwhm_x, fwhm_y, fwhm, chi, chi1, chi2, bin, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    plt.imshow(image - median, origin='lower')
    plt.title(f"Bin: {bin} | X: {round(fwhm_x,2)}, Y: {round(fwhm_y,2)}, FWHM: {round(fwhm,2)} | chi: {round(chi,4)}")

    # Optionally add text box inside plot (commented by default)
    # plt.text(0.8, 0.8,
    #          f"Bin : {bin}",
    #          transform=plt.gca().transAxes,
    #          fontsize=12,
    #          bbox=dict(facecolor='yellow', alpha=0.5))

    avg_chi = round((chi1 + chi2) / 2, 5)
    filepath = os.path.join(output_dir, f"{avg_chi}.jpg")
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()

def sexagesimal_to_decimal(coord_str):
    d, m, s = map(float, coord_str.split(':'))
    return d + m/60 + s/3600
def createstareg(x, y,size):
    """
    Extract a small cutout of a bright star centered at (x, y).
    
    Parameters:
        x (float): X-coordinate of the star's center.
        y (float): Y-coordinate of the star's center.
        
    Returns:
        numpy.ndarray or None: Extracted cutout of the star data or None if the star is in the corner.
    """
    # Define the size of the cutout region
    # size = 7  # Half-size of the region around the star
    x_min, x_max = int(x - size), int(x + size)
    y_min, y_max = int(y - size), int(y + size)


    tolerance_percent = 0.1  # 5% tolerance
    x_tol = data.shape[1] * tolerance_percent
    y_tol = data.shape[0] * tolerance_percent

    # Check if the cutout goes out of bounds
    if x_min < x_tol or x_max > data.shape[1]-x_tol or y_min < y_tol or y_max > data.shape[0]-y_tol:
        # print(f"Star at ({x}, {y}) is too close to the edge or corner. Skipping.")
        return None

    # Extract the cutout region
    star_data = data[y_min:y_max, x_min:x_max]

    return star_data
def fit_gaussian_linecut(linecut):
    x = np.arange(1, len(linecut)+1)
    y = linecut

    # Normalize data
    y_min, y_max = np.min(y), np.max(y)
    y_norm = (y - y_min) / (y_max - y_min)

    # Initial guesses
    A_guess = 1.0
    mu_guess = x[np.argmax(y_norm)]
    sigma_guess = np.std(x)
    p0 = [A_guess, mu_guess, sigma_guess]

    # Fit Gaussian
    try:
        popt, pcov = curve_fit(gaussian, x, y_norm, p0=p0)
        A_fit, mu_fit, sigma_fit = popt

        # Evaluate normalized fit
        y_fit_norm = gaussian(x, *popt)

        # Compute residuals
        residuals = y_norm - y_fit_norm

        # Reduced chi-square (assuming 11 degrees of freedom like your code)
        chi2_red = np.sum(residuals**2) / 11

        return popt, chi2_red
    except RuntimeError:
        # print("Fit failed.")
        pass
        return None, None
def analyze_image_gaussian(image, size):
    # Horizontal cut (row)
    horiz_line = image[size, :]
    popt_h, chi2_h = fit_gaussian_linecut(horiz_line)

    # Vertical cut (column)
    vert_line = image[:, size]
    popt_v, chi2_v = fit_gaussian_linecut(vert_line)

    # Print results
    # print("\n")
    # print("Horizontal Fit Parameters:", popt_h)
    # print("Horizontal Reduced Chi2:", chi2_h)
    # # print("Vertical Fit Parameters:", popt_v)
    # print("Vertical Reduced Chi2:", chi2_v)
    if chi2_h is None:
        chi2_h = 0.0
    if chi2_v is None:
        chi2_v = 0.0
    chi2 = (chi2_h+chi2_v)/2
    # print(chi2)
    return chi2_h,chi2_v
def findsource(bin,data):
    # mean, median, std = (np.mean(data),np.median(data),np.std(data))
    from astropy.stats import sigma_clipped_stats

    mean, median, std = sigma_clipped_stats(data, sigma=3.0, maxiters=5)
    if bin == 2:
        Fwhm = 8.1
    elif bin == 3:
        Fwhm = 5.4
    elif bin == 4:
        Fwhm = 4.1
    else:
        Fwhm = 16.25  # default for binning = 1
    # print(mean)
    daofind = DAOStarFinder(
                threshold=3.0*std,
                fwhm=Fwhm,
                ratio=1,
                sharplo=0.2,
                sharphi=1,
                roundlo=-0.5,
                roundhi=0.5,
                peakmax=55000)
    sources = daofind(data-median)
    return sources,median

def gaussian(x, A, mu, sigma):
    return A * np.exp(- (x - mu)**2 / (2 * sigma**2))

years = [2022]
for year in years:
    year = str(year)
    print(year)
    if not os.path.exists(f"/home/dataarchive/Documents/IQ_june/image/{year}_processed.dat"):
        with open(f"/home/dataarchive/Documents/IQ_june/image/{year}_processed.dat", "w") as f:
            f.write("filename\n")  # write header

    # Step 2: Read existing filenames to avoid duplicates
    with open(f"/home/dataarchive/Documents/IQ_june/image/{year}_processed.dat", "r") as f:
        existing_files = set(line.strip() for line in f.readlines()[1:])  # skip header
    # if not os.path.exists(f"/home/dataarchive/Documents/IQ_june/CSV{year}.csv"):
    #     # Create a new DataFrame with the specified columns
    #     file_det = pd.DataFrame(columns=[ 'file', 'Date', 'JD', 'FILTER1', 'EXPTIME', 'Airmass', 
    #                                     'median_fwhm', 'BINNING', 'Quality', 'Accuracy',"RA","DEC","ALT","Azi","sharpness","roundness1","roundness2"])
    #     file_det.to_csv(f"/home/dataarchive/Documents/IQ_june/CSV/{year}.csv", index=False)
    #     # print('not Exists')
    #     csv_fil = pd.read_csv(f"/home/dataarchive/Documents/IQ_june/CSV/{year}.csv")
    # else:
    #     # Load the existing CSV file
    #     csv_fil = pd.read_csv(f"/home/dataarchive/Documents/IQ_june/CSV_curvefit/{year}.csv")

    # print(np.array(csv_fil["file"]))
    count =0
    # for path,dir,files in os.walk(f"/home/dataarchive/imager_data/rawdata/{year}"):
    for path,dir,files in os.walk(f"/home/dataarchive/imager_data/rawdata/No_ra_dec/{year}"):
        # print("HI")
        for file in files:
            count= count +1
            if file in existing_files:
                print("file exists")
                continue
            csv_fil = pd.read_csv(f"/home/dataarchive/Documents/IQ_june/CSV/{year}.csv")
            if  file in np.array(csv_fil["file"]):
                # print("HI")
                print(f"\n --------------------------------------------------{count}.............................................\n")
                print(file)
                filename = os.path.join(path,file)
                hdul = fits.open(filename)
                header = hdul[0].header
                # date = header["DATE-OBS"]
                try:
                    with fits.open(filename) as hdul:
                        data = hdul[0].data
                except TypeError as e:
                    if "buffer is too small" in str(e):
                        print("❌ ERROR: FITS file may be corrupted or incomplete — buffer is too small for requested array.")
                        os.makedirs(f"/home/dataarchive/imager_data/rawdata/bufferistoosmall/{year}/{os.path.basename(filename)}",exist_ok=True)
                        shutil.move(filename, f"/home/dataarchive/imager_data/rawdata/bufferistoosmall/{year}/{os.path.basename(filename)}")
                        continue
                    else:
                        raise  # re-raise the error if it's not the one we're checking for

                # data   = hdul[0].data
                bin = header["XBINNING"]
                # if "AIRMASS" in header:
                # print("Airmass ഉണ്ട്. ")
                sources,median = findsource(bin,data)
                # print(f"അകെ {len(sources)} source കിട്ടി.")
                if sources is None :
                    print("Source number is zero")
                    os.makedirs(f"/home/dataarchive/imager_data/problem_files/less_sources/{year}/zero", exist_ok=True)
                    shutil.move(filename, f"/home/dataarchive/imager_data/problem_files/less_sources/{year}/zero/{os.path.basename(filename)}")

                    continue
                s= np.sqrt(np.median(sources['npix']))
                size = int(np.ceil(s))+1
                # print(f"ഒരു സ്റ്റാർ ഏകദേശം {size} pixel ഉണ്ട്. ")
                if sources is None :
                    # print(print("None"))
                    os.makedirs(f"/home/dataarchive/imager_data/problem_files/less_sources/{year}/{os.path.basename(filename)}",exist_ok=True)
                    shutil.move(filename, f"/home/dataarchive/imager_data/problem_files/less_sources/{year}/")
                    continue
                # print(f"Source before filter : {len(sources)}")
                # filter = (sources['roundness1'] >= -0.5) & (sources['roundness1'] <= 0.5) & \
                #     (sources['roundness2'] >= -0.5) & (sources['roundness2'] <= 0.5)
                # sources= sources[filter]
                # print(f"Source after filter : {len(sources)}")
                if len(sources) <= 3:
                    os.makedirs(f"/home/dataarchive/imager_data/problem_files/less_sources/{year}", exist_ok=True)
                    shutil.move(filename, f"/home/dataarchive/imager_data/problem_files/less_sources/{year}/{os.path.basename(filename)}")
                    print(f"File moved to '/less_sources/{year}/....'")
                    continue
                else:
                    print("\n Good to analysis chale.. ")
                # if not len(sources) > 5:
                #     os.makedirs(f"/home/dataarchive/imager_data/problem_files/less_sources/{year}/{os.path.basename(filename)}",exist_ok=True)
                #     shutil.move(filename, f"/home/dataarchive/imager_data/problem_files/less_sources/{year}/")
                #     continue

                print("Source identified : ",len(sources))
                x_coords = sources['xcentroid']
                y_coords = sources['ycentroid']
                star_positions = np.vstack((x_coords, y_coords)).T

                tree = cKDTree(star_positions)
                # radius_arcsec = physical distance you care about
                radius_arcsec = 2
                pixel_scale_binned = 0.09549 * bin  # adjust for 2, 3, or 4
                radius_pixels = radius_arcsec / pixel_scale_binned

                neighbors = tree.query_ball_point(star_positions, radius_pixels)

                isolated_stars = [star_positions[i] for i, n in enumerate(neighbors) if len(n) == 1]
                # print(isolated_stars)
                images =[]
                im_cod_array = []
                fwhm_array =[]

                selected_image_count =1
                for i, (x, y) in enumerate(isolated_stars):
                    image = createstareg(x,y,size)
                    # print(image)
                    if image is None:
                        # print('No Image')
                        pass
                    else:
                        chi1, chi2 = analyze_image_gaussian(image, size)
                        chi_diff = abs(chi1 - chi2)
                        chi = (chi1 + chi2) / 2

                        if year in (2018, 2022):
                            threshold = 0.005
                        else:
                            threshold = 0.008

                        tru = sum([
                            chi1 <= threshold,
                            chi2 <= threshold,
                            chi_diff <= threshold
                        ])
                        # tru = sum([
                        #     chi1 <= 0.005,
                        #     chi2 <= 0.005,
                        #     chi_diff <= 0.005
                        # ])
                        fwhm_x, fwhm_y = fit_star_cutout(image, sigma_guess=2.0, plot=True)
                        fwhm = (fwhm_x + fwhm_y) / 2

                        # Proceed only if at least 2 chi conditions are satisfied
                        if tru >= 2:
                            if chi == 0.0:
                                # print("chi zero continued................................................................")
                                pass
                            elif abs(fwhm_x - fwhm_y) > 2:
                                # print("bad : elongated")
                                # output_dir = "/home/dataarchive/Documents/IQ_june/image/bad/elongated/"
                                # save_bad_star_image(image, median, fwhm_x, fwhm_y, fwhm, chi, chi1, chi2, bin, output_dir)
                                pass
                            elif (abs(fwhm_x) >= 20) or ( abs(fwhm_y) >= 20):
                                # print("bad : hifwhm")
                                # output_dir = "/home/dataarchive/Documents/IQ_june/image/bad/hifwhm/"
                                # save_bad_star_image(image, median, fwhm_x, fwhm_y, fwhm, chi, chi1, chi2, bin, output_dir)
                                pass
                            elif abs(fwhm_x) < 1.5 or abs(fwhm_y) < 1.5:
                                # print("bad : Cr")
                                # output_dir = "/home/dataarchive/Documents/IQ_june/image/bad/cr"
                                print(f"............ CR ................ Binning :{bin}.")
                                # print(image)
                                # plt.imshow(image)
                                # plt.show()
                                # save_bad_star_image(image, median, fwhm_x, fwhm_y, fwhm, chi, chi1, chi2, bin, output_dir)
                                pass
                            elif chi != 0.0:
                                # print(image)
                                # plt.imshow(image)
                                # plt.show()
                                print(f"Good : selected Binning :{bin}.")

                                fwhm_array.append(fwhm)
                                images.append(image)
                                im_cod_array.append((x, y))
                                pass

                # print("sources selcted :", len(images))
                coords_df = pd.DataFrame(im_cod_array, columns=['xcentroid', 'ycentroid'])
                
                if len(coords_df)>3:
                    print("selected  sources  :",len(coords_df))
                    sources_df = sources.to_pandas()
                    selected_df = sources_df.merge(coords_df, on=['xcentroid', 'ycentroid'])
                    # print(selected_df.columns)

                    median_fwhm =np.median(fwhm_array)
                    iq = median_fwhm*bin*0.09549
                    accuracy = np.std(fwhm_array)/(np.sqrt(len(fwhm_array)))
                    round1 = np.median(selected_df["roundness1"])
                    round2 = np.median(selected_df["roundness2"])
                    sharpn = np.median(selected_df["sharpness"])

                    print("Filename        : ",file)
                    print("Binning         : ",bin)
                    print("selected stars  : ",len(coords_df))
                    print("Median FWHM     : ",median_fwhm," pixels")
                    print("Image Quality   : ",iq, " arcsecond")
                    print("Accuracy        : ",accuracy)
                    print("Sharpness       : ",sharpn)
                    print("roundness1      : ",round1)
                    print("roundness2      : ",round2)


                    with fits.open(filename) as hdul:
                        # print(hdul.info())
                        hdul.verify('fix')
                        data = hdul[0].data
                        header = hdul[0].header
                        date = header["DATE-OBS"].replace('"T"',"T")
                        filter= header["FILTER1"]
                        exptime = header["EXPTIME"]
                        # airmass = header["AIRMASS"]
                        airmass ="0000"
                        ra = header['OBJCTRA']
                        dec = header['OBJCTDEC']
                        jd = Time(date, format='isot', scale='utc').jd

                        print("Date            : ",date,"JD : ",jd)
                        print("Filter          : ",filter)
                        print("EXPTIME         : ",exptime)
                        print("Airmass         : ",airmass)
                        print("RA              : ",ra )
                        print("DEC             : ",dec )

                    new_data = {
                        'file': file,
                        'Date': date,
                        'JD': jd,
                        'FILTER1':filter,
                        'EXPTIME': exptime,
                        'Airmass': airmass,
                        'median_fwhm': np.median(fwhm_array),
                        'BINNING': bin,
                        'Quality': iq,
                        'Accuracy': accuracy,
                        "RA"   : ra,
                        "DEC" : dec,
                        # "ALT"  : altitude,
                        # "Azi" : azimuth,
                        "sharpness":sharpn,
                        "roundness1":round1,
                        "roundness2": round2,
                        'No of stars': str(len(images)),
                    }

                    csv_fil = pd.read_csv(f"/home/dataarchive/Documents/IQ_june/CSV/{year}.csv")
                    csvd = csv_fil._append(new_data, ignore_index=True)
            #             # print(file_det)
                    csvd.to_csv(f"/home/dataarchive/Documents/IQ_june/CSV/{year}.csv", index=False)


                    # matched_qtable = QTable.from_pandas(selected_df)
                    # positions = list(zip(sources['xcentroid'], sources['ycentroid']))
                    # positions_selected = list(zip(matched_qtable['xcentroid'], matched_qtable['ycentroid']))
                    # apertures = CircularAperture(positions, r=15)
                    # apertures_selected = CircularAperture(positions_selected, r=10)
                    # # Plot
                    # plt.figure(figsize=(10, 10))
                    # plt.imshow(data - median, origin='lower', vmin =25, vmax =75)
                    # apertures.plot(color='red', lw=.6)
                    # apertures_selected.plot(color='lime', lw=.6)
                    # plt.title(f"Bin: {bin},Souce_found: {len(sources)} Selected_source :{len(matched_qtable)} ", fontsize=13)
                    # plt.xlabel("X Pixel")
                    # plt.ylabel("Y Pixel")
                    # # Display file name at top center
                    # plt.text(0.5, 1.02, f"{file}", transform=plt.gca().transAxes,
                    #         fontsize=12, color='black', ha='center', va='bottom')
                    # # Save the figure with a safe filename
                    # safe_filename = os.path.basename(file).replace('.fits', '').replace(' ', '_') + ".jpg"
                    # plt.savefig(f"/home/dataarchive/Documents/IQ_june/image/frames/{year}/{safe_filename}",
                    #             bbox_inches='tight', dpi=600)
                    # plt.close()
        with open(f"/home/dataarchive/Documents/IQ_june/image/{year}_processed.dat", "a") as f:
            f.write(f"{file}\n")
