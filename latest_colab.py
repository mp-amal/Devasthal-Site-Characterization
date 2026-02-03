import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
def plot_cutout_and_ee(
    cutout, radii, ee, xc, yc,
    vmin=None, vmax=None, levels=6,
    pixel_scale=None,  # arcsec/pixel, optional
    savefile=None
):
    """
    Plot image cutout with contours + centroid, and Encircled Energy (EE) curve.

    Parameters
    ----------
    cutout : 2D numpy array
        Image cutout (e.g., PSF or star region).
    radii : 1D numpy array
        Aperture radii (in arcsec if pixel_scale given, else pixels).
    ee : 1D numpy array
        Encircled energy values (fractional, 0–1).
    xc, yc : float
        Centroid position (pixels).
    vmin, vmax : float, optional
        Intensity scaling for image display.
    levels : int, optional
        Number of contour levels.
    pixel_scale : float, optional
        Pixel scale in arcsec/pixel (if provided, x/y axes converted to arcsec).
    savefile : str, optional
        If given, save figure to this path.
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

    # --- Left panel: image cutout ---
    if pixel_scale:
        extent = [
            0, cutout.shape[1] * pixel_scale,
            0, cutout.shape[0] * pixel_scale
        ]
        xcen, ycen = xc * pixel_scale, yc * pixel_scale
        xlabel, ylabel = "ΔRA (arcsec)", "ΔDec (arcsec)"
    else:
        extent = None
        xcen, ycen = xc, yc
        xlabel, ylabel = "X (pix)", "Y (pix)"
    
    im = ax1.imshow(cutout, origin="lower", cmap="gray",
                    vmin=vmin, vmax=vmax, extent=extent)
    ax1.contour(cutout, levels=levels, colors="white", linewidths=0.8, alpha=0.7, extent=extent)
    ax1.plot(xcen, ycen, "rx", ms=6, mew=1.2, label="Centroid")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_aspect("equal")
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Flux [counts]")
    ax1.legend(loc="upper right", fontsize=8, frameon=False)

    # --- Right panel: Encircled Energy curve ---
    ax2.plot(radii, ee, "-o", lw=1.5, ms=5, label="Encircled energy")

    # Highlight common EE fractions
    for frac, color in [(0.5, "r"), (0.8, "g"), (0.9, "b")]:
        if (ee.min() <= frac <= ee.max()):
            # Find radius closest to requested fraction
            idx = np.argmin(np.abs(ee - frac))
            r_val = radii[idx]
            ax2.axhline(frac, ls="--", c=color, lw=1)
            ax2.axvline(r_val, ls="--", c=color, lw=1)
            ax2.text(r_val, frac + 0.03, f"EE{int(frac*100)} = {r_val:.2f}\"",
                     color=color, fontsize=8)

    ax2.set_xlabel("Aperture radius (arcsec)")
    ax2.set_ylabel("Encircled energy")
    ax2.set_xlim(0, radii.max() * 1.05)
    ax2.set_ylim(0, 1.0)
    ax2.legend(frameon=False)

    # Panel labels
    ax1.text(0.05, 0.95, "(a)", transform=ax1.transAxes, fontsize=10, weight="bold", va="top")
    ax2.text(0.05, 0.95, "(b)", transform=ax2.transAxes, fontsize=10, weight="bold", va="top")

    fig.tight_layout()

    if savefile:
        fig.savefig(savefile, dpi=600, bbox_inches="tight")
    return fig, (ax1, ax2)


# --- Standard library ---
import os
import math
import shutil
import warnings

# --- Third-party ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.stats import chisquare, shapiro, multivariate_normal

# --- Astropy ---
from astropy.io import fits
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord, AltAz, EarthLocation, Angle
from astropy.table import QTable
from astropy.utils.exceptions import AstropyWarning

# --- Photutils ---
from photutils.detection import DAOStarFinder
from photutils.aperture import (
    CircularAperture,
    CircularAnnulus,
    aperture_photometry,
)
from photutils.centroids import centroid_com
try:
    from photutils.psf import fit_fwhm  # available in newer photutils
except Exception:
    fit_fwhm = None  # fallback if not present

from matplotlib.ticker import MultipleLocator

star1= np.array([
    [  1,  -3,  -2,  -4,   7,   5,  11,  -1,   6,  -7,   0,  10,   4,   1,  10,   3,  14,  11,  10,   1,  -2,  13,   1,  -2],
    [-16,  -5,   1,   3,   9,  -7,   0,   5,  11,   1,   4,  22,   6,  20,  14,  10,  21,  10,   0,   3,   5,  -2,   2,  -8],
    [ -4,   2,  -3,   7,  10,   7,  12,  15,  20,  21,  28,  20,  20,  22,  19,  21,  19,   6,  10,   1,  16,   9,   4,   7],
    [-12,  -4,   7,  -1,   0,   2,  12,  20,  23,  25,  22,  33,  32,  27,  36,  29,  17,  15,  21,   9,  11,   3,  -3, -16],
    [  1,  -4,  14,   8,  16,   3,  -2,  25,  22,  45,  53,  60,  62,  60,  51,  35,  38,  23,  18,  11,   7,  -1,   1,  17],
    [ 10,   3,  15,  17,  10,  26,  27,  43,  38,  56,  73,  75, 108, 105,  90,  65,  55,  49,  40,  36,  11,   1,   9,   9],
    [ 14,   8,  18,  24,  24,  23,  37,  61,  67,  84, 114, 137, 164, 165, 155, 129,  96,  67,  62,  36,  24,  15,   9,  -1],
    [  2,  11,  16,  16,  26,  42,  59,  76, 104, 123, 179, 232, 278, 295, 267, 217, 173, 110,  97,  51,  34,  21,  17,  15],
    [  7,  10,  17,  34,  43,  58,  79,  91, 160, 211, 272, 382, 510, 537, 483, 354, 251, 181, 113,  79,  53,  45,  33,  14],
    [  1,  16,  15,  25,  45,  75, 103, 156, 232, 332, 474, 757, 1062, 1195, 967, 575, 373, 245, 167, 101,  65,  39,  32,  36],
    [  3,  15,  18,  33,  37,  80, 103, 193, 312, 511, 892, 1674, 2781, 3134, 2093, 1043, 541, 337, 205, 130,  85,  53,  43,  33],
    [  5,  10,  20,  38,  39,  87, 136, 215, 392, 756, 1657, 4098, 7911, 8327, 4898, 1935, 850, 436, 257, 149,  91,  68,  49,  40],
    [  6,  10,  26,  36,  69,  99, 136, 260, 427, 929, 2655, 8053, 16896, 16343, 8356, 3088, 1144, 543, 303, 170, 100,  73,  48,  17],
    [ 16,  16,  35,  48,  63,  98, 156, 260, 451, 944, 2696, 8319, 17790, 16850, 8267, 3167, 1224, 548, 299, 184,  97,  72,  38,  34],
    [  5,  22,  28,  62,  65, 103, 134, 197, 346, 679, 1685, 4436, 8349, 8165, 4432, 1956, 879, 440, 264, 155,  70,  59,  48,  25],
    [  3,  26,  30,  27,  53,  65,  96, 172, 262, 482, 899, 1762, 2660, 2609, 1786,  993, 547, 331, 187, 113,  77,  48,  48,  24],
    [  5,  11,  12,  23,  38,  63,  83, 135, 186, 316, 483, 759, 958, 922, 744,  522, 346, 237, 141,  82,  70,  45,  33,  21],
    [ -8,   3,   9,  17,  29,  32,  70, 107, 148, 211, 290, 386, 461, 447, 381,  281, 230, 150, 109,  65,  69,  46,  40,  29],
    [  6,   0,  11,  17,  17,  34,  50,  88,  95, 124, 185, 228, 246, 261, 231,  174, 120,  94,  65,  59,  50,  34,  31,  21],
    [  4,   6,   6,  19,  24,  31,  34,  53,  71,  92, 120, 135, 137, 136, 122,  117,  82,  69,  34,  43,  39,  29,  24,  16],
    [ -1,  -5,   5,  15,   5,  22,  23,  32,  31,  48,  66,  79,  94,  77,  67,   75,  42,  45,  33,  20,  17,  20,  21,  26],
    [ -7,  -7,   7,  -1,   2,   8,  30,  22,  26,  35,  46,  41,  58,  57,  58,   36,  37,  30,  25,  14,   6,  13,  22,  14],
    [ -6,   0,  -3,   1,  10,  10,   8,   9,  11,  26,  28,  38,  35,  25,  28,   33,  21,  20,  14,  31,   7,   2,  -3,   9],
    [ -2,   6,  -1,   8,  -6,  17,  14,   9,  12,  26,  29,  31,  25,  25,  21,   12,  24,   9,   8,  15,   6,  -6,  -2,   3]
])



star2 = np.array([
    [12,  -4,  -6,   9,  -1,   3, -12,  11,   1,   4,   2,  -5,  -5,  -6,  -5,  -1,  -6,   7,  -4,  -4,  -2,   2,  -5, -17],
    [ 9,  -7,  -3,   1,  -3,   8,  -1,  -5,   6,   6,   9,  -4,   2,   3,  -6,  18,  -1,  -2,  -6,  -2,  -8,   1,  -1,  -4],
    [ 4,   5,  -9,  -1,   1,   5, -10,   7,   6,  -1,   0,   7,   8,   1,   0,   7,   7,  -6,   6,  -1,   7,  11,   1,   2],
    [ 2,  -2,   3,  -9,  15,   0,  -3,  12,  15,   7,  10,   1,   3,   0,   8,   8,   5,   9,   6,   4,  -9,  -8,   7,  13],
    [-4,   9,   5,  -3,  -1,  13,  11,   5,   9,   7,   8,   6,  15,  15,  27,   7,   2,  12,   7,  -4,  -4,   6,   4,   2],
    [ 8,   2,   3,  -6,   4,  11,   5,   4,   2,   3,   7,  14,   7,  14,  12,  15,   8,   2,  18,   5,  -3,   4,  10,   4],
    [ 0,  10,   2,  -5,   8,   3,   0,  14,   5,  20,  18,  23,  25,  28,  19,  25,  21,  17,  -2,  -8,  -8,  -5,  -6,  -1],
    [-6,   9,  -5,  -2,   0,  -6,   0,  14,  21,  19,  31,  27,  31,  52,  48,  17,  26,  17,  11,   6,  15,   6,   1,  -2],
    [-4,   4,   3,  12,   6,  12,   2,   8,  26,  33,  50,  61,  65,  68,  54,  51,  41,  29,  12,  10,   5,   2,  -6,  -2],
    [-3,   9,   4,   7,   6,  -4,  12,   8,  26,  47,  63,  98, 142, 178, 128,  97,  51,  45,  30,  25,  16,  14,  -2,  -7],
    [ 3,   7,  -7,  17,   6,  10,  24,  36,  38,  77, 146, 228, 373, 450, 300, 164,  80,  41,  26,  33,  13,  10,   3,  15],
    [ 6,  -3,   9,  -1,   8,  10,  12,  33,  50, 108, 264, 646,1175,1282, 765, 344, 146,  74,  37,  44,  15,  15,   4,   2],
    [ 7,   2,   9,   5,  10,  29,  24,  41,  66, 150, 422,1290,2690,2643,1423, 551, 202,  81,  52,  29,  17,  12,  11,   9],
    [ 2,   4,   0,  14,  12,  12,  22,  33,  65, 139, 401,1320,2676,2501,1445, 584, 216,  82,  47,  24,  28,  16,  12,  -3],
    [ 9,   8,   6,   7,   6,  24,  23,  31,  60,  94, 251, 615,1028, 959, 647, 339, 164,  81,  37,  27,  32,  12,   5,  10],
    [ 6,  -6,   5,  10,   5,   5,  11,  21,  34,  72, 126, 226, 338, 305, 238, 152,  87,  34,  27,  20,  17,   4,  13,   7],
    [-2,  -8,   3,   0,  12,  11,  10,  20,  25,  46,  55, 100, 127, 122,  91,  67,  53,  42,  25,  12,   8,  11,   6,  -4],
    [-9,   5,  -7,   6,  -7,  13,  13,  13,  13,  19,  26,  62,  59,  54,  54,  42,  28,  25,  15,  21,   5,   5,  14,   4],
    [-1,   4,   0,   3,   9,   7,   9,  15,  17,  31,  11,  34,  42,  31,  33,  20,  24,  10,  15,  14,   6,   3,  10,   5],
    [ 6,  -7,  -5,   3,  -1,   1,   8,   1,   4,   9,  11,  17,   5,  18,  12,  11,  17,  15,   6,   8,   4,  -4,   0,   4],
    [ 5,   9,  13,   4,  -1,   6,  -9,   6,  -1,   2,   9,  16,   3,  13,   1,   6,   9,   4,  -6,  -1,   6,   8,  -2,  11],
    [ 2,  -3,  -1,   1,  -2,  13,  -8,  17,   1,  -1,   5,   5,   9,   6,  16,   9,   8,  11,   6,   4,   4,   9,   0,   0],
    [ 5,  -7,  -6,   4, -14,  -5,   1,   2,   0,   8,   3,   7,   5,   1,  14,   7,   7,   7,   2,   6,   3,   5,  -3,  -1],
    [ 2,  -5,  -3,  -1,  -4,   1,   3,  -2,  14,   2,  18,  -3,  -3,   6,  -1,   4,   7,   7,   6,   1,   1,  -4,  11,   0]
])


import numpy as np

star3 = np.array([
    [ -1,  -7, -10, -13,  -5,   2,  -6,   0, -12,  -4,  -6,  -8,   6,  -3,  -5,  -6,  -6,  -1, -14,  -6, -14,  -6,  -1, -10],
    [ -4,  -6,   3,  -8,  -9,  -5,  -5,  -7,   3,  -4,   0,  -6,  -6,   1,   8,   8,  -1,  -6,  -3,  -2, -20,  -7,   0,   1],
    [ -8,  -4,  -5,  -6,   2,   6,   2,   0,  -4,   1,   1,  -2,  -4,  -3,  -3,  -7,  -9,  -2,   0, -14,  -7, -10,  -4,  -6],
    [  4,  -9,  -2,   3,   2,  -8,   1,  -3,   4,   2,  -4,   2,  -1,  -6,   7,  -2,  -5,   4,   4,  -5,  -3,  -8,  -4,  -9],
    [  7,  -6,   0,  -4,   0,   8,   0, -10, -14,  -6,   8,   5,   2,   6,   7,   8,   4,  -3,   6,   1, -11,  -1,  -8,   2],
    [ -1,   4,   7,  -7,   4,  -5,  -6,   2,   7,  -6,  11,  13,  17,   0,  22,   3,   6,  -7,   2,  -1,  -4,  -9,  -6,  -3],
    [  1, -10,  -1,  -2,   7,   2,  -3,  18,   7,  18,  17,  14,  24,  28,  30,  16,  10,   5,   7,   3,   6,  -3,  -3,   5],
    [  4,   1,  -7,   2,  10,   3,  -3,  -2,   9,  13,  26,  25,  40,  49,  47,  24,  10,  10,   7,   0,   4,   0,  -7,  -9],
    [ -5,   0,   3,   9,   0,   2,   4,  15,  22,  35,  46,  69,  88,  97,  77,  52,  32,  12,   4,   5,   2,  -2,   0,   5],
    [ -9,  -2,  -1, -12,  -2,  11,  15,  22,  42,  44, 100, 150, 255, 260, 192,  84,  58,  40,  17,  13,   4,  -5,   3,  -4],
    [  4,  -8,  -2,  -5,   5,  21,   9,  29,  48,  73, 155, 355, 692, 731, 463, 193,  75,  36,  31,  10,   0,   9,  -3,  -3],
    [  6,   3,   7,   1,   0,   4,  12,  31,  62, 129, 301, 869,1549,1529, 804, 327, 123,  67,  35,  22,   8,   7,  15,   3],
    [ -3,   0,   2,   7,   7,   7,  25,  40,  76, 154, 420,1254,2183,1882, 934, 362, 120,  66,  28,  27,   8,   3,  -2,  11],
    [  2, -11,  13,   5,   0,  10,  21,  35,  58, 133, 451,1178,1824,1329, 679, 278, 113,  59,  34,  12,  21,   2,   2,   5],
    [ -4,   7,   4,   7,  12,   3,  14,  31,  50,  96, 256, 627, 814, 652, 343, 158,  76,  44,  28,  13,  10,  -6,   3,   6],
    [  0,   2,   0,   8,  -1,   9,   1,  25,  32,  63, 123, 224, 253, 223, 141,  97,  40,  40,  11,  17,  12,   6,   3,   3],
    [  2,   2,  -2,  11,   3,  11,   3,   9,  23,  41,  52,  80,  95,  84,  66,  52,  23,  21,  13,   7,  10,   8,  -7,  -5],
    [ -9,  -6, -10,   7,   7,   9,  11,  16,  16,  21,  32,  43,  53,  49,  41,  21,  18,  28,   7,  10,   5,  11,   1,  -1],
    [ -2,   3,   7,   3,   4,  -7,   9,  13,  22,  20,  35,  13,  26,  28,  27,  24,   6,  -1,  -2,  -3,   6,  14,   2,   5],
    [ -8,  -7,  -1,  13,   6,   1,  -6,   1,   6,   9,  17,   7,  20,   9,  17,   0,   4,   4,   8,   1,  -1,  -3,  17,   3],
    [  1,  12,  -2,   7,  12,   1,   2,  -3,  11,  -3,   5,  11,  13,  18,  13,  10,  12,   9,   5,  -3,  -1,   3,   2,   3],
    [ -3,   4,  -1,   5,  -1,  -1,   4,   7,  22,   8,   7,  -2,  -3,   0,   3,   3,  -5,  -6,  16,  13,   4,   6,   2,  13],
    [  8,   4,   7,  -4,  -6,   6,  22,   5,  13,  14,  -3,   7,   5,   4,   4,   0,   8,   7,   6,   2, -11,   4,   9,  16],
    [ -8,   0,  -1,  -6,   0,  -6,   2,   3,   4,   6,   2,   2,   4,  -3,   6,  -5,  -5,   6,  -7,  14,  -4,   6,  25,  38]
])

star4 = np.array([
    [12,  -4,  -6,   9,  -1,   3, -12,  11,   1,   4,   2,  -5,  -5,  -6,  -5,  -1,  -6,   7,  -4,  -4,  -2,   2,  -5, -17],
    [ 9,  -7,  -3,   1,  -3,   8,  -1,  -5,   6,   6,   9,  -4,   2,   3,  -6,  18,  -1,  -2,  -6,  -2,  -8,   1,  -1,  -4],
    [ 4,   5,  -9,  -1,   1,   5, -10,   7,   6,  -1,   0,   7,   8,   1,   0,   7,   7,  -6,   6,  -1,   7,  11,   1,   2],
    [ 2,  -2,   3,  -9,  15,   0,  -3,  12,  15,   7,  10,   1,   3,   0,   8,   8,   5,   9,   6,   4,  -9,  -8,   7,  13],
    [-4,   9,   5,  -3,  -1,  13,  11,   5,   9,   7,   8,   6,  15,  15,  27,   7,   2,  12,   7,  -4,  -4,   6,   4,   2],
    [ 8,   2,   3,  -6,   4,  11,   5,   4,   2,   3,   7,  14,   7,  14,  12,  15,   8,   2,  18,   5,  -3,   4,  10,   4],
    [ 0,  10,   2,  -5,   8,   3,   0,  14,   5,  20,  18,  23,  25,  28,  19,  25,  21,  17,  -2,  -8,  -8,  -5,  -6,  -1],
    [-6,   9,  -5,  -2,   0,  -6,   0,  14,  21,  19,  31,  27,  31,  52,  48,  17,  26,  17,  11,   6,  15,   6,   1,  -2],
    [-4,   4,   3,  12,   6,  12,   2,   8,  26,  33,  50,  61,  65,  68,  54,  51,  41,  29,  12,  10,   5,   2,  -6,  -2],
    [-3,   9,   4,   7,   6,  -4,  12,   8,  26,  47,  63,  98, 142, 178, 128,  97,  51,  45,  30,  25,  16,  14,  -2,  -7],
    [ 3,   7,  -7,  17,   6,  10,  24,  36,  38,  77, 146, 228, 373, 450, 300, 164,  80,  41,  26,  33,  13,  10,   3,  15],
    [ 6,  -3,   9,  -1,   8,  10,  12,  33,  50, 108, 264, 646,1175,1282, 765, 344, 146,  74,  37,  44,  15,  15,   4,   2],
    [ 7,   2,   9,   5,  10,  29,  24,  41,  66, 150, 422,1290,2690,2643,1423, 551, 202,  81,  52,  29,  17,  12,  11,   9],
    [ 2,   4,   0,  14,  12,  12,  22,  33,  65, 139, 401,1320,2676,2501,1445, 584, 216,  82,  47,  24,  28,  16,  12,  -3],
    [ 9,   8,   6,   7,   6,  24,  23,  31,  60,  94, 251, 615,1028, 959, 647, 339, 164,  81,  37,  27,  32,  12,   5,  10],
    [ 6,  -6,   5,  10,   5,   5,  11,  21,  34,  72, 126, 226, 338, 305, 238, 152,  87,  34,  27,  20,  17,   4,  13,   7],
    [-2,  -8,   3,   0,  12,  11,  10,  20,  25,  46,  55, 100, 127, 122,  91,  67,  53,  42,  25,  12,   8,  11,   6,  -4],
    [-9,   5,  -7,   6,  -7,  13,  13,  13,  13,  19,  26,  62,  59,  54,  54,  42,  28,  25,  15,  21,   5,   5,  14,   4],
    [-1,   4,   0,   3,   9,   7,   9,  15,  17,  31,  11,  34,  42,  31,  33,  20,  24,  10,  15,  14,   6,   3,  10,   5],
    [ 6,  -7,  -5,   3,  -1,   1,   8,   1,   4,   9,  11,  17,   5,  18,  12,  11,  17,  15,   6,   8,   4,  -4,   0,   4],
    [ 5,   9,  13,   4,  -1,   6,  -9,   6,  -1,   2,   9,  16,   3,  13,   1,   6,   9,   4,  -6,  -1,   6,   8,  -2,  11],
    [ 2,  -3,  -1,   1,  -2,  13,  -8,  17,   1,  -1,   5,   5,   9,   6,  16,   9,   8,  11,   6,   4,   4,   9,   0,   0],
    [ 5,  -7,  -6,   4, -14,  -5,   1,   2,   0,   8,   3,   7,   5,   1,  14,   7,   7,   7,   2,   6,   3,   5,  -3,  -1],
    [ 2,  -5,  -3,  -1,  -4,   1,   3,  -2,  14,   2,  18,  -3,  -3,   6,  -1,   4,   7,   7,   6,   1,   1,  -4,  11,   0]
])

stars =[star1,star2,star3,star4]

stars = [star1, star2, star3, star4]

# Create the main 2x2 figure
# fig = plt.figure(figsize=(10, 8))
fig = plt.figure(figsize=(18, 8), constrained_layout=True)
outer = fig.add_gridspec(2, 2, wspace=0.3, hspace=0.3)

fig.patch.set_linewidth(0.5)
fig.patch.set_edgecolor("black")
fig.patch.set_facecolor("white")

# Iterate over stars and plot each with 2 subplots inside its panel
for idx, image in enumerate(stars):
    i, j = divmod(idx, 2)  # map index to 2x2 grid

    inner = outer[i, j].subgridspec(1, 2, wspace=0.5)
    ax_left = fig.add_subplot(inner[0], projection="3d")
    ax_right = fig.add_subplot(inner[1])
    # ax_right = fig.add_axes([0.55, 0.55, 0.3, 0.3])

 # --- Plate scale in arcsec/pixel ---

    # --- Estimate the star center automatically ---
    # y_center, x_center = centroid_com(image)
    # print(f"Detected center: x={x_center:.2f}, y={y_center:.2f}")

    peak_value = np.max(image)

    # Find location
    y_center, x_center = np.unravel_index(np.argmax(image), image.shape)




    # --- background from an outer annulus ---
    annulus = CircularAnnulus((x_center, y_center), r_in=25, r_out=30)
    ann_stats = aperture_photometry(image, annulus)
    mean_bg = ann_stats['aperture_sum'][0] / annulus.area

    # --- aperture radii in pixels ---
    # radii_pix = np.arange(1,image.shape[0]*0.9549*2/4, 1)  # 1 to 20 pixels
    # radii_arcsec = radii_pix * plate_scale*2

    # max_radius_pix = 2.5 * fwhm
    max_radius_pix =8
    startpix =max_radius_pix/10
    radii_pix = np.linspace(0.01, max_radius_pix, 10)
    # Convert radii to arcseconds
    plate_scale = 0.09549  # arcsec per pixel
    radii_arcsec = radii_pix * plate_scale*2
    print(radii_arcsec)
    fluxes = []
    for r in radii_pix:
        aperture = CircularAperture((x_center, y_center), r=r)
        phot_table = aperture_photometry(image, aperture)
        flux = phot_table['aperture_sum'][0] - (mean_bg * aperture.area)
        fluxes.append(flux)

    fluxes = np.array(fluxes)
    total_flux = fluxes[-1]
    EE = (fluxes / total_flux)


    Y, X = np.indices(image.shape)

    ny, nx = image.shape

    # Coordinates(x_center, y_center) -> (0,0)
    y = np.arange(ny) - y_center
    x = np.arange(nx) - x_center
    X, Y = np.meshgrid(x, y)
    Y= Y*2*plate_scale
    X= X*2*plate_scale


    # coordinate grid for 3D surface
    y = np.arange(image.shape[0])
    x = np.arange(image.shape[1])
    X, Y = np.meshgrid(x - image.shape[1]//2, y - image.shape[0]//2)

    # 3D surface plot
    surf = ax_left.plot_surface(X, Y, image, cmap='viridis')
    ax_left.view_init(elev=30, azim=225)

    max_radius = int(max(abs(X).max(), abs(Y).max()))
    ax_left.set_xticks(np.linspace(-max_radius, max_radius, 5))
    ax_left.set_yticks(np.linspace(-max_radius, max_radius, 5))

    ax_left.set_xlabel('Radius(")',fontsize =12)
    ax_left.set_ylabel('Radius(")',fontsize =12)
    # ax_left.set_zlabel("Flux",fontsize =12)
    ax_left.set_zlabel("Flux", fontsize=12, rotation=90, labelpad=10)
    ax_left.set_title(f"Source {idx+1}")

    # axes[0].imshow(image, origin='lower',vmin= 15, interpolation='nearest')
    # axes[0].plot(x_center, y_center, 'rx', markersize=8, label='Center')
    # axes[0].set_title(f"{round(qual,2)} {round(fwhm* plate_scale*bin,2)} {bin} {year}")

    # for r in radii_pix:
    #     circ = plt.Circle((x_center, y_center), r, color='yellow', fill=False, lw=0.3, alpha=0.5)
    #     axes[0].add_patch(circ)
    # axes[0].set_title('Star Cutout with Apertures')
    # axes[0].legend()


    ax_right.plot(radii_arcsec, EE, linewidth=1.5, alpha=0.5)
    target_fractions = [0.50, 0.80, 0.90]
    interpolated_radii = np.interp(target_fractions, EE, radii_arcsec)

    ax_right.set_xlim(0,)
    ax_right.set_yticks(np.linspace(0.1, 1, 10))
    # major spacing here is 0.1, so make minor ticks every 0.02 (5 subdivisions)
    # ax_right.yaxis.set_minor_locator(MultipleLocator(0.02))

    # # optional styling
    # ax_right.tick_params(axis='y', which='minor', length=3)
    # ax_right.tick_params(axis='y', which='major', length=6)


    # Major ticks: left & bottom only
    # ax_right.tick_params(axis='both', which='major',
    #                     direction='in',
    #                     top=False, right=False,
    #                     bottom=True, left=True)

    # # Minor ticks: left & bottom only
    # ax_right.tick_params(axis='both', which='minor',
    #                     direction='in',
    #                     top=False, right=False,
    #                     bottom=True, left=True)

    from matplotlib.ticker import AutoMinorLocator

    # create minor ticks
    ax_right.minorticks_on()
    ax_right.xaxis.set_minor_locator(AutoMinorLocator())
    ax_right.yaxis.set_minor_locator(AutoMinorLocator())

    # now styling will work
    ax_right.tick_params(axis='y', which='minor', length=3,direction='in')
    ax_right.tick_params(axis='y', which='major', length=6,direction='in')
    ax_right.tick_params(axis='x', which='minor', length=3,direction='in')
    ax_right.tick_params(axis='x', which='major', length=6,direction='in')









    for r, f, color in zip(interpolated_radii, target_fractions, ["red", "green", "blue"]):
        ax_right.axhline(y=f, color=color, linestyle="--", linewidth=0.5, alpha=0.1)
        ax_right.axvline(x=r, color=color, linestyle="--", linewidth=0.7, alpha=0.9)
        ax_right.text(r, f-0.05, f' E{int(f*100)} = {r:.3f}"', fontsize=7, ha="left", va="bottom", color=color)

    ax_right.set_xlabel("Aperture Radius (arcsec)",fontsize = 12)
    # ax_right.set_ylabel(r"EE = $F(<r)/F_{\\mathrm{total}}$")
    ax_right.set_ylabel(r"EE = $F(<r)/F_{\mathrm{total}}$",fontsize = 12)
    # ax_right.set_title(f"Curve of Growth")

plt.tight_layout()
plt.savefig(f"/home/dataarchive/Documents/Result_august_6/result(jpg)/good_source_collage.png")
plt.savefig(f"/home/dataarchive/Documents/Result_august_6/result(pdf)/good_source_collage.pdf")

    # Example: plot intensity profile (replace with your real analysis)
    # ax_right.plot(image[image.shape[0]//2, :], color="blue")
    # ax_right.set_title(f"Star {idx+1} Profile")


plt.show()


# for image in stars:
#     # --- Plate scale in arcsec/pixel ---

#     # --- Estimate the star center automatically ---
#     y_center, x_center = centroid_com(image)
#     print(f"Detected center: x={x_center:.2f}, y={y_center:.2f}")

#     # --- Estimate background from an outer annulus ---
#     annulus = CircularAnnulus((x_center, y_center), r_in=25, r_out=30)
#     ann_stats = aperture_photometry(image, annulus)
#     mean_bg = ann_stats['aperture_sum'][0] / annulus.area

#     # --- Define aperture radii in pixels ---
#     # radii_pix = np.arange(1,image.shape[0]*0.9549*2/4, 1)  # 1 to 20 pixels
#     # radii_arcsec = radii_pix * plate_scale*2

#     # max_radius_pix = 2.5 * fwhm
#     max_radius_pix =8
#     startpix =max_radius_pix/10
#     radii_pix = np.linspace(0.01, max_radius_pix, 10)
#     # Convert radii to arcseconds
#     plate_scale = 0.09549  # arcsec per pixel
#     radii_arcsec = radii_pix * plate_scale*2

#     fluxes = []
#     for r in radii_pix:
#         aperture = CircularAperture((x_center, y_center), r=r)
#         phot_table = aperture_photometry(image, aperture)
#         flux = phot_table['aperture_sum'][0] - (mean_bg * aperture.area)
#         fluxes.append(flux)

#     fluxes = np.array(fluxes)
#     total_flux = fluxes[-1]
#     EE = (fluxes / total_flux)

#     # --- Plot image and EE curve side by side ---
#     fig, axes = plt.subplots(1, 2, figsize=(10, 4))

#     # Left: Star image with center marked and circular apertures drawn
#     # axes[0].imshow(image, origin='lower',vmin= 15, interpolation='nearest')
#     # axes[0].plot(x_center, y_center, 'rx', markersize=8, label='Center')

#     from mpl_toolkits.mplot3d import Axes3D  # only needed for older matplotlib
#     import numpy as np

#     # --- Replace axes[0] with a true 3D axis in the same grid slot ---
#     # Remove the original 2D axes to avoid overlay ("plot within plot")
#     fig.delaxes(axes[0])
#     axes[0] = fig.add_subplot(1, 2, 1, projection='3d')

#     # Coordinates from pixel indices (X: columns, Y: rows)
#     Y, X = np.indices(image.shape)

#     ny, nx = image.shape

#     # Shift coordinates so that (x_center, y_center) -> (0,0)
#     y = np.arange(ny) - y_center
#     x = np.arange(nx) - x_center
#     X, Y = np.meshgrid(x, y)
#     Y= Y*2*plate_scale
#     X= X*2*plate_scale
#     # Single 3D surface plot in axes[0]
#     surf = axes[0].plot_surface(X, Y, image, cmap='viridis')
#     axes[0].view_init(elev=30, azim=225)

#     max_radius = int(max(abs(X).max(), abs(Y).max()))
#     axes[0].set_xticks(np.linspace(-max_radius, max_radius, 5))  # symmetric xticks
#     axes[0].set_yticks(np.linspace(-max_radius, max_radius, 5))  # symmetric yticks


#     axes[0].set_xlabel('Radius(")')
#     axes[0].set_ylabel('Radius(")')
#     axes[0].set_zlabel("Flux")

#     # fig.colorbar(surf, ax=axes[0], shrink=0.3, aspect=10)


#     axes[1].plot(radii_arcsec, EE ,linewidth =1.5,alpha =0.5)
#     # print(radii_arcsec)
#     # print(EE)
#     target_fractions = [0.50, 0.80, 0.90]
#     interpolated_radii = np.interp(target_fractions, EE, radii_arcsec)
#     # print("Radii :",interpolated_radii)


#     axes[1].set_xlim(0,)

#     # Set 10 y-ticks (from 0 to 100%)
#     axes[1].set_yticks(np.linspace(0.1, 1, 10))

#     # # Add vertical reference lines at 50%, 80%, and 99%
#     # for r, f,color in zip(interpolated_radii, target_fractions*100,["red","green","blue"]):
#     #     plt.text(r, f, f'* ({r:.1f} ", {f:.3f})', fontsize=9, ha="left",color =color, va="bottom")
#     axes[1].set_yticks(np.linspace(0.1, 1, 10))

#     # Add horizontal reference lines at 50%, 80%, and 99%
#     for r, f, color in zip(interpolated_radii, target_fractions*100, ["red", "green", "blue"]):
#         axes[1].axhline(y=f, color=color, linestyle="--", linewidth=0.1,alpha = 0.1)
#         axes[1].axvline(x=r, color=color, linestyle="--", linewidth=0.3,alpha =0.9)
#         axes[1].text(r, f-0.1, f' E{(int(f*100)):.1f} - {r:.3f}"', fontsize=7, ha="left", va="bottom", color=color)
#     # axes[1].set_ylabel("Encircled Energy ()")
#     axes[1].set_xlabel("Aperture Radius (arcsec)")
#     # axes[1].set_ylabel("EE (total_flux/total flux inside radius)")
#     axes[1].set_ylabel(r"EE = $F(<r)/F_{\mathrm{total}}$")

#     # axes[1].set_title("Encircled Energy Curve")
#     # axes[1].grid(True)

#     plt.tight_layout()
#     # selected_image_count = selected_image_count+1
#     # plt.savefig(f"/home/dataarchive/Documents/Result_august_6/EE_image/best/source_{selected_image_count}.pdf")
#     # plt.savefig(f"/home/dataarchive/Documents/Result_august_6/EE_image/{file}_{selected_image_count}.pdf")
#     print(image)
#     plt.show()
#     # plt.close()




    # Assuming you already have your 2D star data in variable `image`
    # # Example: image = np.loadtxt("star_data.txt") or from a FITS file using astropy

    # # Create coordinate grids for the image
    # y = np.arange(image.shape[0])
    # x = np.arange(image.shape[1])
    # X, Y = np.meshgrid(x, y)

    # # Plot 3D surface
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # surf = ax.plot_surface(X, Y, image, cmap='viridis')

    # # Add contour projection
    # ax.contour(X, Y, image, zdir='z', offset=image.min(), cmap='viridis')

    # ax.set_xlabel('X Pixels')
    # ax.set_ylabel('Y Pixels')
    # ax.set_zlabel('Flux')
    # fig.colorbar(surf, shrink=0.5, aspect=10)

    # plt.show()




    # import numpy as np
    # import matplotlib.pyplot as plt
    # from scipy.optimize import curve_fit

    # # Define 2D Gaussian function (for curve_fit: returns a 1D vector)
    # def gaussian_2d(coords, amp, x0, y0, sigma_x, sigma_y, offset):
    #     x, y = coords
    #     g = amp * np.exp(-(((x - x0) ** 2) / (2 * sigma_x ** 2) + ((y - y0) ** 2) / (2 * sigma_y ** 2))) + offset
    #     return g.ravel()

    # # Same Gaussian but returns a 2D grid (for plotting)
    # def gaussian_2d_grid(x, y, amp, x0, y0, sigma_x, sigma_y, offset):
    #     return amp * np.exp(-(((x - x0) ** 2) / (2 * sigma_x ** 2) + ((y - y0) ** 2) / (2 * sigma_y ** 2))) + offset

    # # Example 2D image array (replace with your star image array)
    # # image = np.random.rand(50, 50) * 100  # <-- remove this and supply your own 'image' 2D array
    # # assert image.ndim == 2, "'image' must be a 2D array"

    # # Coordinates
    # y = np.arange(image.shape[0])
    # x = np.arange(image.shape[1])
    # x, y = np.meshgrid(x, y)  # x: cols (nx), y: rows (ny)

    # # Initial guess parameters (more robust)
    # ny, nx = image.shape
    # bg_est = np.median(np.hstack([image[0, :], image[-1, :], image[:, 0], image[:, -1]]))
    # peak_idx = np.unravel_index(np.nanargmax(image), image.shape)
    # amp0 = float(np.nanmax(image) - bg_est)
    # x0 = float(peak_idx[1])
    # y0 = float(peak_idx[0])
    # sx0 = max(1.0, min(nx/4, 3.0))
    # sy0 = max(1.0, min(ny/4, 3.0))
    # off0 = float(bg_est)
    # initial_guess = (amp0, x0, y0, sx0, sy0, off0)

    # # Bounds to keep parameters physical
    # lower = (0.0, 0.0, 0.0, 0.5, 0.5, -np.inf)
    # upper = (np.inf, nx-1, ny-1, max(nx, ny), max(nx, ny), np.inf)

    # # Fit Gaussian
    # popt, _ = curve_fit(gaussian_2d, (x, y), image.ravel(), p0=initial_guess, bounds=(lower, upper), maxfev=20000)
    # popt, _ = curve_fit(gaussian_2d, (x, y), image.ravel(), p0=initial_guess)

    # # Extract fitted parameters
    # amp, x_center, y_center, sigma_x, sigma_y, offset = popt

    # # Generate fitted data on grid (2D)
    # fitted_data = gaussian_2d_grid(x, y, amp, x_center, y_center, sigma_x, sigma_y, offset)

    # # Plot original image and fit
    # fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # # Original image with detected center
    # vmin = max(15, np.nanpercentile(image, 1)) if np.nanmin(image) < 15 else None
    # axes[0].imshow(image, origin='lower', vmin=vmin, interpolation='nearest')
    # axes[0].plot(x_center, y_center, 'rx', markersize=8, label='Center')
    # axes[0].set_title('Original Image with Center')
    # axes[0].legend()

    # # Gaussian fit
    # axes[1].imshow(fitted_data, origin='lower', vmin=vmin, interpolation='nearest')
    # axes[1].plot(x_center, y_center, 'rx', markersize=8, label='Fit Center')
    # axes[1].set_title('2D Gaussian Fit')
    # axes[1].legend()

    # plt.tight_layout()
    # plt.show()
