import concurrent.futures
from os.path import join, dirname
from tkinter import filedialog as fd
from tifffile import imread, imwrite
import math
import numpy as np
import pandas as pd
from scipy import optimize
from skimage.feature import blob_log
from rich.progress import Progress


def Gauss2d(background, height, centerx, centery, sigma):
    # Returns a gaussian *function* with the given parameters. Such *function* can take in arguments. It's a way in python to distinguish arguments from parameters.
    # Therefore, the use of this function will be
    # Gauss2d(parameters)(x, y)
    return lambda x, y: background + height * np.exp(
        -((x - centerx) ** 2 + (y - centery) ** 2) / 2 * (sigma ** 2)
    )


def fit_single_blob(params):
    GaussCrop, blob = params
    initial_x, initial_y, initial_sigma = blob
    # Fitting with BACKGROUND offset
    background_ini = GaussCrop.min()
    height_ini = GaussCrop.max()
    params_ini = [
        background_ini,
        height_ini,
        GaussCrop.shape[0] / 2,
        GaussCrop.shape[0] / 2,
        initial_sigma,
    ]

    errorfunc = lambda params: np.ravel(
        Gauss2d(*params)(*np.indices(GaussCrop.shape)) - GaussCrop
    )

    bounds = (
        [0, 0, GaussCrop.shape[0] * 0.25, GaussCrop.shape[0] * 0.25, 0],
        [
            GaussCrop.max(),
            np.inf,
            GaussCrop.shape[0] * 0.75,
            GaussCrop.shape[0] * 0.75,
            GaussCrop.shape[0] / 2,
        ],
    )
    result = optimize.least_squares(errorfunc, params_ini)
    background, height, centerx, centery, sigma = result.x

    return (
        initial_x + (centerx - math.floor(GaussCrop.shape[0] / 2)),
        initial_y + (centery - math.floor(GaussCrop.shape[0] / 2)),
        sigma,
        height,
    )


def main():
    file = "/Volumes/AnalysisGG/PROCESSED_DATA/CondensateAnalysis/Dcp1a-2x-2s/20221015-UGD-100msexposure-2sperframe-FOV-1.tif"
    blob_log_threshold = 0.001
    max_sig = 10
    chunksize = 1000

    video = imread(file)

    lst_x = []
    lst_y = []
    lst_sigma = []
    lst_height = []
    timestamp = []

    with Progress() as progress:
        task = progress.add_task(
            "Processing...", total=video.shape[0], auto_refresh=False
        )
        for t in np.arange(video.shape[0]):
            frame = video[t]
            blobs = blob_log(
                frame, threshold=blob_log_threshold, exclude_border=5, max_sigma=max_sig
            )
            # Exclude bright PBs
            blobs = blobs[blobs[:, 2] < 3]
            lst_GaussCrop = []
            for initial_x, initial_y, initial_sigma in blobs:
                # crop_size = math.ceil(initial_sigma + 1)
                GaussCrop = frame[
                    int(initial_x) - 3 : int(initial_x) + 4,
                    int(initial_y) - 3 : int(initial_y) + 4,
                ]
                if GaussCrop.size > 0:
                    # This mean a large blob is NOT near boundary and a full crop CAN be obtained
                    lst_GaussCrop.append(GaussCrop)

            arguments = [
                (GaussCrop, blob) for GaussCrop, blob in zip(lst_GaussCrop, blobs)
            ]

            with concurrent.futures.ProcessPoolExecutor() as executor:
                for result in executor.map(
                    fit_single_blob, arguments, chunksize=chunksize
                ):
                    fitted_x, fitted_y, sigma, height = result
                    lst_x.append(fitted_x)
                    lst_y.append(fitted_y)
                    lst_sigma.append(sigma)
                    lst_height.append(height)
                    timestamp.append(t)

            if __name__ == "__main__":
                progress.update(task, advance=1)
                progress.refresh()

    df = pd.DataFrame(
        {
            "t": timestamp,
            "x": lst_x,
            "y": lst_y,
            "sigma": lst_sigma,
            "height": lst_height,
        },
        dtype=float,
    )
    df.to_csv(file.strip(".tif") + ".csv", index=False)


if __name__ == "__main__":
    main()
