import argparse
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
import starfile
from scipy.spatial.transform import Rotation


def get_pixel_size(mrcpath: Path) -> float:
    with mrcfile.open(mrcpath, permissive=True) as mrc:
        header = mrc.header
    # in Angstrom
    pxsize = header["cella"]["x"] / header["nx"]
    return pxsize


def euler_to_zyz(angles: np.ndarray) -> np.ndarray:
    """Convert Euler angles from ZXZ to ZYZ convention

    Args:
        angles (np.ndarray): Euler angles in degrees, shape (N, 3)
    Returns:
        np.ndarray: Converted Euler angles in degrees, shape (N, 3)
    """
    # TODO: Fix Gimbal lock. "Setting third angle to zero since it is not possible to uniquely determine all angles."
    r = Rotation.from_euler("zxz", angles, degrees=True)
    zyz_angles = r.as_euler("zyz", degrees=True)
    return zyz_angles


def nm2px(xy: np.ndarray, pxsize: float, shape: tuple):
    """Convert coordinates from nanometer to pixel

    Args:
        xy (np.ndarray): Coordinates in nanometer, shape (N, 2)
        pxsize (float): Pixel size in Angstrom.
        shape (tuple): Shape of the image.
    Returns:
        np.ndarray: Coordinates in pixel, shape (N, 2)
    """

    x_center, y_center = map(lambda x: x // 2, shape)

    # x, y from nanometer to Angstrom to pixel
    xy_out = (xy * 10 / pxsize).round(0) + np.array([x_center, y_center])

    return xy_out


def txt2star(
    coord_path: Path,
    pxsize: float,
    shape: tuple,
):
    """Generate star file from txt file

    Args:
        coord_path (Path): txt file with coordinates and angles
        pxsize (float): Pixel size in Angstrom.
        shape (tuple): Shape of the image.
    """

    if coord_path is None:
        raise ValueError("Coordinate path must be provided.")
    if pxsize is None:
        raise ValueError("Pixel size must be provided.")

    star_path = coord_path.parent / (coord_path.stem + ".star")
    try:
        coords = pd.read_csv(
            coord_path,
            sep=r"\s+",
            skiprows=3,
            names=["x", "y", "z", "phi", "theta", "psi"],
        )
    except Exception as e:
        raise ValueError(f"Error reading {coord_path}: {e}")

    star = pd.DataFrame()
    star[["rlnCoordinateX", "rlnCoordinateY"]] = nm2px(
        coords[["x", "y"]].values, pxsize=pxsize, shape=shape
    )
    star[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]] = euler_to_zyz(
        coords[["phi", "theta", "psi"]].values
    )

    star["rlnStructure"] = coord_path.stem.split("_")[-1]

    starfile.write(data=star, filename=star_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert simulation coordinates to star file"
    )
    parser.add_argument(
        "--coord_dir",
        type=Path,
        required=True,
        help="Directory containing simulation coordinates txt files",
    )
    parser.add_argument(
        "--pxsize",
        type=float,
        required=True,
        help="Pixel size",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        default=(4096, 4096),
        help="Shape of the image (width height)",
    )
    args = parser.parse_args()

    for coord_file in args.coord_dir.glob("coord*.txt"):
        txt2star(coord_file, args.pxsize, args.shape)
