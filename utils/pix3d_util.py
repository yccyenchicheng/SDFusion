
# import numba
import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator as rgi
from PIL import Image


downsample_uneven_warned = False
def downsample(vox_in, times, use_max=True):
    global downsample_uneven_warned
    if vox_in.shape[0] % times != 0 and not downsample_uneven_warned:
        print('WARNING: not dividing the space evenly.')
        downsample_uneven_warned = True
    return _downsample(vox_in, times, use_max=use_max)


# @numba.jit(nopython=True, cache=True)
def _downsample(vox_in, times, use_max=True):
    dim = vox_in.shape[0] // times
    vox_out = np.zeros((dim, dim, dim))
    for x in range(dim):
        for y in range(dim):
            for z in range(dim):
                subx = x * times
                suby = y * times
                subz = z * times
                subvox = vox_in[subx:subx + times,
                                suby:suby + times, subz:subz + times]
                if use_max:
                    vox_out[x, y, z] = np.max(subvox)
                else:
                    vox_out[x, y, z] = np.mean(subvox)
    return vox_out

def thresholding(V, threshold):
    """
    return the original voxel in its bounding box and bounding box coordinates.
    """
    if V.max() < threshold:
        return np.zeros((2,2,2)), 0, 1, 0, 1, 0, 1
    V_bin = (V >= threshold)
    x_sum = np.sum(np.sum(V_bin, axis=2), axis=1)
    y_sum = np.sum(np.sum(V_bin, axis=2), axis=0)
    z_sum = np.sum(np.sum(V_bin, axis=1), axis=0)

    x_min = x_sum.nonzero()[0].min()
    y_min = y_sum.nonzero()[0].min()
    z_min = z_sum.nonzero()[0].min()
    x_max = x_sum.nonzero()[0].max()
    y_max = y_sum.nonzero()[0].max()
    z_max = z_sum.nonzero()[0].max()
    return V[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1], x_min, x_max, y_min, y_max, z_min, z_max

def interp3(V, xi, yi, zi, fill_value=0):
    x = np.arange(V.shape[0])
    y = np.arange(V.shape[1])
    z = np.arange(V.shape[2])
    interp_func = rgi((x, y, z), V, 'linear', False, fill_value)
    return interp_func(np.array([xi, yi, zi]).T)

def mesh_grid(input_lr, output_size):
    x_min, x_max, y_min, y_max, z_min, z_max = input_lr
    length = max(max(x_max - x_min, y_max - y_min), z_max - z_min)
    center = np.array([x_max - x_min, y_max - y_min, z_max - z_min]) / 2.
    x = np.linspace(center[0] - length / 2, center[0] + length / 2, output_size[0])
    y = np.linspace(center[1] - length / 2, center[1] + length / 2, output_size[1])
    z = np.linspace(center[2] - length / 2, center[2] + length / 2, output_size[2])
    return np.meshgrid(x, y, z)

def downsample_voxel(voxel, threshold, output_size, resample=True):
    if voxel.shape[0] > 100:
        # assert output_size[0] in (32, 128)
        # downsample to 32 before finding bounding box
        if output_size[0] == 32:
            voxel = downsample(voxel, 4, use_max=True)
    if not resample:
        return voxel

    voxel, x_min, x_max, y_min, y_max, z_min, z_max = thresholding(
        voxel, threshold)
    x_mesh, y_mesh, z_mesh = mesh_grid(
        (x_min, x_max, y_min, y_max, z_min, z_max), output_size)
    x_mesh = np.reshape(np.transpose(x_mesh, (1, 0, 2)), (-1))
    y_mesh = np.reshape(np.transpose(y_mesh, (1, 0, 2)), (-1))
    z_mesh = np.reshape(z_mesh, (-1))

    fill_value = 0
    voxel_d = np.reshape(interp3(voxel, x_mesh, y_mesh, z_mesh, fill_value),
                         (output_size[0], output_size[1], output_size[2]))
    return voxel_d