import sys

try:
    import torch
except ImportError as e:
    print(e)

import numpy as np
from pyevtk.hl import imageToVTK, gridToVTK, pointsToVTK
import glob
import os
from joblib import Parallel, delayed
from tqdm import tqdm
import re
import scipy.io as scio

from src.box import GXBox
from src.ops import curl, angles, directions_closure


def get_curl_from_np_box(filename):
    data = np.load(filename)
    bx = data[..., 0]
    by = data[..., 1]
    bz = data[..., 2]
    return curl(bx, by, bz)


def get_image2_from_sav(filename):
    try:
        return scio.readsav(filename).IMAGE2
    except Exception:
        print(f'SAV file {filename} has no IMAGE2 data!')
        return None


def get_refmap_data_from_basemap_sav(filename):
    try:
        return scio.readsav(filename)['mapw']['data'][0]
    except Exception:
        print(f'SAV file {filename} has no basemap data!')
        return None


def clip_neg_and_max_divide(images: list):
    images = np.array(images)
    images = np.clip(images, 0, np.max(images))
    max_per_frame = np.max(images, axis=(1, 2), keepdims=True)
    images = images / max_per_frame
    images *= 255
    return np.round(images).astype(np.uint8)


def save_scalar_data(s, scalar_name, filename):
    imageToVTK(filename, pointData={scalar_name: s})
    return None


def loops_to_vtk_sources(loops, savefile, radius=5, density=3, z_level=5):
    X = []
    Y = []
    Z = []

    for j, t_l in enumerate(loops):
        x, y, z = spherical_grid(t_l[0, 0], t_l[0, 1], z_level, radius, density)
        X.append(x)
        Y.append(y)
        Z.append(z)
        x, y, z = spherical_grid(t_l[-1, 0], t_l[-1, 1], z_level, radius, density)
        X.append(x)
        Y.append(y)
        Z.append(z)

    X = np.array(X).flatten()
    Y = np.array(Y).flatten()
    Z = np.array(Z).flatten()
    data = np.ones(np.shape(X)[0])
    pointsToVTK(savefile, X, Y, Z, {'source': data})
    return None


def source_points(filename, savefile, radius=5, density=3, z_level=5):
    _, endpoints = read_looptrace(filename)
    X = []
    Y = []
    Z = []
    for _, v in endpoints.items():
        for xy in v:
            x, y, z = spherical_grid(xy[0], xy[1], z_level, radius, density)
            X.append(x)
            Y.append(y)
            Z.append(z)
    X = np.array(X).flatten()
    Y = np.array(Y).flatten()
    Z = np.array(Z).flatten()
    data = np.ones(np.shape(X)[0])
    pointsToVTK(savefile, X, Y, Z, {'source': data})
    return None


def regular_grid(savefile, step=20, z=5, size=400, margin=5):
    n = size // step
    X = np.linspace(margin, size-margin, n)
    Y = np.linspace(margin, size-margin, n)
    Z = np.array([z])
    X, Y, Z = np.meshgrid(X, Y, Z)
    X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()
    data = np.ones(np.shape(X)[0])
    pointsToVTK(savefile, X, Y, Z, {'source': data})
    return None


def spherical_grid(x0, y0, z0, r, density):
    # RETURNS SEMI-SPHERICAL GRID
    theta = np.linspace(0, np.pi, density, endpoint=False)
    phi = np.linspace(0, 2 * np.pi, density, endpoint=False)
    theta, phi = np.meshgrid(theta, phi)
    X = r * np.cos(theta) * np.cos(phi) + x0
    Y = r * np.cos(theta) * np.sin(phi) + y0
    Z = r * np.sin(theta) + z0
    # POLAR POINTS
    np.append(X, [x0])
    np.append(Y, [y0])
    np.append(Z, [z0 + r])
    return X.flatten(), Y.flatten(), Z.flatten()


def read_looptrace(filename):
    """READ LOOPTRACING_AUTO4  OUTPUT FILE"""
    loops = {}
    endpoints = {}
    with open(filename, 'r') as f:
        for line in f:
            line = re.sub(' +', ' ', line)
            data = [float(x) for x in line.split()]
            key = int(data[0])
            if key not in loops.keys():
                loops[key] = {'points': [[data[1], data[2]]], 'signal': [data[3]]}
            else:
                loops[key]['points'].append([data[1], data[2]])
                loops[key]['signal'].append(data[3])
    for key in loops.keys():
        endpoints[key] = [loops[key]['points'][0], loops[key]['points'][-1]]
    return loops, endpoints


def save_scalar_cube(cube, data_name, filename, origin=(0., 0., 0.), spacing=(1., 1., 1.)):
    X = np.arange(origin[0], np.shape(cube)[0], spacing[0], dtype='float64')
    Y = np.arange(origin[1], np.shape(cube)[1], spacing[1], dtype='float64')
    Z = np.arange(origin[2], np.shape(cube)[2], spacing[2], dtype='float64')
    gridToVTK(filename, X, Y, Z, pointData={data_name: cube})
    return None


def save_vector_data(vx, vy, vz, vector_name, filename, origin=(0., 0., 0.), spacing=(1., 1., 1.)):
    """CORRECTLY SAVES VECTOR DATA LIKE IN IDL's: sav2vtk.pro"""
    assert np.shape(vx) == np.shape(vy) == np.shape(vz)
    X = np.arange(origin[0], np.shape(vx)[0], spacing[0], dtype='float64')
    Y = np.arange(origin[1], np.shape(vx)[1], spacing[1], dtype='float64')
    Z = np.arange(origin[2], np.shape(vx)[2], spacing[2], dtype='float64')
    gridToVTK(filename, X, Y, Z, pointData={vector_name: (vx, vy, vz)})
    return None


def field_from_box(filename):
    box = GXBox(filename)
    bx = np.array(box.bx, dtype='float64')
    by = np.array(box.by, dtype='float64')
    bz = np.array(box.bz, dtype='float64')
    return bx, by, bz


def energy_density(b_cube):
    e = np.sum(np.power(b_cube, 2), axis=-1)
    return e / 8. / np.pi


def prepare_target_name(filename, target_dir='target'):
    cd = os.path.dirname(filename)
    basename = os.path.basename(filename)
    basename = basename.split('.')[:-1]
    basename = ''.join(basename)  # + '.vtk'
    target_dir = os.path.join(cd, target_dir)
    os.makedirs(target_dir, exist_ok=True)
    return target_dir, basename


def box2vtk(filename, field_name):
    target_dir, basename = prepare_target_name(filename, target_dir='vtk_field')
    vx, vy, vz = field_from_box(filename)
    save_vector_data(vx, vy, vz, field_name, os.path.join(target_dir, basename))
    return None


def box2npy(filename, field_name):
    target_dir, basename = prepare_target_name(filename, target_dir='npy_field')
    vx, vy, vz = field_from_box(filename)
    basename += '.npy'
    basename = field_name + '_' + basename
    vx = np.expand_dims(vx, axis=-1)
    vy = np.expand_dims(vy, axis=-1)
    vz = np.expand_dims(vz, axis=-1)
    v = np.concatenate([vx, vy, vz], axis=-1)
    np.save(os.path.join(target_dir, basename), v)
    return None


def free_energy(pot_dir, high_dir, absolute=True):
    """

    :param pot_dir:
    :param high_dir: Directory with non-potential files
    :param absolute: True, Calculate absolute or relative free energy density
    :return: None
    """
    # PARALLEL IMPLEMENTATION LATER MAYBE
    pot_files = files_list(pot_dir, filter='.npy')
    high_files = files_list(high_dir, filter='.npy')
    assert len(pot_files) == len(high_files)
    # pot_triplets = [pot_files[i-1: i+2] for i in range(len(pot_files))]
    # high_triplets = [high_files[i - 1: i + 2] for i in range(len(high_files))]
    for p, h in tqdm(zip(pot_files, high_files)):
        pot = np.load(p)
        high = np.load(h)
        e_p = energy_density(pot)
        e_np = energy_density(high)
        if absolute:
            free = e_np - e_p
        else:
            free = (e_np - e_p) / e_p
        target_dir, basename = prepare_target_name(h, target_dir='free_energy')
        basename = 'FreeEnergy_' + basename
        save_scalar_cube(free, 'E_free', os.path.join(target_dir, basename))
        basename += '.npy'
        np.save(os.path.join(target_dir, basename), free)
    return None


def box2curl2vtk(filename, field_name):
    """CONVERTS GX BOX TO VTK CURL"""
    target_dir, basename = prepare_target_name(filename, target_dir='vtk_curl')
    vx, vy, vz = field_from_box(filename)
    cx, cy, cz = curl(vx, vy, vz)
    
    average_angle = np.mean(directions_closure(angles(vx, vy, vz, cx, cy, cz)))
    print(f'Average angle between field and curl: {average_angle}')
    save_vector_data(cx, cy, cz, field_name, os.path.join(target_dir, basename))
    return None


def box2curl2grid(filename, origin=(0, 0, 0), spacing=(1, 1, 1)):
    vx, vy, vz = field_from_box(filename)
    cx, cy, cz = curl(vx, vy, vz)

    X = np.arange(origin[0], np.shape(vx)[0], spacing[0], dtype='float64')
    Y = np.arange(origin[1], np.shape(vx)[1], spacing[1], dtype='float64')
    Z = np.arange(origin[2], np.shape(vx)[2], spacing[2], dtype='float64')
    return (cx, cy, cz), (X, Y, Z)


def box2directions2vtk(filename, field_name):
    """CONVERTS GX BOX TO VTK CURL"""
    target_dir, basename = prepare_target_name(filename, target_dir='vtk_closure')
    vx, vy, vz = field_from_box(filename)
    cx, cy, cz = curl(vx, vy, vz)

    average_angle = directions_closure(angles(vx, vy, vz, cx, cy, cz))
    save_scalar_cube(average_angle, field_name, os.path.join(target_dir, basename))
    return None


def convert_folder(path, field_name, func=box2vtk, filter='.sav', n_jobs=8, last=0):
    """YOU SHOULD SPECIFY WHAT FUNCTION TO USE IN PARALLEL"""
    files = files_list(path, filter)
    if last > 0:
        files = files[-last:]
    Parallel(n_jobs=n_jobs)(delayed(func)(file, field)
                            for file, field in tqdm(zip(files, [field_name]*len(files))))
    return None


def convert_folder_serial(path, field_name, func=box2vtk, filter='.sav', n_jobs=8, last=0):
    """YOU SHOULD SPECIFY WHAT FUNCTION TO USE IN PARALLEL"""
    files = files_list(path, filter)
    if last > 0:
        files = files[-last:]
    for file, field in tqdm(zip(files, [field_name]*len(files))):
        try:
            func(file, field)
        except Exception:
            print(f'File {file} can be corrupted')
    return None


def files_list(path, filter):
    print(f'Start searching for files *{filter} in folder: {path}')
    files = sorted(glob.glob(f'{path}/*{filter}'))
    print(f'Found {len(files)} files')
    return files


def torch2vtk(filename):
    data = torch.load(filename)
    data = data.permute(0, 3, 2, 1)  # to npy order zyx -> xyz
    fx, fy, fz = data.unbind()
    basename = os.path.splitext(os.path.basename(filename))[0]
    target_name = os.path.join(os.path.dirname(filename), basename)
    save_vector_data(fx.numpy(), fy.numpy(), fz.numpy(), 'B', target_name)

