import numpy as np


def curl(vx, vy, vz):
    cx = np.gradient(vz, edge_order=2, axis=1) - np.gradient(vy, edge_order=2, axis=2)
    cy = np.gradient(vx, edge_order=2, axis=2) - np.gradient(vz, edge_order=2, axis=0)
    cz = np.gradient(vy, edge_order=2, axis=0) - np.gradient(vx, edge_order=2, axis=1)
    return np.squeeze(cx), np.squeeze(cy), np.squeeze(cz)


def normalize_vectors(vx, vy, vz):
    """
    INPUT: VECTOR FIELD
    OUTPUT: NORMED ON 1 VECTOR FIELD
    """
    vx2 = np.power(vx, 2)
    vy2 = np.power(vy, 2)
    vz2 = np.power(vz, 2)
    norm = np.sqrt(vx2 + vy2 + vz2)
    return vx / norm, vy / norm, vz / norm


def vector_dot(ax, ay, az, bx, by, bz):
    assert np.shape(ax) == np.shape(ay) == np.shape(az) == np.shape(bx) == np.shape(by) == np.shape(bz)
    return ax * bx + ay * by + az * bz


def angles(ax, ay, az, bx, by, bz):
    """RETURNS ANGLES BOX BETWEEN FIELDS A and B"""
    assert np.shape(ax) == np.shape(ay) == np.shape(az) == np.shape(bx) == np.shape(by) == np.shape(bz)
    ax, ay, az = normalize_vectors(ax, ay, az)
    bx, by, bz = normalize_vectors(bx, by, bz)
    scalar_dots = vector_dot(ax, ay, az, bx, by, bz)
    return 180 * np.arccos(scalar_dots) / np.pi


def directions_closure(ang):
    return np.abs(np.abs(90. - ang) - 90.)

