from paraview.simple import *
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt
from scipy.linalg import norm

R = 15.0
O = [203.0, 144.0, 11.0]
field_name = 'Bnlfffe'

source = GetActiveSource() 

data = servermanager.Fetch(source)
vec_vtk = data.GetPointData().GetVectors()
vec_np = vtk_to_numpy(vec_vtk)
nx, ny, nz = data.GetDimensions()
data = vec_np.reshape((nx, ny, nz, 3), order='F')

x = np.arange(nx)
y = np.arange(ny)
z = np.arange(nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
coordinates = np.stack((X, Y, Z), axis=-1)
squared_coords = coordinates ** 2


def calculate_B_mean(data, coordinates, squared_coords, O, R):
    term1 = squared_coords[..., 0] - 2 * coordinates[..., 0] * O[0] + O[0] ** 2
    term2 = squared_coords[..., 1] - 2 * coordinates[..., 1] * O[1] + O[1] ** 2
    term3 = squared_coords[..., 2] - 2 * coordinates[..., 2] * O[2] + O[2] ** 2
    squared_dist = term1 + term2 + term3
    mask = squared_dist <= R ** 2
    B_mean = np.mean(data[mask], axis=0)
    indices = np.where(mask.ravel())[0]
    return B_mean, indices

def create_local_frame(B_mean_):
    B_n = B_mean_ / norm(B_mean_)
    basis_vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    dots = np.abs(basis_vectors @ B_n)
    arbitrary_vec = basis_vectors[np.argmin(dots)]
    x_prime = np.cross(arbitrary_vec, B_n)
    x_prime /= norm(x_prime)
    y_prime = np.cross(B_n, x_prime)
    y_prime /= norm(y_prime)
    return np.vstack([x_prime, y_prime, B_n]), B_n

def transform_field(B_global_, rotation_matrix_):
    return np.dot(B_global_, rotation_matrix_.T)

def plot_all_2d_components(B_local_, O, R, rotation_matrix, title_prefix=""):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    x_prime = rotation_matrix[0]
    y_prime = rotation_matrix[1]
    normal = rotation_matrix[2]

    grid_size = int(2 * R)
    u = np.linspace(-R, R, grid_size)
    v = np.linspace(-R, R, grid_size)
    U, V = np.meshgrid(u, v)

    X_plane = O[0] + U * x_prime[0] + V * y_prime[0]
    Y_plane = O[1] + U * x_prime[1] + V * y_prime[1]
    Z_plane = O[2] + U * x_prime[2] + V * y_prime[2]

    from scipy.interpolate import RegularGridInterpolator
    points = (np.arange(B_local_.shape[0]), np.arange(B_local_.shape[1]), np.arange(B_local_.shape[2]))

    Bx_interp = RegularGridInterpolator(points, B_local_[..., 0], bounds_error=False, fill_value=0)
    By_interp = RegularGridInterpolator(points, B_local_[..., 1], bounds_error=False, fill_value=0)
    Bz_interp = RegularGridInterpolator(points, B_local_[..., 2], bounds_error=False, fill_value=0)

    sample_points = np.stack((X_plane, Y_plane, Z_plane), axis=-1)
    Bx_plane = Bx_interp(sample_points)
    By_plane = By_interp(sample_points)
    Bz_plane = Bz_interp(sample_points)

    r = np.sqrt(U ** 2 + V ** 2)
    phi = np.arctan2(V, U)

    Br_plane = Bx_plane * np.cos(phi) + By_plane * np.sin(phi)
    Bphi_plane = -Bx_plane * np.sin(phi) + By_plane * np.cos(phi)
    Bn_plane = Bz_plane

    vmax = max(np.max(np.abs(Br_plane)), np.max(np.abs(Bphi_plane)), np.max(np.abs(Bn_plane)), 1e-6)

    im1 = axes[0].imshow(Br_plane, cmap='bwr', origin='lower',
                         vmin=-vmax, vmax=vmax,
                         extent=[u[0], u[-1], v[-1], v[0]])
    axes[0].set_title(fr"{title_prefix}$B_r$")
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(Bphi_plane, cmap='bwr', origin='lower',
                         vmin=-vmax, vmax=vmax,
                         extent=[u[0], u[-1], v[-1], v[0]])
    axes[1].set_title(fr"{title_prefix}$B_\phi$")
    plt.colorbar(im2, ax=axes[1])

    im3 = axes[2].imshow(Bn_plane, cmap='bwr', origin='lower',
                         vmin=-vmax, vmax=vmax,
                         extent=[u[0], u[-1], v[-1], v[0]])
    axes[2].set_title(fr"{title_prefix}$B_n$")
    plt.colorbar(im3, ax=axes[2])

    for ax in axes:
        circle = plt.Circle((0, 0), R, color='black',
                            fill=False, linestyle='--', linewidth=2)
        ax.add_patch(circle)
        ax.scatter(0, 0, c='black', s=100, marker='*')
        ax.set_xlim(u[0], u[-1])
        ax.set_ylim(v[-1], v[0])
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.set_xlabel('u')
        ax.set_ylabel('v')

    plt.tight_layout()
    plt.savefig(f'/home/nhan/Pictures/{title_prefix}_fig.png')

def calculate_curl(B):
    gradients = np.stack(np.gradient(B, axis=(0, 1, 2)), axis=-1)
    curl_x = gradients[:, :, :, 2, 1] - gradients[:, :, :, 1, 2]
    curl_y = gradients[:, :, :, 0, 2] - gradients[:, :, :, 2, 0]
    curl_z = gradients[:, :, :, 1, 0] - gradients[:, :, :, 0, 1]
    return np.stack([curl_x, curl_y, curl_z], axis=-1)

B_mean, indices = calculate_B_mean(data, coordinates, squared_coords, O, R)
print(f"Mean magnetic field in a spherical region of radius R={R}:")
print(f"Bx = {B_mean[0]:.2f}, By = {B_mean[1]:.2f}, Bz = {B_mean[2]:.2f}")

points_in_sphere = coordinates.reshape(-1, 3)[indices]
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_in_sphere[:, 0], points_in_sphere[:, 1], points_in_sphere[:, 2],
           s=5, c='blue', alpha=0.3, label=f'Точки в R={R}')
ax.scatter(*O, s=100, c='red', marker='*', label='Center O')
ax.quiver(*O, *B_mean, length=20, color='black',
          arrow_length_ratio=0.2, linewidth=3, label='B_mean')
ax.set_xlabel('Axis X')
ax.set_ylabel('Axis Y')
ax.set_zlabel('Axis Z')
ax.set_title(f'Vicinity of point O={O} with radius R={R}')
plt.legend()
plt.tight_layout()
plt.savefig('/home/nhan/Pictures/field.png')

rotation_matrix, B_n = create_local_frame(B_mean)
print(f"B_mean = {B_mean} (magnitude = {np.linalg.norm(B_mean):.3f})")
print(f"B_n = {B_n} (magnitude = {np.linalg.norm(B_n):.3f})")

B_local = np.apply_along_axis(transform_field, 3, data, rotation_matrix)
plot_all_2d_components(B_local, O, R, rotation_matrix, title_prefix="Local components: ")

curl_global = calculate_curl(data)
curl_local = np.apply_along_axis(transform_field, 3, curl_global, rotation_matrix)
plot_all_2d_components(curl_local, O, R, rotation_matrix, title_prefix="Curl ")


stream_tracer = StreamTracer(Input=source)
stream_tracer.Vectors = ['POINTS', field_name]
stream_tracer.SeedType = 'Point Cloud'
stream_tracer.SeedType.Center = O
stream_tracer.SeedType.Radius = R
stream_tracer.SeedType.NumberOfPoints = 20

UpdatePipeline()
stream_tracer_data = servermanager.Fetch(stream_tracer)

tube = Tube(Input=stream_tracer)

slice_plane = Slice(Input=source)
slice_plane.SliceType = 'Plane'
slice_plane.SliceType.Origin = O
slice_plane.SliceType.Normal = B_mean.tolist()
slice_plane.UpdatePipeline()


Show(stream_tracer)
Show(slice_plane)
Show(tube)

Render()
