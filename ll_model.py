#!/usr/bin/env python3
#
# This is intended as a bare-bones landscape evolution model using landlab

import numpy as np
import scipy as sc
from scipy.spatial import Delaunay

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  as plt3d # allows projection='3d'

import landlab as lab
import landlab.components as lab_components
from landlab.plot.imshow import imshow_grid_at_node

# Model runs
n_steps = 10

# Define map boundaries
map_width = 640
map_height = 640
node_spacing = 1000 #kms

# Filepaths
init_topo_filepath = './images/init_topo.png'
init_uplift_filepath = './images/init_uplift.png'

# Setup landlab
class StaticUplifter:
    def __init__(self, mg):
        self.mg = mg

    def run_one_step(self, dt=None, **kwargs):
        uplift_map = self.mg['node']['uplift_map']
        uplift = uplift_map * (1 if dt is None else dt)

        self.mg['node']['topographic__elevation'] += uplift


def setup_landlab():
    # Load the initial map and format them
    print(f"Loading init maps")
    np_2D_topo_map = plt.imread(init_topo_filepath)
    np_2D_uplift_map = plt.imread(init_uplift_filepath)

    assert np_2D_topo_map.shape == (map_width, map_height)
    assert np_2D_uplift_map.shape == (map_width, map_height)

    np_topo_map = np_2D_topo_map.ravel().astype(np.float)
    np_uplift_map = np_2D_uplift_map.ravel().astype(np.float)

    # Set up landlab grid
    # For now using a raster grid
    print(f"Setting up Landlab grid")
    mg = lab.RasterModelGrid((map_width, map_height), node_spacing)
    mg.add_field('node', 'topographic__elevation', np_topo_map, units='m')
    mg.add_field('node', 'topographic__init_elevation', np_topo_map.copy(), units='m')

    g = mg.calc_grad_at_link('topographic__elevation')
    mg.add_field('link', 'topographic__slope', g)

    # Setup boundaries
    print(f"Setting up boundaries")
    mg.set_closed_boundaries_at_grid_edges(False, False, False, False)
    #mg.set_closed_boundaries_at_grid_edges(True, True, True, True)
    #mg.status_at_node[outlet_node] = mg.BC_NODE_IS_FIXED_VALUE

    # Setup uplift
    print(f"Setting up uplift component")
    mg.add_field('node', 'uplift_map', np_uplift_map, units='m/a')
    uplifter = StaticUplifter(mg)

    # Setup flow router and run it once
    print(f"Setting up flow router component")
    flow_router = lab_components.FlowAccumulator(
            mg, flow_director='FlowDirectorD8',
            depression_finder=lab_components.DepressionFinderAndRouter)
    flow_router.run_one_step()
    
    # Setup fluvial erosion
    print(f"Setting up fluvial erosion component")
    fluvial = lab_components.FastscapeEroder(mg)
    #fluvial = lab_components.ErosionDeposition(mg)

    # Setup diffusion
    print(f"Setting up diffusion component")
    diffusion = lab_components.LinearDiffuser(mg)

    components = [
            uplifter,
            flow_router,
            fluvial,
            diffusion,
        ]

    print(f"Finished setting up")
    print()
    return mg, components


mg, components = setup_landlab()

# Run model
print(f"Running model...")
for i in range(n_steps):
    #for component in components:
    #    component.run_one_step()

    if i%n_steps//10 == 0:
        print(f"{i/n_steps:2.0%} complete ({i}/{n_steps})")

print(f"100% complete ({n_steps}/{n_steps})")
print()

# Plotting
def render_3D_topography(mg, step, ax=None, init=False, show=False):
    """ Plot the topography from a default 3D perspective """

    if init:
        fig_name = 'Initial_topography' 
        plot_title = 'Initial topography' 
    else:
        fig_name = 'Current_topography'
        plot_title = f'Topography at t={step}' 

    if ax is None:
        hsize = 11
        vsize = 7
        fig = plt.figure(fig_name, figsize=(hsize, vsize))
        ax = plt.subplot(111, projection='3d')
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
        ax.set_title(plot_title)
        ax.view_init(35, -80)

    v2r = mg.node_vector_to_raster

    # Get the x, y, z, and initial z data and reshape them.
    # Trim the edges from the rasterized grids
    x = v2r(mg.node_x.copy(), flip_vertically=True)
    y = v2r(mg.node_y.copy(), flip_vertically=True)
    z_mg = mg['node']['topographic__elevation']
    z = v2r(z_mg, flip_vertically=True)
    z_init_mg = mg['node']['topographic__init_elevation']
    z_init = v2r(z_init_mg, flip_vertically=True)

    z_min = min(np.amin(z), np.amin(z_init))
    z_max = max(np.amax(z), np.amax(z_init))

    z_plot = (z_init if init else z).copy()

    # Plot the topography surface
    nrows, ncols = mg.shape
    surf = ax.plot_surface(x, y, z_plot, 
            rcount=nrows, ccount=ncols,
            antialiased=False)
    wire = ax.plot_wireframe(
            x, y, z_plot, color='k', 
            linewidth=0.01, antialiased=False,
            rcount=nrows, ccount=ncols,
            )
    ax.set_zlim3d(z_min, z_max)

    if show:
        plt.show()


print(f"Plotting")
#render_3D_topography(mg, n_steps, show=True)
fig = plt.figure()
imshow_grid_at_node(mg, 'topographic__elevation')

fig = plt.figure()
da = mg['node']['drainage_area']
imshow_grid_at_node(mg, np.log10(da+1))
plt.show()



#plt.imshow(np_uplift_map)
#plt.show()
#assert False


