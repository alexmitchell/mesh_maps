#!/usr/bin/env python3
#
# This is intended as a bare-bones landscape evolution model using landlab

import numpy as np

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  as plt3d # allows projection='3d'
import matplotlib.animation as mpl_animation

import landlab as lab
import landlab.components as lab_components
from landlab.plot.imshow import imshow_grid_at_node

# Model parameters
n_steps = 100
dt = 100 # years
model_duration = n_steps * dt

#map_width = 640
#map_height = 640
node_spacing = 100 # m
max_elevation = 1500.0 # m
max_uplift_rate = 0.1 # m/a

# Filepaths
#resolution = 160
resolution = 320
init_topo_filepath = f'./images/island_topo_{resolution}_{resolution}.png'
init_uplift_filepath = f'./images/island_uplift_{resolution}_{resolution}.png'
# swap the topo and uplift files for funsies
#init_uplift_filepath = f'./images/island_topo_{resolution}_{resolution}.png'
#init_topo_filepath = f'./images/island_uplift_{resolution}_{resolution}.png'

# Setup functions
class StaticUplifter:

    def __init__(self, mg):
        self.mg = mg

    def run_one_step(self, dt=None, **kwargs):
        uplift_map = self.mg['node']['uplift_map']
        uplift = uplift_map * (1 if dt is None else dt)

        self.mg['node']['topographic__elevation'] += uplift


def load_np_maps():
    # Load the initial maps
    print(f"Loading init maps")
    np_2D_topo_map = plt.imread(init_topo_filepath)
    np_2D_uplift_map = plt.imread(init_uplift_filepath)
    assert np_2D_topo_map.shape == np_2D_uplift_map.shape
    
    # Scale map values
    np_2D_topo_map *= max_elevation / np.amax(np_2D_topo_map)
    np_2D_uplift_map *= max_uplift_rate / np.amax(np_2D_uplift_map)

    # Reformat map dimensions to landlab format
    map_shape = np_2D_topo_map.shape
    np_topo_map = np_2D_topo_map.ravel().astype(np.float)
    np_uplift_map = np_2D_uplift_map.ravel().astype(np.float)

    return np_topo_map, np_uplift_map, map_shape

def setup_landlab():

    np_topo_map, np_uplift_map, map_shape = load_np_maps()
    map_width, map_height = map_shape

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
    mg['node']['init_drainage_area'] = mg['node']['drainage_area'].copy()
    
    # Setup fluvial erosion
    print(f"Setting up fluvial erosion component")
    fluvial = lab_components.FastscapeEroder(mg)
    #fluvial = lab_components.ErosionDeposition(mg)

    # Setup diffusion
    print(f"Setting up diffusion component")
    diffusion = lab_components.LinearDiffuser(mg)

    components = {
            "uplift"      : uplifter,
            "flow_router" : flow_router,
            "fluvial"     : fluvial,
            "diffusion"   : diffusion,
        }

    component_order = [
            "uplift",
            "flow_router",
            "fluvial",
            "diffusion",
        ]

    print(f"Finished setting up")
    print()
    return mg, components, component_order


# Plotting functions
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

def render_imshow_comparisons(mg, step, fig=None, axes=None, show=False):
    if fig is None or axes is None:
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15,10))
        axes[1,2].set_visible(False)
    assert axes.shape == (2, 3)

    plt.sca(axes[0,0])
    imshow_grid_at_node(mg, 'topographic__init_elevation')
    plt.title(f"Initial topography")

    plt.sca(axes[0,1])
    imshow_grid_at_node(mg, 'topographic__elevation')
    plt.title(f"Topography at {step} steps")

    plt.sca(axes[0,2])
    z_init = mg['node']['topographic__init_elevation']
    z_final = mg['node']['topographic__elevation']
    imshow_grid_at_node(mg, z_final - z_init)
    plt.title(f"Topographic difference")

    plt.sca(axes[1,0])
    init_da = mg['node']['init_drainage_area']
    imshow_grid_at_node(mg, np.log10(init_da+1))
    plt.title(f"Initial drainage network")

    plt.sca(axes[1,1])
    da = mg['node']['drainage_area']
    imshow_grid_at_node(mg, np.log10(da+1))
    plt.title(f"Drainage network at {step} steps")

    if show:
        plt.show()


class ComparisonAnimator:

    def __init__(self, mg):
        print("Initializing animator object")
        self.mg = mg

        #self.fig, self.axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
        self.fig, self.axes = plt.subplots(nrows=2, ncols=3, figsize=(15,10))
        #self.axes[1,2].set_visible(False)

    def animation_init_func(self):

        print("Initializing animator init function")

        for ax in self.axes.ravel():
            ax.clear()

        ax_z_init  = self.axes[0,0]
        ax_z_last  = self.axes[0,1]
        ax_z_diff  = self.axes[0,2]
        ax_da_init = self.axes[1,0]
        ax_da_last = self.axes[1,1]

        v2r = mg.node_vector_to_raster

        # Plot topography
        z_init = v2r(mg['node']['topographic__init_elevation'])
        z_last = v2r(mg['node']['topographic__elevation'])

        ax_z_init.set_title(f"Initial topography")
        im_z_init = ax_z_init.imshow(z_init, animated=True)

        ax_z_last.set_title(f"Topography at 0 steps")
        im_z_last = ax_z_last.imshow(z_last, animated=True)

        ax_z_diff.set_title(f"Topographic difference")
        im_z_diff = ax_z_diff.imshow(z_last - z_init, animated=True)

        # Plot drainage areas
        da_init = v2r(mg['node']['init_drainage_area'])
        da_last = v2r(mg['node']['drainage_area'])

        ax_da_init.set_title(f"Initial drainage network")
        im_da_init = ax_da_init.imshow(np.log10(da_init + 1), animated=True)

        ax_da_last.set_title(f"Drainage network at 0 steps")
        im_da_last = ax_da_last.imshow(np.log10(da_last+1), animated=True)

        # Return a dict of the image and axis objects
        self.subplot_im_dict = {
                'topo_init' : (im_z_init, ax_z_init),
                'topo_last' : (im_z_last, ax_z_last),
                'topo_diff' : (im_z_diff, ax_z_diff),
                'da_init' : (im_da_init, ax_da_init),
                'da_last' : (im_da_last, ax_da_last),
                }
        return im_z_init, im_z_last, im_z_diff, im_da_init, im_da_last

    def animation_update_func(self, step):

        im_z_init, ax_z_init = self.subplot_im_dict['topo_init']
        im_z_last, ax_z_last = self.subplot_im_dict['topo_last']
        im_z_diff, ax_z_diff = self.subplot_im_dict['topo_diff']
        im_da_init, ax_da_init = self.subplot_im_dict['da_init']
        im_da_last, ax_da_last = self.subplot_im_dict['da_last']

        for ax in [ax_z_init, ax_z_last, ax_z_diff, ax_da_init, ax_da_last]:
            ax.clear()

        mg = self.mg
        v2r = mg.node_vector_to_raster

        # Plot topography
        z_init = v2r(mg['node']['topographic__init_elevation'])
        z_last = v2r(mg['node']['topographic__elevation'])

        ##im_z_init.set_data(z_init)
        #im_z_last.set_data(z_last)
        ##ax_z_last.set_title(f"Topography at {step} steps")
        #im_z_diff.set_data(z_last - z_init) # Not working!
        ax_z_init.set_title(f"Initial topography")
        im_z_init = ax_z_init.imshow(z_init, animated=True)

        ax_z_last.set_title(f"Topography at {step} steps")
        im_z_last = ax_z_last.imshow(z_last, animated=True)

        ax_z_diff.set_title(f"Topographic difference")
        im_z_diff = ax_z_diff.imshow(z_last - z_init, animated=True)

        # Plot drainage areas
        da_init = v2r(mg['node']['init_drainage_area'])
        da_last = v2r(mg['node']['drainage_area'])

        ##im_da_init.set_data(np.log10(da_init + 1))
        #im_da_last.set_data(np.log10(da_last + 1))
        ##ax_da_last.set_title(f"Drainage network at {step} steps")
        ax_da_last.set_title(f"Drainage network at {step} steps")
        im_da_last = ax_da_last.imshow(np.log10(da_last+1), animated=True)

        ax_da_init.set_title(f"Initial drainage network")
        im_da_init = ax_da_init.imshow(np.log10(da_init + 1), animated=True)

        #return im_z_init, im_z_last, im_da_init, im_da_last
        return im_z_init, im_z_last, im_z_diff, im_da_init, im_da_last
        #return im_z_last, im_z_diff, im_da_last


# Run model
print("Setting up...")
mg, components, component_order = setup_landlab()
animator = ComparisonAnimator(mg)

print()
print(f"Running model...")
#for step in range(n_steps):
def run_model_step(step):
    for component_name in component_order:
        if component_name == 'flow_router':
            components[component_name].run_one_step()
        else:
            components[component_name].run_one_step(dt)

    if step%(n_steps//10) == 0:
        print(f"{step/n_steps:2.0%} complete ({step}/{n_steps})")
    if step == n_steps - 1:
        # Get the final drainage network
        components['flow_router'].run_one_step()

        print(f"100% complete ({n_steps}/{n_steps})")
        print()

    return animator.animation_update_func(step)



# Plotting
print(f"Plotting")
#animation = mpl_animation.ArtistAnimation(fig, frames, blit=True)
animation = mpl_animation.FuncAnimation(animator.fig, run_model_step, 
        frames=n_steps, init_func=lambda : animator.animation_init_func(),
        blit=False, repeat=False)
plt.show()
#render_3D_topography(mg, n_steps, show=True)
#render_imshow_comparisons(mg, n_steps, show=True)



#plt.imshow(np_uplift_map)
#plt.show()
#assert False


