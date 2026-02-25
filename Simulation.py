from Medium import Medium
from Beam import Beam
from BoundaryConditions import BoundaryConditions, BoundaryConditionsGmsh
import numpy as np
from collections.abc import Callable
import bisect
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D  # (needed for 3D)
from fipy import CellVariable, Grid2D, Grid3D, TransientTerm, DiffusionTerm, Viewer

_SIGMA_SB = 5.670374419e-8  # Stefan–Boltzmann constant [W/(m^2·K^4)]
# Unit multipliers
unit_factor = {
    "m": 1.0,
    "cm": 1e2,
    "mm": 1e3,
    "um": 1e6, "µm": 1e6,
    "nm": 1e9,
    "km": 1e-3,
}

#### HELPER FUNCTIONS ####
# FUCNTION: Import a FiPy Viewer 2D object and scale the x and y axis to match a certain unit
def scale_viewer_axes_2D(viewer: Viewer,  # FiPy 2D Viewer
                         x_units: str = "mm", y_units: str ="mm",  # Default is m -> mm
                         x_label: str = "x", y_label: str="y",  # Default labels
                         decimals: int = 3,  # How many decimals to display on each axis
                         equal_aspect: bool = True  # Make x width = y height?
                         ) -> matplotlib.axes.Axes:

    # Get the axes from the viewer
    ax = getattr(viewer, "axes", None) or getattr(viewer, "_axes", None)
    if ax is None:
        raise ValueError("Couldn't find Matplotlib axes on the viewer. "
                         "Make sure you're using a Matplotlib-based viewer and call this after viewer.plot().")
    fig = ax.figure
    if fig is not None and fig.canvas is not None:
        fig.canvas.draw_idle()
    fig.set_size_inches((6, 6))

    unit_factor = {
        "m": 1.0,
        "cm": 1e2,
        "mm": 1e3,
        "um": 1e6, "µm": 1e6,
        "nm": 1e9,
        "km": 1e-3,
    }

    if x_units not in unit_factor or y_units not in unit_factor:
        raise ValueError("Unsupported units. Use one of: m, cm, mm, um/µm, nm, km")

    fx = unit_factor[x_units]
    fy = unit_factor[y_units]

    # Create formatters for the ticks that multiply the underlying data
    def _mk_formatter(f, sig):
        fmt = "{:." + str(sig) + "g}"
        return ticker.FuncFormatter(lambda val, pos: fmt.format(val * f))

    # Set the axes
    ax.xaxis.set_major_formatter(_mk_formatter(fx, decimals))
    ax.yaxis.set_major_formatter(_mk_formatter(fy, decimals))
    ax.set_xlabel(f"{x_label} [{x_units}]")
    ax.set_ylabel(f"{y_label} [{y_units}]")

    # Enforce aspect ratio(?)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
        forceAspect(ax)

    return ax


# FUNCTION: Given a set of axes scale the data to match a specific unit
def scale_axes(ax: matplotlib.pyplot.axes,
               plane: str = "xy",  # "xy", "xz", or "yz"
               units: str = ("mm", "mm"),  # (x_units, y_units)
               decimals: int = 3,  # OPTIONAL: tick label precision
               equal_aspect: bool =True,  # OPTIONAL: force equal aspect ratio
               preserve_limits: bool =True,  # OPTIONAL: Keep x and y limits
               labelsize: int = 13,  # OPTIONAL: Axis label font size
               ticklabelsize:int = 11
               ) -> matplotlib.pyplot.axes:    # OPTIONAL: Tick label font size

    # Valide planes
    plane = plane.lower()
    if plane not in ("xy", "xz", "yz"):
        raise ValueError("plane must be one of: 'xy', 'xz', 'yz'")

    x_label_char, y_label_char = {
        "xy": ("x", "y"),
        "xz": ("x", "z"),
        "yz": ("y", "z"),
    }[plane]

    ux, uy = units
    if ux not in unit_factor or uy not in unit_factor:
        raise ValueError("Unsupported units. Use one of: m, cm, mm, um/µm, nm, km")

    fx = unit_factor[ux]
    fy = unit_factor[uy]

    # Keep current limits
    if preserve_limits:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

    # Unit-scaled tick formatters
    def _mk_formatter(scale, sig):
        fmt = "{:." + str(sig) + "g}"
        return ticker.FuncFormatter(lambda val, pos: fmt.format(val * scale))

    ax.xaxis.set_major_formatter(_mk_formatter(fx, decimals))
    ax.yaxis.set_major_formatter(_mk_formatter(fy, decimals))

    # Labels + font sizes
    ax.set_xlabel(f"{x_label_char} [{ux}]", fontsize=labelsize)
    ax.set_ylabel(f"{y_label_char} [{uy}]", fontsize=labelsize)
    ax.tick_params(axis="both", labelsize=ticklabelsize)

    # Equal aspect in displayed units ----
    if equal_aspect:
        ax.set_aspect(fy / fx, adjustable="box")

    fig = ax.figure
    if fig is not None and fig.canvas is not None:
        fig.canvas.draw_idle()

    if preserve_limits:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    return ax


# FUNCTION: Force the plot to be 1:1 aspect ratio
def forceAspect(ax, aspect=1):
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect);


# FUNCTION: Return the region (index) where x lies given a bunch of regions defined by x0
def region_index(x0s: np.ndarray[float] | list[float] | float, # x0s that separate regions
                 x: float                  # Position where you want to find the region its in
                 ) -> float:
    # x0 must be sorted (strictly increasing recommended)
    x0s = np.asarray(x0s, dtype=float)
    m = len(x0s)
    if m == 0:
        raise ValueError("x0s must contain at least one region start.")
    # region start positions: choose last start <= x
    i = bisect.bisect_right(x0s, x) - 1
    if i < 0:
        return 0
    if i >= m:
        return m - 1
    return i

# FUNCTION: Compute the energy deposition gradient at a specific point given a beam and target medium
def compute_dEb_dx( x: float,        # Position to compute SE at (beam is fired along x-axis)
                    x_ref: float,    # Initial x position where values are known
                    dx: float,       # Step size along x
                    beam: Beam,      # Particle beam shot along x-direction
                    medium: Medium | list[Medium],   # Target medium(s) that particle beam is being shot at
                    ):
    if x_ref > x:
        dx = -abs(dx)

    if not np.iterable(medium):
        medium = np.array([medium])

    x_med = np.asarray([med.x0 for med in medium])

    E_inst = float(beam.E_0)   # eV per particle
    I_beam = float(beam.I_0)   # 1/s
    E_beam = E_inst * I_beam
    xi = float(x_ref)
    med_i = 0

    while True:
        if (dx > 0 and xi >= x) or (dx < 0 and xi <= x):
            break
        med_i = region_index(x_med, xi)
        # Step everything
        dEdx = medium[med_i].get_dEdx(E_inst)
        dIdx = medium[med_i].get_dIdx(E_inst, I_beam)
        dEdx_beam = dEdx * I_beam + E_inst * dIdx
        I_beam = I_beam + dIdx * dx
        E_beam = max(E_beam + dEdx_beam * dx, 0)
        E_inst = E_beam / I_beam
        xi += dx

    dEdx = medium[med_i].get_dEdx(E_inst)
    return -(I_beam * dEdx), E_inst


#### SIMULATION FUNCTIONS ####

# FUNCTION: Simulate heat generation for a particle beam irradiating a solid material(s) in 2D.
# Given boundary conditions and material(s) properties.
def heateq_solid_2d(
        beam: Beam,
        medium: Medium | list[Medium],  # Supports connected materials
        BC: BoundaryConditions,
        Ly: float,                      # Define the height and width of the sim box (NO SUPPORT FOR MATERIALS OF DIFFERENT HEIGHTS)
        t: float,                       # Total simulation time [s]
        T0: float = 298,                # OPTIONAL: Initial simulation temperature [K]
        SE = None,                      # OPTIONAL: Can give a pre-computed source energy term, otherwise it will compute it for you
        x_shift=None, y_shift=None,     # OPTIONAL: How much to shift the origin by
        alpha = 0, beta = 0,            # OPTIONAL: Beam divergence in y (alpha) and z (beta) directions
        dx: float = 1e-4, dy: float = 1e-4, # OPTIONAL: Cell widths and heights
        dt: float = 1e-3,           # OPTIONAL: Time interval between steps
        view: bool = False,         # OPTIONAL: Enable viewer?
        view_freq: int = 20,        # OPTIONAL: Update viewer every N steps
        dT_target: float = None,    # OPTIONAL: Scale dt so that a specific dT between steps can be achieved
        dt_ramp: float = None,      # OPTIONAL: Scaling factor to ramp dt by every step
        dt_max: float = 1,          # OPTIONAL: Set a maximum value that dt can ramp to
        x_units: str = 'mm', y_units: str = 'mm', # OPTIONAL: Scale viewer axes to a specific unit
        debug: bool = False
        ):

    # If only single material given, make it a list for ease of use later
    if not np.iterable(medium):
        medium = [medium]
    Lx = 0
    for med in medium:
        Lx += med.Lx
    # Create the mesh
    nx = int(np.floor(Lx / dx))
    ny = int(np.floor(Ly / dy))
    mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)
    if x_shift is None:
        x_shift = 0
    if y_shift is None:
        y_shift = -Ly / 2

    mesh += ((x_shift,), (y_shift,))  # shift mesh
    cx = mesh.cellCenters[0].value
    cy = mesh.cellCenters[1].value
    CX = cx.reshape((nx, ny), order='F')  # shape (nx, ny)
    CY = cy.reshape((nx, ny), order='F')

    # Mask regions of the mesh for each medium
    x = mesh.cellCenters[0]
    masks = []
    for i, med in enumerate(medium):
        xL = med.x0
        xR = medium[i + 1].x0 if i + 1 < len(medium) else (med.x0 + med.Lx)
        masks.append((x >= xL) & (x < xR))

    # Compute SE if not given
    if SE is None:
        cj = CX[:, 0]
        dEb_dx = np.zeros_like(cj)  # eV/(m·s)
        E_inst = np.zeros_like(cj)

        for l in range(nx - 1):
            dEb_dx[l], E_inst[l] = compute_dEb_dx(cj[l], cj[0], dx, beam, medium)
            if E_inst[l] <= 1.0 and debug:  # Cutoff at 1 eV to avoid float noise
                print(f"Beam stopped at x = {cj[l]:.2e} m (Cell {l}/{nx})")
                dEb_dx[l + 1:] = 0.0  # Explicitly zero out the rest
                E_inst[l + 1:] = 0.0
                break
        dEb_dx *= 1.602176634e-19  # ev to J
        phi_free = np.array(beam.PD(CX, CY, 0.0, alpha, beta))
        SE = dEb_dx[:, None] * phi_free * 1 / (float(beam.E_0 * beam.I_0))
        SE /= medium[0].Lz

        if(debug):
            # Plot energy Gradient
            plt.plot(cj*unit_factor[x_units], dEb_dx*1e-3, 'bo', markersize=2)
            plt.xlabel(rf'x [{x_units}]', size='x-large')
            plt.ylabel(r"$dE_{b}/dx$ $[keV/m · s^{-1}$)]", size='x-large')
            ax = plt.gca()
            # Loop over regions, label each one by their name
            for i, med in enumerate(medium[1:]):
                ax.axvline(med.x0*unit_factor[x_units], ls='--', color='k', lw=1)  # seperator
            plt.show()

            # Plot beam energy
            plt.plot(cj*unit_factor[x_units], E_inst*1e-3, 'bo', markersize = 2)
            plt.xlabel(rf'x [{x_units}]', size='x-large')
            plt.ylabel(r"$E_{inst}$ [keV]", size='x-large')
            ax = plt.gca()
            for i, med in enumerate(medium[1:]):
                ax.axvline(med.x0 * unit_factor[x_units], ls='--', color='k', lw=1)  # seperator
            plt.show()

            plt.figure(figsize=(10, 4))
            plt.plot(cj*unit_factor[x_units], SE[:, ny // 2], label="Source Term (W/m^3)")
            ax = plt.gca()
            for i, med in enumerate(medium[1:]):
                ax.axvline(med.x0 * unit_factor[x_units], ls='--', color='k', lw=1)  # seperator
            plt.xlabel(rf'x [{x_units}]', size='x-large')
            plt.ylabel("Heat Source Intensity", size='x-large')
            plt.title("Beam Energy Deposition Profile (Centerline)")
            plt.legend()
            plt.show()
            P_per_m = float((SE.reshape(-1, order='F') * mesh.cellVolumes).sum())  # W/m
            print("Injected power [W/m] =", P_per_m)
            print("Injected power [W]   =", P_per_m * Ly)

        SE = SE.reshape(-1, order='F')



    # Create FiPy variables
    T0 = float(T0)
    SE = CellVariable(mesh=mesh, value=SE, name=r"$S_{E}$")
    T = CellVariable(mesh=mesh, value=T0, name="Temperature [K]", hasOld=True)

    rhoC = CellVariable(mesh=mesh, name="rhoC")
    k_cell = CellVariable(mesh=mesh, name="k")
    # Define the manual update function
    def _manual_refresh_props():
        # We use .value to get the current temperature array
        current_T = T.value

        # Initialize temp arrays
        new_rhoC = np.zeros(mesh.numberOfCells)
        new_k = np.zeros(mesh.numberOfCells)

        for i, med in enumerate(medium):
            # Evaluate material functions using temperature
            cp_vals = med.get_C(current_T)
            k_vals = med.get_k(current_T)

            # Use the masks to fill the arrays
            m = np.array(masks[i])
            new_rhoC[m] = med.rho * cp_vals[m]
            new_k[m] = k_vals[m]

        # Push the values back into the FiPy variables
        rhoC.setValue(new_rhoC)
        k_cell.setValue(new_k)

    k_face = k_cell.harmonicFaceValue
    eq = TransientTerm(coeff=rhoC) == DiffusionTerm(coeff=k_face) + SE

    if view:
        try:
            viewer = Viewer(vars=(T,), title="Temperature Distribution")
            ax = viewer.axes

            # Get extents of coordinates
            y_min = mesh.cellCenters[1].min()
            y_max = mesh.cellCenters[1].max()
            y_text = y_min + 0.03 * (y_max - y_min)  # small offset from bottom

            # Loop over regions, label each one by their name
            for i, med in enumerate(medium):
                ax.axvline(med.x0, ls = '--', color = 'k', lw = 1) # seperator
                if med.name is None:
                    continue

                # Region boundaries
                x_left = med.x0
                if i < len(medium) - 1:
                    x_right = medium[i + 1].x0
                else:
                    x_right = mesh.cellCenters[0].max()

                x_mid = 0.5 * (x_left + x_right)

                ax.text(
                    x_mid,
                    y_text,
                    med.name,
                    ha='center',
                    va='bottom',
                    fontsize=11,
                    color='gray',
                    alpha=0.8,
                    transform=ax.transData
                )
        except Exception:
            viewer = None

    t_elapsed = 0.0
    step = 0

    while t_elapsed < t:
        # Update variables and values
        T.updateOld()
        T_old = T.value.copy()

        retry = True
        while retry:
            # Attempt to solve
            res = 1e10
            inner_steps = 0
            # Sweep loop for nonlinearity
            while res > 1e-3 and inner_steps < 5:
                bcs = BC.update(mesh, T)
                _manual_refresh_props()
                res = eq.sweep(var=T, dt=dt, boundaryConditions=bcs)
                inner_steps += 1

            # Check if dT target is violated
            if dT_target is not None:
                dT_inf = np.max(np.abs(T.value - T_old))

                if dT_inf > dT_target:
                    print(f"dT ({dT_inf:.2f}K) > target. Retrying with dt/2.")
                    dt *= 0.5
                    T.setValue(T_old)  # RESET T BEFORE RETRYING!
                    continue  # Restart the 'while retry' loop

            # If we get here, the step was successful
            retry = False

        # Increment time variably
        t_elapsed += dt
        if (dt_ramp is not None and dt < dt_max):
            dt *= dt_ramp
        if (dt > dt_max):
            dt = dt_max
        if (t_elapsed + dt > t):
            dt = t - t_elapsed

        # Troubleshooting stuff
        if (step % max(1, view_freq)) == 0:
            if(BC.T_amb is not None):
                Tamb = float(BC.T_amb)
                err = float(np.max(np.abs(T.value - Tamb)))
                print(f"t={t_elapsed:.3f}s  Tmax={T.value.max():.6f}  Tmin={T.value.min():.6f}  max|T-Tamb|={err:.6e}")

        # Display simulation
        if viewer is not None:
            Tmin = float(T.value.min())
            Tmax = float(T.value.max())
            print(f"t={t_elapsed + dt:0.3f}s  T[min,max]=[{Tmin:.2f}, {Tmax:.2f}]")
            scale_viewer_axes_2D( viewer,
                                  x_units=x_units, y_units=y_units,
                                  x_label="x", y_label="y",
                                  decimals=4)
            viewer.plot()
        step += 1 # For viewing frequency

# FUNCTION: Simulate heat generation for a particle beam irradiating a solid in 3D.
# Given boundary conditions and material properties.
def heateq_solid_3d(beam: Beam,
                    medium: Medium | list[Medium],
                    BC: BoundaryConditions,
                    Ly: float, Lz: float,       # YZ Dimensions of medium
                    t: float,                   # Total simulation time [s]
                    T0: float = 298,            # OPTIONAL: Initial simulation temperature [K]
                    SE=None,                    # OPTIONAL: Can give a pre-computed source energy term, otherwise it will compute it for you
                    x_shift=None, y_shift=None, z_shift = None,  # OPTIONAL: How much to shift the origin by
                    alpha=0, beta=0,            # OPTIONAL: Beam divergence in y (alpha) and z (beta) directions
                    dx: float = 1e-4, dy: float = 1e-4, dz: float = 1e-4,           # OPTIONAL: Cell widths and heights
                    dt: float = 1e-3,           # OPTIONAL: Time interval between steps
                    view: bool = False,         # OPTIONAL: Enable viewer?
                    view_freq: int = 2,         # OPTIONAL: Update viewer every N steps
                    dT_target: float = None,    # OPTIONAL: Scale dt so that a specific dT between steps can be achieved
                    dt_ramp: float = None,      # OPTIONAL: Scaling factor to ramp dt by every step
                    dt_max: float = 1,          # OPTIONAL: Set a maximum value that dt can ramp to
                    x_units: str = 'mm', y_units: str = 'mm', z_units: str = 'mm',  # OPTIONAL: Scale viewer axes to a specific unit
                    debug: bool = True
                    ) -> [float, float]:

    import numpy as np

    # If only single material given, make it a list for ease of use later
    if not np.iterable(medium):
        medium = [medium]

    Lx = 0
    for med in medium:
        Lx += med.Lx

    # Making mesh
    if x_shift is None: x_shift = 0
    if y_shift is None: y_shift = -Ly / 2
    if z_shift is None: z_shift = -Lz / 2
    import numpy as np
    nx = int(np.floor(Lx / dx))
    ny = int(np.floor(Ly / dy))
    nz = int(np.floor(Lz / dz))
    mesh = Grid3D(nx=nx, dx=dx, ny=ny, dy=dy, nz=nz, dz=dz)
    mesh += ((x_shift,), (y_shift,), (z_shift,))

    # Getting cells in each direction
    cx = mesh.cellCenters[0].value
    cy = mesh.cellCenters[1].value
    cz = mesh.cellCenters[2].value
    CX = cx.reshape((nx, ny, nz), order='F')
    CY = cy.reshape((nx, ny, nz), order='F')
    CZ = cz.reshape((nx, ny, nz), order='F')

    # Mask regions of the mesh for each medium
    x = mesh.cellCenters[0]
    masks = []
    for i, med in enumerate(medium):
        xL = med.x0
        xR = medium[i + 1].x0 if i + 1 < len(medium) else (med.x0 + med.Lx)
        masks.append((x >= xL) & (x < xR))

    # Compute SE if not given
    if SE is None:
        cj = CX[:, 0, 0]
        dEb_dx = np.zeros_like(cj)  # eV/(m·s)
        E_inst = np.zeros_like(cj)
        for l in range(nx - 1):
            dEb_dx[l], E_inst[l] = compute_dEb_dx(cj[l], cj[0], dx, beam, medium)

        dEb_dx *= 1.602176634e-19  # ev to J

        phi_free = np.array(beam.PD(CX, CY, CZ, alpha, beta))
        SE = dEb_dx[:, None, None] * phi_free * 1 / (beam.E_0 * beam.I_0)

        if (debug):
            # Plot energy Gradient
            plt.title("1D Beam Energy Gradient Profile")
            plt.plot(cj * unit_factor[x_units], dEb_dx * 1e-3, 'bo', markersize=2)
            plt.xlabel(rf'x [{x_units}]', size='x-large')
            plt.ylabel(r"$dE_{b}/dx$ [$W·m^{-1}$]", size='x-large')
            ax = plt.gca()
            # Loop over regions, label each one by their name
            for i, med in enumerate(medium[1:]):
                ax.axvline(med.x0 * unit_factor[x_units], ls='--', color='k', lw=1)  # seperator
            plt.show()

            # Plot beam energy
            plt.title("Beam Instantaneous Energy")
            plt.plot(cj * unit_factor[x_units], E_inst * 1e-3, 'bo', markersize=2)
            plt.xlabel(rf'x [{x_units}]', size='x-large')
            plt.ylabel(r"$E_{inst}$ [keV]", size='x-large')
            ax = plt.gca()
            for i, med in enumerate(medium[1:]):
                differences = np.abs(cj - med.x0)
                ax.axvline(med.x0 * unit_factor[x_units], ls='--', color='k', lw=1,
                           label=r'$E_{inst} \approx$' + f"{E_inst[ differences.argmin()]*1e-3:.2f} keV")  # seperator
            plt.legend(fontsize='large')
            plt.show()

            # Plot deposited energy
            plt.plot(cj * unit_factor[x_units], SE[:, ny // 2, nz //2], 'bo', markersize=2)
            ax = plt.gca()
            for i, med in enumerate(medium[1:]):
                ax.axvline(med.x0 * unit_factor[x_units], ls='--', color='k', lw=1)  # seperator
            plt.xlabel(rf'x [{x_units}]', size='x-large')
            plt.ylabel(r"$S_{E}$ [$W\cdot m^{-3}$]", size='x-large')
            plt.title("Beam Volumetric Energy Deposition (Centerline z = y = 0)")
            plt.legend(fontsize='large')
            plt.show()
            P = float((SE.reshape(-1, order='F') * mesh.cellVolumes).sum())  # W/m
            print("Total power [W] =", P)

        SE = SE.reshape(-1, order='F')  # Need Fortran ordering

    # Create FiPy variables
    T0 = float(T0)
    SE = CellVariable(mesh=mesh, value=SE, name=r"$S_{E}$")
    T = CellVariable(mesh=mesh, value=T0, name="Temperature [K]", hasOld=True)
    rhoC = CellVariable(mesh=mesh, name="rhoC")
    k_cell = CellVariable(mesh=mesh, name="k")

    # Define the manual update function
    def _manual_refresh_props():
        current_T = T.value # Current temp array

        new_rhoC = np.zeros(mesh.numberOfCells)
        new_k = np.zeros(mesh.numberOfCells)

        for i, med in enumerate(medium):
            # Evaluate material functions \
            cp_vals = med.get_C(current_T)
            k_vals = med.get_k(current_T)

            # Use the masks to fill the arrays
            m = np.array(masks[i])
            new_rhoC[m] = med.rho * cp_vals[m]
            new_k[m] = k_vals[m]

        # Set values in FiPy variables
        rhoC.setValue(new_rhoC)
        k_cell.setValue(new_k)

    k_face = k_cell.harmonicFaceValue
    eq = TransientTerm(coeff=rhoC) == DiffusionTerm(coeff=k_face) + SE

    def viewer():
        # Three orthogonal mid-slices: xy, xz, yz
        ix, iy, iz = 0, ny // 2, nz // 2
        x0, x1 = x_shift, x_shift + Lx
        y0, y1 = y_shift, y_shift + Ly
        z0, z1 = z_shift, z_shift + Lz

        # Use manual spacing
        fig, axes = plt.subplots(
            1, 3,
            figsize=(12.5, 4.0),
            constrained_layout=True
        )

        fig.tight_layout()

        A = T.value.reshape((nx, ny, nz), order='F')

        # Plot three slices
        im_xy = axes[0].imshow(
            A[:, :, iz].T, origin='lower',
            extent=[x0, x1, y0, y1], aspect='auto', cmap='turbo'
        )
        axes[0].set_title(r'XY plane at z = $L_{z}/2$', fontsize=14)

        im_xz = axes[1].imshow(
            A[:, iy, :].T, origin='lower',
            extent=[x0, x1, z0, z1], aspect='auto', cmap='turbo'
        )
        axes[1].set_title(r'XZ plane at y = $L_{y}/2$', fontsize=14)

        im_yz = axes[2].imshow(
            A[ix, :, :].T, origin='lower',
            extent=[y0, y1, z0, z1], aspect='auto', cmap='turbo'
        )
        axes[2].set_title('YZ plane at x = 0', fontsize=14)

        # Scale/label each panel correctly
        scale_axes(axes[0], plane="xy", units=(x_units, y_units), decimals=3,
                   equal_aspect=False, preserve_limits=True, labelsize=14, ticklabelsize=12)
        scale_axes(axes[1], plane="xz", units=(x_units, z_units), decimals=3,
                   equal_aspect=False, preserve_limits=True, labelsize=14, ticklabelsize=12)
        scale_axes(axes[2], plane="yz", units=(y_units, z_units), decimals=3,
                   equal_aspect=False, preserve_limits=True, labelsize=14, ticklabelsize=12)
        for ax in axes:
            ax.set_box_aspect(1)
            # Get extents of coordinates
        y_min = mesh.cellCenters[1].min()
        y_max = mesh.cellCenters[1].max()
        z_min = mesh.cellCenters[2].min()
        z_max = mesh.cellCenters[2].max()
        y_text = y_min + 0.03 * (y_max - y_min)  # small offset from bottom
        z_text = z_min + 0.03 * (z_max - z_min)  # small offset from bottom
        # Loop over regions, label each one by their name
        for i, med in enumerate(medium):
            axes[0].axvline(med.x0, ls = '--', color = 'k', lw = 1) # seperator
            axes[1].axvline(med.x0, ls = '--', color = 'k', lw = 1) # seperator
            if med.name is None:
                continue

            # Region boundaries
            x_left = med.x0
            if i < len(medium) - 1:
                x_right = medium[i + 1].x0
            else:
                x_right = mesh.cellCenters[0].max()

            x_mid = 0.5 * (x_left + x_right)

            axes[0].text(
                x_mid,
                y_text,
                med.name,
                ha='center',
                va='bottom',
                fontsize=11,
                color='black',
                alpha=0.8,
                transform=axes[0].transData
            )
            axes[1].text(
                x_mid,
                z_text,
                med.name,
                ha='center',
                va='bottom',
                fontsize=11,
                color='black',
                alpha=0.8,
                transform=axes[1].transData
            )

        cbar = fig.colorbar(im_yz)
        cbar.set_label("T [K]", fontsize=14)
        cbar.ax.tick_params(labelsize=12)
        fig.subplots_adjust(left=0.06, right=0.94, bottom=0.18, top=0.88)
        return fig, A

    t_elapsed = 0.0
    step = 0
    T_maxes = []
    ts = []

    while t_elapsed < t:
        T.updateOld()
        T_old = T.value.copy()
        _manual_refresh_props()
        bcs = BC.update(mesh, T)

        if dT_target is not None:
            while True:
                eq.solve(var=T, dt=dt, boundaryConditions=bcs)
                dT_inf = np.max(np.abs(T.value - T_old))
                if dT_inf > dT_target:
                    print("Time step too large for required dT, reducing by 50%")
                    T.setValue(T_old)
                    dt *= 0.5
                else:
                    break
        else:
            eq.solve(var=T, dt=dt, boundaryConditions=bcs)

        t_elapsed += dt
        step += 1

        # Ramp dt
        if (dt_ramp is not None and dt < dt_max):
            dt *= dt_ramp
        if (dt > dt_max):
            dt = dt_max
        if (t_elapsed + dt > t):
            dt = t - t_elapsed

        # Viewer update every view_freq steps ----
        if view and (step % view_freq == 0):
            fig, A = viewer()

            # Force PyCharm to "see" it as a completed figure
            fig.canvas.draw()

            # This is what makes SciView create a new plot entry
            plt.show()

            # Prevent memory blow-up
            plt.close(fig)

            vmin = A.min()
            vmax = A.max()
            T_maxes.append(vmax)
            ts.append(t_elapsed)
            print(f"step={step}  t={t_elapsed:.4e}s  T[min,max]=[{vmin:.2f}, {vmax:.2f}]  ΔT={vmax - vmin:.2f}")

    if view:
        plt.ioff()
        plt.show()

    return T_maxes, ts

from fipy import Gmsh3D

def heateq_solid_3d_test(
    beam,
    medium,
    mesh,
    BC,
    t,
    T0=298.0,
    SE=None,
    x_shift=0.0, y_shift=0.0, z_shift=0.0,
    alpha=0.0, beta=0.0,
    dt=1e-3,
    view=False,
    view_freq=2,
    dT_target=None,
    dt_ramp=None,
    dt_max=1.0,
    x_units="mm", y_units="mm", z_units="mm",
    debug=True,
):
    import numpy as np
    import matplotlib.pyplot as plt
    from fipy import CellVariable, TransientTerm, DiffusionTerm

    if not np.iterable(medium):
        medium = [medium]

    # cell center coords
    x = mesh.cellCenters[0].value
    y = mesh.cellCenters[1].value
    z = mesh.cellCenters[2].value
    nCells = mesh.numberOfCells

    x_cm = x * 1e-2
    y_cm = y * 1e-2
    z_cm = z * 1e-2

    # build material masks
    tag_map = np.asarray(mesh.physicalCellMap).astype(int)  # (nCells,)

    # If you used explicit ids in gmsh:
    names = []
    for name in mesh.physicalCells:
        names.append(name)

    masks = []
    for med in medium:
        if med.name not in names:
            raise KeyError(f"Medium name '{med.name}' not in physical volumes={names}")
        tag = names[med.name]
        m = (tag_map == tag)
        masks.append(m)

    covered = np.zeros(nCells, dtype=int)
    for m in masks:
        covered += m.astype(int)
    if (covered == 0).any() or (covered > 1).any():
        raise RuntimeError(
            f"Material masks bad: uncovered={(covered==0).sum()} overlap={(covered>1).sum()} "
            f"tags={np.unique(tag_map)}"
        )

    # compute SE
    if SE is None:
        # 1D profile along +x
        x_grid = np.linspace(0.0, float(x_cm.max()), 2000)
        dEb_dx = np.zeros_like(x_grid)   # whatever compute_dEb_dx returns (likely eV/(m*s))
        E_inst = np.zeros_like(x_grid)

        for i, xi in enumerate(x_grid):
            dEb_dx[i], E_inst[i] = compute_dEb_dx(xi, 0.0, 1e-4, beam, medium)

        # interpolate onto cells by their x-position (meters)
        dEb_dx_cells = np.interp(x_cm, x_grid, dEb_dx, left=0.0, right=0.0)
        dEb_dx_cells *= 1.602176634e-19  # eV -> J  (keep only if compute_dEb_dx is eV/(m*s))

        # beam profile evaluated at cell centers (use same unit system your beam expects)
        phi_free = np.asarray(beam.PD(x_cm, y_cm, z_cm, alpha, beta))

        SE_cells = dEb_dx_cells * phi_free / (beam.E_0 * beam.I_0)

        if debug:
            plt.figure()
            plt.title("1D Beam Energy Gradient Profile")
            plt.plot(x_grid, dEb_dx, "b.")
            plt.xlabel("x [m]")
            plt.ylabel("dEb/dx (raw units)")
            plt.show()

            plt.figure()
            plt.title("Cell-wise volumetric source term SE")
            plt.plot(x_cm, SE_cells, "b.", markersize=2)
            plt.xlabel("x [m]")
            plt.ylabel("SE [W/m^3] (as coded)")
            plt.show()
    else:
        SE_cells = np.asarray(SE, dtype=float)
        if SE_cells.shape != (nCells,):
            raise ValueError(f"Provided SE must be shape {(nCells,)}, got {SE_cells.shape}")

    # --- FiPy variables ---
    T = CellVariable(mesh=mesh, value=float(T0), name="Temperature [K]", hasOld=True)
    SE_var = CellVariable(mesh=mesh, value=SE_cells, name=r"$S_E$")
    rhoC = CellVariable(mesh=mesh, name="rhoC")
    k_cell = CellVariable(mesh=mesh, name="k")

    def _manual_refresh_props():
        current_T = T.value
        new_rhoC = np.zeros(nCells)
        new_k = np.zeros(nCells)

        for med, m in zip(medium, masks):
            cp_vals = med.get_C(current_T)
            k_vals = med.get_k(current_T)
            new_rhoC[m] = med.rho * cp_vals[m]
            new_k[m] = k_vals[m]

        if np.any(new_rhoC <= 0) or not np.isfinite(new_rhoC).all():
            raise RuntimeError(f"rhoC bad: min={new_rhoC.min()}, zeros={(new_rhoC==0).sum()}")
        if np.any(new_k <= 0) or not np.isfinite(new_k).all():
            raise RuntimeError(f"k bad: min={new_k.min()}, zeros={(new_k==0).sum()}")

        rhoC.setValue(new_rhoC)
        k_cell.setValue(new_k)

    k_face = k_cell.harmonicFaceValue
    eq = TransientTerm(coeff=rhoC) == DiffusionTerm(coeff=k_face) + SE_var

    if(view):
        viewer = Viewer(vars=(T,))

    # --- time loop ---
    t_elapsed = 0.0
    step = 0
    T_maxes = []
    ts = []

    while t_elapsed < t:
        T.updateOld()
        T_old = T.value.copy()

        _manual_refresh_props()
        bcs = BC.update(mesh, T)
        fixed_mask = np.asarray(BC.mats["Tantalum"]["Fixed"], dtype=bool)  # example
        if dT_target is not None:
            inner = 0
            while True:
                inner += 1
                eq.solve(var=T, dt=dt, boundaryConditions = bcs)
                dT_inf = float(np.max(np.abs(T.value - T_old)))
                Tf = np.asarray(T.faceValue)

                if dT_inf > dT_target:
                    T.setValue(T_old)
                    dt *= 0.5
                    if dt < 1e-30:
                        raise RuntimeError("dt underflow while enforcing dT_target")
                else:
                    break

                if inner > 50:
                    raise RuntimeError("Failed to satisfy dT_target after 50 inner attempts")
        else:
            eq.solve(var=T, dt=dt, boundaryConditions=bcs)

        t_elapsed += dt
        step += 1

        # ramp dt
        if dt_ramp is not None and dt < dt_max:
            dt *= dt_ramp
        if dt > dt_max:
            dt = dt_max
        if t_elapsed + dt > t:
            dt = t - t_elapsed

        if view and (step % max(1, view_freq) == 0):
            viewer.plot()

            vmin = float(T.value.min())
            vmax = float(T.value.max())
            T_maxes.append(vmax)
            ts.append(t_elapsed)
            print(f"step={step}  t={t_elapsed:.4e}s  T[min,max]=[{vmin:.2f}, {vmax:.2f}]  ΔT={vmax - vmin:.2f}")

    if view:
        plt.ioff()
        plt.show()

    return T_maxes, ts