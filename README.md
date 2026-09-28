# Magnetic Coupling Framework for Soft Robots

This [SOFA](https://www.sofa-framework.org/) example maps magnetic torque in tetrahedral soft robots to equivalent nodal forces. It accompanies the MARSS submission *A FEM Framework for Simulating Magnetic Soft Continuum Robots via Consistent Torque-to-Nodal Force Mapping*. Please cite the source if you use it.

Two scenes are provided: `magnetic_rod_bending_withCppPlugin.py` loads the C++ `MagneticPlugin`; `magnetic_rod_bending_onlyPythonScript.py` uses a Python controller and `ConstantForceField`. **Neither scene can run from a fresh checkout:** both reference `data/mesh/magnetic_rod.stl`, which is not in the repository.

## Versions and dependencies

| Item | Known from this repository | Still to verify locally |
| --- | --- | --- |
| SOFA | No release, commit or precision pinned | Exact release/commit, install/build origin, precision, OS and compiler |
| C++ plugin | `MagneticPlugin` version `1.0` (not a SOFA version); CMake >= 3.22 | Compatible SOFA ABI, compiler and runtime loading |
| Build packages | Eigen3, `Sofa.Framework`, `Sofa.Helper`, `Sofa.Core`, `Sofa.Component.Topology.Container.Dynamic` | Eigen version, CMake generator and build type |
| Python | SofaPython3, NumPy and Gmsh; Numba is optional in the Python-only scene | Python/SofaPython3/NumPy/Gmsh/Numba versions and JIT behavior |
| Example data | Referenced STL absent; no reference trajectory | Geometry source/hash, mesh and numerical results |

SOFA's [plugin build guide](https://sofa-framework.github.io/doc/plugins/build-a-plugin-from-sources/) explains out-of-tree CMake configuration. The [runSofa guide](https://sofa-framework.github.io/doc/using-sofa/runsofa/) explains Python scene loading and command options. These commands are a proposed procedure, **not a claim that any specific SOFA version has been tested**.

Both scenes request these SOFA plugins/components: `Sofa.Component.Constraint.Projective`, `Sofa.Component.Engine.Select`, `Sofa.Component.IO.Mesh`, `Sofa.Component.LinearSolver.Iterative`, `Sofa.Component.Mapping.Linear`, `Sofa.Component.Mass`, `Sofa.Component.ODESolver.Backward`, `Sofa.Component.SolidMechanics.FEM.Elastic`, `Sofa.Component.StateContainer`, `Sofa.Component.Topology.Container.Dynamic`, `Sofa.Component.Topology.Mapping`, `Sofa.Component.Visual`, `Sofa.GL.Component.Rendering3D` and `MultiThreading`. The C++ scene additionally requests `MagneticPlugin` and `Sofa.Component.LinearSolver.Direct`. SofaPython3 must also be available to run `.py` scenes. Verify that your SOFA distribution supplies `ParallelTetrahedronFEMForceField`, the selected linear solver and rendering components.

## Prepare the example mesh

From a Unix-like shell, start at the repository root. Substitute your own **closed, tetrahedralizable rod surface** with a coordinate scale consistent with the clamp box. Record its source, units and SHA-256 hash. The missing STL makes the repository incomplete as a standalone experiment.

```bash
git clone https://github.com/hyzhao955/Magnetic-Coupling-Framework-for-Soft-Robots.git
cd Magnetic-Coupling-Framework-for-Soft-Robots
git rev-parse HEAD
mkdir -p data/mesh
cp /absolute/path/to/your/magnetic_rod.stl data/mesh/magnetic_rod.stl
test -s data/mesh/magnetic_rod.stl
sha256sum data/mesh/magnetic_rod.stl
```

Make `numpy` and `gmsh` importable by the **interpreter embedded in SofaPython3**. The Python-only scene optionally uses `numba`; its import fallback and non-JIT execution still require local validation. After activating the Python environment used by SOFA, a possible installation and version check is:

```bash
python -m pip install numpy gmsh numba
python -c 'import sys,numpy,gmsh; print(sys.version); print(numpy.__version__,gmsh.__version__)'
```

Both scripts tetrahedralize the STL using Gmsh, write ASCII VTK alongside it and reuse any existing VTK without checking freshness. The C++ scene uses `data/mesh/magnetic_rod_h1p5.vtk`; Python-only uses `data/mesh/magnetic_rod.vtk`. Remove the relevant VTK when the STL, mesh size or Gmsh version changes. **Caution:** both converters assume consecutive Gmsh node tags starting at 1 (they calculate `tag - 1`); inspect connectivity if your mesh differs. They fall back to an expected VTK path after conversion failure, but do not provide that file.

## Build and load the C++ plugin

Set `SOFA_ROOT` to a writable SOFA install prefix; adjust paths if building against a SOFA build tree. Build with the **same** SOFA installation that runs the scene:

```bash
export SOFA_ROOT=/absolute/path/to/sofa/install
cmake -S 'MagneticForceField (C++ Plugin)' -B build/magnetic-plugin \
  -DCMAKE_PREFIX_PATH="$SOFA_ROOT/lib/cmake" \
  -DCMAKE_INSTALL_PREFIX="$SOFA_ROOT" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/magnetic-plugin --parallel
cmake --install build/magnetic-plugin
```

The CMake target is `MagneticPlugin` (typically `libMagneticPlugin.so` on Linux). It links `Sofa.Helper`, `Sofa.Core`, `Sofa.Framework`, `Sofa.Component.Topology.Container.Dynamic` and `Eigen3::Eigen`, and defines `SOFA_BUILD_MAGNETICPLUGIN`. No SOFA version, C++ standard, RPATH or runtime search path is pinned. If SOFA is not writable, install to another prefix and configure SOFA's PluginManager to find the shared library. Confirm that `RequiredPlugin(name='MagneticPlugin')` loads it and `MagneticTetraForceField` is registered; a successful compilation alone does not verify runtime compatibility.

## Run

Run from the repository root because the STL and CSV paths are relative to the **working directory**. These are finite headless smoke-test commands, not measured successful runs. Check `runSofa -h` if your SOFA release uses different options. For a GUI run, omit `-g batch -a -n ...` and start animation in the GUI.

```bash
"$SOFA_ROOT/bin/runSofa" -l SofaPython3 -g batch -a -n 50 magnetic_rod_bending_withCppPlugin.py
"$SOFA_ROOT/bin/runSofa" -l SofaPython3 -g batch -a -n 100 magnetic_rod_bending_onlyPythonScript.py
```

The C++ scene declares `MagneticPlugin` in its `RequiredPlugin` list. Loading SofaPython3 at the command line enables Python scene parsing. Calling `python script.py` directly does not construct and animate the SOFA scene.

## Scene inputs

| Input | C++ scene | Python-only scene |
| --- | --- | --- |
| Mesh | STL above; Gmsh size `1.5`; `magnetic_rod_h1p5.vtk` | Same STL; size `2.0`; `magnetic_rod.vtk` |
| Time and gravity | `dt=0.03` s; `[0,0,0]` | `dt=0.01` s; `[0,0,0]` |
| Placement and clamp | translation `[1,0,0]`; `fixingBox=[1,0,0,10,15,20]` | Same |
| Mechanics | mass `0.0024`, Young's modulus `180000`, Poisson ratio `0.3`, large-deformation parallel tetrahedral FEM; implicit Euler Rayleigh mass `0.4`, stiffness `0.02` | Same |
| Solver | `SparseLDLSolver` | `CGLinearSolver(iterations=300,tolerance=1e-10,threshold=1e-6)` |
| Magnetic input | `B=[0,0.001,0]`, `M0=[249000,0,0]`, `scaleFactor=1.0` | Same vectors; controller `scale=1.0` |
| Profiling | component defaults: `profileWindow=1.0` s, `profileOutput='magnetic_addforce_profile.csv'`, `profileSampleStride=10` | No CSV profiling |

Change these literals in the scene's `createScene` / `MagneticRod`; change C++ profiling values by passing component Data fields to `addObject("MagneticTetraForceField", ...)`. The repository does **not** define physical units for mesh coordinates, mass, modulus, `B` or `M0`; confirm experimental units before physical comparison. The C++ field computes `M=R M0`, `tau=M × B` and equivalent nodal face loads. Its `addDForce` and `addKToMatrix` do not add magnetic stiffness. The Python controller updates loads at `onAnimateEndEvent`, whereas C++ contributes in `addForce`. Different meshes, time steps, solvers and update timing prevent an apples-to-apples performance comparison. The Numba `prange` loop accumulates into shared nodal entries; check numerical repeatability before quantitative use.

## Outputs and acceptance checks

- Meshing should print `[Mesh] Nodes=..., Tets=...` and create/reuse the corresponding VTK. Counts depend on the supplied STL and Gmsh; none are asserted here. Verify a nonempty tetrahedral topology and nonempty `BoxROI.indices`.
- In a GUI, verify the rod loads and deforms as animation advances. Neither scene saves displacement, stress or force history by default, and no reference numerical trajectory is committed. Add a recorder and document its observable, units and time interval for a quantitative result.
- The C++ field can save `magnetic_addforce_profile.csv` in the launch directory. Header: `wall_time_s,sim_time,addForce_ms,rotation_ms,face_ms,sample_count,sample_stride`. It records at most one `addForce` sample per simulation time within `profileWindow`, then writes on a subsequent handled event. Run beyond the default 1 s window and check for data rows. `rotation_ms` and `face_ms` extrapolate sampled tetrahedra; timings are hardware dependent. An early stop, no subsequent event, `profileWindow<=0`, empty output path or write failure may prevent the CSV.
- Python-only should print `[MagLoad] Initialized: nodes=..., tets=...` when the controller's `init()` succeeds. Verify the callback is invoked and forces update; no CSV is created.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| CMake cannot find `Sofa.Framework` or `Eigen3` | Correct `CMAKE_PREFIX_PATH`; install Eigen development files; use matching SOFA build and runtime. |
| `MagneticPlugin` / `MagneticTetraForceField` missing | Locate the installed shared library in PluginManager, inspect linker dependencies and verify SOFA ABI/precision. |
| Python scene or `Sofa` module unavailable | Build/install and load SofaPython3; use the Python environment embedded by SOFA. |
| `gmsh` / `numpy` import error | Install into that same Python environment; verify the import there. |
| STL missing, VTK loader error or empty topology | Supply the absent STL, run from repository root, check watertightness and Gmsh meshing; remove stale VTK. |
| No deformation | Check fixed ROI indices, mesh scale, animation, magnetic field direction and plugin/controller errors; parallel `M0` and `B` yield zero initial torque. |
| FEM, solver, GL or `MultiThreading` component unavailable | Confirm required SOFA components are built and loadable; the scene's `RequiredPlugin` does not install them. |
| CSV missing or Python `[MagLoad][ERROR]` | Run C++ beyond the profile window, permit another event and check write permissions; inspect Python callback/JIT error and version compatibility. |

## Local verification still required

1. Publish one **tested** SOFA/SofaPython3 release or commit, OS, precision, compiler, CMake, Python and package versions with a successful build and both launch logs.
2. Supply or link a redistributable STL with hash, geometry and units. Verify Gmsh tag mapping, VTK connectivity, node/tet counts and fixed nodes. Until then the repo documents a procedure, not a self-contained reproducible experiment.
3. Record a numerical observable (for example tip position at specified simulation times), its units, an expected range/tolerance and the C++ CSV; verify the controller callback and CSV writing locally.
4. Harmonize mesh, `dt`, solver, force update and profiling scope before comparing C++ with Python. Validate force direction/magnitude against an analytical case and check Numba repeatability.
