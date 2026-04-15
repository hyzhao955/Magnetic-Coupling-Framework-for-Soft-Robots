# magnetic_rod_bending_withCppPlugin.py
# C++ plugin-based magnetic-force scene.
# 基于 C++ 插件的磁力场场景。

import os
import numpy as np
import Sofa
import Sofa.Core as SC

# ========== STL -> VTK ==========
# Convert a surface STL into a tetrahedral VTK mesh for SOFA.
# 将表面 STL 转换为 SOFA 可用的四面体 VTK 网格。
def convert_stl_to_vtk(stl_path, vtk_path=None, mesh_size=2.0):
    """将 STL 转成四面体网格，, convert stl to tetraheral mesh"""
    import math
    try:
        import gmsh
    except ImportError as e:
        print(f"[Mesh][ERROR] {e}")
        return None

    if not os.path.exists(stl_path):
        print(f"[Mesh][ERROR] STL not found: {stl_path}")
        return None
    h = float(mesh_size) if np.isfinite(mesh_size) and mesh_size > 0 else 1.0
    if vtk_path is None:
        mesh_tag = f"{h:g}".replace(".", "p")
        vtk_path = os.path.splitext(stl_path)[0] + f"_h{mesh_tag}.vtk"
    if os.path.exists(vtk_path):
        print(f"[Mesh] Using existing VTK: {os.path.abspath(vtk_path)}")
        return vtk_path

    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("mesh")
        gmsh.merge(stl_path)

        try:
            gmsh.model.mesh.classifySurfaces(45 * math.pi / 180.0, True, True, True)
            gmsh.model.mesh.createGeometry()
        except:
            pass

        surfaces = gmsh.model.getEntities(2)
        if not surfaces:
            gmsh.finalize()
            return None

        sl = gmsh.model.geo.addSurfaceLoop([s[1] for s in surfaces])
        gmsh.model.geo.addVolume([sl])
        gmsh.model.geo.synchronize()

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
        gmsh.model.mesh.field.add("MathEval", 1)
        gmsh.model.mesh.field.setString(1, "F", f"{h}")
        gmsh.model.mesh.field.setAsBackgroundMesh(1)

        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Algorithm3D", 4)
        gmsh.model.mesh.generate(3)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        coords = np.array(node_coords, dtype=float).reshape(-1, 3)

        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(3)
        tet_idx = None
        for i, et in enumerate(elem_types):
            if et == 4:
                tet_idx = i
                break
        if tet_idx is None:
            gmsh.finalize()
            return None

        tets = np.array(elem_node_tags[tet_idx], dtype=np.int64).reshape(-1, 4) - 1
        print(f"[Mesh] Nodes={coords.shape[0]}, Tets={tets.shape[0]}")

        with open(vtk_path, "w") as f:
            f.write("# vtk DataFile Version 2.0\nTetrahedral mesh\nASCII\nDATASET UNSTRUCTURED_GRID\n")
            f.write(f"POINTS {coords.shape[0]} float\n")
            for p in coords:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            f.write(f"\nCELLS {tets.shape[0]} {tets.shape[0] * 5}\n")
            for c in tets:
                f.write(f"4 {c[0]} {c[1]} {c[2]} {c[3]}\n")
            f.write(f"\nCELL_TYPES {tets.shape[0]}\n")
            f.write(("10\n") * tets.shape[0])

        gmsh.finalize()
        print(f"[Mesh] ✅ Saved: {vtk_path}")
        return vtk_path
    except Exception as e:
        print(f"[Mesh][ERROR] {e}")
        try:
            gmsh.finalize()
        except:
            pass
        return None


def MagneticRod(parentNode=None, name="MagneticRod",
           rotation=[0.0, 0.0, 0.0], translation=[0.0, 0.0, 0.0],
           fixingBox=[1.0, 0.0, 0.0, 10.0, 15.0, 20.0],
           stl_file="data/mesh/magnetic_rod.stl",
           mesh_size=2.0):
    # Build the deformable magnetic-rod subtree and attach the C++ magnetic force field.
    # 构建可变形磁棒子树，并挂接 C++ 实现的磁力场。
    """Create the magnetic rod soft body - final optimized version."""

    magnetic_rod = parentNode.addChild(name)

    # 网格
    vtk_file = convert_stl_to_vtk(stl_file, mesh_size=mesh_size)
    if vtk_file is None:
        vtk_file = stl_file.replace(".stl", ".vtk")

    # 求解器（优化参数）
    magnetic_rod.addObject('EulerImplicitSolver', rayleighMass=0.4, rayleighStiffness=0.02)
    magnetic_rod.addObject('SparseLDLSolver')
    # Optional iterative solver alternative: CGLinearSolver

    # 拓扑和力学
    magnetic_rod.addObject('MeshVTKLoader', name='loader', filename=vtk_file,
                           rotation=rotation, translation=translation)    
    magnetic_rod.addObject('TetrahedronSetTopologyContainer', name='topology', src='@loader')
    magnetic_rod.addObject('TetrahedronSetTopologyModifier')
    magnetic_rod.addObject('MechanicalObject', name='dofs', src='@loader')
    magnetic_rod.addObject('UniformMass', totalMass=0.0024)
    
    # 使用并行FEM
    magnetic_rod.addObject('ParallelTetrahedronFEMForceField', 
                           name='FEM',
                           youngModulus=180000, 
                           poissonRatio=0.3, 
                           method='large')
    
    # 固定约束
    magnetic_rod.addObject('BoxROI', name='boxROI', box=fixingBox, drawBoxes=True)
    magnetic_rod.addObject('FixedProjectiveConstraint', indices='@boxROI.indices')

    # Magnetic force field (C++ plugin)
    magnetic_rod.addObject("MagneticTetraForceField",
                           name="MagneticForceField",
                           B=[0.0, 1e-3, 0.0],
                           M0=[2.49e5, 0.0, 0.0],
                           scaleFactor=1.0)

    # === 可视化子节点（简洁版）===
    visu = magnetic_rod.addChild("Visual")
    visu.addObject("TriangleSetTopologyContainer", name="visuTopo")
    visu.addObject("TriangleSetTopologyModifier")
    visu.addObject("Tetra2TriangleTopologicalMapping", 
                   input="@../topology",  
                   output="@visuTopo")
    visu.addObject("OglModel", 
                   name="visualModel",
                   color=[0.9, 0.6, 0.2, 1.0])
    visu.addObject("IdentityMapping", input="@../dofs", output="@.")

    # Legacy contact node removed
    return magnetic_rod


# ========== 场景 ==========
def createScene(rootNode):
    # Assemble the full SOFA scene graph for the magnetic rod example.
    # 组装磁棒算例所需的完整 SOFA 场景图。
    # 必需插件
    required_plugins = [
        'MagneticPlugin',
        'Sofa.Component.Constraint.Projective',
        'Sofa.Component.Engine.Select',
        'Sofa.Component.IO.Mesh',
        'Sofa.Component.LinearSolver.Iterative',
        'Sofa.Component.Mapping.Linear',
        'Sofa.Component.Mass',
        'Sofa.Component.ODESolver.Backward',
        'Sofa.Component.SolidMechanics.FEM.Elastic',
        'Sofa.Component.StateContainer',
        'Sofa.Component.Topology.Container.Dynamic',
        'Sofa.Component.Topology.Mapping',
        'Sofa.Component.Visual',
        'Sofa.GL.Component.Rendering3D',
        'MultiThreading',
        'Sofa.Component.LinearSolver.Direct'
    ]
    
    for plugin in required_plugins:
        rootNode.addObject('RequiredPlugin', name=plugin)
    
    rootNode.addObject('DefaultAnimationLoop')
    
    # 启用力场可视化（这会显示力矢量，有助于理解应力分布）
    rootNode.addObject('VisualStyle', 
                       displayFlags='showBehavior showForceFields')
    
    rootNode.gravity = [0.0, 0.0, 0.0]
    rootNode.dt = 0.03
    
    # Legacy contact pipeline removed

    # Create magnetic rod
    MagneticRod(rootNode,
                translation=[1.0, 0.0, 0.0],
                stl_file="data/mesh/magnetic_rod.stl",
                mesh_size=1.5)

    return rootNode

