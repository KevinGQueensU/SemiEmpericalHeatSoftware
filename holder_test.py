import numpy as np

from Beam import Beam
from Medium import Atom, Medium
from BoundaryConditions import BoundaryConditions, BoundaryConditionsGmsh
import Simulation as sim
from fipy import CellVariable, Gmsh3D, Viewer, TransientTerm, DiffusionTermCorrection
import fipy
from fipy.tools import numerix as nx
import scipy as sp
import gmsh
from fipy.meshes import gmshMesh
from fipy.viewers.vtkViewer import VTKViewer

print(gmshMesh.gmshVersion())
mesh = Gmsh3D("3D_Holder_Test.msh")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

cc = mesh.cellCenters  # shape (3, nCells)
x, y, z = cc[0], cc[1], cc[2]
masks = mesh.physicalCells
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x, y, z, s=2)
plt.show()
names = []
for mat in mesh.physicalFaces:
    names.append(mat)

BoundaryConditions_Gmsh(mesh)


