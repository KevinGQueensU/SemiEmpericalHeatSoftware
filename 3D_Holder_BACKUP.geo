// Gmsh project created on Sun Feb 22 21:20:14 2026
SetFactory("OpenCASCADE");
Mesh.MeshSizeFactor = 0.1;
OptimizeMesh = "Netgen";
RefineMesh;
//+
Box(1) = {0, -1, -1, 0.001, 2, 2};
//+
Box(2) = {0.001, -1, -1, 0.01, 2, 2};
//+
Box(3) = {-0.4, 0.6, -1, 2/5, 2/5, 2};
//+
Box(4) = {-0.4, -1, -1, 2/5, 2/5, 2};
//+
BooleanFragments{ Volume{3}; Volume{1}; Volume{2}; Volume{4}; Delete; }{ }
//+
Physical Volume("Tantalum", 55) = {4, 3};
//+
Physical Volume("Olivine", 56) = {1};
//+
Physical Volume("Aluminum", 57) = {2};
//+
Physical Surface("Fixed", 54) = {6, 23, 5, 22, 14};
//+
Physical Surface("BBR", 52) = {8, 19, 21, 20, 4, 11, 16, 1, 3, 13, 18, 10, 15, 9, 17};
