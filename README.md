# headmodel-conversion

Converts a CHARM head model (a multi-material tetrahedral mesh) into one STEP
file per tissue, as B-rep solids that import cleanly into Ansys Maxwell. The tet
volume is remeshed with MMG3D under a Hausdorff bound, which keeps a tissue's two
walls from crossing and so yields valid, non-overlapping solids even for the
folded tissues.

```sh
git clone --recurse-submodules <repo> && cd headmodel-conversion && uv sync
headmodel-convert path/to/m2m_sub-001 path/to/out
```
