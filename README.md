# VGE---3D-mesh-smoothing

Models downloaded from https://graphics.stanford.edu/data/3Dscanrep/
Noise added in Blender using Displace Modifier (https://docs.blender.org/manual/en/2.81/modeling/modifiers/deform/displace.html) or using the uncommented code in `main.py`

Implemented methods:
- Simple Laplacian smoothing
- Taubin smoothing (G. Taubin 1995. "Curve and surface smoothing without shrinkage")
- Bilateral denoising (S. Fleishman, I. Drori, and D. Cohen-Or. 2003. "Bilateral mesh denoising")
