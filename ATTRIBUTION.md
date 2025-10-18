# Third-Party Data Attribution

This repository includes derivative / converted meshes from two datasets.  
This file satisfies CC attribution requirements: credit, license link, and change notes.  
_No endorsement_: the original authors/licensors do **not** endorse us or our use.

---

## Dataset A — Hearing Anything Anywhere (DiffRIR)

- **Title:** Hearing Anything Anywhere (DiffRIR dataset)  
- **Authors:** Mason Long Wang, Ryosuke Sawata, Samuel Clarke, Ruohan Gao, Shangzhe Wu, Jiajun Wu  
- **Source:** Project page (links to Zenodo & code) and dataset record  
- **License:** **CC BY 4.0** — https://creativecommons.org/licenses/by/4.0/  

**Our Changes for HAA files:**  
Format conversion only — we converted Python hard-coded room geometry (vertices/faces) into Wavefront `.obj`. No geometry edits beyond serialization.

| File (under `mesh/`)      | Source dataset | License  | Changes summary                            |
|---------------------------|----------------|----------|--------------------------------------------|
| `classroomBase.obj`       | HAA (DiffRIR)  | CC BY 4.0| Converted from Python geometry to `.obj`.  |
| `complexBase.obj`         | HAA (DiffRIR)  | CC BY 4.0| Converted from Python geometry to `.obj`.  |
| `dampenedBase.obj`        | HAA (DiffRIR)  | CC BY 4.0| Converted from Python geometry to `.obj`.  |
| `hallwayBase.obj`         | HAA (DiffRIR)  | CC BY 4.0| Converted from Python geometry to `.obj`.  |

---

## Dataset B — Real Acoustic Fields (RAF)

- **Title:** Real Acoustic Fields: An Audio-Visual Room Acoustics Dataset and Benchmark  
- **Authors:** Ziyang Chen, Israel D. Gebru, Christian Richardt, Anurag Kumar, William Laney, Andrew Owens, Alexander Richard  
- **Source:** Project page & GitHub repository (includes mesh download links)  
- **License:** **CC BY-NC 4.0** — https://creativecommons.org/licenses/by-nc/4.0/  

**Our Changes for RAF files:**  
Mesh editing — we capped several openings with planar patches to close holes (local triangles changed accordingly). We then exported/zipped the `.obj`.

| File (under `mesh/`)        | Source dataset | License     | Changes summary                                   |
|-----------------------------|----------------|-------------|---------------------------------------------------|
| `EmptyRoom.obj.zip`         | RAF            | CC BY-NC 4.0| Capped openings with planar patches; re-exported. |
| `FurnishedRoom.obj.zip`     | RAF            | CC BY-NC 4.0| Capped openings with planar patches; re-exported. |
