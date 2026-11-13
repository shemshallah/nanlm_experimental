# FA1 R7.24 — One curved, mirrored C(256,2/3/4) Diamond

**Status:** experimental architecture on the existing `master` R7 lineage. No accepted model promotion. The inherited accepted learning step is **14,416**. R7.23's prior selected candidate is already known to fail a sealed independent-book test; that test must not be reused for selecting this extension.

## Architecture and combinatorial occupancy

The original 256 R7 point representations `H` and pair tensors `P/M` are read **in place**. The exact R7.23 combinadic triple table and its 256-sector meta-mind are preserved. R7.24 adds a differentiable geometric treatment of edges, triangular faces, and tetrahedral closures **inside** the same upward/downward Diamond; it is not a second prediction model.

| Level | Globally addressable cells | Active cells for 8 contextual vertices |
|---|---:|---:|
| 0-simplex, points | 256 | 8 |
| 1-simplex, unordered pairs with oriented channels | C(256,2) = 32,640 | C(8,2) = 28 |
| 2-simplex, unordered triples with oriented channels | C(256,3) = 2,763,520 | C(8,3) = 56 |
| 3-simplex, tetrahedra | C(256,4) = 174,792,640 | C(8,4) = 70 |

Every active edge belongs to 6 active faces, every face to 5 tetrahedra, and each face has 3 edges. Global tetrahedra use exact *rankable addresses*, but **not** a 174,792,640-entry dense trainable table. The computational graph only instantiates the 70 tetrahedral closures for the 8 active vertices per sample.

## S+ / S- mirror geometry

For distinct vertices `i,j`, `P_ij` is the existing trainable R7 pair code and `M_ij` its mirror code; `s_ij` is +1 when `i<j` and -1 otherwise. Construct a symmetric edge state from `(x_i+x_j)/2` and `P_ij`, and an oriented antisymmetric state from `s_ij * f(s_ij*(x_j-x_i),M_ij)`. The invariant canonical basis is processed only once, then multiplied by orientation. Swapping endpoints is an involution: S+ is unchanged, S- changes sign. The same even/odd construction is applied to triangular triples, using the parity of their vertex permutation.

A learned oriented connection is `theta_ij = s_ij * pi * tanh(f_edge(S+, canonical_S-))`. It reverses sign under edge reversal. Its triangle holonomy is

    K_ijk = theta_ij + theta_jk - theta_ik .

A separate spherical projection of the contextual point vectors gives an *oriented spherical solid angle*:

    Omega_ijk = 2 atan2(det(u_i,u_j,u_k),
                       1 + u_i·u_j + u_j·u_k + u_k·u_i).

The face also has an independently learned alternating curvature 2-form `beta_ijk`, using its existing exact indexed triple code. Trigonometric features use sine/cosine, avoiding discontinuous raw angle wrapping in the forward path.

## Tetrahedral closure and the subtle d²=0 trap

Pure edge holonomy is an *exact exterior derivative*: `K=d(theta)`. Its tetrahedral alternating sum `d(K)=d²(theta)` is **identically zero**. Therefore a naïve "tetrahedron curvature loss" on only edge-derived phase would be an empty tautology, not learnable supervision.

R7.24 instead learns an **independent** triangle 2-form, and measures its tetrahedral coboundary:

    D_ijkl = beta_jkl - beta_ikl + beta_ijl - beta_ijk .

The nontrivial consistency penalty is `mean(1 - cos(D))`. The face deltas and tetrahedral aggregated messages also flow into the LM path; the regularizer by itself is explicitly not proof of better prediction. The spherical-area boundary provides an additional geometric input to the tetra shared message operator.

## Meta-mind specialization and no shortcut

The original 256 apex sectors choose top-4 operators per causal query. R7.24 gives these sectors **distinct learned rank-4 transformations** of the apex state, conditioned on a summary of the 70 active tetrahedra. This makes "specialization" actual trainable computation, not just 256 static sector vectors.

Signal path: original R7 causal near/far modal tokens + byte/word/sentence towers + point geometry -> 28 mirror edges -> 56 oriented curved triangles -> 70 tetra closure feedback -> top-4 of 256 specialized apex processors -> 56 faces -> 28 mirror pairs -> 8 points -> inherited R7 hierarchical decoder. The correction is mathematically blocked when the down-pair ablation is enabled: no independent shortcut writes to the LM output.

The new geometric integration gate starts at **exactly zero**. All inherited R7.23 predictions match bit for bit before new training, while the inherited R7.22 model weights and five Adam optimizers are preserved without resets. The geometry parameters append a ninth group *inside the existing fifth optimizer*, not a sixth optimizer.

## Verified local checks

- Antisymmetric mirror parity of edges, faces, connection holonomy, solid angle, and tetra boundary under an odd permutation.
- Symmetry of mirror-even edge and face states under that permutation.
- Full count/uniqueness of 70 active tetra addresses and 5 tetra incidences per face; 174,792,640 total combinadic addresses.
- `d²(theta)=0` for connection-derived holonomy; independent learned 2-form has a **nonzero, differentiable** tetra closure.
- Gradient reachability through original frozen `P/M`, learned phase, triple curvature, tetra feedback and sparse apex specialists.
- No-pair ablation produces exactly the inherited hidden state, even with nonzero geometric gate.
- Earlier 128-update geometry-only CPU pilot: closure loss 0.3626 -> 0.00625, but no qualifying LM/history gains. **Rejected**.

## Evaluation and GPU decision

All quoted losses are word-target NLL, never byte-accurate BPB. The six previously used R7.23 test books are **spent** and may not be used again for model selection. New code and data are on `master`; no new architecture branch. Evaluate book-disjoint dev examples and causal donor controls, then exactly one fresh sealed external-book test. Only a reproducible, positive book-cluster confidence interval with useful ablations should justify GPU scale. Geometry closure improvements without language prediction gains are insufficient.