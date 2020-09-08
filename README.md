# ConvexMaxOverlap

Compute the optimal translational alignment between two convex polygons.

TODO(yycho0108): Cleanup README.

Based on the following reference:

```
Mark De Berg, Olivier Devillers, Marc Van Kreveld, Otfried Schwarzkopf, Monique Teillaud.
Computing the Maximum Overlap of Two Convex Polygons Under Translations. RR-2832, INRIA. 1996.
inria-00073859
```

Also implements the subroutine for selecting the Nth element in a sorted matrix:

The function generally works, except when it does not. I am still in the process of debugging.

```
@article{Frederickson1984GeneralizedSA,
    title={Generalized Selection and Ranking: Sorted Matrices},
    author={G. Frederickson and D. B. Johnson},
    journal={SIAM J. Comput.},
    year={1984},
    volume={13},
    pages={14-30}
}
```

## NOTE

The subroutine for finding the maximum overlap constrained along a vector has not been exactly replicated from the original paper,

which points to `On the Sectional Area of Convex Polytopes`. Instead, a coarse approximation is employed with a golden section search.

Generally converges in about 10 explicit intersection area evaluations (for a rather simple convex hull).

The golden section search has been copy-pasted from the [wikipedia article](https://en.wikipedia.org/wiki/Golden-section_search).
