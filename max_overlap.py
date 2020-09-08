#!/usr/bin/env python3

import sys

import numpy as np
from scipy.spatial import ConvexHull
from matplotlib import pyplot as plt

from gss import gss
from shapely.geometry import Polygon


def anorm(x):
    return (x+np.pi) % (2*np.pi)-np.pi


def generate_data():
    pa = np.random.normal(scale=5, size=(1, 2)) + \
        np.random.normal(size=(512, 2))
    ha = ConvexHull(pa)
    ha = pa[ha.vertices]

    #plt.plot(pa[...,0], pa[...,1], 'gx')
    #plt.plot(pa2[...,0], pa2[...,1], 'r+')
    # plt.show()
    # sys.exit(0)
    # pb = np.random.normal(scale=5, size=(1, 2)) + \
    #    np.random.normal(size=(128, 2))
    pb = np.random.normal(scale=5, size=(1, 2)) + pa[:32]
    hb = ConvexHull(pb)
    hb = pb[hb.vertices]
    return ha, hb


def compute_kth_naive(A, B, k):
    M = A[:, None] - B[None, :]
    return np.partition(M.ravel(), k)[k]


def split_cell(cell):
    i0, j0, i1, j1 = cell

    di, dj = i1-i0, j1-j0
    ri, rj = di//2, dj//2

    im = i0+ri
    jm = j0+rj

    # Compute block sizes.
    # b00 = im * jm
    # b01 = im * (j1 - jm)
    # b10 = (i1-im) * jm
    # b11 = (i1-im) * (j1-jm)

    return [
        (i0, j0, im, jm),
        (i0, jm, im, j1),
        (im, j0, i1, jm),
        (im, jm, i1, j1)
    ]


def mark_active_cells(n, m, cells, ri, rj):
    M = np.zeros(shape=(n, m), dtype=np.uint8)
    for i0, j0 in cells:
        M[i0:i0+ri, j0:j0+rj] = 1
    return M


def compute_kth(A, B, k):
    # Assume implicit matrix:
    # M = A[:,None] - B[None,:]
    n = len(A)
    m = len(B)
    if n < m:
        print('invert')
        # Require that n >= m.
        # However, since the problem now becomes B-A,
        # invert both the index order and the sign.
        return -compute_kth(B, A, n*m-k-1)
    S = n*m

    # num_iter == how many quad partitions are required.
    num_iter = int(np.ceil(np.log2(S)/2))
    cells = [[0, 0]]  # only store beginning index.
    cells = np.asarray(cells)

    # ri, rj tracks current cell sizes.
    ri, rj = n, m

    for p in range(1, num_iter+1):
        # Compute params
        Bp = int(min(m, 2**(p+1)-1))
        u = 4 ** p

        # Split cells into quadrants.
        if ri > 1 and rj > 1:
            # Thick cells.
            ri //= 2
            rj //= 2
            cells = np.concatenate([
                cells,
                cells + [[0, rj]],
                cells + [[ri, 0]],
                cells + [[ri, rj]]
            ])
        else:
            # Thin cells.
            if ri > 1:
                ri //= 4
                cells = np.concatenate([
                    cells,
                    cells + [[ri, 0]],
                    cells + [[2*ri, 0]],
                    cells + [[3*ri, 0]]
                ])
            elif rj > 1:
                rj //= 4
                cells = np.concatenate([
                    cells,
                    cells + [[0, rj]],
                    cells + [[0, 2*rj]],
                    cells + [[0, 3*rj]]
                ])

        # Process upper bound.
        q = int(np.ceil(k*u/S)) + Bp
        if q < len(cells):
            # ref = `index` of min element.
            # == M(i0, j0) == A(i0) - B(j0)
            ref = A[cells[..., 0]] - B[cells[..., 1]]
            idx = np.argpartition(ref, q)
            print('upper bound = {}'.format(ref[idx[q]]))
            cells = cells[idx[:q]]

        # Process lower bound ...
        r = int(np.floor(k*u/S)) - Bp
        if r >= 0:
            ref = A[cells[..., 0]+ri-1] - B[cells[..., 1]+rj-1]
            idx = np.argpartition(ref, r)
            print('lower bound = {}'.format(ref[idx[r]]))
            cells = cells[idx[r:]]
            k -= r * (S // u)

        # Optionally visualize the currently active cells.
        # M = mark_active_cells(n, m, cells, ri, rj)
        # print(M)

    values = A[cells[..., 0]] - B[cells[..., 1]]
    # print('v', values)
    print(values.shape, k)
    value = np.partition(values, k)[k]
    return value


def compute_max_t(pa, pb_, y):
    """
    Not the best possible implementation ...
    """
    # NOTE(yycho0108): offset pb by input parameter y.
    # TODO(yycho0108): Consider avoiding copy here.
    pb = pb_ + [[0, y]]

    # NOTE(yycho0108): It is possible to further bound this search area
    # by limiting ax... to the intersection within the strip.
    bound_search = False
    if bound_search:
        ay = np.sort(pa[..., 1])
        by = np.sort(pb[..., 1])

        y0, y1 = np.sort([ay[0], ay[-1], by[0], by[-1]])[1:3]
        print(y0, y1, ay[0], ay[-1])
        ayi0 = np.clip(np.searchsorted(ay, y0)-1, 0, len(ay)-1)
        ayi1 = np.clip(np.searchsorted(ay, y1)+1, ayi0+1, len(ay))
        byi0 = np.clip(np.searchsorted(by, y0)-1, 0, len(by)-1)
        byi1 = np.clip(np.searchsorted(by, y1)+1, byi0+1, len(by))
        print('{}-{}/{}, {}-{}/{}'.format(ayi0,
                                          ayi1, len(ay), byi0, byi1, len(by)))
        ax0 = pa[ayi0:ayi1, 0].min()
        ax1 = pa[ayi0:ayi1, 0].max()
        bx0 = pb[byi0:byi1, 0].min()
        bx1 = pb[byi0:byi1, 0].max()
    else:
        ax0 = pa[..., 0].min()
        ax1 = pa[..., 0].max()
        bx0 = pb[..., 0].min()
        bx1 = pb[..., 0].max()

    dx0 = ax0-bx1  # aligns bx1 with ax0
    dx1 = ax1-bx0  # aligns bx0 with ax1

    poly_pa = Polygon(pa)

    # global count
    # count = 0
    def compute_overlap_area(dx):
        # global count
        # count += 1
        # print(count)
        poly_pb = Polygon(pb+[[dx, 0]])
        return -poly_pa.intersection(poly_pb).area

    xs = np.linspace(dx0, dx1)
    # ars = [compute_overlap_area(x) for x in xs]
    # plt.plot(xs, ars)
    # plt.show()
    # sys.exit(0)

    b0, b1 = gss(compute_overlap_area,
                 dx0, dx1, tol=1e-1)
    dx = (b0+b1)/2
    return -compute_overlap_area(dx), dx


def pad_to_2(x: np.ndarray):
    n = x.shape[0]
    n2 = int(np.exp2(np.ceil(np.log2(n))))
    # return np.pad(x, (0, n2-n), mode='constant', constant_values=np.inf)
    return np.pad(x, (0, n2-n), mode='edge')


def locate_hstrip(pa: np.ndarray, pb: np.ndarray):
    n = pa.shape[0]
    m = pb.shape[0]
    ya = pa[..., 1]
    yb = pb[..., 1]

    A = pad_to_2(np.sort(ya))
    B = pad_to_2(np.sort(yb)[::-1])

    k_min = 0
    k_max = n * m
    k = (k_min + k_max) // 2

    # Test compute_kth ...
    if False:
        for _ in range(32):
            A = np.sort(np.random.normal(size=10))
            B = np.sort(np.random.normal(size=8))
            k = np.random.randint(A.shape[0]*B.shape[0])
            A = pad_to_2(A)
            B = pad_to_2(B[::-1])
            # print(A.shape)
            # print(B.shape)
            out1 = compute_kth_naive(A, B, k)
            print(out1)
            out2 = compute_kth(A, B, k)
            print(out1-out2)
        sys.exit(0)

    # Proceed with binary search ...
    y = None
    while True:
        # Determine current bisection anchor.
        k = ((k_min + k_max) // 2)

        # 1) Compute kth lagest entry of M.
        # FIXME(yycho0108): compute_kth doesn't work sometimes for some reason.
        #y_k0 = compute_kth(A, B, k)
        #y_k1 = compute_kth(A, B, k + 1)
        y_k0 = compute_kth_naive(A, B, k)
        y_k1 = compute_kth_naive(A, B, k + 1)

        # Lemma 3.3
        # print('k0,k1', k_min, k_max)
        max_t_wy_k0_t = compute_max_t(pa, pb, y_k0)[0]
        max_t_wy_k1_t = compute_max_t(pa, pb, y_k1)[0]
        # print(max_t_wy_k0_t)
        # print(max_t_wy_k1_t)

        # Update bounds.
        if max_t_wy_k0_t <= max_t_wy_k1_t:
            k_min = k
            y = y_k1
        if max_t_wy_k0_t >= max_t_wy_k1_t:
            k_max = k
            y = y_k0

        # Check convergence.
        if k_max <= k_min + 1:
            break

    # This is technically the second stage of the algorithm,
    # but it's also possible to return the coarse result here.
    dx = compute_max_t(pa, pb, y)[1]
    return dx, y


def main():
    show_hull = True
    # seed = 7
    seed = np.random.randint(0, 65536)
    # seed = 22421
    print('seed = {}'.format(seed))

    np.random.seed(seed)
    pa, pb = generate_data()
    print('optimize...')
    dx, dy = locate_hstrip(pa, pb)

    if show_hull:
        plt.plot(pa[..., 0], pa[..., 1], 'rx-', label='pa')
        plt.plot(pb[..., 0], pb[..., 1], 'b+-', label='pb')
        plt.plot(pb[..., 0]+dx, pb[..., 1]+dy, 'g.--', label='pb+delta')
        plt.axis('equal')
        plt.grid()
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
