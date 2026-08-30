# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from scipy.special import comb as n_over_k

from mmocr.utils.typing_utils import ArrayLike


def bezier_coefficient(n, t, k):
    return t**k * (1 - t)**(n - k) * n_over_k(n, k)


def bezier_coefficients(time, point_num, ratios):
    return [[bezier_coefficient(time, ratio, num) for num in range(point_num)]
            for ratio in ratios]


def linear_interpolation(point1: np.ndarray,
                         point2: np.ndarray,
                         number: int = 2) -> np.ndarray:
    t = np.linspace(0, 1, number + 2).reshape(-1, 1)
    return point1 + (point2 - point1) * t


def curve2bezier(curve: ArrayLike):
    curve = np.array(curve).reshape(-1, 2)
    if len(curve) == 2:
        return linear_interpolation(curve[0], curve[1])
    diff = curve[1:] - curve[:-1]
    distance = np.linalg.norm(diff, axis=-1)
    norm_distance = distance / distance.sum()
    norm_distance = np.hstack(([0], norm_distance))
    cum_norm_dis = norm_distance.cumsum()
    pseudo_inv = np.linalg.pinv(bezier_coefficients(3, 4, cum_norm_dis))
    control_points = pseudo_inv.dot(curve)
    return control_points


def bezier2curve(bezier: np.ndarray, num_sample: int = 10):
    bezier = np.asarray(bezier)
    t = np.linspace(0, 1, num_sample)
    return np.array(bezier_coefficients(3, 4, t)).dot(bezier)


def reorder_quad_to_reading_axis(poly) -> np.ndarray:
    """Reorder a quad to [TL, TR, BR, BL].

    ICDAR-style quads are given in clockwise order but may start at any
    corner (e.g. bottom-left). ``poly2bezier`` treats the first two points
    as the reading (top) edge, so without this normalization a *short*
    (left/right) edge would be used as the reading axis and the resulting
    ``BezierAlign`` crops would be rotated ~90 degrees.

    The quad's reading axis is assumed to be the longer opposite edge pair
    (text spans along its length). Each of the two reading-parallel edges is
    then oriented left->right so the output is traversed clockwise starting
    from the top-left corner.

    Args:
        poly (ArrayLike): quad of shape (4, 2) in cyclic boundary order.

    Returns:
        np.ndarray: quad of shape (4, 2) as [TL, TR, BR, BL].
    """
    p = np.array(poly).reshape(4, 2).copy()
    edges = [np.linalg.norm(p[(i + 1) % 4] - p[i]) for i in range(4)]
    if edges[1] + edges[3] > edges[0] + edges[2]:
        p = p[[1, 2, 3, 0]]
    top, bottom = p[:2].copy(), p[2:].copy()
    for edge in (top, bottom):
        if edge[1, 0] < edge[0, 0] or (edge[1, 0] == edge[0, 0]
                                       and edge[1, 1] < edge[0, 1]):
            edge[0], edge[1] = edge[1].copy(), edge[0].copy()
    return np.vstack((top, bottom[::-1]))


def poly2bezier(poly):
    poly = np.array(poly).reshape(-1, 2)
    if len(poly) == 5:
        poly = poly[:4]
    elif len(poly) == 4:
        poly = reorder_quad_to_reading_axis(poly)
    points_num = len(poly)
    up_curve = poly[:points_num // 2]
    down_curve = poly[points_num // 2:]
    up_bezier = curve2bezier(up_curve)
    down_bezier = curve2bezier(down_curve)
    up_bezier[0] = up_curve[0]
    up_bezier[-1] = up_curve[-1]
    down_bezier[0] = down_curve[0]
    down_bezier[-1] = down_curve[-1]
    return np.vstack((up_bezier, down_bezier)).flatten().tolist()


def bezier2poly(bezier, num_sample=20):
    bezier = bezier.reshape(2, 4, 2)
    curve_top = bezier2curve(bezier[0], num_sample)
    curve_bottom = bezier2curve(bezier[1], num_sample)
    return np.vstack((curve_top, curve_bottom)).flatten().tolist()
