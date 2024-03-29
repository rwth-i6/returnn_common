"""
Image utils, like warping, interpolation
"""

from typing import Tuple
from ... import nn


def dense_image_warp(
    image: nn.Tensor,
    *,
    spatial_dims: Tuple[nn.Dim, nn.Dim],
    flow: Tuple[nn.Tensor, nn.Tensor],
) -> nn.Tensor:
    """
    Image warping using per-pixel flow vectors.

    Adapted from RETURNN tf_util.dense_image_warpm, which is
    adapted from tensorflow.contrib.image.dense_image_warp,
    from newer TF version which supports variable-sized images.

    :param image: float `Tensor` with shape `[batch..., spatial_dims[0], spatial_dims[1], channels...]`.
    :param spatial_dims: spatial dims of image, e.g. height, width
    :param flow: Each a float `Tensor` with shape `[batch..., spatial_dims[0], spatial_dims[1]]`.
      E.g. via :func:`create_random_warp_flow_2d`.
      How much to warp the x/y coordinates. 0.0 means no warp.
    :returns: A float `Tensor` with shape`[batch..., spatial_dims[0], spatial_dims[1], channels...]`
      and same type as input image.
    """
    assert len(spatial_dims) == len(flow) == 2
    # The flow is defined on the image grid. Turn the flow into a list of query points in the grid space.
    query_points = tuple([nn.range_over_dim(spatial_dims[i], dtype=flow[i].dtype) - flow[i] for i in [0, 1]])
    # Compute values at the query points, then reshape the result back to the image grid.
    interpolated = interpolate_bilinear(image, spatial_dims=spatial_dims, query_points=query_points)
    return interpolated


def interpolate_bilinear(
    image: nn.Tensor,
    *,
    spatial_dims: Tuple[nn.Dim, nn.Dim],
    query_points: Tuple[nn.Tensor, nn.Tensor],
) -> nn.Tensor:
    """
    Find values for query points on a grid using bilinear interpolation.
    Uses the 4 pixels around the query_points.

    Similar to Matlab's interp2 function.
    Adapted from RETURNN tf_util.interpolate_bilinear, which is
    adapted from tensorflow.contrib.image.dense_image_warp,
    from newer TF version which supports variable-sized images.

    :param image: a float `Tensor` of shape `[batch..., spatial_dims[0], spatial_dims[1], channels...]`.
    :param spatial_dims: spatial dims of image, e.g. height, width
    :param query_points: each is float `Tensor` of N points with shape `[batch..., output_dims...]`.
      Indices into the image.
    :returns: a `Tensor` with shape `[batch..., output_dims..., channels...]`
    """
    assert len(query_points) == len(spatial_dims) == 2
    query_dtype = query_points[0].dtype
    img_dtype = image.dtype

    assert all([q.dtype == query_dtype for q in query_points])
    assert all([q.dims_set == query_points[0].dims_set for q in query_points])  # not really necessary but reasonable

    alphas = []
    floors = []
    ceilings = []

    for dim in [0, 1]:
        queries = query_points[dim]

        size_in_indexing_dimension = nn.dim_value(spatial_dims[dim])

        # Note: In the original code, we used max_floor = size_in_indexing_dimension - 2,
        # so that max_floor + 1 is still a valid index into the grid.
        # We changed that, and clip the ceiling again below.
        # This should only really happen when the index is out of bounds anyway.
        max_floor = nn.cast(size_in_indexing_dimension - 1, dtype=query_dtype)
        min_floor = nn.constant(0.0, dtype=query_dtype)
        floor = nn.clip_by_value(nn.floor(queries), min_floor, max_floor)
        int_floor = nn.cast(floor, dtype="int32")
        floors.append(int_floor)
        ceiling = int_floor + 1
        ceiling = nn.clip_by_value(ceiling, 0, size_in_indexing_dimension - 1)
        ceilings.append(ceiling)

        # alpha has the same type as the grid, as we will directly use alpha
        # when taking linear combinations of pixel values from the image.
        alpha = nn.cast(queries - floor, dtype=img_dtype)
        min_alpha = nn.constant(0.0, dtype=img_dtype)
        max_alpha = nn.constant(1.0, dtype=img_dtype)
        alpha = nn.clip_by_value(alpha, min_alpha, max_alpha)
        alphas.append(alpha)
    assert len(alphas) == len(floors) == len(ceilings) == 2

    flattened_img, flat_dim = nn.merge_dims(image, axes=spatial_dims)
    width = nn.dim_value(spatial_dims[1])

    # This wraps gather using the flattened image and flattened indices.
    def _gather(y_coords: nn.Tensor, x_coords: nn.Tensor) -> nn.Tensor:
        linear_coordinates = y_coords * width + x_coords
        gathered_values = nn.gather(flattened_img, position=linear_coordinates, axis=flat_dim)
        return gathered_values

    # grab the pixel values in the 4 corners around each query point
    top_left = _gather(floors[0], floors[1])
    top_right = _gather(floors[0], ceilings[1])
    bottom_left = _gather(ceilings[0], floors[1])
    bottom_right = _gather(ceilings[0], ceilings[1])

    # now, do the actual interpolation
    interp_top = alphas[1] * (top_right - top_left) + top_left
    interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
    interp = alphas[0] * (interp_bottom - interp_top) + interp_top

    return interp
