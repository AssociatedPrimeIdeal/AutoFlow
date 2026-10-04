"""CPU/Torch masked graph construction and 3D/4D PUMA recovery."""

from __future__ import annotations

import numpy as np

from ._puma import puma


def get_3d_masked_edges(masked_array):
    """
    This function processes a 3D masked array to extract nodes and edges based on the masked values. It creates a mapping
    from each masked coordinate to a unique node ID and then identifies edges between adjacent nodes based on a 6-connectivity
    model (neighbors in the cardinal directions). The function is useful for converting a 3D volume into a graph representation,
    where each masked voxel represents a node and edges represent direct adjacency between these voxels.

    Parameters:
    - masked_array (numpy.ndarray): A 3D boolean array where True values indicate masked points that will be converted to nodes.

    Returns:
    - tuple:
        - numpy.ndarray: Array of values from the masked positions of the input array.
        - numpy.ndarray: Array of edges, where each edge is represented by a pair of node IDs indicating connectivity.
        - dict: A dictionary mapping from 3D coordinates to node IDs.

    Notes:
    - The function assumes the input array is strictly three-dimensional and boolean.
    - Edges are only established between directly adjacent nodes in the 3D space along the axes, not diagonally.
    """
    # Create a dictionary to map coordinates to node IDs
    coord_to_node = {}
    node_id = 0

    nodes = []
    values = []
    # Iterate through the 3D masked array
    for z in range(masked_array.shape[0]):
        for y in range(masked_array.shape[1]):
            for x in range(masked_array.shape[2]):
                if masked_array[z, y, x]:
                    # Add a node for the masked value
                    coord_to_node[(z, y, x)] = node_id
                    # g.add_node(node_id)
                    nodes.append(node_id)
                    values.append(masked_array[z, y, x])
                    node_id += 1

    # Define 3D neighborhood offsets for 6-connectivity
    neighbor_offsets = [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]

    # Iterate through the 3D masked array again to create edges
    edges = []
    for z in range(masked_array.shape[0]):
        for y in range(masked_array.shape[1]):
            for x in range(masked_array.shape[2]):
                if masked_array[z, y, x]:
                    # Get the current node ID
                    current_node_id = coord_to_node[(z, y, x)]

                    # Create edges to adjacent masked values
                    for offset in neighbor_offsets:
                        new_z, new_y, new_x = z + offset[0], y + offset[1], x + offset[2]
                        if (new_z, new_y, new_x) in coord_to_node:
                            adjacent_node_id = coord_to_node[(new_z, new_y, new_x)]

                            edges.append([current_node_id, adjacent_node_id])

    return np.array(values), np.array(edges), coord_to_node


def get_4d_masked_edges(masked_array):
    """
    This function processes a 4D masked array to extract nodes and edges based on the masked values. It creates a mapping
    from each masked coordinate to a unique node ID and then identifies edges between adjacent nodes based on 8-connectivity
    (neighbors in the cardinal directions across spatial and temporal axes). This function is ideal for converting a 4D volume
    (3D spatial plus temporal or another dimension) into a graph representation, where each masked voxel represents a node
    and edges represent direct adjacency between these voxels.

    Parameters:
    - masked_array (numpy.ndarray): A 4D boolean array where True values indicate masked points that will be converted to nodes.

    Returns:
    - tuple:
        - numpy.ndarray: Array of values from the masked positions of the input array.
        - numpy.ndarray: Array of edges, where each edge is represented by a pair of node IDs indicating connectivity.
        - dict: A dictionary mapping from 4D coordinates to node IDs.

    Notes:
    - The function assumes the input array is strictly four-dimensional and boolean.
    - Edges are only established between directly adjacent nodes in the 4D space, not diagonally across any dimension.
    """
    # Create a dictionary to map coordinates to node IDs
    coord_to_node = {}
    node_id = 0

    nodes = []
    values = []
    # Iterate through the 3D masked array
    for z in range(masked_array.shape[0]):
        for y in range(masked_array.shape[1]):
            for x in range(masked_array.shape[2]):
                for t in range(masked_array.shape[3]):
                    if masked_array[z, y, x, t]:
                        # Add a node for the masked value
                        coord_to_node[(z, y, x, t)] = node_id
                        nodes.append(node_id)
                        values.append(masked_array[z, y, x, t])
                        node_id += 1

    # Define 3D neighborhood offsets for 6-connectivity
    neighbor_offsets = [
        (0, 0, 0, 1), (0, 0, 0, -1), (0, 0, 1, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, -1, 0, 0),
        (1, 0, 0, 0), (-1, 0, 0, 0)
    ]

    # Iterate through the 3D masked array again to create edges
    edges = []
    for z in range(masked_array.shape[0]):
        for y in range(masked_array.shape[1]):
            for x in range(masked_array.shape[2]):
                for t in range(masked_array.shape[3]):
                    if masked_array[z, y, x, t]:
                        # Get the current node ID
                        current_node_id = coord_to_node[(z, y, x, t)]

                        # Create edges to adjacent masked values
                        for offset in neighbor_offsets:
                            new_z, new_y, new_x, new_t = z + offset[0], y + offset[1], x + offset[2], t + offset[3]
                            if (new_z, new_y, new_x, new_t) in coord_to_node:
                                adjacent_node_id = coord_to_node[(new_z, new_y, new_x, new_t)]

                                edges.append([current_node_id, adjacent_node_id])

    return np.array(values), np.array(edges), coord_to_node


def gc3D_unwrap(phi_w, mask, start_index=0, period=2 * np.pi, **kwargs):
    """
    Unwraps a 3D phase array ('phi_w') using graph-based methods. This function applies the unwrapping only to the
    regions specified by the mask, leveraging the connectivity and discontinuity properties within those regions.

    Parameters:
    - phi_w (numpy.ndarray): A 3-dimensional array containing wrapped phase values that need to be unwrapped.
    - mask (numpy.ndarray): A binary mask array that indicates the regions within `phi_w` where unwrapping should be applied.
      The unwrapping is performed only where mask is True.
    - start_index (int, optional): The index of the node from which to normalize the resulting unwrapped phases. Default is 0.
    - period (float, optional): The period of the phase wrap-around, typically 2*pi for phase data. Default is 2*pi.
    - **kwargs: Additional keyword arguments passed to the `puma` function for phase unwrapping optimization.

    Returns:
    - numpy.ndarray: The unwrapped phase array, where only the regions specified by the mask have been unwrapped.

    Raises:
    - ValueError: If the input array `phi_w` is not three-dimensional as required.

    Notes:
    - The function first masks the input phase array with the provided mask, then extracts the edges and nodes necessary
      for graph construction via the `get_3d_masked_edges` function.
    - It then uses the `puma` function to optimize the unwrapping, ensuring that the phase continuity is maintained across
      the masked region.
    - The `puma` function returns normalized phase values, which are then scaled back to the original period and adjusted
      to ensure a continuous phase across the entire volume.
    - This function modifies the input array `phi_w` directly by updating the phase values at the locations specified by the mask.
    """
    masked_to_unwrap = np.ma.masked_array(phi_w, 1 - mask)

    if phi_w.ndim != 3:
        raise ValueError("Input array phi_w must have 3 dimensions.")

    x, edges, coord_to_node = get_3d_masked_edges(masked_to_unwrap)

    m = puma(x / period, edges, period, **kwargs)
    m -= m[start_index]

    for index, loc in enumerate(coord_to_node.keys()):
        phi_w[loc] = m[index] * period + phi_w[loc]

    return phi_w


def gc4D_unwrap(phi_w, mask, start_index=0, period=2 * np.pi, **kwargs):
    """
    Unwraps a 4D phase array ('phi_w') using graph-based methods specifically tailored for four-dimensional data. This
    function applies unwrapping only to the regions specified by the mask, leveraging the connectivity and discontinuity
    properties within those regions across both spatial and temporal dimensions.

    Parameters:
    - phi_w (numpy.ndarray): A 4-dimensional array containing wrapped phase values that need to be unwrapped.
    - mask (numpy.ndarray): A binary mask array that indicates the regions within `phi_w` where unwrapping should be applied.
      Only the elements of `phi_w` corresponding to a True value in `mask` are considered for unwrapping.
    - start_index (int, optional): The index of the node from which to normalize the resulting unwrapped phases. Default is 0.
    - period (float, optional): The period of the phase wrap-around, typically 2*pi for phase data. Default is 2*pi.
    - **kwargs: Additional keyword arguments passed to the `puma` function for phase unwrapping optimization.

    Returns:
    - numpy.ndarray: The unwrapped phase array, where only the regions specified by the mask have been unwrapped.

    Raises:
    - ValueError: If the input array `phi_w` is not four-dimensional as required.

    Notes:
    - The function first converts the input phase array to a masked array where non-masked regions are ignored. It then
      extracts the edges and nodes necessary for graph construction via the `get_4d_masked_edges` function.
    - It uses the `puma` function to optimize the unwrapping, ensuring that phase continuity is maintained across the
      masked region in all four dimensions.
    - The `puma` function returns normalized phase values, which are then scaled back to the original period and adjusted
      to ensure a continuous phase across the entire dataset.
    - This function modifies the input array `phi_w` directly by updating the phase values at the locations specified by the mask.
    """
    masked_to_unwrap = np.ma.masked_array(phi_w, 1 - mask)

    if phi_w.ndim != 4:
        raise ValueError("Input array phi_w must have 4 dimensions.")

    x, edges, coord_to_node = get_4d_masked_edges(masked_to_unwrap)

    m = puma(x / period, edges, period, **kwargs)
    m -= m[start_index]

    for index, loc in enumerate(coord_to_node.keys()):
        phi_w[loc] = m[index] * period + phi_w[loc]

    return phi_w


def _gc3d_torch_edges(phi_w: np.ndarray, mask: np.ndarray, device: str):
    """Build the legacy gc3D masked graph with torch, preserving node order.

    The max-flow/PUMA solve remains the original CPU implementation.  Keeping
    that solver unchanged avoids numerical changes while moving the expensive
    coordinate/edge discovery to CUDA.
    """
    import torch

    phase = np.asarray(phi_w)
    include = np.asarray(mask, dtype=bool) & (phase != 0)
    dev = torch.device(device)
    include_t = torch.as_tensor(include, device=dev)
    coords = torch.nonzero(include_t, as_tuple=False)
    n_nodes = int(coords.shape[0])
    if n_nodes == 0:
        return np.asarray([], dtype=phase.dtype), np.empty((0, 2), dtype=np.int64), {}
    node_ids = torch.full(include_t.shape, -1, dtype=torch.int64, device=dev)
    node_ids[tuple(coords.T)] = torch.arange(n_nodes, device=dev, dtype=torch.int64)
    neighbor_ids = []
    for axis, direction in ((2, 1), (2, -1), (1, 1), (1, -1), (0, 1), (0, -1)):
        shifted = torch.full_like(node_ids, -1)
        src = [slice(None)] * 3
        dst = [slice(None)] * 3
        if direction > 0:
            src[axis] = slice(0, -1); dst[axis] = slice(1, None)
        else:
            src[axis] = slice(1, None); dst[axis] = slice(0, -1)
        shifted[tuple(src)] = node_ids[tuple(dst)]
        neighbor_ids.append(shifted[tuple(coords.T)])
    neigh = torch.stack(neighbor_ids, dim=1)
    src_ids = torch.arange(n_nodes, device=dev, dtype=torch.int64).view(-1, 1).expand(-1, 6)
    edge_pairs = torch.stack((src_ids, neigh), dim=-1).reshape(-1, 2)
    edge_pairs = edge_pairs[edge_pairs[:, 1] >= 0]
    coords_cpu = coords.detach().cpu().numpy()
    edges_cpu = edge_pairs.detach().cpu().numpy().astype(np.int64, copy=False)
    coord_to_node = {tuple(int(v) for v in coord): int(i) for i, coord in enumerate(coords_cpu)}
    return np.asarray(phase[include], dtype=phase.dtype), edges_cpu, coord_to_node


def _gc3d_unwrap_torch(phi_w: np.ndarray, mask: np.ndarray, device: str):
    from ._puma import puma

    values, edges, coord_to_node = _gc3d_torch_edges(phi_w, mask, device)
    out = np.array(phi_w, copy=True)
    if values.size == 0 or edges.size == 0:
        return out
    shifts = puma(values / (2.0 * np.pi), edges, 2.0 * np.pi)
    shifts = shifts - shifts[0]
    for index, loc in enumerate(coord_to_node.keys()):
        out[loc] = shifts[index] * (2.0 * np.pi) + out[loc]
    return out
