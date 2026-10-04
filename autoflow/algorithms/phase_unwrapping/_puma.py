"""PUMA graph-cut solver with optional PyMaxflow dependency.

Derived from the MIT-licensed PUDIP-Flow TradMethod sources.
"""

import numpy as np
try:
    import maxflow
except ImportError:
    print("PyMaxflow not found, graph-cut unwrapping will fail")


def puma(psi: np.ndarray, edges: np.ndarray, period, max_jump: int = 1, p: float = 1, **kwargs):
    """
    Based on the git repository: "https://github.com/yoyololicon/kamui/tree/dev".
    The `puma` function implements a phase unwrapping method that utilizes graph cuts to minimize a potential
    function over a given network structure defined by nodes (psi values) and edges. It iteratively adjusts the
    unwrapping by minimizing the potential energy across edges, which can be thought of as 'cuts' in the graph.

    Parameters:
        psi (numpy.ndarray): A 1-dimensional array containing the phase values at each node of the graph.
        edges (numpy.ndarray): A 2-dimensional array where each row represents an edge connecting two nodes, with the nodes
            indexed according to their positions in `psi`.
        period (unused in this snippet, typically used to define the cycle length of phase values).
        max_jump (int): The maximum jump magnitude allowed in a single graph cut iteration. Default is 1.
        p (float): The p-norm used in the potential energy calculation. Default is 1.
        **kwargs: Additional keyword arguments for adjusting the potential function V, such as 'potential_mode', 'delta', and 'lam'.

    Returns:
        numpy.ndarray: An array the same size as `psi` containing the adjusted phase values after the graph cut optimization.

    Notes:
        - This function requires the `maxflow` library to create and manipulate the graph.
        - The potential function V can be configured via `kwargs` to use different modes of potential ('truncated', 'smooth', 'tanh').
        - The function returns the phase values adjusted for discontinuities across the graph defined by `edges`.
    """

    if max_jump > 1:
        # jump_steps = list(range(1, max_jump + 1)) * 5
        jump_steps = list(range(1, max_jump + 1))
    else:
        jump_steps = [max_jump]

    total_nodes = psi.size

    def V(x, potential_mode='truncated', delta=1 / np.sqrt(2), lam=0.5, **kwargs):
        """
        Computes the potential value for a given difference 'x' between nodes, based on a specified mode. This function is
        used within graph-based optimization to evaluate the cost or potential associated with a particular phase difference.

        Parameters:
        - x (numpy.ndarray): Array of differences for which the potential is calculated.
        - potential_mode (str): The mode of potential calculation. It can be 'truncated', 'smooth', or 'tanh'.
        - delta (float): A parameter that influences the scaling of differences in the potential calculation. Default is 1/sqrt(2).
        - lam (float): A parameter that sets the maximum potential value in some modes or scales the potential in others. Default is 0.5.

        Returns:
        - numpy.ndarray: The computed potential values for each element in 'x'.

        Raises:
        - Exception: If an unknown 'potential_mode' is specified.

        Notes:
        - 'truncated': Potential is quadratic up to a limit 'lam', beyond which it is capped.
        - 'smooth': Potential is a smooth quadratic function that becomes less steep as 'x' increases.
        - 'tanh': Potential uses a hyperbolic tangent function to provide a smooth transition between values, capped by 'lam'.
        """
        if potential_mode == 'truncated':
            potential = x ** 2 / (2 * delta ** 2)
            potential[potential > lam] = lam

        elif potential_mode == 'smooth':
            potential = lam * x ** 2 / (2 * delta ** 2 + x ** 2)

        elif potential_mode == 'tanh':
            potential = lam * np.tanh(x ** 2 / (2 * delta ** 2))

        else:
            raise Exception(f'Potential mode {potential_mode} not implemented...')

        return potential

    K = np.zeros_like(psi)

    def cal_Ek(K, psi, i, j):
        """
        Computes the total energy of the graph for a given configuration of node potentials ('K') and phase values ('psi')
        across specified edges. This function sums up the potential values for all the edges defined by indices arrays 'i' and 'j'.

        Parameters:
        - K (numpy.ndarray): An array containing the potential values at each node. This represents the current state of the graph.
        - psi (numpy.ndarray): An array containing the original phase values at each node.
        - i (numpy.ndarray): An array of starting indices for the edges, referencing positions in 'K' and 'psi'.
        - j (numpy.ndarray): An array of ending indices for the edges, referencing positions in 'K' and 'psi'.
        - **kwargs: Additional keyword arguments that may be passed to the potential function 'V'.

        Returns:
        - float: The total energy calculated as the sum of potentials across all defined edges. The potential for each edge is
        determined by the difference in the adjusted potentials ('K') and original phase differences ('psi') between nodes.

        Notes:
        - This function leverages the potential function 'V', which calculates the potential based on the difference in the
        total phase shift (adjusted potential difference plus original phase difference) across an edge.
        - The energy computation is central to the optimization in graph-based unwrapping, guiding the iterative adjustment of
        the node potentials to minimize the overall system energy.
        """
        return np.sum(V(K[j] - K[i] - psi[i] + psi[j], **kwargs))

    prev_Ek = cal_Ek(K, psi, edges[:, 0], edges[:, 1])

    energy_list = []

    for step in jump_steps:
        while 1:
            energy_list.append(prev_Ek)
            G = maxflow.Graph[float]()
            G.add_nodes(total_nodes)

            i, j = edges[:, 0], edges[:, 1]
            psi_diff = psi[i] - psi[j]
            a = (K[j] - K[i]) - psi_diff
            e00 = e11 = V(a)
            e01 = V(a - step, **kwargs)
            e10 = V(a + step, **kwargs)
            weight = np.maximum(0, e10 + e01 - e00 - e11)

            G.add_edges(edges[:, 0], edges[:, 1], weight, np.zeros_like(weight))

            tmp_st_weight = np.zeros((2, total_nodes))

            for i in range(edges.shape[0]):
                u, v = edges[i]
                tmp_st_weight[0, u] += max(0, e10[i] - e00[i])
                tmp_st_weight[0, v] += max(0, e11[i] - e10[i])
                tmp_st_weight[1, u] -= min(0, e10[i] - e00[i])
                tmp_st_weight[1, v] -= min(0, e11[i] - e10[i])

            for i in range(total_nodes):
                G.add_tedge(i, tmp_st_weight[0, i], tmp_st_weight[1, i])

            G.maxflow()

            partition = G.get_grid_segments(np.arange(total_nodes))
            K[~partition] += step

            energy = cal_Ek(K, psi, edges[:, 0], edges[:, 1])

            if energy < prev_Ek:
                prev_Ek = energy
            else:
                K[~partition] -= step
                break

    return K
