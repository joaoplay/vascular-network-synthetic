"""
Blood flow estimation for vascular networks.

Python implementation based on the FlowEstimateV1 algorithm by Secomb.

Units: flows in nl/min, pressures in mmHg, viscosities in cP,
       shear stress in dyn/cm^2, lengths and diameters in microns.
"""

import numpy as np
import networkx as nx
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import cg as conjugate_gradient
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FlowEstimateParams:
    """
    parameters for the flow estimation algorithm
    """
    target_pressure: float = 31.0          # mmHg
    target_shear_stress: float = 63.5      # dyn/cm^2
    vary_target_shear: bool = True         # in secomb code, its either 0 or 1
    ktau_start: float = 0.002
    ktau_steps: int = 9
    max_inside_iterations: int = 5
    tolerance: float = 1e-6
    max_cg_iterations: int = 200000
    const_viscosity: float = 3.0           #all of these were taken from rheolparams.dat
    plasma_viscosity: float = 1.0466       # 
    const_hematocrit: float = 0.4          #
    vary_viscosity: bool = False           # its 0 or 1 in the .dat file, and its 0 so == false
    mcv: float = 55.0                      # mean cell volume
    known_flow_weight: float = 10.0
    cpar: np.ndarray = field(default_factory=lambda: np.array([0.80, -0.075, -11.0, 12.0]))
    viscpar: np.ndarray = field(default_factory=lambda: np.array([6.0, -0.085, 3.2, -2.44, -0.06, 0.645]))
    optw: float = 1.1
    optlam: float = 0.5


@dataclass
class FlowEstimateResult:
    """
    results of flow estimation
    """
    node_pressures: dict[int, float] = field(default_factory=dict)
    edge_flows: dict[tuple, float] = field(default_factory=dict)
    edge_shear_stress: dict[tuple, float] = field(default_factory=dict)
    edge_velocity: dict[tuple, float] = field(default_factory=dict)
    edge_viscosity: dict[tuple, float] = field(default_factory=dict)


def viscor(diameter: float, hematocrit: float, params: FlowEstimateParams) -> float:
    """
    viscor.cpp from secomb 
    """
    mcvcorr = (92.0 / params.mcv) ** (1.0 / 3.0)
    dcorr = diameter * mcvcorr
    cp = params.cpar
    vp = params.viscpar

    c = ((cp[0] + np.exp(cp[1] * dcorr))
         * (-1.0 + 1.0 / (1.0 + 10.0 ** cp[2] * dcorr ** cp[3]))
         + 1.0 / (1.0 + 10.0 ** cp[2] * dcorr ** cp[3]))

    eta45 = (vp[0] * np.exp(vp[1] * dcorr)
             + vp[2] + vp[3] * np.exp(vp[4] * dcorr ** vp[5]))

    hdref = 0.45
    hdfac = ((1.0 - hematocrit) ** c - 1.0) / ((1.0 - hdref) ** c - 1.0)

    w = params.optw
    etarel = (1.0 + (eta45 - 1.0) * hdfac * (dcorr / (dcorr - w)) ** 2) * (dcorr / (dcorr - w)) ** 2

    return etarel * params.plasma_viscosity


def estimate_flows(graph: nx.Graph,
                   constants: Optional[FlowEstimateParams] = None,
                   boundary_conditions: Optional[dict] = None) -> FlowEstimateResult:
    """
    estimate blood flows, pressures, and shear stresses in a vascular network graph of choosing 

    param graph: NetworkX graph with:
        eode attribute 'node_label': [x, y, z] coordinates
        edge attribute 'avgRadiusAvg': vessel radius
    param constants: FlowEstimateParams() 
    param boundary_conditions: optional dict mapping node_id -> {'type': 'pressure'|'flow', 'value': float}.
        type 'pressure': known boundary pressure (mmHg).
        type 'flow': known boundary inflow (nl/min, positive = inflow).
    return: FlowEstimateResult with node pressures, edge flows, shear stresses, velocities.
    """
    constants = FlowEstimateParams()
    if boundary_conditions is None:
        boundary_conditions = {}

    
    
    #sorted list of nodes and edges
    nodes = sorted(graph.nodes())
    edges = list(graph.edges())
    nnod = len(nodes)
    nseg = len(edges)

    if nseg == 0 or nnod == 0:
        return FlowEstimateResult()

    node_to_idx = {n: i for i, n in enumerate(nodes)}
    idx_to_node = {i: n for i, n in enumerate(nodes)}

    #coordinates of each node
    coords = np.zeros((nnod, 3))
    for n in nodes:
        label = graph.nodes[n].get('node_label', [0, 0, 0])
        coords[node_to_idx[n]] = label[:3]

    #node start and end indices for each vessel and its diameter and hematocrit
    ista = np.zeros(nseg, dtype=int)  #start node index for each segment
    iend = np.zeros(nseg, dtype=int)  #end node index for each segment
    diameters = np.zeros(nseg)
    hd = np.full(nseg, constants.const_hematocrit)

    for i, (u, v) in enumerate(edges):
        ista[i] = node_to_idx[u]
        iend[i] = node_to_idx[v]
        radius = float(graph.edges[u, v].get('avgRadiusAvg', 3.0) or 3.0)
        diameters[i] = 2.0 * radius

    #vessel lengths and the total length of the network
    seg_lengths = np.sqrt(np.sum((coords[iend] - coords[ista]) ** 2, axis=1))
    seg_lengths = np.maximum(seg_lengths, 1e-3)
    total_length = seg_lengths.sum()

    #node type classification and connectivity
    #nodtyp[i] = number of segments connected to node i
    #this part is based on analyzenet.cpp from secomb
    nodtyp = np.zeros(nnod, dtype=int)
    nodseg = [[] for _ in range(nnod)]  #segments connected to each node
    nodnod = [[] for _ in range(nnod)]  #neighbor nodes for each node

    for i in range(nseg):
        i1, i2 = ista[i], iend[i]
        nodtyp[i1] += 1
        nodtyp[i2] += 1
        nodseg[i1].append(i)
        nodseg[i2].append(i)
        nodnod[i1].append(i2)
        nodnod[i2].append(i1)

    #node length weights
    length_weight = np.zeros(nnod)
    for inod in range(nnod):
        for si in nodseg[inod]:
            length_weight[inod] += 0.5 * seg_lengths[si]


    #0 = known pressure boundary
    #1 = internal node
    #2 = known flow boundary
    #3 = unknown boundary 
    knowntyp = np.ones(nnod, dtype=int)
    nodeinflow = np.zeros(nnod)
    n_known_press = 0
    n_unknown = 0

    boundary_node_indices = []
    for inod in range(nnod):
        if nodtyp[inod] == 1:  #boundary node
            boundary_node_indices.append(inod)
            orig_node = idx_to_node[inod]
            bc = boundary_conditions.get(orig_node)
            if bc is not None and bc['type'] == 'pressure':
                knowntyp[inod] = 0
                n_known_press += 1
            elif bc is not None and bc['type'] == 'flow':
                knowntyp[inod] = 2
                nodeinflow[inod] = bc['value']
            else:
                knowntyp[inod] = 3  #unknown boundary
                n_unknown += 1

    matrixdim = 2 * nnod - n_unknown - n_known_press

    #for those with lagrange multipliers
    nodelambda = np.zeros(nnod, dtype=int)
    counter = nnod
    for inod in range(nnod):
        if knowntyp[inod] in (1, 2):  # has conservation constraint
            nodelambda[inod] = counter
            counter += 1


    #compute conductances and shear factors
    pi = np.pi

    #constants from flow.cpp
    facfp = pi * 1333.0 / 128.0 / 0.01 * 60.0 / 1.0e6
    shearconstant = 32.0 / pi * 1.0e4 / 60.0

    cond = np.zeros(nseg)       # conductances
    shearfac = np.zeros(nseg)   # shear stress factors

    for i in range(nseg):
        if constants.vary_viscosity:
            viscosity = viscor(diameters[i], hd[i], constants)
        else:
            viscosity = constants.const_viscosity
        cond[i] = facfp * diameters[i] ** 4 / seg_lengths[i] / viscosity
        shearfac[i] = shearconstant * viscosity / diameters[i] ** 3

    #initialize pressure with target pressure and noise 
    rng = np.random.default_rng(42)
    nodpress = np.full(nnod, constants.target_pressure) + rng.uniform(-5, 5, nnod)

    #known pressures nodes
    for inod in range(nnod):
        if knowntyp[inod] == 0:
            orig_node = idx_to_node[inod]
            nodpress[inod] = boundary_conditions[orig_node]['value']

    #flow direction: initialized to +1
    flow_direction = np.ones(nseg)
    known_flow_dir = np.zeros(nseg, dtype=int)

    kpress = 1.0
    kappa = 1.0  # bias correction factor, updated each iteration
    lambdas = np.zeros(nnod)

    ktau = constants.ktau_start

    for ktau_step in range(constants.ktau_steps):
        for inside_it in range(constants.max_inside_iterations):
            #compute hfactor1 and hfactor2
            sheartarget = np.zeros(nseg)
            hfactor1 = np.zeros(nseg)
            hfactor2 = np.zeros(nseg)

            for si in range(nseg):
                if constants.vary_target_shear:
                    vp = (nodpress[ista[si]] + nodpress[iend[si]]) / 2.0
                    vp = max(vp, 10.0)
                    vt = 100.0 - 86.0 * np.exp(-5000.0 * (np.log10(np.log10(vp))) ** 5.4)
                    sheartarget[si] = flow_direction[si] * vt
                else:
                    sheartarget[si] = flow_direction[si] * constants.target_shear_stress

                if known_flow_dir[si] == 0:
                    hfactor1[si] = ktau * seg_lengths[si] * shearfac[si] * kappa * cond[si] * sheartarget[si]
                    hfactor2[si] = ktau * seg_lengths[si] * shearfac[si] ** 2 * cond[si] ** 2
                else:
                    hfactor1[si] = constants.known_flow_weight * shearfac[si] * kappa * cond[si] * sheartarget[si]
                    hfactor2[si] = constants.known_flow_weight * shearfac[si] ** 2 * cond[si] ** 2

            #Ax = b
            #here i use the second solvetyp
            #this part is from Amatrix.cpp
            A = lil_matrix((matrixdim, matrixdim))
            b = np.zeros(matrixdim)

            #build hmat and kmat per node (diagonal and off-diagonal)
            hmat_diag = np.zeros(nnod)
            kmat_diag = np.zeros(nnod)
            hmat_off = [[0.0] * len(nodseg[inod]) for inod in range(nnod)]
            kmat_off = [[0.0] * len(nodseg[inod]) for inod in range(nnod)]

            scale_factor = max(1.0, 1000.0 * ktau)

            for inod in range(nnod):
                hmat_diag[inod] = kpress * length_weight[inod]
                kmat_diag[inod] = 0.0
                for i, si in enumerate(nodseg[inod]):
                    hmat_diag[inod] += hfactor2[si]
                    hmat_off[inod][i] = -hfactor2[si]
                    kmat_diag[inod] += cond[si] * scale_factor
                    kmat_off[inod][i] = -cond[si] * scale_factor

            #symmetric preconditioner
            precond = np.ones(matrixdim)
            for inod in range(nnod):
                if knowntyp[inod] != 0:
                    precond[inod] = 1.0 / np.sqrt(hmat_diag[inod]) if hmat_diag[inod] > 0 else 1.0
                    if knowntyp[inod] != 3 and kmat_diag[inod] != 0:
                        precond[nodelambda[inod]] = np.sqrt(hmat_diag[inod]) / kmat_diag[inod]

            #build RHS vector b
            for inod in range(nnod):
                if knowntyp[inod] == 0:
                    b[inod] = nodpress[inod]
                else:
                    b[inod] = kpress * length_weight[inod] * constants.target_pressure
                    if knowntyp[inod] != 3:
                        b[nodelambda[inod]] = nodeinflow[inod] * scale_factor
                    for i, si in enumerate(nodseg[inod]):
                        if ista[si] == inod:
                            b[inod] += hfactor1[si]
                        if iend[si] == inod:
                            b[inod] -= hfactor1[si]
                        neighbor = nodnod[inod][i]
                        if knowntyp[neighbor] == 0:
                            b[inod] -= hmat_off[inod][i] * nodpress[neighbor]
                            if knowntyp[inod] != 3:
                                b[nodelambda[inod]] -= kmat_off[inod][i] * nodpress[neighbor]

            #apply preconditioner to b
            b *= precond

            #build matrix A
            for inod in range(nnod):
                if knowntyp[inod] == 0:
                    A[inod, inod] = 1.0
                else:
                    A[inod, inod] = hmat_diag[inod] * precond[inod] ** 2
                    if knowntyp[inod] != 3:
                        lam_idx = nodelambda[inod]
                        val = kmat_diag[inod] * precond[inod] * precond[lam_idx]
                        A[inod, lam_idx] = val
                        A[lam_idx, inod] = val
                    for i, si in enumerate(nodseg[inod]):
                        neighbor = nodnod[inod][i]
                        if knowntyp[neighbor] != 0:
                            val_h = hmat_off[inod][i] * precond[inod] * precond[neighbor]
                            A[inod, neighbor] = val_h
                            if knowntyp[neighbor] != 3:
                                val_k = kmat_off[inod][i] * precond[inod] * precond[nodelambda[neighbor]]
                                A[inod, nodelambda[neighbor]] = val_k
                            if knowntyp[inod] != 3:
                                val_k2 = kmat_off[inod][i] * precond[nodelambda[inod]] * precond[neighbor]
                                A[nodelambda[inod], neighbor] = val_k2

            #solve the linear system using conjugate gradient
            A_csr = A.tocsr()

            #initial guess
            x0 = np.zeros(matrixdim)
            for inod in range(nnod):
                x0[inod] = nodpress[inod]
            for inod in range(nnod):
                if knowntyp[inod] in (1, 2):
                    x0[nodelambda[inod]] = lambdas[inod]

            x, info = conjugate_gradient(A_csr, b, x0=x0, rtol=constants.tolerance,
                                         maxiter=constants.max_cg_iterations)

            #recover nodal pressures and lambdas
            for inod in range(nnod):
                if knowntyp[inod] != 0:
                    nodpress[inod] = x[inod] * precond[inod]
                if knowntyp[inod] in (1, 2):
                    lambdas[inod] = x[nodelambda[inod]] * precond[nodelambda[inod]]

            #compute flows, shear stresses
            q = np.zeros(nseg)
            tau = np.zeros(nseg)
            for i in range(nseg):
                q[i] = (nodpress[ista[i]] - nodpress[iend[i]]) * cond[i]
                tau[i] = (nodpress[ista[i]] - nodpress[iend[i]]) * 1333.0 * diameters[i] / seg_lengths[i] / 4.0

            #update kappa
            kappasum1 = np.sum(seg_lengths * np.abs(tau))
            kappasum2 = np.sum(seg_lengths * tau ** 2)
            mean_tau = kappasum1 / total_length
            mean_tau2 = kappasum2 / total_length
            if mean_tau > 0:
                kappa = mean_tau2 / mean_tau ** 2
            else:
                kappa = 1.0

            #check if flow directions changed; update
            directions_changed = False
            for i in range(nseg):
                new_dir = 1.0 if q[i] >= 0 else -1.0
                if new_dir != flow_direction[i]:
                    flow_direction[i] = new_dir
                    directions_changed = True

            if not directions_changed:
                break

        #double ktau
        ktau *= 2.0

    result = FlowEstimateResult()

    for inod in range(nnod):
        result.node_pressures[idx_to_node[inod]] = float(nodpress[inod])

    for i, (u, v) in enumerate(edges):
        result.edge_flows[(u, v)] = float(q[i])
        result.edge_shear_stress[(u, v)] = float(tau[i])
        #V=Q/A
        radius = diameters[i] / 2.0
        area = pi * radius ** 2  
        #convert flow from nl/min to microns^3/s: 1 nl = 1e6 microns^3
        flow_um3_per_s = q[i] * 1.0e6 / 60.0
        result.edge_velocity[(u, v)] = float(flow_um3_per_s / area) if area > 0 else 0.0
        if constants.vary_viscosity:
            result.edge_viscosity[(u, v)] = float(viscor(diameters[i], hd[i], constants))
        else:
            result.edge_viscosity[(u, v)] = float(constants.const_viscosity)

    return result


def annotate_graph_with_flows(graph: nx.Graph,
                              params: Optional[FlowEstimateParams] = None,
                              boundary_conditions: Optional[dict] = None) -> nx.Graph:
    
    result = estimate_flows(graph, constants=params, boundary_conditions=boundary_conditions)

    nx.set_node_attributes(graph, result.node_pressures, 'pressure')

    for (u, v), flow in result.edge_flows.items():
        graph.edges[u, v]['flow'] = flow
    for (u, v), ss in result.edge_shear_stress.items():
        graph.edges[u, v]['shear_stress'] = ss
    for (u, v), vel in result.edge_velocity.items():
        graph.edges[u, v]['velocity'] = vel
    for (u, v), visc in result.edge_viscosity.items():
        graph.edges[u, v]['viscosity'] = visc

    return graph


def compute_radius_from_flow(flow: float, u: int, v: int, graph: nx.Graph,
                             min_radius: float = 1.0, default_viscosity: float = 3.0) -> float:
    """Compute a vessel radius from Poiseuille's law

    Uses the pressure drop across the edge (from node pressures), the edge
    viscosity, and the edge length (from node coordinates).

    
    The conversion factor 'facfp' from the solver is applied so units are
    consistent: cond = facfp * d^4 / (L * μ), Q = cond * deltaP.

    params:
        flow: Flow through the edge (nl/min).
        u: incoming node id.
        v: outgoing node id.
        graph: NetworkX graph
        min_radius: Minimum allowed radius (microns).
        default_viscosity: Fallback viscosity in cP.

    Returns:
        Estimated radius in microns.
    """
    abs_flow = abs(flow)
    if abs_flow < 1e-10:
        return min_radius

    #pressure drop across the edge
    p_u = graph.nodes[u].get('pressure', 0)
    p_v = graph.nodes[v].get('pressure', 0)
    delta_p = abs(p_u - p_v)
    if delta_p < 1e-10:
        return min_radius

    #edge length from coordinates
    coord_u = np.array(graph.nodes[u].get('node_label', [0, 0, 0])[:3], dtype=float)
    coord_v = np.array(graph.nodes[v].get('node_label', [0, 0, 0])[:3], dtype=float)
    length = float(np.linalg.norm(coord_v - coord_u))
    if length < 1e-3:
        length = 1e-3

    viscosity = float(graph.edges[u, v].get('viscosity', default_viscosity) or default_viscosity)

    # facfp is the unit-conversion factor from the solver:
    # cond = facfp * d^4 / (L * mu),  Q = cond * dP
    # so Q = facfp * (2r)^4 / (L * mu) * dP
    # solving for r: r = ( Q * L * mu / (facfp * 16 * dP) )^(1/4)
    facfp = np.pi * 1333.0 / 128.0 / 0.01 * 60.0 / 1.0e6

    radius = (abs_flow * length * viscosity / (facfp * 16.0 * delta_p)) ** 0.25
    return max(float(radius), min_radius)
