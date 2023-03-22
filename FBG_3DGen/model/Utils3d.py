import torch 
import numpy as np 
import warnings


def outer(x, y):
    """ outer product between input tensors """
    return x[..., None] @ y[..., None, :]


def skew(x):
    """
        returns skew symmetric 3x3 form of a 3 dim vector
    """
    assert len(x.shape) > 1, "`x` requires at least 2 dimensions"
    zero = torch.zeros(*x.shape[:-1]).to(x)
    a, b, c = x[..., 0], x[..., 1], x[..., 2]
    s = torch.stack(
        [
            torch.stack([zero, c, -b], dim=-1),
            torch.stack([-c, zero, a], dim=-1),
            torch.stack([b, -a, zero], dim=-1),
        ],
        dim=-1,
    )
    return s

def det2x2(a):
    """ batch determinant of a 2x2 matrix """
    return a[..., 0, 0] * a[..., 1, 1] - a[..., 1, 0] * a[..., 0, 1]

def det3x3(a):
    """ batch determinant of a 3x3 matrix """
    return (torch.cross(a[..., 0, :], a[..., 1, :], dim=-1) * a[..., 2, :]).sum(dim=-1)

def tripod(p1, p2, p3, eps=1e-7, raise_warnings=True, enforce_boundaries=True):
    """ computes a unique orthogonal basis for input points """
    e1 = p2 - p1
    e1_norm = torch.norm(e1, dim=-1, keepdim=True)

    if raise_warnings:
        if torch.any(e1_norm < eps):
            warnings.warn("singular division in computing orthogonal basis")
    if enforce_boundaries:
        e1_norm = e1_norm.clamp_min(eps)

    e1 = e1 / e1_norm
    u = p3 - p1
    e2 = torch.cross(u, e1)
    e2_norm = torch.norm(e2, dim=-1, keepdim=True)

    if raise_warnings:
        if torch.any(e2_norm < eps):
            warnings.warn("singular division in computing orthogonal basis")
    if enforce_boundaries:
        e2_norm = e2_norm.clamp_min(eps)

    e2 = e2 / e2_norm
    e3 = torch.cross(e2, e1)
    return -e3, -e2, e1

def orientation(p1, p2, p3, eps=1e-7, raise_warnings=True, enforce_boundaries=True):
    """ computes unique orthogonal basis transform for input points """
    return torch.stack(tripod(p1, p2, p3, eps, raise_warnings, enforce_boundaries), dim=-1)

def dist_deriv(x1, x2, eps=1e-7, enforce_boundaries=True, raise_warnings=True):
    """
        computes distance between input points together with
        the Jacobian wrt to `x1`
    """
    r = x2 - x1
    rnorm = torch.norm(r, dim=-1, keepdim=True)

    if raise_warnings:
        if torch.any(rnorm < eps):
            warnings.warn("singular division in distance computation")
    if enforce_boundaries:
        rnorm = rnorm.clamp_min(eps)

    dist = rnorm[..., 0]
    J = -r / rnorm
    # J = _safe_div(-r, rnorm)
    return dist, J

def angle_deriv(x1, x2, x3, eps=1e-7, enforce_boundaries=True, raise_warnings=True):
    """
        computes angle between input points together with
        the Jacobian wrt to `x1`
    """
    r12 = x1 - x2
    r12_norm = torch.norm(r12, dim=-1, keepdim=True)

    if raise_warnings:
        if torch.any(r12_norm < eps):
            warnings.warn("singular division in angle computation")
    if enforce_boundaries:
        r12_norm = r12_norm.clamp_min(eps)

    rn12 = r12 / r12_norm

    J = (torch.eye(3).to(x1) - outer(rn12, rn12)) / r12_norm[..., None]

    r32 = x3 - x2
    r32_norm = torch.norm(r32, dim=-1, keepdim=True)

    if raise_warnings:
        if torch.any(r32_norm < eps):
            warnings.warn("singular division in angle computation")
    if enforce_boundaries:
        r32_norm = r32_norm.clamp_min(eps)

    rn32 = r32 / r32_norm

    cos_angle = torch.sum(rn12 * rn32, dim=-1)
    J = rn32[..., None, :] @ J

    if raise_warnings:
        if torch.any((cos_angle < -1. + eps) & (cos_angle > 1. - eps)):
            warnings.warn("singular radians in angle computation")
    if enforce_boundaries:
        cos_angle = cos_angle.clamp(-1. + eps, 1. - eps)

    a = torch.acos(cos_angle)

    J = -J / torch.sqrt(1.0 - cos_angle.pow(2)[..., None, None])

    return a, J[..., 0, :]

def torsion_deriv(x1, x2, x3, x4, eps=1e-7, enforce_boundaries=True, raise_warnings=True):
    """
        computes torsion angle between input points together with
        the Jacobian wrt to `x1`.
    """
    b0 = -1.0 * (x2 - x1)

    # TODO not used can be removed in next refactor
    # db0_dx1 = torch.eye(3).to(x1)

    b1 = x3 - x2
    b2 = x4 - x3
    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1norm = torch.norm(b1, dim=-1, keepdim=True)

    if raise_warnings:
        if torch.any(b1norm < eps):
            warnings.warn("singular division in distance computation")
    if enforce_boundaries:
        b1norm = b1norm.clamp_min(eps)

    b1_normalized = b1 / b1norm

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    #
    # dv_db0 = jacobian of v wrt b0

    v = b0 - torch.sum(b0 * b1_normalized, dim=-1, keepdim=True) * b1_normalized
    dv_db0 = torch.eye(3)[None, None, :, :].to(x1) - outer(b1_normalized, b1_normalized)

    w = b2 - torch.sum(b2 * b1_normalized, dim=-1, keepdim=True) * b1_normalized

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    #
    # dx_dv = jacobian of x wrt v
    x = torch.sum(v * w, dim=-1, keepdim=True)
    dx_dv = w[..., None, :]

    # b1xv = fast cross product between b1_normalized and v
    # given by multiplying v with the skew of b1_normalized
    # (see https://en.wikipedia.org/wiki/Skew-symmetric_matrix#Cross_product)
    #
    # db1xv_dv = Jacobian of b1xv wrt v
    A = skew(b1_normalized)
    b1xv = (A @ (v[..., None]))[..., 0]
    db1xv_dv = A

    # y = dot product of b1xv and w
    # dy_db1xv = Jacobian of v wrt b1xv
    y = torch.sum(b1xv * w, dim=-1, keepdim=True)
    dy_db1xv = w[..., None, :]

    x = x[..., None]
    y = y[..., None]

    # a = torsion angle spanned by unit vector (x, y)
    # xysq = squared norm of (x, y)
    # da_dx = Jacobian of a wrt xysq
    a = torch.atan2(y, x)
    xysq = x.pow(2) + y.pow(2)

    if raise_warnings:
        if torch.any(xysq < eps):
            warnings.warn("singular division in torsion computation")
    if enforce_boundaries:
        xysq = xysq.clamp_min(eps)

    da_dx = -y / xysq
    da_dy = x / xysq

    # compute derivative with chain rule
    J = da_dx @ dx_dv @ dv_db0 + da_dy @ dy_db1xv @ db1xv_dv @ dv_db0
    return a[..., 0, 0], J[..., 0, :]

def decompose_z_matrix(z_matrix, fixed):
    """
    Decompose the z-matrix into blocks to allow parallel (batched) reconstruction
    of cartesian coordinates starting from the fixed atoms.
    Parameters
    ----------
    z_matrix : np.ndarray
        Z-matrix definition for the internal coordinate transform.
        Each row in the z-matrix defines a (proper or improper) torsion by specifying the atom indices
        forming this torsion. Atom indices are integers >= 0.
        The shape of the z-matrix is (n_conditioned_atoms, 4).

    fixed : np.ndarray
        Fixed atoms that are used to seed the reconstruction of Cartesian from internal coordinates.
    Returns
    -------
    blocks : list of np.ndarray
        Z-matrix for each stage of the reconstruction. The shape for each block is
        (n_conditioned_atoms_in_block, 4).
    index2atom : np.ndarray
        index2atom[i] specifies the atom index of the atom that is placed by the i-th row in the original Z-matrix.
        The shape is (n_conditioned_atoms, ).
    atom2index : np.ndarray
        atom2index[i] specifies the row in the original z-matrix that is responsible for placing the i-th atom.
        The shape is (n_conditioned_atoms, ).
    index2order : np.ndarray
        order in which the reconstruction is applied, where i denotes a row in the Z-matrix.
        The shape is (n_conditioned_atoms, ).
    """
    atoms = [fixed]
    blocks = []  # blocks of Z-matrices. Each block corresponds to one stage of Cartesian reconstruction.
    given = np.sort(fixed)  # atoms that were already visited
    # filter out conditioned variables

    non_given = ~np.isin(z_matrix[:, 0], given)
    z_matrix = z_matrix[non_given]

    # prepend the torsion index to each torsion in the z matrix

    z_matrix = np.concatenate([np.arange(len(z_matrix))[:, None], z_matrix], axis=1)
    order = []  # torsion indices
    while len(z_matrix) > 0:
        can_be_placed_in_this_stage = np.all(np.isin(z_matrix[:, 2:], given), axis=-1)
        # torsions, where atoms 2-4 were already visited
        if (not np.any(can_be_placed_in_this_stage)) and len(z_matrix) > 0:
            raise ValueError(
                f"Z-matrix decomposition failed. "
                f"The following atoms were not reachable from the fixed atoms: \n{z_matrix[:,1]}"
            )
        pos = z_matrix[can_be_placed_in_this_stage, 0]
        atom = z_matrix[can_be_placed_in_this_stage, 1]
        atoms.append(atom)
        order.append(pos)
        blocks.append(z_matrix[can_be_placed_in_this_stage][:, 1:])
        given = np.union1d(given, atom)
        z_matrix = z_matrix[~can_be_placed_in_this_stage]
    index2atom = np.concatenate(atoms)
    atom2index = np.argsort(index2atom)
    index2order = np.concatenate(order)
    return blocks, index2atom, atom2index, index2order

def ic2xyz_deriv(p1, p2, p3, d14, a124, t1234,
                 eps=1e-7,
                 enforce_boundaries=True,
                 raise_warnings=True):
    """ computes the xyz coordinates from internal coordinates
        relative to points `p1`, `p2`, `p3` together with its
        jacobian with respect to `p1`.
    """
    v1 = p1 - p2
    v2 = p1 - p3

    n = torch.cross(v1, v2, dim=-1)
    nn = torch.cross(v1, n, dim=-1)

    n_norm = torch.norm(n, dim=-1, keepdim=True)

    if raise_warnings:
        if torch.any(n_norm < eps):
            warnings.warn("singular norm in xyz reconstruction")
    if enforce_boundaries:
        n_norm = n_norm.clamp_min(eps)

    n_normalized = n / n_norm

    nn_norm = torch.norm(nn, dim=-1, keepdim=True)

    if raise_warnings:
        if torch.any(nn_norm < eps):
            warnings.warn("singular norm in xyz reconstruction")
    if enforce_boundaries:
        nn_norm = nn_norm.clamp_min(eps)

    nn_normalized = nn / nn_norm

    n_scaled = n_normalized * -torch.sin(t1234)
    nn_scaled = nn_normalized * torch.cos(t1234)

    v3 = n_scaled + nn_scaled
    v3_norm = torch.norm(v3, dim=-1, keepdim=True)

    if raise_warnings:
        if torch.any(v3_norm < eps):
            warnings.warn("singular norm in xyz reconstruction")
    if enforce_boundaries:
        v3_norm = v3_norm.clamp_min(eps)

    v3_normalized = v3 / v3_norm
    #print (v3_normalized.shape,d14.shape)
    v3_scaled = v3_normalized * d14 * torch.sin(a124)

    v1_norm = torch.norm(v1, dim=-1, keepdim=True)

    if raise_warnings:
        if torch.any(v1_norm < eps):
            warnings.warn("singular norm in xyz reconstruction")
    if enforce_boundaries:
        v1_norm = v1_norm.clamp_min(eps)

    v1_normalized = v1 / v1_norm
    v1_scaled = v1_normalized * d14 * torch.cos(a124)

    position = p1 + v3_scaled - v1_scaled

    J_d = v3_normalized * torch.sin(a124) - v1_normalized * torch.cos(a124)
    J_a = v3_normalized * d14 * torch.cos(a124) + v1_normalized * d14 * torch.sin(a124)

    J_t1 = (d14 * torch.sin(a124))[..., None]
    J_t2 = (
        1.0
        / v3_norm[..., None]
        * (torch.eye(3)[None, :].to(p1) - outer(v3_normalized, v3_normalized))
    )

    J_n_scaled = n_normalized * -torch.cos(t1234)
    J_nn_scaled = nn_normalized * -torch.sin(t1234)
    J_t3 = (J_n_scaled + J_nn_scaled)[..., None]

    J_t = (J_t1 * J_t2) @ J_t3

    J = torch.stack([J_d, J_a, J_t[..., 0]], dim=-1)

    return position, J


def ic2xy0_deriv(p1, p2, d14, a124, eps=1e-7, enforce_boundaries=True, raise_warnings=True):
    """ computes the xy coordinates (z set to 0) for the given
        internal coordinates together with the Jacobian
        with respect to `p1`.
    """
    t1234 = torch.Tensor([[0.5 * np.pi]]).to(p1)
    p3 = torch.Tensor([[0, -1, 0]]).to(p1)
    xyz, J = ic2xyz_deriv(p1, p2, p3, d14, a124, t1234, eps=eps, enforce_boundaries=enforce_boundaries, raise_warnings=raise_warnings)
    J = J[..., [0, 1, 2], :][..., [0, 1]]
    return xyz, J


