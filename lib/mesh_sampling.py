import math
import heapq
import numpy as np
import scipy.sparse as sp

from opendr.topology import get_vert_connectivity, get_vertices_per_edge

from menpo.shape import PointCloud, TriMesh
from menpo3d.vtkutils import trimesh_from_vtk, trimesh_to_vtk, VTKClosestPointLocator

from vtk.util.numpy_support import vtk_to_numpy


def vertex_quadrics(mesh):
    """Computes a quadric for each vertex in the Mesh.

    Returns:
       v_quadrics: an (N x 4 x 4) array, where N is # vertices.
    """

    # Allocate quadrics
    v_quadrics = np.zeros((len(mesh.points), 4, 4,))

    # For each face...
    for f_idx in range(len(mesh.trilist)):
        # Compute normalized plane equation for that face
        vert_idxs = mesh.trilist[f_idx]
        verts = np.hstack((mesh.points[vert_idxs], np.array([1, 1, 1]).reshape(-1, 1)))
        u, s, v = np.linalg.svd(verts)
        eq = v[-1, :].reshape(-1, 1)
        eq = eq / (np.linalg.norm(eq[0:3]))

        # Add the outer product of the plane equation to the
        # quadrics of the vertices for this face
        for k in range(3):
            v_quadrics[mesh.trilist[f_idx, k], :, :] += np.outer(eq, eq)

    return v_quadrics


def setup_deformation_transfer(source, target):
    # Get closest points on mesh and trilist indices
    vtk_mesh = trimesh_to_vtk(source)
    closest_pts_on_mesh = VTKClosestPointLocator(vtk_mesh)
    nearest_pts, _ = closest_pts_on_mesh(target.points)

    bcs, bc_idxs = source.barycentric_coordinates_of_pointcloud(PointCloud(nearest_pts))

    # Make sparse colummn matrix to hold upsampling values
    rows = np.floor(np.arange(0, target.points.shape[0], 1.0/3.0)).ravel()
    cols = source.trilist[bc_idxs].ravel()
    coeffs_v = bcs.ravel()

    matrix = sp.csc_matrix((coeffs_v, (rows, cols)), shape=(target.points.shape[0], source.points.shape[0]))
    return matrix


def setup_deformation_transfer_orig(source, target, use_normals=False):
    rows = np.zeros(3 * target.points.shape[0])
    cols = np.zeros(3 * target.points.shape[0])
    coeffs_v = np.zeros(3 * target.points.shape[0])
    coeffs_n = np.zeros(3 * target.points.shape[0])

    nearest_faces, nearest_parts, nearest_vertices = source.compute_aabb_tree().nearest(target.points, True)
    nearest_faces = nearest_faces.ravel().astype(np.int64)
    nearest_parts = nearest_parts.ravel().astype(np.int64)
    nearest_vertices = nearest_vertices.ravel()

    for i in range(target.points.shape[0]):
        # Closest triangle index
        f_id = nearest_faces[i]
        # Closest triangle vertex ids
        nearest_f = source.trilist[f_id]

        # Closest surface point
        nearest_v = nearest_vertices[3 * i:3 * i + 3]
        # Distance vector to the closest surface point
        # dist_vec = target.points[i] - nearest_v

        rows[3 * i:3 * i + 3] = i * np.ones(3)
        cols[3 * i:3 * i + 3] = nearest_f

        n_id = nearest_parts[i]
        if n_id == 0:
            # Closest surface point in triangle
            A = np.vstack((source.points[nearest_f])).T
            coeffs_v[3 * i:3 * i + 3] = np.linalg.lstsq(A, nearest_v)[0]
        elif 0 < n_id <= 3:
            # Closest surface point on edge
            A = np.vstack((source.points[nearest_f[n_id - 1]], source.points[nearest_f[n_id % 3]])).T
            tmp_coeffs = np.linalg.lstsq(A, target.points[i])[0]
            coeffs_v[3 * i + n_id - 1] = tmp_coeffs[0]
            coeffs_v[3 * i + n_id % 3] = tmp_coeffs[1]
        else:
            # Closest surface point a vertex
            coeffs_v[3 * i + n_id - 4] = 1.0

    matrix = sp.csc_matrix((coeffs_v, (rows, cols)), shape=(target.points.shape[0], source.points.shape[0]))
    return matrix


def qslim_decimator_transformer(mesh, factor=None, n_verts_desired=None):
    """Return a simplified version of this mesh.

    A Qslim-style approach is used here.

    :param mesh: mesh to be simplified
    :param factor: fraction of the original vertices to retain
    :param n_verts_desired: number of the original vertices to retain
    :returns: new_faces: An Fx3 array of faces, mtx: Transformation matrix
    """

    if factor is None and n_verts_desired is None:
        raise Exception('Need either factor or n_verts_desired.')

    if n_verts_desired is None:
        n_verts_desired = math.ceil(len(mesh.points) * factor)

    Qv = vertex_quadrics(mesh)

    # fill out a sparse matrix indicating vertex-vertex adjacency
    # from psbody.mesh.topology.connectivity import get_vertices_per_edge
    vert_adj = get_vertices_per_edge(mesh.points, mesh.trilist)
    # vert_adj = sp.lil_matrix((len(mesh.points), len(mesh.points)))
    # for f_idx in range(len(mesh.trilist)):
    #     vert_adj[mesh.trilist[f_idx], mesh.trilist[f_idx]] = 1

    vert_adj = sp.csc_matrix((vert_adj[:, 0] * 0 + 1, (vert_adj[:, 0], vert_adj[:, 1])),
                             shape=(len(mesh.points), len(mesh.points)))
    vert_adj = vert_adj + vert_adj.T
    vert_adj = vert_adj.tocoo()

    def collapse_cost(Qv, r, c, v):
        Qsum = Qv[r, :, :] + Qv[c, :, :]
        p1 = np.vstack((v[r].reshape(-1, 1), np.array([1]).reshape(-1, 1)))
        p2 = np.vstack((v[c].reshape(-1, 1), np.array([1]).reshape(-1, 1)))

        destroy_c_cost = p1.T.dot(Qsum).dot(p1)
        destroy_r_cost = p2.T.dot(Qsum).dot(p2)
        result = {
            'destroy_c_cost': destroy_c_cost,
            'destroy_r_cost': destroy_r_cost,
            'collapse_cost': min([destroy_c_cost, destroy_r_cost]),
            'Qsum': Qsum}
        return result

    # construct a queue of edges with costs
    queue = []
    for k in range(vert_adj.nnz):
        r = vert_adj.row[k]
        c = vert_adj.col[k]

        if r > c:
            continue

        cost = collapse_cost(Qv, r, c, mesh.points)['collapse_cost']
        heapq.heappush(queue, (cost, (r, c)))

    # decimate
    collapse_list = []
    nverts_total = len(mesh.points)
    faces = mesh.trilist.copy()
    while nverts_total > n_verts_desired:
        e = heapq.heappop(queue)
        r = e[1][0]
        c = e[1][1]
        if r == c:
            continue

        cost = collapse_cost(Qv, r, c, mesh.points)
        if cost['collapse_cost'] > e[0]:
            heapq.heappush(queue, (cost['collapse_cost'], e[1]))
            # print 'found outdated cost, %.2f < %.2f' % (e[0], cost['collapse_cost'])
            continue
        else:

            # update old vert idxs to new one,
            # in queue and in face list
            if cost['destroy_c_cost'] < cost['destroy_r_cost']:
                to_destroy = c
                to_keep = r
            else:
                to_destroy = r
                to_keep = c

            collapse_list.append([to_keep, to_destroy])

            # in our face array, replace "to_destroy" vertidx with "to_keep" vertidx
            np.place(faces, faces == to_destroy, to_keep)

            # same for queue
            which1 = [idx for idx in range(len(queue)) if queue[idx][1][0] == to_destroy]
            which2 = [idx for idx in range(len(queue)) if queue[idx][1][1] == to_destroy]
            for k in which1:
                queue[k] = (queue[k][0], (to_keep, queue[k][1][1]))
            for k in which2:
                queue[k] = (queue[k][0], (queue[k][1][0], to_keep))

            Qv[r, :, :] = cost['Qsum']
            Qv[c, :, :] = cost['Qsum']

            a = faces[:, 0] == faces[:, 1]
            b = faces[:, 1] == faces[:, 2]
            c = faces[:, 2] == faces[:, 0]

            # remove degenerate faces
            def logical_or3(x, y, z):
                return np.logical_or(x, np.logical_or(y, z))

            faces_to_keep = np.logical_not(logical_or3(a, b, c))
            faces = faces[faces_to_keep, :].copy()

        nverts_total = (len(np.unique(faces.flatten())))

    new_faces, mtx = _get_sparse_transform(faces, len(mesh.points))
    return new_faces, mtx


def _get_sparse_transform(faces, num_original_verts):
    verts_left = np.unique(faces.flatten())
    IS = np.arange(len(verts_left))
    JS = verts_left
    data = np.ones(len(JS))

    mp = np.arange(0, np.max(faces.flatten()) + 1)
    mp[JS] = IS
    new_faces = mp[faces.copy().flatten()].reshape((-1, 3))

    ij = np.vstack((IS.flatten(), JS.flatten()))
    mtx = sp.csc_matrix((data, ij), shape=(len(verts_left), num_original_verts))

    return new_faces, mtx


def generate_transform_matrices(mesh, factors):
    """Generates len(factors) meshes, each of them is scaled by factors[i] and
       computes the transformations between them.
    
    Returns:
       M: a set of meshes downsampled from mesh by a factor specified in factors.
       A: Adjacency matrix for each of the meshes
       D: Downsampling transforms between each of the meshes
       U: Upsampling transforms between each of the meshes
    """
    factors = map(lambda x: 1.0/x, factors)
    M, A, D, U = [], [], [], []
    A.append(get_vert_connectivity(mesh.points, mesh.trilist))
    M.append(mesh)

    for factor in factors:
        ds_f, ds_D = qslim_decimator_transformer(M[-1], factor=factor)
        D.append(ds_D)
        new_mesh_v = ds_D.dot(M[-1].points)
        new_mesh = TriMesh(points=new_mesh_v, trilist=ds_f)
        M.append(new_mesh)
        A.append(get_vert_connectivity(new_mesh.points, new_mesh.trilist))
        U.append(setup_deformation_transfer(M[-1], M[-2]))

    return M, A, D, U
