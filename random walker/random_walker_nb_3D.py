# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:44:59 2019

@author: nblon
"""

"""
Random walker segmentation algorithm
from *Random walks for image segmentation*, Leo Grady, IEEE Trans
Pattern Anal Mach Intell. 2006 Nov;28(11):1768-83.
Installing pyamg and using the 'cg_mg' mode of random_walker improves
significantly the performance.
"""

import numpy as np
from scipy import sparse, ndimage as ndi
from skimage._shared.utils import warn
#import matplotlib.pyplot as plt
import functions as pl


# executive summary for next code block: try to import umfpack from
# scipy, but make sure not to raise a fuss if it fails since it's only
# needed to speed up a few cases.
# See discussions at:
# https://groups.google.com/d/msg/scikit-image/FrM5IGP6wh4/1hp-FtVZmfcJ
# http://stackoverflow.com/questions/13977970/ignore-exceptions-printed-to-stderr-in-del/13977992?noredirect=1#comment28386412_13977992
try:
    from scipy.sparse.linalg.dsolve import umfpack
    old_del = umfpack.UmfpackContext.__del__

    def new_del(self):
        try:
            old_del(self)
        except AttributeError:
            pass
    umfpack.UmfpackContext.__del__ = new_del
    UmfpackContext = umfpack.UmfpackContext()
except:
    UmfpackContext = None

try:
    from pyamg import ruge_stuben_solver
    amg_loaded = True
except ImportError:
    amg_loaded = False
from scipy.sparse.linalg import cg
from skimage.util import img_as_float
from skimage.filters import rank_order

#-----------Laplacian--------------------


def _make_graph_edges_3d(n_x, n_y, n_z):
    """Returns a list of edges for a 3D image.
    Parameters
    ----------
    n_x: integer
        The size of the grid in the x direction.
    n_y: integer
        The size of the grid in the y direction
    n_z: integer
        The size of the grid in the z direction
    Returns
    -------
    edges : (2, N) ndarray
        with the total number of edges::
            N = n_x * n_y * (nz - 1) +
                n_x * (n_y - 1) * nz +
                (n_x - 1) * n_y * nz
        Graph edges with each column describing a node-id pair.
    """
    vertices = np.arange(n_x * n_y * n_z).reshape((n_x, n_y, n_z))
    edges_deep = np.vstack((vertices[:, :, :-1].ravel(),
                            vertices[:, :, 1:].ravel()))
    edges_right = np.vstack((vertices[:, :-1].ravel(),
                             vertices[:, 1:].ravel()))
    edges_down = np.vstack((vertices[:-1].ravel(), vertices[1:].ravel()))
    edges = np.hstack((edges_deep, edges_right, edges_down))
#    a, b = np.unique(edges,return_counts=True)
#    k ,j = np.unique(b,return_counts=True)
#    print(k)
#    print(j)
    return edges


def _compute_weights_3d(data, spacing, eps=1.e-6,alpha=0.3, beta =0.3, gamma=0.4, a=130.0, b=10.0, c=800.0):

    intensity_weights = _compute_intensity_weight_3d(data, spacing, eps, a)
    magnitude_weights = _compute_magnitude_weight_3d(data, spacing, eps, b)
    directional_weights = _compute_directional_weight_3d(data, spacing, eps, c)
    
    weights = alpha*intensity_weights + beta*magnitude_weights + gamma*directional_weights
    
#    weights = directional_weights
#    weights = magnitude_weights
#    weights = intensity_weights
    
#    print(np.amax(intensity_weights))
#    print(np.amin(intensity_weights))
#    print(np.amax(magnitude_weights))
#    print(np.amin(magnitude_weights))
#    print(np.amax(directional_weights))
#    print(np.amin(directional_weights))
    
    return weights

def _compute_intensity_weight_3d(data, spacing,eps=1.e-6, a=130.0):
    #split the calculation of the exponentials of the intensity, velocity magnitude and velocity direction part
    #calculate intensity part here
    delta_I_sq = 0    
    delta_I_sq = _compute_delta_I_3d(data[...,0], spacing) ** 2
#    print(delta_I_sq)
#    plt.hist(gradients.ravel(),bins=100)
#    plt.show()
    delta_I_sq *= a/data[...,0].std()
    intensity_weights = np.exp(- delta_I_sq)
    intensity_weights += eps
    return intensity_weights

def _compute_magnitude_weight_3d(data, spacing,eps=1.e-6, b=10.0):
    #split the calculation of the exponentials of the intensity, velocity magnitude and velocity direction part
    #calculate intensity part here
    
    B_data = np.zeros(data.shape)
    B_data_1 = np.zeros(data.shape)

    B_coords_1 = np.nonzero(data[...,0]>0.1)
    B_data_1[B_coords_1] = True
    B_data = B_data_1*data
#    B_data = data
    
    delta_v = 0
    delta_v = _compute_eucl_dist_3d(B_data, spacing)
    delta_v *= b/data[...,1:].std()
#    print('Delta V: max=' +str(np.amax(delta_v)) + ' min=' + str(np.amin(delta_v)))
#    plt.hist(gradients.ravel(),bins=100)
#    plt.show()
    magnitude_weights = np.exp(- delta_v)
    magnitude_weights += eps
    return magnitude_weights

def _compute_directional_weight_3d(data, spacing,eps=1.e-6, c=800.0):
    #split the calculation of the exponentials of the intensity, velocity magnitude and velocity direction part
    #calculate intensity part here
    dot_product = 0
    
    B_data = np.zeros(data.shape)
    B_data_1 = np.zeros(data.shape)
    
    B_coords_1 = np.nonzero(data[...,0]>0.1)
    B_data_1[B_coords_1] = True
    B_data = B_data_1*data
    
    dot_product = _compute_dotproduct_3d(B_data, spacing)
    dot_product = -(dot_product-1)
    dot_product *= c

    directional_weights = np.exp(-dot_product)
    directional_weights += eps
    return directional_weights
    
def _compute_delta_I_3d(data, spacing):
    gr_deep = np.abs(data[:, :, :-1] - data[:, :, 1:]).ravel() / spacing[2]
#    print(gr_deep)
    gr_right = np.abs(data[:, :-1] - data[:, 1:]).ravel() / spacing[1]
#    print(gr_right)
    gr_down = np.abs(data[:-1] - data[1:]).ravel() / spacing[0]
#    print(gr_down)
#    print(data.shape)
    return np.r_[gr_deep, gr_right, gr_down]


def _compute_eucl_dist_3d(data,spacing):
    #compute the euclidean distances between all adjacent points
    #compute distance between adjacent points in v_x array
    dist_x_deep = np.abs(data[:, :, :-1, 1] - data[:, :, 1:, 1]).ravel() / spacing[2]
    dist_x_right = np.abs(data[:, :-1, :, 1] - data[:, 1:, :, 1]).ravel() / spacing[1]
    dist_x_down = np.abs(data[:-1, :, :, 1] - data[1:, :, :, 1]).ravel() / spacing[0]
    #compute distance between adjacent points in v_y array
    dist_y_deep = np.abs(data[:, :, :-1, 2] - data[:, :, 1:, 2]).ravel() / spacing[2]
    dist_y_right = np.abs(data[:, :-1, :, 2] - data[:, 1:, :, 2]).ravel() / spacing[1]
    dist_y_down = np.abs(data[:-1, :, :, 2] - data[1:, :, :, 2]).ravel() / spacing[0]
    #compute distance between adjacent points in v_z array
    dist_z_deep = np.abs(data[:, :, :-1, 3] - data[:, :, 1:, 3]).ravel() / spacing[2]
    dist_z_right = np.abs(data[:, :-1, :, 3] - data[:, 1:, :, 3]).ravel() / spacing[1]
    dist_z_down = np.abs(data[:-1, :, :, 3] - data[1:, :, :, 3]).ravel() / spacing[0]
    #combine arrays to form list of all distances of 3D vectors
    dist_deep = np.sqrt(np.square(dist_x_deep) + np.square(dist_y_deep) + np.square(dist_z_deep))
    dist_right = np.sqrt(np.square(dist_x_right) + np.square(dist_y_right) + np.square(dist_z_right))
    dist_down = np.sqrt(np.square(dist_x_down) + np.square(dist_y_down) + np.square(dist_z_down))
    
    return np.r_[dist_deep,dist_right,dist_down]

def _compute_dotproduct_3d(data, spacing):
    #compute the dot product and normalize data to obtain the cos of the enclosed angle which constrains the data to [-1,1]
    dot_deep = (data[:,:,:-1,1]*data[:,:,1:,1]+data[:,:,:-1,2]*data[:,:,1:,2]+data[:,:,:-1,3]*data[:,:,1:,3]).ravel() / spacing[2]
    dot_right = (data[:,:-1,:,1]*data[:,1:,:,1]+data[:,:-1,:,2]*data[:,1:,:,2]+data[:,:-1,:,3]*data[:,1:,:,3]).ravel() / spacing[1]
    dot_down = (data[:-1,:,:,1]*data[1:,:,:,1]+data[:-1,:,:,2]*data[1:,:,:,2]+data[:-1,:,:,3]*data[1:,:,:,3]).ravel() / spacing[0]
    
    #normalize the dot products
    dot_deep_magn = ((pl.norm(data[:,:,:-1,1],data[:,:,:-1,2],data[:,:,:-1,3])).ravel()*(pl.norm(data[:,:,1:,1],data[:,:,1:,2],data[:,:,1:,3])).ravel())
    dot_deep_magn += 1e-6
    dot_right_magn = ((pl.norm(data[:,:-1,:,1],data[:,:-1,:,2],data[:,:-1,:,3])).ravel()*(pl.norm(data[:,1:,:,1],data[:,1:,:,2],data[:,1:,:,3])).ravel())
    dot_right_magn += 1e-6
    dot_down_magn = ((pl.norm(data[:-1,:,:,1],data[:-1,:,:,2],data[:-1,:,:,3])).ravel()*(pl.norm(data[1:,:,:,1],data[1:,:,:,2],data[1:,:,:,3])).ravel())
    dot_down_magn += 1e-6
    
    dot_deep_norm = dot_deep/dot_deep_magn
    dot_right_norm = dot_right/dot_right_magn
    dot_down_norm = dot_down/dot_down_magn
    
    return np.r_[dot_deep_norm, dot_right_norm, dot_down_norm]



def _make_laplacian_sparse(edges, weights):
    """
    Sparse implementation
    """
    pixel_nb = edges.max() + 1
    diag = np.arange(pixel_nb)
    i_indices = np.hstack((edges[0], edges[1]))
    j_indices = np.hstack((edges[1], edges[0]))
    data = np.hstack((-weights, -weights))
    lap = sparse.coo_matrix((data, (i_indices, j_indices)),
                            shape=(pixel_nb, pixel_nb))
    connect = - np.ravel(lap.sum(axis=1))
    lap = sparse.coo_matrix(
        (np.hstack((data, connect)), (np.hstack((i_indices, diag)),
                                      np.hstack((j_indices, diag)))),
        shape=(pixel_nb, pixel_nb))
    return lap.tocsr()


def _clean_labels_ar(X, labels, copy=False):
    X = X.astype(labels.dtype)
    if copy:
        labels = np.copy(labels)
    labels = np.ravel(labels)
    labels[labels == 0] = X
    return labels


def _buildAB(lap_sparse, labels):
    """
    Build the matrix A and rhs B of the linear system to solve.
    A and B are two block of the laplacian of the image graph.
    """
    labels = labels[labels >= 0] #labels smaller than 0 are discarded as per the definition of the random walker (-1 labelled px should be ignored)
    indices = np.arange(labels.size) #returns evenly spaces values (1), lenght of vector is given by numer of elements in array "labels"
    unlabeled_indices = indices[labels == 0] #all indices = 0 are classified as unlabeled
    seeds_indices = indices[labels > 0] #all indices >0 are seeds, in our case we have seeds with values 1 (fg) and 2 (bg)
    # The following two lines take most of the time in this function
    B = lap_sparse[unlabeled_indices][:, seeds_indices] #lap_sparse got passed to the _buildAB function from line 456
    lap_sparse = lap_sparse[unlabeled_indices][:, unlabeled_indices]
    nlabels = labels.max()
    rhs = []
    for lab in range(1, nlabels + 1):
        mask = (labels[seeds_indices] == lab)
        fs = sparse.csr_matrix(mask)
        fs = fs.transpose()
        rhs.append(B * fs)
    return lap_sparse, rhs


def _mask_edges_weights(edges, weights, mask):
    """
    Remove edges of the graph connected to masked nodes, as well as
    corresponding weights of the edges.
    """
    mask0 = np.hstack((mask[:, :, :-1].ravel(), mask[:, :-1].ravel(),
                       mask[:-1].ravel()))
    mask1 = np.hstack((mask[:, :, 1:].ravel(), mask[:, 1:].ravel(),
                       mask[1:].ravel()))
    ind_mask = np.logical_and(mask0, mask1)
    edges, weights = edges[:, ind_mask], weights[ind_mask]
    max_node_index = edges.max()
    # Reassign edges labels to 0, 1, ... edges_number - 1
    order = np.searchsorted(np.unique(edges.ravel()),
                            np.arange(max_node_index + 1))
    edges = order[edges.astype(np.int64)]
    return edges, weights


def _build_laplacian(data, spacing, mask=None,
                     alpha=0.3, beta =0.3, gamma=0.4, a=130.0, b=10.0, c=800.0):
    l_x, l_y, l_z = tuple(data.shape[i] for i in range(3))
    edges = _make_graph_edges_3d(l_x, l_y, l_z)
    weights = _compute_weights_3d(data, spacing, eps=1.e-10,alpha=alpha, beta=beta, gamma=gamma, a=a, b=b, c=c)
    if mask is not None:
        edges, weights = _mask_edges_weights(edges, weights, mask)
    lap = _make_laplacian_sparse(edges, weights)
    del edges, weights
    return lap


#----------- Random walker algorithm --------------------------------


def random_walker(data, labels, mode='bf', tol=1.e-3, copy=True,
                  return_full_prob=False, spacing=None, alpha=0.3, beta =0.3, gamma=0.4, a=130.0, b=10.0, c=800.0):
    """Random walker algorithm for segmentation from markers.
    Random walker algorithm is implemented for gray-level or multichannel
    images.
    Parameters
    ----------
    data : array_like
        Image to be segmented in phases. Gray-level `data` can be two- or
        three-dimensional; multichannel data can be three- or four-
        dimensional (multichannel=True) with the highest dimension denoting
        channels. Data spacing is assumed isotropic unless the `spacing`
        keyword argument is used.
    labels : array of ints, of same shape as `data` without channels dimension
        Array of seed markers labeled with different positive integers
        for different phases. Zero-labeled pixels are unlabeled pixels.
        Negative labels correspond to inactive pixels that are not taken
        into account (they are removed from the graph). If labels are not
        consecutive integers, the labels array will be transformed so that
        labels are consecutive. In the multichannel case, `labels` should have
        the same shape as a single channel of `data`, i.e. without the final
        dimension denoting channels.
    beta : float
        Penalization coefficient for the random walker motion
        (the greater `beta`, the more difficult the diffusion).
    mode : string, available options {'cg_mg', 'cg', 'bf'}
        Mode for solving the linear system in the random walker algorithm.
        If no preference given, automatically attempt to use the fastest
        option available ('cg_mg' from pyamg >> 'cg' with UMFPACK > 'bf').
        - 'bf' (brute force): an LU factorization of the Laplacian is
          computed. This is fast for small images (<1024x1024), but very slow
          and memory-intensive for large images (e.g., 3-D volumes).
        - 'cg' (conjugate gradient): the linear system is solved iteratively
          using the Conjugate Gradient method from scipy.sparse.linalg. This is
          less memory-consuming than the brute force method for large images,
          but it is quite slow.
        - 'cg_mg' (conjugate gradient with multigrid preconditioner): a
          preconditioner is computed using a multigrid solver, then the
          solution is computed with the Conjugate Gradient method.  This mode
          requires that the pyamg module (http://pyamg.org/) is
          installed. For images of size > 512x512, this is the recommended
          (fastest) mode.
    tol : float
        tolerance to achieve when solving the linear system, in
        cg' and 'cg_mg' modes.
    copy : bool
        If copy is False, the `labels` array will be overwritten with
        the result of the segmentation. Use copy=False if you want to
        save on memory.
    multichannel : bool, default False
        If True, input data is parsed as multichannel data (see 'data' above
        for proper input format in this case)
    return_full_prob : bool, default False
        If True, the probability that a pixel belongs to each of the labels
        will be returned, instead of only the most likely label.
    spacing : iterable of floats
        Spacing between voxels in each spatial dimension. If `None`, then
        the spacing between pixels/voxels in each dimension is assumed 1.
    Returns
    -------
    output : ndarray
        * If `return_full_prob` is False, array of ints of same shape as
          `data`, in which each pixel has been labeled according to the marker
          that reached the pixel first by anisotropic diffusion.
        * If `return_full_prob` is True, array of floats of shape
          `(nlabels, data.shape)`. `output[label_nb, i, j]` is the probability
          that label `label_nb` reaches the pixel `(i, j)` first.
    See also
    --------
    skimage.morphology.watershed: watershed segmentation
        A segmentation algorithm based on mathematical morphology
        and "flooding" of regions from markers.
    Notes
    -----
    Multichannel inputs are scaled with all channel data combined. Ensure all
    channels are separately normalized prior to running this algorithm.
    The `spacing` argument is specifically for anisotropic datasets, where
    data points are spaced differently in one or more spatial dimensions.
    Anisotropic data is commonly encountered in medical imaging.
    The algorithm was first proposed in *Random walks for image
    segmentation*, Leo Grady, IEEE Trans Pattern Anal Mach Intell.
    2006 Nov;28(11):1768-83.
    The algorithm solves the diffusion equation at infinite times for
    sources placed on markers of each phase in turn. A pixel is labeled with
    the phase that has the greatest probability to diffuse first to the pixel.
    The diffusion equation is solved by minimizing x.T L x for each phase,
    where L is the Laplacian of the weighted graph of the image, and x is
    the probability that a marker of the given phase arrives first at a pixel
    by diffusion (x=1 on markers of the phase, x=0 on the other markers, and
    the other coefficients are looked for). Each pixel is attributed the label
    for which it has a maximal value of x. The Laplacian L of the image
    is defined as:
       - L_ii = d_i, the number of neighbors of pixel i (the degree of i)
       - L_ij = -w_ij if i and j are adjacent pixels
    The weight w_ij is a decreasing function of the norm of the local gradient.
    This ensures that diffusion is easier between pixels of similar values.
    When the Laplacian is decomposed into blocks of marked and unmarked
    pixels::
        L = M B.T
            B A
    with first indices corresponding to marked pixels, and then to unmarked
    pixels, minimizing x.T L x for one phase amount to solving::
        A x = - B x_m
    where x_m = 1 on markers of the given phase, and 0 on other markers.
    This linear system is solved in the algorithm using a direct method for
    small images, and an iterative method for larger images.
    Examples
    --------
    >>> np.random.seed(0)
    >>> a = np.zeros((10, 10)) + 0.2 * np.random.rand(10, 10)
    >>> a[5:8, 5:8] += 1
    >>> b = np.zeros_like(a)
    >>> b[3, 3] = 1  # Marker for first phase
    >>> b[6, 6] = 2  # Marker for second phase
    >>> random_walker(a, b)
    array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 2, 2, 2, 1, 1],
           [1, 1, 1, 1, 1, 2, 2, 2, 1, 1],
           [1, 1, 1, 1, 1, 2, 2, 2, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=int32)
    """
    # Parse input data
    if mode is None:
        if amg_loaded:
            mode = 'cg_mg'
        elif UmfpackContext is not None:
            mode = 'cg'
        else:
            mode = 'bf'

#    if UmfpackContext is None and mode == 'cg':
#        warn('"cg" mode will be used, but it may be slower than '
#             '"bf" because SciPy was built without UMFPACK. Consider'
#             ' rebuilding SciPy with UMFPACK; this will greatly '
#             'accelerate the conjugate gradient ("cg") solver. '
#             'You may also install pyamg and run the random_walker '
#             'function in "cg_mg" mode (see docstring).')

    if (labels != 0).all():
        warn('Random walker only segments unlabeled areas, where '
             'labels == 0. No zero valued areas in labels were '
             'found. Returning provided labels.')

        if return_full_prob:
            # Find and iterate over valid labels
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels > 0]

            out_labels = np.empty(labels.shape + (len(unique_labels),),
                                  dtype=np.bool)
            for n, i in enumerate(unique_labels):
                out_labels[..., n] = (labels == i)

        else:
            out_labels = labels
        return out_labels

    # This algorithm expects 4-D arrays of floats, where the first three
    # dimensions are spatial and the final denotes channels. 2-D images have
    # a singleton placeholder dimension added for the third spatial dimension,
    # and single channel images likewise have a singleton added for channels.
    # The following block ensures valid input and coerces it to the correct
    # form.
    if data.ndim < 3:
        raise ValueError('Data must have 3 or 4 '
                         'dimensions.')
    dims = data[..., 0].shape  # To reshape final labeled result
    data = img_as_float(data)
    if data.ndim == 3:  # 2D multispectral, needs singleton in 3rd axis
        data = data[:, :, np.newaxis, :]

    # Spacing kwarg checks
    if spacing is None:
        spacing = np.asarray((1.,) * 3)
    elif len(spacing) == len(dims):
        if len(spacing) == 2:  # Need a dummy spacing for singleton 3rd dim
            spacing = np.r_[spacing, 1.]
        else:                  # Convert to array
            spacing = np.asarray(spacing)
    else:
        raise ValueError('Input argument `spacing` incorrect, should be an '
                         'iterable with one number per spatial dimension.')

    if copy:
        labels = np.copy(labels)
    label_values = np.unique(labels)

    # Reorder label values to have consecutive integers (no gaps)
    if np.any(np.diff(label_values) != 1):
        mask = labels >= 0
        labels[mask] = rank_order(labels[mask])[0].astype(labels.dtype)
    labels = labels.astype(np.int32)

    # If the array has pruned zones, be sure that no isolated pixels
    # exist between pruned zones (they could not be determined)
    if np.any(labels < 0):
        filled = ndi.binary_propagation(labels > 0, mask=labels >= 0)
        labels[np.logical_and(np.logical_not(filled), labels == 0)] = -1
        del filled
    labels = np.atleast_3d(labels)
    if np.any(labels < 0):
        lap_sparse = _build_laplacian(data, spacing, mask=labels >= 0,
                                      alpha=alpha, beta=beta, gamma=gamma, a=a, b=b, c=c)
    else:
        lap_sparse = _build_laplacian(data, spacing,
                                      alpha=alpha, beta=beta, gamma=gamma, a=a, b=b, c=c)
    lap_sparse, B = _buildAB(lap_sparse, labels)

    # We solve the linear system
    # lap_sparse X = B
    # where X[i, j] is the probability that a marker of label i arrives
    # first at pixel j by anisotropic diffusion.
    if mode == 'cg':
        X = _solve_cg(lap_sparse, B, tol=tol,
                      return_full_prob=return_full_prob)
    if mode == 'cg_mg':
        if not amg_loaded:
            warn("""pyamg (http://pyamg.org/)) is needed to use
                this mode, but is not installed. The 'cg' mode will be used
                instead.""")
            X = _solve_cg(lap_sparse, B, tol=tol,
                          return_full_prob=return_full_prob)
        else:
            X = _solve_cg_mg(lap_sparse, B, tol=tol,
                             return_full_prob=return_full_prob)
    if mode == 'bf':
        X = _solve_bf(lap_sparse, B,
                      return_full_prob=return_full_prob)

    # Clean up results
    if return_full_prob:
        labels = labels.astype(np.float)
        X = np.array([_clean_labels_ar(Xline, labels, copy=True).reshape(dims)
                      for Xline in X])
#        print(X[0,93,118,6])
#        print(labels[93,118,6])
        for i in range(1, int(labels.max()) + 1):
            mask_i = np.squeeze(labels == i)
            X[:, mask_i] = 0
            X[i - 1, mask_i] = 1
#        print(X[0,93,118,6])
#        print(labels[93,118,6])
#        print(labels)
    else:
        X = _clean_labels_ar(X + 1, labels).reshape(dims)
    return X


def _solve_bf(lap_sparse, B, return_full_prob=False):
    """
    solves lap_sparse X_i = B_i for each phase i. An LU decomposition
    of lap_sparse is computed first. For each pixel, the label i
    corresponding to the maximal X_i is returned.
    """
    lap_sparse = lap_sparse.tocsc()
    solver = sparse.linalg.factorized(lap_sparse.astype(np.double))
    X = np.array([solver(np.array((-B[i]).todense()).ravel())
                  for i in range(len(B))])
    if not return_full_prob:
        X = np.argmax(X, axis=0)
    return X


def _solve_cg(lap_sparse, B, tol, return_full_prob=False):
    """
    solves lap_sparse X_i = B_i for each phase i, using the conjugate
    gradient method. For each pixel, the label i corresponding to the
    maximal X_i is returned.
    """
    lap_sparse = lap_sparse.tocsc()
    X = []
    for i in range(len(B)):
        x0 = cg(lap_sparse, -B[i].todense(), tol=tol)[0]
        X.append(x0)
    if not return_full_prob:
        X = np.array(X)
        X = np.argmax(X, axis=0)
    return X


def _solve_cg_mg(lap_sparse, B, tol, return_full_prob=False):
    """
    solves lap_sparse X_i = B_i for each phase i, using the conjugate
    gradient method with a multigrid preconditioner (ruge-stuben from
    pyamg). For each pixel, the label i corresponding to the maximal
    X_i is returned.
    """
    X = []
    ml = ruge_stuben_solver(lap_sparse)
    M = ml.aspreconditioner(cycle='V')
    for i in range(len(B)):
        x0 = cg(lap_sparse, -B[i].todense(), tol=tol, M=M, maxiter=30)[0]
        X.append(x0)
    if not return_full_prob:
        X = np.array(X)
        X = np.argmax(X, axis=0)
    return X