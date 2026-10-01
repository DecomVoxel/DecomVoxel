import torch
import os
import trimesh
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt
import json
from sklearn.base import BaseEstimator, RegressorMixin


class GeneralPlaneRegressor(BaseEstimator, RegressorMixin):
    """
    General plane regressor using ax + by + cz + d = 0 representation
    """
    def __init__(self, alpha=1.0, prior_normal=None):
        """
        Args:
            alpha: Regularization strength
            prior_normal: Prior normal direction [n1, n2, n3], should be normalized
        """
        self.alpha = alpha
        self.prior_normal = prior_normal
        self.coef_ = None  # [a, b, c, d] coefficients
        
    def fit(self, X, y=None):
        """
        Fit general plane equation ax + by + cz + d = 0
        
        Args:
            X: Points array of shape [N, 3]
            y: Not used (for sklearn compatibility)
        """
        points = X  # Shape [N, 3]
        N = points.shape[0]
        
        if self.prior_normal is not None:
            # Use regularized fitting with normal constraint
            self.coef_ = self._fit_with_normal_constraint(points)
        else:
            # Use SVD-based fitting
            self.coef_ = self._fit_with_svd(points)
            
        return self
    
    def _fit_with_svd(self, points):
        """
        Fit plane using SVD (Principal Component Analysis approach)
        """
        # Center the points
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        
        # Compute SVD
        U, S, Vt = np.linalg.svd(centered_points, full_matrices=False)
        
        # The normal vector is the last column of V (row of Vt)
        normal = Vt[-1, :]  # Shape [3]
        
        # Compute d: ax + by + cz + d = 0, so d = -(a*cx + b*cy + c*cz)
        d = -np.dot(normal, centroid)
        
        return np.array([normal[0], normal[1], normal[2], d])
    
    def _fit_with_normal_constraint(self, points):
        """
        Fit plane with normal direction constraint using optimization
        """

        points = points.astype(np.float64)
        if self.prior_normal is not None:
            self.prior_normal = np.array(self.prior_normal, dtype=np.float64)

        def objective(params):
            a, b, c, d = params
            
            # Normalize the normal vector
            normal_norm = np.sqrt(a*a + b*b + c*c)
            if normal_norm < 1e-8:
                return 1e10
            
            a_norm, b_norm, c_norm = a/normal_norm, b/normal_norm, c/normal_norm
            d_norm = d/normal_norm
            
            # Compute point-to-plane distances
            distances = np.abs(points[:, 0] * a_norm + 
                             points[:, 1] * b_norm + 
                             points[:, 2] * c_norm + d_norm)
            
            # Main loss: mean squared distance
            mse_loss = np.mean(distances ** 2)
            
            # Regularization: constrain normal directions to be collinear
            if self.prior_normal is not None:
                current_normal = np.array([a_norm, b_norm, c_norm])
                dot_product = np.dot(self.prior_normal, current_normal)
                # Regularization: minimize (1 - |dot_product|)^2
                regularization = self.alpha * (1.0 - np.abs(dot_product)) ** 2
            else:
                regularization = 0.0
            
            return mse_loss + regularization
        
        # Initial guess using SVD
        initial_coef = self._fit_with_svd(points)
        
        # If prior normal is available, adjust initial guess
        if self.prior_normal is not None:
            # Use prior normal as initial normal, compute d from centroid
            centroid = np.mean(points, axis=0)
            initial_coef[:3] = self.prior_normal
            initial_coef[3] = -np.dot(self.prior_normal, centroid)
        
        # Optimization with normalization constraint
        def constraint_fun(params):
            a, b, c, d = params
            return a*a + b*b + c*c - 1.0  # ||normal|| = 1
        
        from scipy.optimize import minimize
        constraint = {'type': 'eq', 'fun': constraint_fun}
        
        result = minimize(objective, initial_coef, method='SLSQP', constraints=constraint)
        
        if result.success:
            return result.x.astype(np.float64)
        else:
            # Fallback to SVD result
            return initial_coef
    
    def predict(self, X):
        """
        Compute signed distances to the plane (for compatibility with RANSAC)
        """
        if self.coef_ is None:
            raise ValueError("Model not fitted yet")
        
        a, b, c, d = self.coef_
        # Normalize coefficients
        normal_norm = np.sqrt(a*a + b*b + c*c)
        if normal_norm < 1e-8:
            return np.zeros(X.shape[0])
        
        a_norm, b_norm, c_norm, d_norm = a/normal_norm, b/normal_norm, c/normal_norm, d/normal_norm
        
        # Return signed distances
        return X[:, 0] * a_norm + X[:, 1] * b_norm + X[:, 2] * c_norm + d_norm
    
    def get_plane_params(self):
        """
        Get normalized plane parameters
        """
        if self.coef_ is None:
            raise ValueError("Model not fitted yet")
        
        a, b, c, d = self.coef_
        normal_norm = np.sqrt(a*a + b*b + c*c)
        if normal_norm < 1e-8:
            return np.array([0, 0, 1]), 0
        
        normal = np.array([a, b, c]) / normal_norm
        if abs(c) > 1e-5:
            z_offset = -1.0 * d / c
            center = np.array([0, 0, z_offset])
        else:
            if abs(b) > 1e-5:
                y_offset = -1.0 * d / b
                center = np.array([0, y_offset, 0])
            else:
                x_offset = -1.0 * d / a
                center = np.array([x_offset, 0, 0])

        return normal, center


def normals_cluster_1d(valid_normals_1d, n_init_clusters=8, n_clusters=6, min_size_ratio=0.004):
    """
    Cluster 1D normal vectors and return 1D cluster masks
    
    Args:
        valid_normals_1d: Normal vectors from valid region (N, 3)
        n_init_clusters: Initial number of clusters for KMeans
        n_clusters: Number of clusters to keep after filtering
        min_size_ratio: Minimum size ratio for valid clusters
    
    Returns:
        cluster_masks: List of 1D cluster masks (each is boolean array of length N)
        cluster_centers: Cluster centers (normal directions)
    """
    min_cluster_size = valid_normals_1d.shape[0] * min_size_ratio

    # KMeans clustering on 1D normals
    kmeans = KMeans(n_clusters=n_init_clusters, random_state=0, n_init=1).fit(valid_normals_1d)
    pred_1d = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    # Select top clusters by size
    count_values = np.bincount(pred_1d)
    topk = np.argpartition(count_values, -n_clusters)[-n_clusters:]
    sorted_topk_idx = np.argsort(count_values[topk])
    sorted_topk = topk[sorted_topk_idx][::-1]
    
    cluster_masks = []
    cluster_centers = []
    
    for cluster_id in sorted_topk:
        # Create 1D mask for this cluster
        cluster_mask_1d = (pred_1d == cluster_id)
        
        # Filter by minimum size
        if cluster_mask_1d.sum() < min_cluster_size:
            continue
        
        cluster_masks.append(cluster_mask_1d)
        
        # Normalize cluster center
        center_norm = centers[cluster_id] / np.linalg.norm(centers[cluster_id])
        cluster_centers.append(center_norm)
    
    return cluster_masks, np.array(cluster_centers)


def fit_plane_ransac(pnts, threshold=0.01, min_samples=3, max_trials=1000, 
                    alpha=1.0, prior_normal=None):
    """
    Fit a plane to 3D points using RANSAC with general plane equation ax + by + cz + d = 0
    
    Args:
        pnts: numpy array of shape [N, 3], representing N 3D points
        threshold: Distance threshold to determine inliers
        min_samples: Minimum number of data points to fit the model
        max_trials: Maximum number of iterations for RANSAC
        use_normal_constraint: Whether to use normal direction constraints
        alpha: Regularization strength (only used when use_normal_constraint=True)
        prior_normal: Prior normal direction [n1, n2, n3] (only used when use_normal_constraint=True)
    
    Returns:
        plane_normal: Normal vector of the plane [a, b, c] (normalized)
        plane_center: Center point of the plane [x, y, z]
        inlier_mask: Boolean mask of inliers, shape [N,]
    """

    if prior_normal is not None:
        use_normal_constraint = True
    else:
        use_normal_constraint = False
    
    # Normalize prior_normal if provided
    if prior_normal is not None:
        prior_normal = np.array(prior_normal)
        prior_normal = prior_normal / np.linalg.norm(prior_normal)
    
    # Create general plane regressor
    if use_normal_constraint:
        base_regressor = GeneralPlaneRegressor(alpha=alpha, prior_normal=prior_normal)
    else:
        base_regressor = GeneralPlaneRegressor()
    
    # Use RANSAC for robust fitting
    ransac = RANSACRegressor(
        estimator=base_regressor,
        residual_threshold=threshold,
        min_samples=min_samples,
        max_trials=max_trials,
        random_state=42
    )
    
    # Fit using all 3D coordinates (no y needed for general plane fitting)
    ransac.fit(pnts, np.zeros(pnts.shape[0]))  # Dummy y for sklearn compatibility
    
    # Get plane parameters
    plane_normal, plane_center = ransac.estimator_.get_plane_params()
    
    return plane_normal, plane_center, ransac.inlier_mask_


def get_plane_loss_by_ransac(pnts, normals, plane_mask):
    """
    pnts: [H, W, 3]
    normals: [H, W, 3]
    plane_mask: [H, W]
    """

    valid_pnts = pnts[plane_mask]
    valid_normals = normals[plane_mask]

    _, cluster_centers = normals_cluster_1d(valid_normals.detach().cpu().numpy())
    most_frequent_normal = cluster_centers[0]

    plane_normal, plane_center, _ = fit_plane_ransac(valid_pnts.detach().cpu().numpy(), prior_normal=most_frequent_normal)
    plane_normal_t = torch.from_numpy(plane_normal).to(valid_pnts.device).to(torch.float32)
    plane_normal_t = plane_normal_t / (torch.norm(plane_normal_t) + 1e-8)
    plane_center_t = torch.from_numpy(plane_center).to(valid_pnts.device).to(torch.float32)

    # Calculate the distance from all valid_pnts to the plane
    # Distance formula: |(p - p0) · n|
    signed_dist = torch.sum((valid_pnts - plane_center_t) * plane_normal_t, dim=-1)
    dist = (torch.abs(signed_dist)).mean()

    return dist