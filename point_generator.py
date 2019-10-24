import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import NonlinearConstraint
from scipy.spatial import distance
from scipy.cluster.vq import vq, kmeans
import matplotlib.pyplot as plt
import pathos.multiprocessing as mp

class PointGenerator():
    """Generates points within a constrained multi-dimensional space."""

    def __init__(self, n_dim, n_points, constraints, known_bounds):
        """
        Construct a PointGenerator.

        :param n_dim: Number of dimensions (int)
        :param n_points: Number of expected points to be generated (int)
        :param constraints: Constraints read in from a constraint file (strings)
        :param known_bounds: Known min/max bounds on variables (dict)
        """
        assert(n_dim > 1)
        assert(n_points > 0)

        self.run_optimizer = True # Set to False to just do Monte Carlo
        self.do_parallel = True # Run the optimizer on clusters in parallel?
        self.target_points_per_cluster = 100 # Set to a high number to avoid clusters
        # Use 50 iters for generating the nice figures in the Solution Explanation
        self.maxiter = 20

        self.plot_2D = False
        self.n_dim = n_dim
        self.n_points = n_points
        self.constraints = []
        for cstr in constraints:
            self.constraints.append(compile(cstr, "<string>", "eval"))
        self.known_bounds = known_bounds

    def _initial_points(self, n_points):
        """
        Find an initial set of points by random sampling
        :param n_points: Number of points to generate (int)
        """

        x0 = np.zeros((n_points, self.n_dim))
        cmin = self.known_bounds[0]
        cmax = self.known_bounds[1]
        i = 0
        while i < n_points:
            # Test a lot of points at once, it's faster
            x = np.random.random((n_points*1000, self.n_dim))
            # If cmin/cmax known bounds exist, then scale the input points
            for k in range(0, self.n_dim):
                xk_min = 0 
                xk_max = 1
                if k in cmin:
                    xk_min = cmin[k]
                if k in cmax:
                    xk_max = cmax[k]
                x[:, k] = x[:, k] * (xk_max - xk_min) + xk_min

            x_valid = np.array(np.ones((x.shape[0],), dtype=bool))
            for constraint in self.constraints:
                x_valid = np.logical_and(x_valid, eval(constraint) >= 0)

            # Only keep the valid points
            i_incr = min(np.sum(x_valid), n_points - i)
            x0[i:i+i_incr, :] = x[x_valid, :][0:i_incr, :]
            i += i_incr
        
        return x0

    def _test_invalid(self, x):
        """
        Test whether point x violates any constraint.
        
        :param x: One test point of shape (1, n_dim)
        """

        if np.any(x < 0) or np.any(x > 1):
            return True
        
        x = np.expand_dims(x, 0) # turn x into a matrix with 1 row
        for constraint in self.constraints:
            if not eval(constraint) >= 0:
                return True
                
        return False

    def _generate_loss(self, n_points_cluster):
        """
        Generate the loss function for one cluster
        
        :param n_points_cluster: Number of points in one cluster (int)
        """

        def loss_fun(x_vec):
            """
            Sum U=1/d for all point pairs
            
            :param x_vec: All variables being optimized, squished into a vector
            """
            x = np.reshape(x_vec, (n_points_cluster, self.n_dim))
            dist_pairs = distance.pdist(x)
            dist_pairs[dist_pairs == 0] = 1e-6  # avoid div by 0
            return np.sum(np.reciprocal(dist_pairs))

        return loss_fun

    def _generate_jac(self, n_points_cluster):
        """
        Generate the Jacobian function for one cluster
        
        :param n_points_cluster: Number of points in one cluster (int)
        """

        def jac_fun(x_vec):
            """
            Compute the Jacobian as dU/dx = delta_x/d^3

            :param x_vec: All variables being optimized, squished into a vector
            """
            x = np.reshape(x_vec, (n_points_cluster, self.n_dim))
            jac = np.zeros_like(x)

            for i in range(0, n_points_cluster-1):
                for j in range(i+1, n_points_cluster):
                    dx = x[j, :] - x[i, :]
                    r2 = np.sum(dx*dx)
                    r2 = max(r2, 1e-6) # avoid div by 0
                    frac = dx / np.power(r2, 1.5) # equivalent to 1/r^3
                    jac[i, :] += frac
                    jac[j, :] -= frac # symmetric response w.r.t the two points
            jac = np.reshape(jac, (-1,))
            return jac

        return jac_fun

    def _generate_bounds(self, x0_cluster):
        """
        Generate fixed bounds for a cluster, to keep points approximately within this cluster

        :param x0_cluster: Initial location of all variables being optimized in this cluster
        """
        n_points_cluster = x0_cluster.shape[0]
        # Find the min/max values of each dimension in this cluster
        min_x = np.min(x0_cluster, axis=0)
        max_x = np.max(x0_cluster, axis=0)
        # If close to the boundary, just set to the boundary
        diff = max_x - min_x
        min_x[min_x - 0 < diff/3] = 0
        max_x[1 - max_x < diff/3] = 1
        # Copy the min/max values so they are repeated for each point in x0
        min_x = np.repeat(np.expand_dims(min_x, axis=0), n_points_cluster, axis=0)
        max_x = np.repeat(np.expand_dims(max_x, axis=0), n_points_cluster, axis=0)
        # Reshape to the format expected by Bounds()
        min_x = np.reshape(min_x, (-1,))
        max_x = np.reshape(max_x, (-1,))
        assert(np.all(x0_cluster >= 0.0))
        assert(np.all(x0_cluster <= 1.0))
        return Bounds(min_x, max_x, keep_feasible=False)

    def _generate_nl_constraints(self, n_points_cluster):
        """
        Generate the nonlinear constraint functions for a cluster.

        :param n_points_cluster: Number of points in one cluster (int)
        """
        def nl_const(x_vec):
            """
            Evaluate the nonlinear constraints

            :param x_vec: All variables being optimized, squished into a vector
            """
            x = np.reshape(x_vec, (n_points_cluster, self.n_dim))
            cmat = np.zeros((n_points_cluster, len(self.constraints)))
            for i, constraint in enumerate(self.constraints):
                cmat[:, i] = eval(constraint)
                # return eval(constraint)
            cvec = np.reshape(cmat, (-1,))
            return cvec
        return NonlinearConstraint(nl_const, 0, np.inf, keep_feasible=False)


    def generate_points(self):
        """Generate points within the feasible space"""

        x0 = self._initial_points(self.n_points)
        if not self.run_optimizer:
            if self.n_dim == 2 and self.plot_2D:
                plt.plot(x0[:, 0], x0[:, 1], '.')
                plt.show()
            return x0

        # Use k-means clustering to subdivide the points, to solve several 
        # smaller optimization problems instead of one large problem.
        n_clusters = round(self.n_points / self.target_points_per_cluster)
        if n_clusters > 1:
            centroids, _ = kmeans(x0, n_clusters)
            idx, _ = vq(x0, centroids)
        else:
            n_clusters = 1
            idx = np.zeros((self.n_points,))

        if self.n_dim == 2 and self.plot_2D:
            for c in range(0, n_clusters):
                plt.plot(x0[idx == c, 0], x0[idx == c, 1], '.')

        x_sol = np.zeros_like(x0)

        # shared with child threads
        self._pool_shared = (x0, idx)

        if self.do_parallel:

            with mp.Pool() as p:
                p_results = p.map(self._minimize_cluster, range(0, n_clusters))

            for c, cx in p_results:
                # Copy the solution to the shared solution vector
                x_sol[idx == c,:] = cx

        else:
            for c in range(0, n_clusters):
                _, cx = self._minimize_cluster(c)
                x_sol[idx == c,:] = cx

        # Compare initial and final loss
        print(self._generate_loss(self.n_points)(x0))
        print(self._generate_loss(self.n_points)(x_sol))

        if self.n_dim == 2 and self.plot_2D:
            plt.figure()
            plt.plot(x_sol[:, 0], x_sol[:, 1], 'r.')
            plt.show()
        
        return x_sol
        
    def _minimize_cluster(self, c):
        """
        Run an optimization problem on cluster c to find the optimal point locations.
        
        :param c: The cluster index (int)
        """
        
        x0, idx = self._pool_shared
        # Get the subset of points corresponding to just this cluster
        x0_cluster = x0[idx == c, :]
        # Number of points in this cluster
        n_points_cluster = x0_cluster.shape[0]
        # Optimize the points placement for just this cluster
        res = minimize(fun=self._generate_loss(n_points_cluster),
                        x0=np.reshape(x0_cluster, (-1,)),
                        jac=self._generate_jac(n_points_cluster),
                        method='trust-constr',
                        bounds=self._generate_bounds(x0_cluster),
                        constraints=self._generate_nl_constraints(n_points_cluster),
                        options={'verbose':3, 'maxiter':self.maxiter})
        cx = np.reshape(res.x, (n_points_cluster, self.n_dim))
        self._fix_invalid(cx)
        return (c, cx)

    def _fix_invalid(self, x):
        """Bring points in x to the centroid until they are all valid points"""

        n = x.shape[0]
        centroid = np.mean(x, axis=0)

        # assume the centroid is a valid point
        # assume any invalid point is not very far outside the valid space
        for i in range(0, n):
            iter = 0
            while self._test_invalid(x[i, :]) and iter < 10:
                x[i, :] -= (x[i, :] - centroid) * 0.25
                iter += 1
            # backup plan: random sampling
            if iter == 10:
                x[i, :] = self._initial_points(1)



