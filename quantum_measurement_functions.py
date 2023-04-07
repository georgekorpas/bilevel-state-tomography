import numpy as np
from qutip import Qobj

#_______________________________________________

def measure_counts(rho, operator, samples):
    """
    Simulate the measurement of a quantum state (represented by the density matrix rho) 
    with respect to a given operator. Returns the counts for each eigenvalue of the operator
    after 'samples' number of measurements.

    Parameters
    ----------
    rho : Qobj
        The density matrix of the quantum state.
    operator : Qobj
        The operator to measure the quantum state against (e.g., Pauli X, Y, or Z).
    samples : int
        The number of measurements to perform.

    Returns
    -------
    counts : dict
        A dictionary containing the counts for each eigenvalue of the operator after
        'samples' number of measurements.

    Example
    -------
    >>> rho = Qobj([[0.5, 0.5], [0.5, 0.5]])
    >>> operator = sigmax()
    >>> samples = 1000
    >>> measure_counts(rho, operator, samples)
    {-1.0: 498, 1.0: 502}
    """

    # Compute the eigenvalues and eigenvectors of the given operator
    eigenvalues, eigenvectors = operator.eigenstates()

    # Compute the probabilities of obtaining each eigenvalue using the Born rule
    probabilities = [(eigenvector.dag() * rho * eigenvector)[0, 0] for eigenvector in eigenvectors]

    # Generate a list of 'samples' number of outcomes, each of which is one of the eigenvalues,
    # chosen randomly according to the calculated probabilities
    outcomes = np.random.choice(eigenvalues, size=samples, p=np.real(probabilities))

    # Count the occurrences of each eigenvalue in the list of outcomes and store them in a dictionary
    counts = {round(eigenvalue): np.count_nonzero(outcomes == eigenvalue) for eigenvalue in eigenvalues}

    return counts


#_______________________________________________

def counts_dict_to_array(counts_dict):
    """
    Convert a dictionary containing counts for two eigenvalues to a 2D NumPy array.

    Parameters
    ----------
    counts_dict : dict
        A dictionary where the keys are the two eigenvalues and the values are the corresponding counts.

    Returns
    -------
    np.ndarray
        A 2D NumPy array with the counts in the positions [0, 0] and [1, 0], corresponding to the two eigenvalues.

    Notes
    -----
    This function assumes that the input dictionary contains only two keys, corresponding to the two possible
    eigenvalues of a quantum state. The function first converts the input dictionary to a sorted list of tuples
    and then extracts the counts for the two eigenvalues from the sorted list. The function then returns a 2D NumPy
    array with the counts in separate rows, corresponding to the two eigenvalues.

    Examples
    --------
    >>> counts = {-0.9999999999999999: 49887, 0.9999999999999996: 50113}
    >>> counts_array = counts_dict_to_array(counts)
    >>> print(counts_array)
    [[49887]
     [50113]]
    """
    # Convert the counts dictionary to a sorted list of tuples
    counts_list = sorted(counts_dict.items())

    # Extract the values from the sorted list of tuples
    counts_values = [count[1] for count in counts_list]

    # Convert the counts values to a two-dimensional NumPy array
    counts_array = np.array(counts_values).reshape(2, 1)

    return counts_array

#_______________________________________________

def sample_spherical_gaussians(mu1, mu2, n1, n2):
    """
    Samples two spherical Gaussians in two dimensions.

    Parameters
    ----------
    mu1 : array_like
        Mean of the first Gaussian. Must be a 2D NumPy array of shape (2,).
    mu2 : array_like
        Mean of the second Gaussian. Must be a 2D NumPy array of shape (2,).
    n1 : int
        Number of samples to draw from the first Gaussian.
    n2 : int
        Number of samples to draw from the second Gaussian.

    Returns
    -------
    tuple
        A tuple containing two lists, one with the points sampled from the first Gaussian and one with
        the points sampled from the second Gaussian.

    Raises
    ------
    ValueError
        If `mu1` or `mu2` are not 2D NumPy arrays of shape (2,), or if `n1` or `n2` are not positive integers.

    Notes
    -----
    This function assumes that the covariance matrix for each spherical Gaussian is the identity matrix.
    """

    # Check input parameters
    if not isinstance(mu1, np.ndarray) or mu1.shape != (2,):
        raise ValueError("mu1 must be a 2D NumPy array of shape (2,)")
    if not isinstance(mu2, np.ndarray) or mu2.shape != (2,):
        raise ValueError("mu2 must be a 2D NumPy array of shape (2,)")
    if not isinstance(n1, int) or n1 <= 0:
        raise ValueError("n1 must be a positive integer")
    if not isinstance(n2, int) or n2 <= 0:
        raise ValueError("n2 must be a positive integer")

    # Define covariance matrix for spherical Gaussian
    cov = np.eye(2)

    # Draw samples from first Gaussian
    x1 = np.random.multivariate_normal(mu1, cov, n1)

    # Draw samples from second Gaussian
    x2 = np.random.multivariate_normal(mu2, cov, n2)

    return x1.tolist(), x2.tolist()

#_______________________________________________

def estimate_gaussian_means(data, max_iterations=100, tol=1e-4):
    """
    Estimate the means of two spherical Gaussian distributions using a simple
    Maximum Likelihood Estimation (MLE) algorithm.

    This function assumes that the input data is generated from two spherical
    Gaussian distributions and estimates their means using an iterative
    procedure.

    Parameters
    ----------
    data : numpy.ndarray, shape (n_samples, n_features)
        The input data points.
    max_iterations : int, optional, default: 100
        The maximum number of iterations for the algorithm.
    tol : float, optional, default: 1e-4
        The convergence tolerance for the algorithm.

    Returns
    -------
    mean1 : numpy.ndarray, shape (n_features,)
        The estimated mean of the first Gaussian distribution.
    mean2 : numpy.ndarray, shape (n_features,)
        The estimated mean of the second Gaussian distribution.

    Example
    -------
    >>> data = np.array([(1, 2), (2, 4), (3, 6), (4, 8),
    ...                 (9, 10), (10, 12), (11, 14), (12, 16)])
    >>> mean1, mean2 = estimate_gaussian_means(data)
    >>> print("Estimated means:", mean1, mean2)
    """
    # Initialize means randomly
    mean1 = random.choice(data)
    mean2 = random.choice(data)

    for iteration in range(max_iterations):
        # Assign each point to the closest Gaussian mean
        assignments = []
        for point in data:
            distance1 = np.linalg.norm(point - mean1)
            distance2 = np.linalg.norm(point - mean2)
            assignment = 0 if distance1 < distance2 else 1
            assignments.append(assignment)

        # Update the means based on the assigned points
        cluster1 = [data[i] for i in range(len(data)) if assignments[i] == 0]
        cluster2 = [data[i] for i in range(len(data)) if assignments[i] == 1]

        new_mean1 = np.mean(cluster1, axis=0)
        new_mean2 = np.mean(cluster2, axis=0)

        # Check for convergence
        if np.linalg.norm(new_mean1 - mean1) < tol and np.linalg.norm(new_mean2 - mean2) < tol:
            break

        mean1 = new_mean1
        mean2 = new_mean2

    return mean1, mean2
