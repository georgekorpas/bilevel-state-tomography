import numpy as np
from qutip import Qobj

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
    counts = {eigenvalue: np.count_nonzero(outcomes == eigenvalue) for eigenvalue in eigenvalues}

    return counts




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

