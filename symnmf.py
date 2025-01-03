import sys
import numpy as np;
import symnmfmodule as sm
np.random.seed(0)

def get_data(file_name):
    """
    read the data from the given filename

    Parameters:
    filename (string): the path to the file

    Returns:
    List: the data in the file
    """

    data = []
    with open(file_name, 'r') as file:
        for line in file:
            row = [float(num) for num in line.strip().split(',')]
            data.append(row)
    return data

def handle_error():
    """
    handle errors according to the instuctions
    """

    print("An Error Has Occurred")
    exit()

def sym(X):
    """
    Calculate and output the similarity matrix as described

    Parameters:
    X (List): A matrix of data

    Returns:
    List: the similarty matrix
    """
    return sm.sym(X, len(X), len(X[0]))

def ddg(X):
    """
    Calculate and output the Diagonal Degree Matrix as described

    Parameters:
    X (List): A matrix of data

    Returns:
    List: the Diagonal Degree Matrix
    """
    return sm.ddg(X, len(X), len(X[0]))

def norm(X):
    """
    Calculate and output the normalized similarity matrix as described

    Parameters:
    X (List): A matrix of data

    Returns:
    List: the normalized similarity matrix
    """
    return sm.norm(X, len(X), len(X[0]))

def symnmf(X, k):
    """
    Perform full the symNMF as described and output H

    Parameters:
    X (List): A matrix of data
    k (int): the number of clusters 

    Returns:
    List: H optimized as described
    """
    # get the normalized similarity matrix 
    W = norm(X)
    #  the average of all entries of W
    m = np.mean(np.array(W))
    # initialize H of size len(X) * k
    init_H = [[2.0 * np.sqrt(m/k) * np.random.uniform() for i in range(k)] for j in range(len(X))]

    return sm.symnmf(init_H, W)

def print_matrix(matrix):
    """
    Prints a matrix formatted to 4 decimal places, separated by commas.

    Parameters:
    matrix (list of lists of float): The matrix to be printed.

    Returns:
    None
    """
    for row in matrix:
        print(",".join(f"{x:.4f}" for x in row))

if __name__ == "__main__":
    goals = ["symnmf", "sym", "norm", "ddg"]

    # check number of arguments
    if(len(sys.argv) != 4):
        handle_error()

    # initialze arguments
    try:
        k = int(sys.argv[1])
        goal = sys.argv[2]
        file_name = sys.argv[3]
    except:
        handle_error()
    
    # check if goal is allowed
    if(goal not in goals):
        handle_error()
    
    # check if the filename is correct
    if(not file_name.endswith(".txt")):
        handle_error()
    
    # get data
    X = get_data(file_name)

    # check if k < N and K > 1
    if(k <= 1 or k >= len(X)):
        handle_error()
    if(goal == "sym"):
        res = sym(X)
    elif(goal == "ddg"):
        res = ddg(X)
    elif(goal == "norm"):
        res = norm(X)
    else:
        res = symnmf(X, k)
    print_matrix(res)




