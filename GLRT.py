from galahad import glrt
import numpy as np
import torch
import time

def cubic_f(fk, gk, hk, y, M, order = "01"):
    """
    Compute the cubic regularized function value and gradient at point y.
    
    Parameters:
    y (torch.Tensor): The point at which to evaluate the function.
    M (float): The cubic regularization parameter.
    order (str): The order of derivatives to compute ("0" for function value, "01" for function value and gradient).
    
    Returns:
    If order == "0": returns the function value at y.
    If order == "01": returns a tuple (function value, gradient) at y.
    """
    norm_y = torch.norm(y)
    hk_y = torch.mv(hk, y)
    cf = fk + torch.dot(gk, y) + torch.dot(hk_y, y) / 2 + M * (norm_y ** 3) / 6
    if "0" == order: 
        return cf
    if "01" == order:
        cg = gk + hk_y + M * norm_y * y / 2
        return cf, cg
    if "012" == order:
        cg = gk + hk_y + M * norm_y * y / 2
        ch = hk + M * (torch.outer(y, y) / (2 * norm_y) + norm_y * torch.eye(len(y)) / 2) 
        return cf, cg, ch

if __name__ == "__main__":
    np.set_printoptions(precision=4,suppress=True,floatmode='fixed')
    print("\n** python test: glrt")

    start = time.time()

    power = 3.0
    n = 100
    fk = torch.randn(1, dtype = torch.float64)
    gk = torch.randn(n, dtype = torch.float64)
    hk = torch.randn(n,n, dtype = torch.float64)
    hk = hk.T @ hk
    M = 1e3
    NAME = "GLRT_test"
    v = torch.zeros(n, dtype = torch.float64)

    # set default options
    options = glrt.initialize()

    # set some non-default options
    options['unitm'] = False
    options['print_level'] = 0
    #print("options:", options)

    # load non-default options
    glrt.load_options(options)

    # first problem

    print(" solve problem 1")
    status = 1
    weight = 1e3
    hvk = torch.mv(hk, v)
    r = gk.numpy()
    v = v.numpy()
    hvk = hvk.numpy()
    i = 1
    res = {}

    while status > 0:
        status, x, r, v = glrt.solve_problem(status, n, power, weight, r, v)
        inform = glrt.information()
        res[i] = {"inform": inform, "x": x}
        i += 1
        if status == 2: # precondition v -> v/2
            for i in range(n):
                v[i] = v[i] / 2.0;
        elif status == 3: # form v -> H v
            hvk[0] = 2.0 * v[0] + v[1]
            for i in range(1,n-1):
                hvk[i] = v[i-1] + 2.0 * v[i] + v[i+1]
            hvk[n-1] = v[n-2] + 2.0 * v[n-1]
            for i in range(n):
                v[i] = hvk[i]
        elif status == 4: # reset r
            r = gk.numpy()

    # get information

    end = time.time()
    print(" time taken:", end-start)
    print("Results:", res)
    with open("Results/glrt"+NAME+".json", "w") as f:
        f.write(f"Results: {res}\n")
        f.write(f"Time taken: {end - start}\n")
