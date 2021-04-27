def sphere_function(n_dim, A=1):
    def _sphere_function_(params):
        loss = 0
        for dim in range(n_dim):
            dim_str = "x" + str(dim)
            x = params[dim_str]

            loss += A * x * x

        score = -loss
        return score

    _sphere_function_.__name__ = sphere_function.__name__
    return _sphere_function_
