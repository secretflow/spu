import jax.numpy as jnp

def generate_ring_sequence(n):
    # init
    upper_row = list(range(0, n, 2))
    lower_row = list(range(1, n+1, 2))
    
    index_pairs = []

    loop = n - (n - 1) % 2
    
    for step in range(loop):
        # record index
        pairs = []
        for i in range((n+1) // 2):
            a, b = upper_row[i], lower_row[i]
            if(max(a,b) < n):
                pairs.append((min(a, b), max(a, b)))

        index_pairs.append(pairs)

        swap = step//2
        upper_row[swap], lower_row[swap] = lower_row[swap], upper_row[swap]

        # right shift
        lower_row = [lower_row[-1]] + lower_row[:-1]
    
    return index_pairs

def serial_jacobi_evd(
        A,           
        J, 		        # rotate matrix，init as jnp.eye(A.shape[0])
        max_jacobi_iter
    ):

    n = A.shape[0]
    eigenvectors = jnp.eye(n)
    
    for _ in range(max_jacobi_iter):
        # Select n/2 pairs of non-diagonal elements that don't share rows/columns
        selected_pairs = generate_ring_sequence(n)

        for pair in selected_pairs:

            # Combine rotation matrices for selected pairs
            ks, ls = zip(*pair)
            k_list = jnp.array(ks)
            l_list = jnp.array(ls)
            mask = jnp.not_equal(A[k_list, l_list], 0)
            diff = A[l_list, l_list] - A[k_list, k_list]

            '''
            # fitting the rotate matrix elements by sqrt & rsqrt (derived from the trigonometric functions)
            # But the fitting accuracy is lower than the trigonometric method
            
            # cos2theta = jnp.where(mask, diff / jnp.sqrt((4 * A[k_list, l_list]**2 + diff **2) ), 0)
            # cos_squrare = 0.5 * (1 + cos2theta)
            # sin_squrare = 0.5 * (1 - cos2theta)
            # combined_squares = jnp.stack([cos_squrare, sin_squrare], axis=0)
            # sqrt_combined = jnp.sqrt(combined_squares)
            # c = sqrt_combined[0]
            # s = sqrt_combined[1] * jnp.sign(A[k_list, l_list])
            '''

            # trigonometric method
            theta = jnp.where(mask, 0.5 * jnp.arctan2(2 * A[k_list, l_list],  diff ), 0)
            theta_cosine = 0.5 * jnp.pi - theta
            combined_theta = jnp.stack([theta_cosine, theta], axis=0)
            sin_combined = jnp.sin(combined_theta)
            c = sin_combined[0]
            s = sin_combined[1]
                  
            J_combined = J.copy()
            for i in range(len(pair)):
                if(mask.at[i]):
                    k, l = ks[i], ls[i]
                    J_combined = J_combined.at[k, k].set(c[i]).at[l, l].set(c[i]).at[k, l].set(s[i]).at[l, k].set(-s[i])

            # Update A and eigenvectors using the rotation matrix
            A = jnp.dot(J_combined.T, jnp.dot(A, J_combined))
            eigenvectors = jnp.dot(eigenvectors, J_combined) 

            '''
            # Update the Matrix A with the mapping of the chosen rotation matrix and using the hardmard product instead of matrix multiplication
            # This method will introduce more communication overhead (6n^2) and communication round
            
            # r = np.arange(n)
            # c_r = [1.0]*n
            # s_r = [0.0]*n

            #     if(mask.at[i]):
            #         k, l = ks[i], ls[i]
            #         c_r[k] = c[i]
            #         c_r[l] = c[i]
            #         s_r[l] = s[i]
            #         s_r[k] = -s[i]
    
            # c_list = jnp.array(c_r)
            # s_list = jnp.array(s_r)

            # # Update matrix A and eigenvectors using hardmard product
            # A_row = (c_list * s_list[:,None]) * A[r,:]
            # A = (c_list * c_list[:,None]) * A + A_row + A_row.T + (s_list * s_list[:,None]) * A[r,:][:,r]
            # eigenvectors = c_list * eigenvectors + s_list * eigenvectors[:,r]
            '''
            
    eigenvalues = jnp.diag(A)
    return eigenvalues, eigenvectors