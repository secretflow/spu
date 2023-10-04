import jax
import jax.numpy as jnp
import time

INF = float('inf')

class SMO():

    def __init__(self,
                 size,
                 C: float,
                 tol: float = 1e-3,
                 eps: float = 1e-5,
                 tau: float = 1e-12) -> None:
        self.size = size
        self.C = C
        self.tol = tol
        self.tau = tau
        self.Cs = jnp.array([self.C]*size)
        self.zeros = jnp.array([0]*size)
        '''[label: pass plain parameters]
        There is a tip here.
        In spu, the parameters in plain are also need to pass in the function parameters,
        e.g. the index i,j of working_set_select are opened, but should pass them in the function parameters.
        Do not use this: define `self.i` and `self.j` here then update `smo.i, smo.j = i, j`.
        That will be incorrect.
        '''
    def working_set_select_i(self, alpha, y, neg_y_grad):

        alpha_lower_C, alpha_upper_0, y_lower_0, y_upper_0 = jnp.array([self.Cs,alpha,self.zeros,y]) > jnp.array([alpha,self.zeros,y,self.zeros])
        Zup = (alpha_lower_C & y_upper_0) | (alpha_upper_0 & y_lower_0)
        Mup = (1-Zup)*jnp.min(neg_y_grad)
        i = jnp.argmax(neg_y_grad * Zup + Mup)
        return i


    def working_set_select_j(self, i, alpha, y, neg_y_grad, Q):
        alpha_lower_C, alpha_upper_0, y_lower_0, y_upper_0 = jnp.array([self.Cs,alpha,self.zeros,y]) > jnp.array([alpha,self.zeros,y,self.zeros])
        Zlow = (alpha_lower_C & y_lower_0) | (alpha_upper_0 & y_upper_0)

        m = neg_y_grad[i]

        Zlow_m = Zlow & (neg_y_grad < m)
        Qi = Q[i]
        Qj = Q.diagonal()
        quad_coef = Qi[i] + Qj - 2 * Q[i]
        quad_coef = (quad_coef > 0)*quad_coef + (1-(quad_coef > 0))*self.tau
        Ft = -(m - neg_y_grad)**2 / (quad_coef)
        Mlow_m = (1-Zlow_m)*jnp.max(Ft)
        j = jnp.argmin(Ft*Zlow_m + Mlow_m)
        return j
        
       
    def working_set_select(self,alpha,y,neg_y_grad,Q):
        alpha_lower_C, alpha_upper_0, y_lower_0, y_upper_0 = jnp.array([self.Cs,alpha,self.zeros,y]) > jnp.array([alpha,self.zeros,y,self.zeros])
        Zup = (alpha_lower_C & y_upper_0) | (alpha_upper_0 & y_lower_0)
        Mup = (1-Zup)*jnp.min(neg_y_grad)
        i = jnp.argmax(neg_y_grad * Zup + Mup)
        Zlow = (alpha_lower_C & y_lower_0) | (alpha_upper_0 & y_upper_0)
        m = neg_y_grad[i]
        Zlow_m = Zlow & (neg_y_grad < m)
        Qi = Q[i]
        Qj = Q.diagonal()
        quad_coef = Qi[i] + Qj - 2 * Q[i]
        quad_coef = (quad_coef > 0)*quad_coef + (1-(quad_coef > 0))*self.tau
        Ft = -(m - neg_y_grad)**2 / (quad_coef)
        Mlow_m = (1-Zlow_m)*jnp.max(Ft)
        j = jnp.argmin(Ft*Zlow_m + Mlow_m)
        return i,j
        

    def update(self, i, j, Q, y, alpha, neg_y_grad):
        Qi, Qj = Q[i], Q[j]
        yi, yj = y[i], y[j]
        alpha_i, alpha_j = alpha[i]+0, alpha[j]+0
        alpha_i0, alpha_j0 = alpha_i+0, alpha_j+0

        quad_coef = Qi[i] + Qj[j] - 2 * yi * yj * Qi[j]
        quad_coef = (quad_coef > 0)*quad_coef + (1-(quad_coef > 0))*self.tau

        yi_mul_yj = yi*yj
        yi_neq_yj = yi != yj
        n_yi_neq_yj = 1 - yi_neq_yj

        delta = (- yi_mul_yj * neg_y_grad[i] * yi + neg_y_grad[j] * yj) / quad_coef
        diff_sum = alpha_i + yi_mul_yj*alpha_j
        alpha_i = alpha_i + (-1*yi_mul_yj*delta)
        alpha_j = alpha_j + delta

        # first cal
        diff_sum_upper_0, diff_sum_upper_C, alpha_i_lower_0, alpha_j_lower_0, alpha_i_upper_C = jnp.array([diff_sum, diff_sum, 0, 0, alpha_i]) > jnp.array([0, self.C, alpha_i, alpha_j, self.C])
        outer = jnp.array([yi_neq_yj,yi_neq_yj,1-yi_neq_yj,1-yi_neq_yj]) * jnp.array([diff_sum_upper_0,1-diff_sum_upper_0,diff_sum_upper_C,1-diff_sum_upper_C])
        update_condition = jnp.array([alpha_j_lower_0,alpha_i_lower_0,alpha_i_upper_C,alpha_j_lower_0]*2)
        update_from = jnp.array([alpha_i,alpha_i,alpha_i,alpha_i,alpha_j,alpha_j,alpha_j,alpha_j])
        update_to = jnp.array([diff_sum,0,self.C,diff_sum,0,-diff_sum,diff_sum-self.C,0])
        inner = (update_from + update_condition*(update_to-update_from)).reshape(2,-1)
        alpha_i,alpha_j = jnp.dot(inner,outer.T)
        
        
        # second cal
        alpha_i_lower_0,alpha_i_upper_C,alpha_j_upper_C = jnp.array([0, alpha_i, alpha_j]) > jnp.array([alpha_i, self.C, self.C])
        update_condition = jnp.array([alpha_i_upper_C,alpha_j_upper_C,alpha_j_upper_C,alpha_i_lower_0]*2)
        update_from = jnp.array([alpha_i,alpha_i,alpha_i,alpha_i,alpha_j,alpha_j,alpha_j,alpha_j])
        update_to = jnp.array([self.C,self.C+diff_sum,diff_sum-self.C,0,self.C-diff_sum,self.C,self.C,diff_sum])
        inner = (update_from + update_condition*(update_to-update_from)).reshape(2,-1)
        alpha_i,alpha_j = jnp.dot(inner,outer.T)

        delta_i = alpha_i - alpha_i0
        delta_j = alpha_j - alpha_j0
        
        neg_y_grad = neg_y_grad - y*(jnp.dot(jnp.array([delta_i,delta_j]),jnp.array([Q[i],Q[j]])))
        alpha = alpha.at[jnp.array([i,j])].set(jnp.array([alpha_i,alpha_j]))     

        return neg_y_grad,alpha


    def calculate_b(self,alpha,neg_y_grad,y) -> float:
       
        alpha_lower_C = alpha < self.C- self.tol
        alpha_equal_C = jnp.abs(alpha - self.C)<self.tol
        alpha_equal_0 = jnp.abs(alpha)<self.tol
        alpha_upper_0 = alpha > 0
        y_lower_0 = y < 0
        y_upper_0 = y > 0

        alpha_upper_0_and_lower_C = alpha_upper_0 & alpha_lower_C
        sv_sum = jnp.sum(alpha_upper_0_and_lower_C)

        rho_0 = -1 * (neg_y_grad*alpha_upper_0_and_lower_C).sum() / sv_sum

        Zub = (alpha_equal_0 & y_lower_0) | (alpha_equal_C & y_upper_0)
        Zlb = (alpha_equal_0 & y_upper_0) | (alpha_equal_C & y_lower_0)
        rho_1 = -((neg_y_grad*Zub).min() + (neg_y_grad*Zlb).max()) / 2

        rho = (sv_sum>0)*rho_0 + (1-(sv_sum>0))*rho_1

        b = -1*rho
        return b