import numpy as np

def lm_dot_2B_propul(F, G, H, K, L, P, T, g0, l_F, l_G, l_H, l_K, l_L, l_P, m, m0, mu):
    # Constants
    t2 = 1.0 / mu
    t3 = P * t2
    t4 = np.sqrt(t3)
    t5 = np.conj(t4)  # In Python, np.conj is used, but it has no effect on real numbers
    t6 = np.sin(L)
    t7 = np.cos(L)
    t8 = H * t6
    t9 = F * t7
    t10 = G * t6
    t11 = t9 + t10 + 1.0
    t12 = 1.0 / t11
    t32 = K * t7
    t13 = t8 - t32
    t14 = F * t7 * 2.0
    t15 = G * t6 * 2.0
    t16 = t14 + t15 + 2.0
    t17 = 1.0 / t16
    t18 = H**2
    t19 = K**2
    t20 = t18 + t19 + 1.0
    t33 = l_H * t5 * t7 * t17 * t20
    t34 = G * l_F * t5 * t12 * t13
    t35 = l_K * t5 * t6 * t17 * t20
    t21 = np.abs(t33 - t34 + t35 + l_L * t5 * t12 * t13 + F * l_G * t5 * t12 * t13)
    t22 = l_G * t5 * t7
    t37 = l_F * t5 * t6
    t23 = t22 - t37
    t24 = np.abs(t23)
    t29 = F + t7
    t30 = t12 * t29
    t31 = t7 + t30
    t39 = l_F * t5 * t31
    t40 = G + t6
    t41 = t12 * t40
    t42 = t6 + t41
    t43 = l_G * t5 * t42
    t44 = P * l_P * t5 * t12 * 2.0
    t45 = t39 + t43 + t44
    t25 = np.abs(t45)
    t26 = 1.0 / g0
    t27 = 1.0 / m**2
    t28 = 1.0 / m0
    t36 = np.abs(t33 - t34 + t35 + l_L * t5 * t12 * (t8 - t32) + F * l_G * t5 * t12 * (t8 - t32))
    t38 = t24**2
    t46 = t25**2
    t47 = -t8 + t32
    t49 = l_L * t5 * t12 * t47
    t50 = F * l_G * t5 * t12 * t47
    t51 = G * l_F * t5 * t12 * t47
    t52 = t33 + t35 - t49 - t50 + t51
    t48 = np.abs(t52)
    t53 = t48**2
    t54 = t38 + t46 + t53
    t55 = 1.0 / np.sqrt(t54)

    lam_m_dot = (
        l_G * (
            T * t4 * t7 * t23 * t26 * t27 * t28 * t55 +
            T * t4 * t26 * t27 * t28 * t42 * t45 * t55 -
            F * T * t4 * t12 * t26 * t27 * t28 * t47 * t52 * t55
        ) +
        l_F * (
            -T * t4 * t6 * t23 * t26 * t27 * t28 * (1.0 / np.sqrt(t38 + t46 + t21**2)) +
            T * t4 * t26 * t27 * t28 * t31 * t45 * (1.0 / np.sqrt(t38 + t46 + t36**2)) +
            G * T * t4 * t12 * t26 * t27 * t28 * t47 * t52 * t55
        ) +
        P * T * l_P * t4 * t12 * t26 * t27 * t28 * t45 * t55 * 2.0 -
        T * l_L * t4 * t12 * t26 * t27 * t28 * t47 * t52 * t55 +
        T * l_H * t4 * t7 * t17 * t20 * t26 * t27 * t28 * t52 * t55 +
        T * l_K * t4 * t6 * t17 * t20 * t26 * t27 * t28 * t52 * t55
    )
    
    return lam_m_dot
