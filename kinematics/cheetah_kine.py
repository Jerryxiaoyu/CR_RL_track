from kinematics.transform import rotvec2tr,transl
import math
import numpy as np

def fkine(robot, q):
    """
    q = [0.2,0.4,0.5,0.1,0.3,-0.2]
    (T_p1, T_p2) = fkine('cheeta', q)
    
    T0 = transl(0,0,0.6)
    print(T0*T_p1)
    print(T0*T_p2)
    
    :param robot: 'cheeta'
    :param q:   joint vector
    :return:
    """
    if robot == 'cheeta':
        T_bfoot = transl(-0.5, 0, 0) * rotvec2tr(q[0], [0, 1, 0]) * transl(0.16, 0, -0.25) * rotvec2tr(q[1], [0, 1, 0]) \
                  * transl(-0.28, 0, -0.14) * rotvec2tr(q[2], [0, 1, 0]) * transl(0.03, 0, -0.097) * rotvec2tr(-0.27,
                                                                                                               [0, 1,
                                                                                                                0])
    
        T_e1 = rotvec2tr(math.pi, [0, 1, 0]) * transl(0, 0, 0.094)
    
        T_ffoot = transl(0.5, 0, 0) * rotvec2tr(q[3], [0, 1, 0]) * transl(-0.14, 0, -0.24) * rotvec2tr(q[4], [0, 1, 0]) \
                  * transl(0.13, 0, -0.18) * rotvec2tr(q[5], [0, 1, 0]) * transl(0.045, 0, -0.07) * rotvec2tr(-0.6,
                                                                                                              [0, 1, 0])
        T_e2 = rotvec2tr(math.pi, [0, 1, 0]) * transl(0, 0, 0.07)
    
        T_p1 = T_bfoot * T_e1
        T_p2 = T_ffoot * T_e2
        
    return (T_p1, T_p2)

def fkine_pos(com_point, q):
    """
    :param com_point: Root frame origin coordinate e.g: [0,0,0]  x z theta
    :param q:
    :return:
    """
    
    (T_p1, T_p2) = fkine('cheeta', q)
    T0 = transl(0, 0, 0.6)*transl(com_point[0], 0, com_point[1])*  rotvec2tr(com_point[2], [0, 1, 0])
    #T0 = transl(com_point[0], com_point[1], com_point[2])
    # print(T0 * T_p1)
    # print(T0 * T_p2)

    return transl(T0 ), transl(T0*T_p1) , transl(T0*T_p2)



def desired_Body(current_time, switch_time):
    
    qz =[0,0,0,0,0,0]
    
    desired_speed = 0.2    #0.2m/s
    
    
    com_p = [desired_speed*(current_time+switch_time), 0, 0 ]

    com, p1, p2 = fkine_pos(com_p, qz)
    
    return com, p1 ,p2


def desired_Body_pos(pos_x):
    qz = [0, 0, 0, 0, 0, 0]
    
    desired_speed = 0.2  # 0.2m/s
    
    com_p = [pos_x, 0, 0]
    
    com, p1, p2 = fkine_pos(com_p, qz)
    
    return com, p1, p2

def _iner_test():
    import matplotlib.pyplot as plt
    
    t = np.linspace(0, 1000,1000)
    com_P = 0.2  *t
    
    F1,F2 =[],[]
    for i in range(1000):
        p1, p2 = desired_Body(i)
        F1.append(p1 )
        F2.append(p2 )
        
    F1 = np.array(F1).reshape((1000,3))
    F2 = np.array(F2).reshape((1000,3))
  
    plt.plot(t, com_P, label = 'com')
    plt.plot(t, F1[:,0], label= 'F1')
    plt.plot(t, F2[:,0], label='F2')
    plt.legend()
    plt.show()

# qz=[0,0,0,0,0,0]
# print(fkine_pos([0,0,0],qz))
#
# print(rotvec2tr(1, [0,1,0]))