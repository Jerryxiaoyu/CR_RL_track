
from numpy import *


def ishomog(tr):
    """
    True if C{tr} is a 4x4 homogeneous transform.

    @note: Only the dimensions are tested, not whether the rotation submatrix
    is orthonormal.

    @rtype: boolean
    """
    
    return tr.shape == (4, 4)


################ misc support functions

def arg2array(arg):
    """
    Convert a 1-dimensional argument that is either a list, array or matrix to an
    array.

    Useful for functions where the argument might be in any of these formats:::
            func(a)
            func(1,2,3)

            def func(*args):
                if len(args) == 1:
                    v = arg2array(arg[0]);
                elif len(args) == 3:
                    v = arg2array(args);
             .
             .
             .

    @rtype: array
    @return: Array equivalent to C{arg}.
    """
    if isinstance(arg, (matrix, ndarray)):
        s = arg.shape
        if len(s) == 1:
            return array(arg)
        if min(s) == 1:
            return array(arg).flatten()
    
    elif isinstance(arg, list):
        return array(arg)
    
    elif isinstance(arg, (int, float, float32, float64)):
        return array([arg])
    
    raise ValueError

def rotvec2r(theta, v):
    """
    Rotation about arbitrary axis.  Compute a rotation matrix representing
    a rotation of C{theta} about the vector C{v}.

    @type v: 3-vector
    @param v: rotation vector
    @type theta: number
    @param theta: the rotation angle
    @rtype: 3x3 orthonormal matrix
    @return: rotation

    @see: L{rotx}, L{roty}, L{rotz}
    """
    v = arg2array(v);
    ct = cos(theta)
    st = sin(theta)
    vt = 1 - ct
    r = mat([[ct, -v[2] * st, v[1] * st], \
             [v[2] * st, ct, -v[0] * st], \
             [-v[1] * st, v[0] * st, ct]])
    return v * v.T * vt + r
def rotvec2tr(theta, v):
    """
    Rotation about arbitrary axis.  Compute a rotation matrix representing
    a rotation of C{theta} about the vector C{v}.

    @type v: 3-vector
    @param v: rotation vector
    @type theta: number
    @param theta: the rotation angle
    @rtype: 4x4 homogeneous matrix
    @return: rotation

    @see: L{trotx}, L{troty}, L{trotz}
    """
    return r2t(rotvec2r(theta, v))
###################################### translational transform


def transl(x, y=None, z=None):
    """
    Create or decompose translational homogeneous transformations.

    Create a homogeneous transformation
    ===================================

        - T = transl(v)
        - T = transl(vx, vy, vz)

        The transformation is created with a unit rotation submatrix.
        The translational elements are set from elements of v which is
        a list, array or matrix, or from separate passed elements.

    Decompose a homogeneous transformation
    ======================================


        - v = transl(T)

        Return the translation vector
    """
    
    if y == None and z == None:
        x = mat(x)
        try:
            if ishomog(x):
                return x[0:3, 3].reshape(3, 1)
            else:
                return concatenate((concatenate((eye(3), x.reshape(3, 1)), 1), mat([0, 0, 0, 1])))
        except AttributeError:
            n = len(x)
            r = [[], [], []]
            for i in range(n):
                r = concatenate((r, x[i][0:3, 3]), 1)
            return r
    elif y != None and z != None:
        return concatenate((concatenate((eye(3), mat([x, y, z]).T), 1), mat([0, 0, 0, 1])))


def rotx(theta):
    """
    Rotation about X-axis

    @type theta: number
    @param theta: the rotation angle
    @rtype: 3x3 orthonormal matrix
    @return: rotation about X-axis

    @see: L{roty}, L{rotz}, L{rotvec}
    """
    
    ct = cos(theta)
    st = sin(theta)
    return mat([[1, 0, 0],
                [0, ct, -st],
                [0, st, ct]])


def roty(theta):
    """
    Rotation about Y-axis

    @type theta: number
    @param theta: the rotation angle
    @rtype: 3x3 orthonormal matrix
    @return: rotation about Y-axis

    @see: L{rotx}, L{rotz}, L{rotvec}
    """
    
    ct = cos(theta)
    st = sin(theta)
    
    return mat([[ct, 0, st],
                [0, 1, 0],
                [-st, 0, ct]])


def rotz(theta):
    """
    Rotation about Z-axis

    @type theta: number
    @param theta: the rotation angle
    @rtype: 3x3 orthonormal matrix
    @return: rotation about Z-axis

    @see: L{rotx}, L{roty}, L{rotvec}
    """
    
    ct = cos(theta)
    st = sin(theta)
    
    return mat([[ct, -st, 0],
                [st, ct, 0],
                [0, 0, 1]])


def trotx(theta):
    """
    Rotation about X-axis

    @type theta: number
    @param theta: the rotation angle
    @rtype: 4x4 homogeneous matrix
    @return: rotation about X-axis

    @see: L{troty}, L{trotz}, L{rotx}
    """
    return r2t(rotx(theta))


def troty(theta):
    """
    Rotation about Y-axis

    @type theta: number
    @param theta: the rotation angle
    @rtype: 4x4 homogeneous matrix
    @return: rotation about Y-axis

    @see: L{troty}, L{trotz}, L{roty}
    """
    return r2t(roty(theta))


def trotz(theta):
    """
    Rotation about Z-axis

    @type theta: number
    @param theta: the rotation angle
    @rtype: 4x4 homogeneous matrix
    @return: rotation about Z-axis

    @see: L{trotx}, L{troty}, L{rotz}
    """
    return r2t(rotz(theta))


def r2t(R):
    """
    Convert a 3x3 orthonormal rotation matrix to a 4x4 homogeneous transformation::

        T = | R 0 |
            | 0 1 |

    @type R: 3x3 orthonormal rotation matrix
    @param R: the rotation matrix to convert
    @rtype: 4x4 homogeneous matrix
    @return: homogeneous equivalent
    """
    
    return concatenate((concatenate((R, zeros((3, 1))), 1), mat([0, 0, 0, 1])))