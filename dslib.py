# dstools.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numbers

def solve_de(f, t0, x0, tmin, tmax, tdelta=0.01, args=()):
    def fsystem(x, t, *args):
        return np.array([f(t, x[0], *args)])

    tvecp = np.arange(t0, tmax + tdelta, tdelta)
    tvecn = np.arange(t0, tmin - tdelta, -tdelta)
    tvec = np.hstack([tvecn[-1:0:-1], tvecp])
 
    init = np.array([x0])
    xsolp = odeint(fsystem, init, tvecp, args=args)[:,0]
    xsoln = odeint(fsystem, init, tvecn, args=args)[:,0]
    xsol = np.hstack([xsoln[-1:0:-1], xsolp])

    return tvec, xsol


def solve_de_system(f, t0, x0, tmin, tmax, tdelta=0.01, args=()):
    def fsystem(x, t, *args):
        return f(t, x, *args)

    tvecp = np.arange(t0, tmax + tdelta, tdelta)
    tvecn = np.arange(t0, tmin - tdelta, -tdelta)
    tvec = np.hstack([tvecn[-1:0:-1], tvecp])

    xsolp = odeint(fsystem, x0, tvecp, args=args)
    xsoln = odeint(fsystem, x0, tvecn, args=args)
    xsol = np.vstack([xsoln[-1:0:-1], xsolp]).transpose()

    return tvec, xsol


def plot_direction_field(f, hbounds, vbounds, *args, tvalue=0.0, **kw):
    xvalues = np.linspace(*hbounds)
    yvalues = np.linspace(*vbounds)
    xymesh = np.meshgrid(xvalues, yvalues)
    xvalues, yvalues = xymesh
    uvalues, vvalues = f(tvalue, xymesh, *args)
    scale = np.sqrt(uvalues**2+vvalues**2)
    scale[scale == 0] = 1 
    uvalues /= scale
    vvalues /= scale
    return plt.quiver(xvalues, yvalues, uvalues, vvalues, **kw)


def plot_slope_field(f, hbounds, vbounds, **kw):
    def fsystem(t, xyvec):
        x = xyvec[0].copy()
        y = xyvec[1].copy()
        return [1, f(y, x)]
    return plot_direction_field(fsystem, hbounds, vbounds, 
                                headwidth=0, headlength=0.001, 
                                headaxislength=0, pivot='middle', **kw)


def plot_phase_line(f=None, stable=[], unstable=[], xmin=None, xmax=None, param=0,
                    half_stable_l=[], half_stable_r=[], 
                    left=[], right=[], singular=[], markersize=12):
    allpoints = (stable + unstable + 
                 half_stable_l + half_stable_r + 
                 right + left + singular)
    isNone=False
    if xmax is None:
        isNone = True
        xmax = max(allpoints)
    if xmin is None:
        isNone = True
        xmin = min(allpoints)
    w = xmax - xmin
    if isNone:
        xmax += 0.2*w
        xmin -= 0.2*w
    msz = markersize
    plt.axhline(param, color='black', lw=1)
    if f is not None:
        xvalues = np.linspace(xmin, xmax, 300)
        yvalues = f(xvalues)
        plt.plot(xvalues, yvalues, color='blue')
    else:
        plt.axis([xmin, xmax, -0.1*w, 0.1*w])
        plt.gca().set_yticks([])        
    plt.plot(stable, len(stable)*[0], 'o', 
            mfc='red', color='black', markersize=msz, zorder=1000)
    plt.plot(unstable, len(unstable)*[0], 'o', 
            mfc='white', color='black', markersize=msz, zorder=1000)
    plt.plot(half_stable_l, len(half_stable_l)*[0], 'o', 
            mfc='white', color='black', markersize=msz, zorder=1000)
    plt.plot(half_stable_l, len(half_stable_l)*[0], 'o', 
            mfc='red', color='black', markersize=msz, 
            fillstyle='left', zorder=1000)
    plt.plot(half_stable_r, len(half_stable_r)*[0], 'o', 
            mfc='white', color='black', markersize=msz, zorder=1000)
    plt.plot(half_stable_r, len(half_stable_r)*[0], 'o', 
            mfc='red', color='black', markersize=msz, 
            fillstyle='right', zorder=1000)
    plt.plot(right, len(right)*[0], '>', mfc='red', 
            color='black', markersize=msz, zorder=1000)
    plt.plot(left, len(left)*[0], '<', mfc='red', 
            color='black', markersize=msz, zorder=1000)
    plt.plot(singular, len(singular)*[0], 's', mfc='white',
            color='green', markersize=msz, zorder=1000)
    return plt.gca()

def plot_solutions_1d(f, t0=0.0, inits=[0], tmin=-10, tmax=10, fixed_points=[], npoints=300, **kwargs):
    if 'color' not in kwargs:
        kwargs['color'] = 'blue'
    for init in inits:
        t0f = t0
        try:
            t0f, x0f = init
        except TypeError:
            x0f = init
        if not (isinstance(t0f, numbers.Number) 
                and isinstance(x0f, numbers.Number)):
            raise ValueError('initial condition must be a number or a 2-tuple of numbers')
        if tmax>t0f:
            tvalues = np.linspace(t0f, tmax, npoints)
            sol = odeint(f, x0f, tvalues)
            plt.plot(tvalues, sol, **kwargs)
        if tmin < t0f:
            tvalues = np.linspace(t0f, tmin, npoints)
            sol = odeint(f, x0f, tvalues)
            plt.plot(tvalues, sol, **kwargs)
    for fp in fixed_points:
        plt.axhline(fp, **kwargs)
    return plt.gca()




















