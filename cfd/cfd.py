#!/usr/bin/env python3

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pandas

global h, mu, rho, p, ma, particle_ids, particle_positions, particle_velocities


def calculate_accelerations(vi, xi, ei):
    global ma, rho, p, mu

    c = 15 / (7 * np.pi * h ** 3)  # smoothing function constant

    acc = np.zeros((2, 1))  # 2D problem
    for alpha in range(2):
        for beta in range(2):
            term_1 = 0
            term_2 = 0
            for j_particle_id in particle_ids:
                vj = np.array([particle_velocities[str(j_particle_id)]]).T
                xj = np.array([particle_positions[str(j_particle_id)]]).T

                # if xi = xj, particles are in same position (same particle) and will cause divide by zero
                if (xi == xj).all():
                    continue

                # need to check location condition of particle j to particle i
                R = get_R(xi, xj)
                if 0 <= R < 1:
                    del_W = (c * (xi - xj) / np.linalg.norm((xi - xj)) * (3 * R ** 2 / 2 - 2 * R))
                elif 1 <= R < 2:
                    del_W = (c * (xi - xj) / np.linalg.norm((xi - xj)) * -(2 - R) ** 2 / 2)
                else:
                    continue  # no contribution

                term_1 += ma * (2*p/rho**2) * del_W[alpha, 0].item(0)
                ej = get_strain_rate(vj, xj)
                term_2 += ma * (mu * ei[alpha, beta]/rho**2 + mu * ej[alpha, beta]/rho**2) * del_W[beta, 0].item(0)

            acc[alpha, 0] = -term_1 + term_2

    return acc


def get_strain_rate(vi, xi):
    global ma, rho

    c = 15 / (7 * np.pi * h ** 3)  # smoothing function constant

    e = np.zeros((2, 2))  # 2D problem
    for alpha in range(2):
        for beta in range(2):
            for gamma in range(2):
                term_1 = 0
                term_2 = 0
                term_3 = 0
                for j_particle_id in particle_ids:
                    vj = np.array([particle_velocities[str(j_particle_id)]]).T
                    xj = np.array([particle_positions[str(j_particle_id)]]).T

                    # if xi = xj, particles are in same position (same particle) and will cause divide by zero
                    if (xi == xj).all():
                        continue

                    # need to check location condition of particle j to particle i
                    R = get_R(xi, xj)
                    if 0 <= R < 1:
                        del_W = (c * (xi - xj) / np.linalg.norm((xi - xj)) * (3 * R ** 2 / 2 - 2 * R))
                    elif 1 <= R < 2:
                        del_W = (c * (xi - xj) / np.linalg.norm((xi - xj)) * -(2 - R) ** 2 / 2)
                    else:
                        continue  # no contribution

                    term_1 += ma/rho * (vj[alpha, 0].item(0) - vi[alpha, 0].item(0)) * del_W[alpha, 0].item(0)
                    term_2 += ma/rho * (vj[alpha, 0].item(0) - vi[alpha, 0].item(0)) * del_W[beta, 0].item(0)
                    if alpha == beta:  # kronecker delta
                        term_3 += ma/rho * (vj[gamma, 0].item(0) - vi[gamma, 0].item(0)) * del_W[gamma, 0].item(0)
                    else:
                        continue  # no contribution
                e[alpha, beta] = term_1 + term_2 - (2/3 * term_3)
    return e


def calculate_velocity_gradient(vi, xi):
    # arguments assumed to be numpy arrays, could add check for completeness

    global h

    c = 15/(7*np.pi*h**3)  # smoothing function constant
    del_v = np.zeros((2, 2))
    for j_particle_id in particle_ids:
        vj = np.array([particle_velocities[str(j_particle_id)]]).T
        xj = np.array([particle_positions[str(j_particle_id)]]).T

        # if xi = xj, particles are in same position (same particle) and will cause divide by zero
        if (xi == xj).all():
            continue

        # need to check location condition of particle j to particle i
        R = get_R(xi, xj)
        if 0 <= R < 1:
            W = c * h * (R**3/2 - R**2 - (2/3))
            del_W = (c * (xi - xj) / np.linalg.norm((xi - xj)) * (3 * R ** 2 / 2 - 2 * R))
        elif 1 <= R < 2:
            W = c * h * ((2 - R)**2/6)
            del_W = (c * (xi - xj) / np.linalg.norm((xi - xj)) * -(2 - R)**2/2)
        else:
            continue  # no contribution
        del_v += np.outer((ma * (vj - vi)), del_W.T)

    # density of fluid particle is set
    return 1/rho * del_v


def get_R(xi, xj):
    # arguments assumed to be numpy arrays, could add check for completeness

    global h

    return np.linalg.norm((xi - xj)) / h


def main():

    global h, mu, rho, p, ma, particle_ids, particle_positions, particle_velocities

    # fluid params
    mu = 0.1  # viscosity
    rho = 1.0  # density
    p = 10  # pressures
    ma = 1  # mass, assuming 1

    # fluid discretization
    num_particles = 100
    m = 10  # number rows
    n = 10  # number columns
    h = 1.3

    # PART (1): Compute velocity gradient =========================================================

    particle_ids = np.linspace(1, num_particles, num_particles)
    x_poses = np.zeros((1, num_particles))
    y_poses = np.zeros((1, num_particles))

    # create a matrix of particle ids
    particle_field = np.zeros((m, n))
    idp = 0
    for i in range(m):
        for j in range(n):
            particle_field[i][j] = particle_ids[idp]
            idp += 1
    #print(particle_field)

    particle_positions = dict()
    for i, particle_id in enumerate(particle_ids):
        position = np.where(particle_field == particle_id)
        # for plotting
        x_poses[0, i] = position[0]
        y_poses[0, i] = position[1]
        # add to dictionary
        x = position[0]
        y = position[1]
        particle_positions[str(particle_id)] = [x.item(0), y.item(0)]
    #print(particle_positions)

    particle_velocities = dict()
    for particle_id in particle_ids:
        x = particle_positions[str(particle_id)][0]
        m = x + 1
        y = particle_positions[str(particle_id)][1]
        n = y + 1
        velocity = np.array([-m * np.exp(-(m ** 2 + n ** 2)), -n * np.exp(-(m ** 2 + n ** 2))])
        # add to dictionary
        vx = velocity[0]
        vy = velocity[1]
        particle_velocities[str(particle_id)] = [vx.item(0), vy.item(0)]
    #print(particle_velocities)

    particle_vel_gradients = dict()
    # calculate velocity gradient for each particle
    for i_particle_id in particle_ids:
        vi = particle_velocities[str(i_particle_id)]
        xi = particle_positions[str(i_particle_id)]
        # function will calculate velocity gradient of particle i by looping through all j particles in vicinity
        del_v = calculate_velocity_gradient(np.array([vi]).T, np.array([xi]).T)
        # store in dictionary as list
        particle_vel_gradients[str(i_particle_id)] = [del_v[0].item(0), del_v[0].item(1),
                                                      del_v[1].item(0), del_v[1].item(1)]  # row order
    # output data into text file
    data = particle_vel_gradients
    headers = ["dv1dx1", "dv1dx2", "dv2dx1", "dv2dx2"]
    pandas.set_option("display.max_rows", None, "display.max_columns", None)
    f = open("velocityGradients.txt", "w")
    f.write(str(pandas.DataFrame(data, headers)))
    f.close()

    # plot velocity gradient field, x direction
    # looping through dictionary to separate U and V for plotting
    U_x = np.zeros((1, num_particles))
    V_x = np.zeros((1, num_particles))
    for i, i_particle_id in enumerate(particle_ids):
        U_x[0, i] = particle_vel_gradients[str(i_particle_id)][0]
        V_x[0, i] = particle_vel_gradients[str(i_particle_id)][2]

    fig, ax = plt.subplots()
    q = ax.quiver(x_poses, y_poses, U_x, V_x, units='xy', scale=0.01)
    plt.grid()
    ax.set_aspect('equal')
    plt.xlim(-2.5, 10)
    plt.ylim(-2.5, 10)
    plt.title('X Velocity gradient, numerical')
    plt.show()

    # plot velocity gradient field, y direction
    # looping through dictionary to separate U and V for plotting
    U_y = np.zeros((1, num_particles))
    V_y = np.zeros((1, num_particles))
    for i, i_particle_id in enumerate(particle_ids):
        U_y[0, i] = particle_vel_gradients[str(i_particle_id)][1]
        V_y[0, i] = particle_vel_gradients[str(i_particle_id)][3]

    fig, ax = plt.subplots()
    q = ax.quiver(x_poses, y_poses, U_y, V_y, units='xy', scale=0.01)
    plt.grid()
    ax.set_aspect('equal')
    plt.xlim(-2.5, 10)
    plt.ylim(-2.5, 10)
    plt.title('Y Velocity gradient, numerical')
    plt.show()

    # PART (2): Compare with analytical ===========================================================

    # now check with analytical (x=m-1, y=n-1)
    # v1_x = (2*x**2 + 4*x + 1) * exp(-((x+1)**2+(y+1)**2))
    # v1_y = 2*(x+1)*(y+1) * exp(-((x+1)**2+(y+1)**2))
    # v2_x = 2*(x+1)*(y+1) * exp(-((x+1)**2+(y+1)**2))
    # v2_y = (2*y**2 + 4*y + 1) * exp(-((x+1)**2+(y+1)**2))
    v1_x = np.zeros((1, num_particles))
    v1_y = np.zeros((1, num_particles))
    v2_x = np.zeros((1, num_particles))
    v2_y = np.zeros((1, num_particles))
    particle_vel_gradients_analytical = dict()
    for i, particle_id in enumerate(particle_ids):
        x = particle_positions[str(particle_id)][0]
        y = particle_positions[str(particle_id)][1]
        v1_x[0, i] = (2*x**2 + 4*x + 1) * np.exp(-((x+1)**2+(y+1)**2))
        v1_y[0, i] = 2*(x+1)*(y+1) * np.exp(-((x+1)**2+(y+1)**2))
        v2_x[0, i] = 2*(x+1)*(y+1) * np.exp(-((x+1)**2+(y+1)**2))
        v2_y[0, i] = (2*y**2 + 4*y + 1) * np.exp(-((x+1)**2+(y+1)**2))
        particle_vel_gradients_analytical[str(particle_id)] = [v1_x[0, i].item(0), v1_y[0, i].item(0),
                                                      v2_x[0, i].item(0), v2_y[0, i].item(0)]

    # output data as a text file
    data = particle_vel_gradients_analytical
    headers = ["dv1dx1", "dv1dx2", "dv2dx1", "dv2dx2"]
    f = open("velocityGradientsAnalytical.txt", "w")
    f.write(str(pandas.DataFrame(data, headers)))
    f.close()


    # analytical: plot velocity gradient field, x direction
    # looping through dictionary to separate U and V for plotting
    fig, ax = plt.subplots()
    q = ax.quiver(x_poses, y_poses, v1_x, v2_x, units='xy', scale=0.01)
    plt.grid()
    ax.set_aspect('equal')
    plt.xlim(-2.5, 10)
    plt.ylim(-2.5, 10)
    plt.title('X Velocity gradient, analytical')
    plt.show()

    # analytical: plot velocity gradient field, y direction
    # looping through dictionary to separate U and V for plotting
    fig, ax = plt.subplots()
    q = ax.quiver(x_poses, y_poses, v1_y, v2_y, units='xy', scale=0.01)
    plt.grid()
    ax.set_aspect('equal')
    plt.xlim(-2.5, 10)
    plt.ylim(-2.5, 10)
    plt.title('Y Velocity gradient, analytical')
    plt.show()

    # PART (4): Calculate time derivative =========================================================

    particle_accelerations = dict()
    # calculate velocity gradient for each particle
    for i_particle_id in particle_ids:
        vi = particle_velocities[str(i_particle_id)]
        xi = particle_positions[str(i_particle_id)]
        # function will calculate velocity gradient of particle i by looping through all j particles in vicinity
        ei = get_strain_rate(np.array([vi]).T, np.array([xi]).T)
        v_dot = calculate_accelerations(np.array([vi]).T, np.array([xi]).T, ei)
        # # store in dictionary as list
        particle_accelerations[str(i_particle_id)] = [v_dot[0].item(0), v_dot[1].item(0)]
    #print(particle_accelerations)

    # output data as a text file
    data = particle_accelerations
    headers = ["dv1dt", "dv2dt"]
    f = open("accelerations.txt", "w")
    f.write(str(pandas.DataFrame(data, headers)))
    f.close()

    # Plotting particle positions =================================================================

    # # plot positions on grid for sanity check
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(x_poses, y_poses, 'o', color='blue')
    id = 1
    for i in range(100):
        ax.annotate('%s' % id, xy=(x_poses[0, i], y_poses[0, i]), textcoords='data')
        id += 1
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
