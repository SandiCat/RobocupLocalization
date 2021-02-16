import numpy as np
import scipy.optimize


def localize_ball(x, y, radius, ball_radius, focal_length):
    """
    Does geometry to find the center of the ball given its projection to the camera, i.e. the fitted circle
    :param x: x coordinate of the circle
    :param y: y coordinate of the circle
    :param radius: radius of the circle
    :param ball_radius: radius of the real-life ball
    :param focal_length: focal length of the camera
    :return: a numpy vector of the center of the real-life ball **in the camera's coordinate system**
    """
    # the y axis is flipped in camera coordinates
    y = -y

    # find three points on the circle
    center = (x, y)
    circle_points = [
        (0, -radius),
        (radius * np.cos(np.pi/6), radius * np.sin(np.pi/6)),
        (radius * np.cos(5/6 * np.pi), radius * np.sin(5/6 * np.pi)),
    ]
    npoints = 3

    # calculate the direction vectors of the camera rays that contain all the possible projected points
    # the formula is derived by reversing the pinhole camera model, intersecting two planes and calculating the
    # cross product of their normals
    line_directions = [np.array([u * focal_length, -v * focal_length, focal_length ** 2]) for (u, v) in circle_points]

    # solve a nonlinear equation to find the center of the sphere
    def unpack(x):
        # pull out the intersection points of the rays and the sphere and the center of the sphere from the root
        p = [x[i * 3:i * 3 + 3] for i in range(npoints)]
        s = x[-3:]
        return p, s

    def equation(x):
        p, s = unpack(x)
        print(p, s)

        ret = np.ravel(np.array([
            [
                # distance from the line (easier than checking if the point is on the line and better for the solver)
                np.linalg.norm(np.cross(p[i], line_directions[i])) / np.linalg.norm(line_directions[i]),
                # distance between the intersection point and the center of the ball should be its radius
                np.linalg.norm(p[i] - s) - ball_radius,
                # the direction of the ray and the radius vector to the intersection point should be orthogonal
                np.dot(s - p[i], line_directions[i])
            ]
            for i in range(npoints)
        ]))
        print(ret)
        return ret

    x0 = scipy.optimize.broyden1(equation, 100*np.random.random(npoints*3+3) - 50)
    _, s = unpack(x0)

    # flip the point if it's behind the camera projection plane (there are two solutions and we need that one)
    if s[2] < 0:
        s = -s

    return s


def camera_coords_to_world_coords(point, cam_height, cam_angle):
    """
    Transforms from a coordinate system with origin at the camera's aperture (see pinhole camera model on wikipedia)
    to a coordinate system aligned with the plane that the camera is pointed towards. This coordinate system will rotate
    as the camera pans. It is assumed that the camera's position is the position of the aperture. The axis order doesn't
    follow the same convention as the camera coordinate system, but instead the standard convention with the z axis
    pointing up and the x axis in the direction of the camera.
    :param point: point to transform, in the camera's coordinate system
    :param cam_height: how high the aperture of the camera is from the plane
    :param cam_angle: the angle of the direction vector of the camera and the plane, between 0 and pi/2, exclusive
    :return: transformed point
    """

    # adjust the axis order
    point = np.array([point[2], point[0], point[1]])

    # calculate the vectors of the camera axis in the desired coordinate system
    cam_direction = np.array([np.cos(cam_angle), 0, -np.sin(cam_angle)])
    z = cam_direction
    x = np.cross(np.array([0, 0, 1]), cam_direction)
    y = np.cross(z, x)

    # transposed rotation matrix
    rotation = np.vstack([x, y, z])

    # translation vector
    translation = np.array([0, 0, cam_height])

    return rotation @ (point - translation)
