import queue

import numpy as np
import random
from svgpathtools import Line,Path
from scipy.interpolate import splprep,insert
import time


a = 1

def c(inner_radius, outer_radius, alpha, n, width_factor_s, width_factor_f):
    """
    Generates points for left and right sides of a logarithmic spiral with widths as specified based on given parameters.

    Parameters:
    inner_radius (float): The inner radius of the shape.
    outer_radius (float): The outer radius of the shape.
    alpha (float): The angle in degrees.
    n (int): The number of segments.
    width_factor_s (float): The width factor at the start.
    width_factor_f (float): The width factor at the end.

    Returns:
    tuple: Two lists of points for the left and right sides of the shape.
    """
    points_left = []
    points_right = []
    dir = np.sign(alpha) if alpha != 0 else 1
    alpha = abs(alpha)
    k = np.tan(np.radians(90 - alpha))
    b = b_start = np.log(inner_radius / a) / k
    b_end = np.log(outer_radius / a) / k
    width = np.pi / (2 * n)

    while b < b_end:
        t = (b - b_start) / (b_end - b_start)
        offset = width * (width_factor_s * (1 - t) + width_factor_f * t)
        r = a * np.exp(k * b)
        points_left.append((r * np.cos(b + offset - b_start), dir * (r * np.sin(b + offset - b_start))))
        points_right.append((r * np.cos(b - offset - b_start), dir * (r * np.sin(b - offset - b_start))))
        b += (np.log(np.sin(np.radians(90 - alpha)) / r + 1)) / (2*k)

    return points_left, points_right


def close_shape(model, type, left, right):
    """
    Closes a shape by connecting the left and right points and adds it to the model.

    Parameters:
    model (object): The model to add the shape to (DXF or SVG).
    type (str): The type of model ("dxf" or "svg").
    left (list): The list of points for the left side.
    right (list): The list of points for the right side.

    Returns:
    None
    """
    total_path = [(p.real, p.imag) if isinstance(p, complex) else p for p in left + list(reversed(right))]

    if type == 'svg':
        path = model.path(stroke='none', fill='black')
        path.push('M', total_path[0])  # Move to the first point
        for p in total_path[1:]:
            path.push('L', p)  # Draw a line segment to the next point
        path.push('L', total_path[0])  # Close the path
        model.add(path)
    elif type == 'dxf':
        hatch = model.add_hatch(color=7, dxfattribs={'hatch_style': 1})  # Use hatch style 1 for solid fill
        hatch.paths.add_polyline_path(total_path, is_closed=True, flags=1)  # Set flag for external path


def rotate_and_offset_bboxes(bboxes, theta, offset_x, offset_y):
    """
    Rotates and offsets the coordinates of multiple bounding boxes.

    Parameters:
    bboxes (list of lists of tuples): A list of bounding boxes, each defined by four corner points.
    theta (float): The angle to rotate the bounding boxes.
    offset_x (float): The x offset to apply after rotation.
    offset_y (float): The y offset to apply after rotation.

    Returns:
    list of lists: A list of rotated and offset bounding box coordinates.
    """
    rotated_bboxes = []

    for bbox in bboxes:
        # Rotate each corner point
        rotated_corners = []
        for x, y in bbox:
            x_rotated = x * np.cos(theta) - y * np.sin(theta)
            y_rotated = x * np.sin(theta) + y * np.cos(theta)
            rotated_corners.append((x_rotated + offset_x, y_rotated + offset_y))

        rotated_bboxes.append(rotated_corners)

    return rotated_bboxes
def check_and_correct(paths, dx, dy, draw=None):
    """
    Translates and corrects paths, ensuring they are continuous and closed.

    Parameters:
    paths (list): List of paths to check and correct.
    dx (float): The x translation to apply.
    dy (float): The y translation to apply.
    draw (svgwrite.Drawing, optional): The drawing object to add corrected paths to.

    Returns:
    None
    """
    for p in range(len(paths)):
        # Translate the path
        paths[p] = paths[p].translated(complex(dx, dy)) # = path.translate(paths[p], complex(dx, dy))
        # Remove zero-length segments
        paths[p] = remove_zero_length_segments(paths[p])
        if paths[p].iscontinuous() and paths[p].isclosed():
            if draw is not None:
                draw.add(draw.path(d=paths[p].d(), fill='black'))
        else:
            sub_paths = paths[p].continuous_subpaths()
            path = []
            for i in range(len(sub_paths)):
                if sub_paths[i].start != sub_paths[i].end:
                    sub_paths[i].append(Line(sub_paths[i].end, sub_paths[i].start))
                path.extend(sub_paths[i])
            paths[p] = Path(*path)
            if draw is not None:
                draw.add(draw.path(d=paths[p].d(), fill='black'))
    if draw is not None:
        draw.save()

def find_bounding_box(points_list1, points_list2):
    """
    Finds the bounding box for two lists of points.

    Parameters:
    points_list1 (list): The first list of points.
    points_list2 (list): The second list of points.

    Returns:
    tuple: The bounding box coordinates (min_x, max_x, min_y, max_y).
    """
    # Combine both lists of points
    all_points = points_list1 + points_list2

    # Find the minimum and maximum x and y coordinates
    min_x, min_y = min(p[0] for p in all_points), min(p[1] for p in all_points)
    max_x, max_y = max(p[0] for p in all_points), max(p[1] for p in all_points)

    return min_x, max_x, min_y, max_y

def get_edges(points):
    """
    Calculates edges (vectors) from a list of points.

    Parameters:
    points (list): The list of points.

    Returns:
    list: The list of edges (vectors).
    """
    edges = []
    for i in range(len(points)):
        edge = np.array(points[(i + 1) % len(points)]) - np.array(points[i])
        edges.append(edge)
    return edges


def project_points_onto_axis(points, axis):
    """
    Projects a list of points onto a given axis.

    Parameters:
    points (list): List of points to project.
    axis (array): The axis to project the points onto.

    Returns:
    tuple: The minimum and maximum projections of the points onto the axis.
    """
    # Calculate the projection of each point onto the axis
    projections = [np.dot(np.array(p), axis) for p in points]
    return min(projections), max(projections)


def overlap_on_axis(points1, points2, axis):
    """
    Checks if the projections of two sets of points overlap on a given axis.

    Parameters:
    points1 (list): The first set of points.
    points2 (list): The second set of points.
    axis (array): The axis to project the points onto.

    Returns:
    bool: True if the projections overlap, False otherwise.
    """
    # Project both sets of points onto the axis
    min1, max1 = project_points_onto_axis(points1, axis)
    min2, max2 = project_points_onto_axis(points2, axis)
    # Check if the projections overlap
    return max1 >= min2 and max2 >= min1


def boxes_collide(boxes_list, target_box):
    """
    Checks if any bounding box in a list collides with a target bounding box using the Separating Axis Theorem (SAT).

    Parameters:
    boxes_list (list of lists): A list of bounding boxes, each defined by a list of points.
    target_box (list): The points defining the target bounding box.

    Returns:
    Integer: Returns the collision range between the two bounding boxes
    """

    def check_collision(total_edges, box_points, target_box):
        for edge in total_edges:
            # Calculate the perpendicular axis to the edge
            axis = np.array([-edge[1], edge[0]])  # Perpendicular to the edge
            axis = axis / np.linalg.norm(axis)  # Normalize the axis
            if not overlap_on_axis(box_points, target_box, axis):
                return False
        # If no separating axis is found, the boxes collide
        return True

    # Get the edges of the target bounding box
    f_index = s_index = 0
    target_edges = get_edges(target_box)
    collision = False
    for box_points in boxes_list:        # Get the edges of the current bounding box
        box_edges = get_edges(box_points)
        total_edges = box_edges+target_edges
        if check_collision(total_edges, box_points, target_box):
            collision = True
            break
        s_index += 1
    if collision:
        f_index = len(boxes_list)
        for box_points in boxes_list[::-1]:            # Get the edges of the current bounding box
            box_edges = get_edges(box_points)
            total_edges = box_edges + target_edges
            # Check for overlap on all axes perpendicular to the edges
            if check_collision(total_edges, box_points, target_box):
                return s_index, f_index
            f_index -= 1
        # If no collisions are found
    return (0,0)


def remove_zero_length_segments(path):
    """
    Removes zero-length segments from an svgpathtools Path.

    Args:
        path (Path): The input path.

    Returns:
        Path: The modified path without zero-length segments.
    """
    non_zero_segments = [seg for seg in path if seg.length() > 0]
    return Path(*non_zero_segments)

# def is_point_in_fill(path, point1):
#     """
#     Determines if a point is inside the fill of a given path using the ray-casting algorithm.
#
#     Parameters:
#     path (Path): The path to check against.
#     point1 (complex): The point to check.
#
#     Returns:
#     bool: True if the point is inside the fill, False otherwise.
#     """
#     # Get the bounding box of the path and expand it slightly
#     min_x, max_x, min_y, max_y = path.bbox()
#     min_x, max_x, min_y, max_y = min_x - 1, max_x + 1, min_y - 1, max_y + 1
#
#     # Find a point outside the bounding box to cast a ray
#     diff = 0
#     while True:
#         t = random.uniform(0, 1)
#         point2 = path.point(t)
#         diff = point1 - point2
#         unit_slope = diff / abs(diff)
#         if abs(unit_slope - path.unit_tangent(t)) > 0.001:
#             break
#
#     # Calculate the slope of the line from point1 to point2
#     slope = diff.imag / diff.real if diff.real != 0 else np.inf
#
#     if slope != np.inf:
#         # Intersection with left or right edge
#         if diff.real < 0:
#             intersect_x = min_x - 1
#         else:
#             intersect_x = max_x + 1
#         intersect_y = point1.imag + slope * (intersect_x - point1.real)
#     else:
#         # Vertical line (slope is infinite)
#         intersect_x = point2.real
#         intersect_y = min(max_y, max(min_y, point1.imag))  # Clamp within bounding box
#
#     # Create a line from point1 to the intersection point
#     line = Path(Line(point1, complex(intersect_x, intersect_y)))
#
#     # Check the number of intersections with the path
#     return len(path.intersect(line)) % 2 == 1

def is_point_in_fill(path, point1):
    """
    Determines if a point is inside the fill of a given path using the ray-casting algorithm.

    Parameters:
    path (Path): The path to check against.
    point1 (complex): The point to check.

    Returns:
    bool: True if the point is inside the fill, False otherwise.
    """
    # Get the bounding box of the path
    min_x, max_x, min_y, max_y = path.bbox()

    # Cast a horizontal ray to the right from point1
    intersect_x = max_x + 1
    intersect_y = point1.imag

    # Create a line from point1 to the intersection point
    line = Path(Line(point1, complex(intersect_x, intersect_y)))

    # Check the number of intersections with the path
    return len(path.intersect(line)) % 2 == 1

def b_spline_to_bezier_series(tck):
    """Convert a parametric b-spline into a sequence of Bezier curves of the same degree.

    Inputs:
    tck : (t,c,k) tuple of b-spline knots, coefficients, and degree returned by splprep.

    Output:
    A list of Bezier curves of degree k that is equivalent to the input spline.
    Each Bezier curve is an array of shape (k+1,d) where d is the dimension of the
    space; thus the curve includes the starting point, the k-1 internal control
    points, and the endpoint, where each point is of d dimensions.
    """
    t,c,k = tck
    t = np.asarray(t)
    try:
        c[0][0]
    except:
        # I can't figure out a simple way to convert nonparametric splines to
        # parametric splines. Oh well.
        raise TypeError("Only parametric b-splines are supported.")
    new_tck = tck
    knots_to_consider = np.unique(t[k+1:-k-1])
    desired_multiplicity = k+1
    for x in knots_to_consider:
        current_multiplicity = sum(t == x)
        remainder = current_multiplicity%desired_multiplicity
        if remainder != 0:
          # add enough knots to bring the current multiplicity up to the desired multiplicity
          number_to_insert = desired_multiplicity - remainder
          new_tck = insert(x, new_tck, number_to_insert, False)
    tt,cc,kk = new_tck
    # strip off the last k+1 knots, as they are redundant after knot insertion
    bezier_points = np.transpose(cc)[:-desired_multiplicity]
    # print("------------------------------------")
    res = []
    if len(bezier_points)>0:
        bezier_curves = np.split(bezier_points, len(bezier_points) / desired_multiplicity, axis=0)
        # print("first print: ", bezier_curves)
        res = [[[float(x),float(y)] for x,y in curve] for curve in bezier_curves]
    else:
        X=[]
        y=[]
        # print("first print: C:", c)
        if isinstance(c[0], (np.ndarray, list, tuple)):
            X = [float(x) for x in c[0]]
        if isinstance(c[1], (np.ndarray, list, tuple)):
            Y = [float(y) for y in c[1]]
        res = [[[xi, yi] for xi, yi in zip(X, Y)]]
    # print("second print ", res)
    # print("------------------------------------")
    return res


def rotate_bezier_curves(bezier_curves, center, d_angle, offset, n, order="normal"):
    """
    Rotates a list of Bezier curves around a center point by a specified angle.

    Parameters:
    bezier_curves (list): List of Bezier curves to rotate.
    center (tuple): The center point around which to rotate the curves.
    d_angle (float): The angle increment for each rotation.
    offset (float): The initial rotation offset.
    n (int): The number of rotations to perform.
    order (str): The order of the points in the rotated curves ("normal" or reversed).

    Returns:
    list: A grid of rotated Bezier curves.
    """
    grid = []
    for i in range(n):
        angle = d_angle * i + offset
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        rotated_bezier_curves = []
        for bezier_curve in bezier_curves:
            rotated_bezier = []
            for point in bezier_curve:
                x, y = point
                x = float(x)
                y = float(y)
                # Rotate the point around the center
                rotated_x = center[0] + x * cos_angle - y * sin_angle
                rotated_y = center[1] + x * sin_angle + y * cos_angle
                res = [float(rotated_x), float(rotated_y)]
                rotated_bezier.append(res)
            if order == "normal":
                rotated_bezier_curves.append(rotated_bezier)
            else:
                rotated_bezier_curves.append(rotated_bezier[::-1])
        if order == "normal":
            grid.append(rotated_bezier_curves)
        else:
            grid.append(rotated_bezier_curves[::-1])
    return grid


def new_add_grid(model, in_r, out_r, n, offset, alpha, width_factor_s, width_factor_f, center, id=''):
    """
    Adds a grid of rotated Bezier curves to an SVG model.

    Parameters:
    model (svgwrite.Drawing): The SVG model to add the grid to.
    in_r (float): The inner radius for the grid.
    out_r (float): The outer radius for the grid.
    n (int): The number of rotations to perform.
    offset (float): The initial rotation offset.
    alpha (float): The angle for the Bezier curves.
    width_factor_s (float): The width factor for the start of the curves.
    width_factor_f (float): The width factor for the end of the curves.
    center (tuple): The center point for the rotations.
    id (str): The ID for the SVG path element.
    """
    log_curve = model.path(stroke='none', fill='black', id=id) if id else model.path(stroke='none', fill='black')
    d_angle = 2 * np.pi / n
    points_left, points_right = c(in_r, out_r, alpha, n, width_factor_s, width_factor_f)

    # Convert points to Bezier curves
    points = np.array(points_left)
    x = points[:, 0]
    y = points[:, 1]
    tck, _ = splprep([x, y], k=3, s=3)
    bezier_curves_left = rotate_bezier_curves(b_spline_to_bezier_series(tck), center, d_angle, offset, n)

    points = np.array(points_right)
    x = points[:, 0]
    y = points[:, 1]
    tck, _ = splprep([x, y], k=3, s=3)
    outer_angle_dir = '-' if alpha >= 0 else '+'
    inner_angle_dir = '-' if outer_angle_dir == '+' else '+'
    bezier_curves_right = rotate_bezier_curves(b_spline_to_bezier_series(tck), center, d_angle, offset, n,
                                               order="reversed")

    total_grouped_curves = [left + right for left, right in zip(bezier_curves_left, bezier_curves_right)]

    for group in total_grouped_curves:
        log_curve.push('M', *group[0][0])
        log_curve.push('C', *group[0][1:4])
        prev_end = group[0][3]

        for bez in group[1:]:
            if prev_end != bez[0]:
                log_curve.push_arc(target=bez[0], rotation=0, r=(out_r, out_r), large_arc=False,
                                   angle_dir=outer_angle_dir, absolute=True)
            log_curve.push('C', *bez[1:4])
            prev_end = bez[3]

        log_curve.push_arc(target=group[0][0], rotation=0, r=(in_r, in_r), large_arc=False, angle_dir=inner_angle_dir,
                           absolute=True)

    model.add(log_curve)


def old_add_grid(model, type, in_r, out_r, n, offset, alpha, width_factor_s, width_factor_f, center, id=''):
    """
    Adds a grid of rotated points to a DXF or SVG model using the old method.

    Parameters:
    model (object): The model to add the grid to (DXF or SVG).
    type (str): The type of model ("dxf" or "svg").
    in_r (float): The inner radius for the grid.
    out_r (float): The outer radius for the grid.
    n (int): The number of rotations to perform.
    offset (float): The initial rotation offset.
    alpha (float): The angle for the points.
    width_factor_s (float): The width factor for the start of the points.
    width_factor_f (float): The width factor for the end of the points.
    center (tuple): The center point for the rotations.
    id (str): The ID for the SVG path element.
    """
    d_angle = 2 * np.pi / n
    points_left, points_right = c(in_r, out_r, alpha, n, width_factor_s, width_factor_f)
    log_curve = ''
    if type == "svg":
        log_curve = model.path(stroke='none', fill='black', id=id) if id else model.path(stroke='none', fill='black')

    for i in range(n):
        temp_points_left = []
        temp_points_right = []
        for t in range(len(points_left)):
            # Rotate points for the left side
            temp_points_left.append((
                center[0] + points_left[t][0] * np.cos(d_angle * i + offset) - points_left[t][1] * np.sin(
                    d_angle * i + offset),
                center[1] + points_left[t][0] * np.sin(d_angle * i + offset) + points_left[t][1] * np.cos(
                    d_angle * i + offset)
            ))
            # Rotate points for the right side
            temp_points_right.append((
                center[0] + points_right[t][0] * np.cos(d_angle * i + offset) - points_right[t][1] * np.sin(
                    d_angle * i + offset),
                center[1] + points_right[t][0] * np.sin(d_angle * i + offset) + points_right[t][1] * np.cos(
                    d_angle * i + offset)
            ))
        if type == "svg":
            log_curve.push('M', *temp_points_left[0])  # Move to the first point
            for x, y in temp_points_left[1:]:
                log_curve.push('L', x, y)  # Draw a line segment to the next point
            for x, y in temp_points_right[::-1]:
                log_curve.push('L', x, y)  # Draw a line segment to the next point
            log_curve.push('L', *temp_points_left[0])  # Close the path
        elif type == "dxf":
            close_shape(model, type, temp_points_left, temp_points_right)
    if type == "svg":
        model.add(log_curve)


def add_grid(model, type, in_r, out_r, n, offset, alpha, width_factor_s, width_factor_f, center, id='', old_add=False):
    """
    Adds a grid of rotated points to a model (DXF or SVG) using either the old or new method.

    Parameters:
    model (object): The model to add the grid to (DXF or SVG).
    type (str): The type of model ("dxf" or "svg").
    in_r (float): The inner radius for the grid.
    out_r (float): The outer radius for the grid.
    n (int): The number of rotations to perform.
    offset (float): The initial rotation offset.
    alpha (float): The angle for the points.
    width_factor_s (float): The width factor for the start of the points.
    width_factor_f (float): The width factor for the end of the points.
    center (tuple): The center point for the rotations.
    id (str): The ID for the SVG path element.
    old_add (bool): Flag to indicate if the grid should be added using the old method.
    """
    if type == "svg" and not old_add:
        new_add_grid(model, in_r, out_r, n, offset, alpha, width_factor_s, width_factor_f, center, id=id)
    elif type == "dxf" or old_add:
        old_add_grid(model, type, in_r, out_r, n, offset, alpha, width_factor_s, width_factor_f, center, id)


def rotating_calipers(hull):
    """
    Finds the minimal area bounding box for a set of convex hull points.

    Parameters:
    hull (list of tuples): List of points defining the convex hull.

    Returns:
    list of tuples: The coordinates of the minimal area bounding box.
    """
    min_area = float('inf')
    best_box = None

    # Convert list of tuples to numpy array for vectorized operations
    hull_array = np.array(hull)

    for i in range(len(hull)):
        edge = hull_array[(i + 1) % len(hull)] - hull_array[i]
        edge_angle = np.arctan2(edge[1], edge[0])
        rotation_matrix = np.array([
            [np.cos(-edge_angle), -np.sin(-edge_angle)],
            [np.sin(-edge_angle), np.cos(-edge_angle)]
        ])
        rotated_points = np.dot(hull_array, rotation_matrix)
        min_x, min_y = np.min(rotated_points, axis=0)
        max_x, max_y = np.max(rotated_points, axis=0)
        area = (max_x - min_x) * (max_y - min_y)

        if area < min_area:
            min_area = area
            best_box = [
                tuple(np.dot([min_x, min_y], rotation_matrix.T)),
                tuple(np.dot([max_x, min_y], rotation_matrix.T)),
                tuple(np.dot([max_x, max_y], rotation_matrix.T)),
                tuple(np.dot([min_x, max_y], rotation_matrix.T))
            ]
    return best_box

def graham_scan(points):
    """Computes the convex hull of a set of 2D points using the Graham Scan algorithm."""

    # Sort the points lexicographically (tuples compare lexicographically).
    points = sorted(points)

    # Function to determine the orientation of the triplet (p, q, r).
    def orientation(p, q, r):
        return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])

    # Build the lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and orientation(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    # Build the upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and orientation(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenate lower and upper hull to get the full hull
    # The last point of each half is omitted because it's repeated at the beginning of the other half
    return lower[:-1] + upper[:-1]


import concurrent.futures
import threading

def process_combined_points(combined_points):
    hull_points = graham_scan(combined_points)
    bbox = rotating_calipers(hull_points)
    return bbox

def n_optimal_bboxes(set1, set2, n):
    """Processes two sets of points as described."""
    bboxes = [None] * n  # Initialize a list to store bounding boxes in order
    k, m = divmod(len(set1), n)
    subsets1 = [None] * n
    subsets2 = [None] * n
    def process_and_store(i):
        combined_points = []
        subsets1[i] = set1[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
        subsets2[i] = set2[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
        combined_points.extend(subsets1[i])
        combined_points.extend(subsets2[i])
        bbox = process_combined_points(combined_points)
        bboxes[i] = bbox

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_and_store, i) for i in range(n)]
        concurrent.futures.wait(futures)

    return subsets1, subsets2, bboxes

def combine_path(left,right):
    return [(p.real, p.imag) if isinstance(p, complex) else p for p in left + list(reversed(right))]


def close_shape_aux(model, type, total_path):
    """
    Closes a shape by connecting the left and right points and adds it to the model.

    Parameters:
    model (object): The model to add the shape to (DXF or SVG).
    type (str): The type of model ("dxf" or "svg").
    left (list): The list of points for the left side.
    right (list): The list of points for the right side.

    Returns:
    None
    """
    if type == 'svg':
        path = model.path(stroke='none', fill='black')
        path.push('M', total_path[0])  # Move to the first point
        for p in total_path[1:]:
            path.push('L', p)  # Draw a line segment to the next point
        path.push('L', total_path[0])  # Close the path
        model.add(path)
    elif type == 'dxf':
        hatch = model.add_hatch(color=7, dxfattribs={'hatch_style': 1})  # Use hatch style 1 for solid fill
        hatch.paths.add_polyline_path(total_path, is_closed=True, flags=1)  # Set flag for external path



def cut_grid(model, type, paths, in_r, out_r, n, offset, alpha, width_factor_s, width_factor_f, center):
    """
    Cuts a grid of rotated points from paths in a DXF or SVG model.

    Parameters:
    model (object): The model to add the grid to (DXF or SVG).
    type (str): The type of model ("dxf" or "svg").
    paths (list): List of paths to cut from.
    in_r (float): The inner radius for the grid.
    out_r (float): The outer radius for the grid.
    n (int): The number of rotations to perform.
    offset (float): The initial rotation offset.
    alpha (float): The angle for the points.
    width_factor_s (float): The width factor for the start of the points.
    width_factor_f (float): The width factor for the end of the points.
    center (tuple): The center point for the rotations.
    """
    # check = svgwrite.Drawing(f'check_new_algo_{in_r}_{out_r}.svg', profile='full', size=("2000px","2000px"))
    d_angle = 2 * np.pi / n
    points_left, points_right = c(in_r, out_r, alpha, n, width_factor_s, width_factor_f)

    # Tried varius threading techniques non were faster that single thread. only headache

    # Create a lock object
    # lock = threading.Lock()
    # cut_path_queue = queue.Queue()
    # Create a condition for notification
    # condition = threading.Condition()
    # semaphore = threading.Semaphore(0)


    # c_bbox_shell = find_bounding_box(points_left, points_right)
    # x_min, x_max, y_min, y_max = c_bbox_shell
    # c_bbox_shell = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]

    deviders = 25
    start_time = time.time()
    seg_points_left,seg_points_right, c_optimal_bboxes_proto = n_optimal_bboxes(points_left,points_right, deviders)
    end_time = time.time()
    print("elapsed_time: ", (end_time - start_time))
    lines_checked = 0
    total_paths_for_closing = [[] for _ in range(n)]
    #
    # def add_to_queue(left_side, right_side):
    #     cut_path_queue.put((left_side, right_side))
    #     semaphore.release()

    # def fine_grain_collision_segment(i):
    for i in range(n):
        theta = d_angle * i + offset

        # c_bbox = rotate_and_offset_bboxes([c_bbox_shell], theta, center[0], center[1])

        c_optimal_bboxes = rotate_and_offset_bboxes(c_optimal_bboxes_proto, theta, center[0], center[1])

        for path in paths:
            x_min, x_max, y_min, y_max = path.bbox()
            p_bbox = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]

            # collision_segment = boxes_collide(c_bbox, p_bbox)
            collision_segments = boxes_collide(c_optimal_bboxes, p_bbox)
            collision_start, collision_end = collision_segments
            if not collision_start and not collision_end:
                continue

            # if not boxes_collide(c_bbox, p_bbox):
            #     continue

            # check.add(check.path(d=path.d(), fill='none', stroke = 'black'))

            # for t in range(deviders_len * (collision_segment-1), len(points_left)):
            left_side, right_side = [], []
            for s in range(collision_start,collision_end):
                for p_l, p_r in zip(seg_points_left[s],seg_points_right[s]):
                    left = complex(
                        (center[0] + p_l[0] * np.cos(theta) - p_l[1] * np.sin(theta)),
                        (center[1] + p_l[0] * np.sin(theta) + p_l[1] * np.cos(theta))
                    )
                    # Rotate points for the right side
                    right = complex(
                        (center[0] + p_r[0] * np.cos(theta) - p_r[1] * np.sin(theta)),
                        (center[1] + p_r[0] * np.sin(theta) + p_r[1] * np.cos(theta))
                    )
                    iso_line = Path(Line(left, right))
                    intersections = iso_line.intersect(path)

                    # iso_path = check.path(stroke='red', fill='none', stroke_width = 0.3)
                    # iso_path.push("M", left.real, left.imag)
                    # iso_path.push("L", right.real, right.imag)
                    # check.add(iso_path)

                    if len(intersections) == 0:
                        if is_point_in_fill(path, left):
                            left_side.append(left)
                            right_side.append(right)
                        else:
                            if len(left_side) > 0:
                                # if lock.acquire(blocking=False):
                                #     try:
                                close_shape(model, type, left_side, right_side)
                                #     finally:
                                #         lock.release()
                                # else:
                                #     total_paths_for_closing[i].append(combine_path( left_side, right_side))
                                # with condition:
                                #     cut_path_queue.put((left_side, right_side))
                                #     condition.notify()
                                # add_to_queue(left_side,right_side)
                                # with lock:
                                #     close_shape(model, type, left_side, right_side)
                                left_side, right_side = [], []
                    elif len(intersections) == 1:
                        a, _ = intersections[0]
                        T1, seg1, t1 = a
                        if is_point_in_fill(path, left):
                            left_side.append(left)
                            right_side.append(seg1.point(t1))
                        elif is_point_in_fill(path, right):
                            left_side.append(seg1.point(t1))
                            right_side.append(right)
                        else:
                            left_side.append(seg1.point(t1))
                    elif len(intersections) == 2:
                        a, _ = intersections[0]
                        b, _ = intersections[-1]
                        T1, seg1, t1 = a
                        T2, seg2, t2 = b
                        point1 = seg1.point(t1)
                        point2 = seg2.point(t2)
                        middle_point = (point1+point2)/2
                        if not is_point_in_fill(path, middle_point):
                            continue
                        # Check if left_side is not empty
                        if left_side:
                            last_left_point = left_side[-1]
                            if abs(last_left_point - point1) < abs(last_left_point - point2):
                                left_side.append(point1)
                                right_side.append(point2)
                            else:
                                left_side.append(point2)
                                right_side.append(point1)
                        else:
                            # If left_side is empty, append points to left and right respectively
                            left_side.append(point1)
                            right_side.append(point2)
                    elif len(intersections) > 2:
                        continue
            if len(left_side) > 0:
                # if lock.acquire(blocking=False):
                #     try:
                close_shape(model, type, left_side, right_side)
                #     finally:
                #         lock.release()
                # else:
                #     total_paths_for_closing[i].append(combine_path(left_side,right_side))
                # with lock:
                #     close_shape(model, type, left_side, right_side)
                # with condition:
                #     cut_path_queue.put((left_side, right_side))
                #     condition.notify()
                # add_to_queue(left_side, right_side)

    # for j in range(len(total_paths_for_closing[i])):
        #     with lock:
        #         close_shape_aux(model, type, total_paths_for_closing[i][j])
    #
    # wait_for_workers = threading.Event()
    # wait_for_workers.set()
    # def process_items():
    #     while True:
    #         if semaphore.acquire(blocking=False):
    #             cut_path = cut_path_queue.get()
    #             try:
    #                 close_shape(model, type, cut_path[0], cut_path[1])
    #                 cut_path_queue.task_done()
    #             finally:
    #                 semaphore.release()
    #         else:
    #             if not wait_for_workers.is_set():  # Check if workers are finished
    #                 break  # Exit the loop if workers are finished
    #             semaphore.acquire()  # Block until the semaphore is available
    #
    # processing_thread = threading.Thread(target=process_items, daemon=True)
    # processing_thread.start()

    # with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    #     futures = [executor.submit(fine_grain_collision_segment, i) for i in range(n)]
    #     concurrent.futures.wait(futures)
    #     wait_for_workers.clear()
    # cut_path_queue.join()
    # for i in range(n):
    #     if total_paths_for_closing[i]:
    #         for j in range(len(total_paths_for_closing[i])):
    #             close_shape_aux(model,type,total_paths_for_closing[i][j])

    # print("lines checked: ", lines_checked)
    # check.save()

def is_prime(num: int) -> bool:
    """Check if a number is prime."""
    if num < 2:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

def first_n_primes(n: int) -> list:
    """Return the first n prime numbers in ascending order (including 1)."""
    primes = [2]
    num = 3
    while len(primes) < n:
        if is_prime(num):
            primes.append(num)
        num += 1
    return primes

def add_crosshair_to_center(model, center):
    center_x , center_y = center
    width , height = center_x*2, center_y*2

    # Add horizontal line of the crosshair
    model.add_line((center_x - width / 2, center_y), (center_x + width / 2, center_y))

    # Add vertical line of the crosshair
    model.add_line((center_x, center_y - height / 2), (center_x, center_y + height / 2))

