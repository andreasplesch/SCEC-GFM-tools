import copy

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from CFM_TS import CFM_TS


def segment_angle(v1, v2, acute):
    # v1 is your first vector
    # v2 is your second vector
    angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
    # np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    if acute:
        return angle
    else:
        return 2 * np.pi - angle


def CFM_connections(triangles):
    ta = np.array(triangles)  # Collect all the triangles in a single array
    # stack all the connections in ta
    t0 = np.vstack((np.sort(ta[:, [0, 1]]), np.sort(ta[:, [1, 2]]), np.sort(ta[:, [0, 2]])))
    # keep the index of the original triangle
    fa = np.array(range(len(ta)))
    f0 = np.hstack((fa, fa, fa))
    # Unique connections
    unique_connections = np.unique(t0, axis=0, return_index=True, return_counts=True)
    # Form the connections dictionary.
    connections = {}
    # connections['triangles']=f0[unique_connections[1]]
    connections['vertices'] = unique_connections[0]
    connections['number'] = unique_connections[2]
    trg = []
    for i in range(len(unique_connections[2])):
        trg.append(np.array(np.where(np.all(t0 == unique_connections[0][i], axis=1))).flatten().tolist())
    connections['triangles'] = trg
    return connections


def CFM_edges(surface):
    # Which unique connection is a surface edge?
    edge_connections = np.where(surface.connections['number'] == 1)
    edge_points = surface.connections['vertices'][edge_connections]
    # order the sequence of points as a loop
    i_now = edge_points[0]
    idx_now = np.where(edge_points == i_now)  # Find instances of i_now
    row_now = idx_now[0][0]  # row of the first instance
    li_sorted = edge_points[row_now, :]  # start the sorted array
    while edge_points.shape[0] > 1:  # iterate
        i_now = li_sorted[-1]  # update latest point index
        edge_points = np.delete(edge_points, row_now, 0)  # remove last recorded segment
        idx_now = np.where(edge_points[:, :] == i_now)
        row_now = idx_now[0][0]
        li_sorted = np.append(li_sorted, edge_points[row_now, 1 - idx_now[1]])
    # print(surface.connections['vertices'][edge_connections])
    # print(li_sorted)

    pt = surface.vertices
    pt_sorted = np.array(list(surface.vertices[i] for i in li_sorted))  # points sorted along the edge
    pdiff_sorted = np.diff(pt_sorted, axis=0)  # segment length
    successive_angles = np.zeros([pt_sorted.shape[0] - 1, 1])
    # First angle, comes from looping points, assuming closed outline
    successive_angles[0] = segment_angle(pdiff_sorted[-1, :], pdiff_sorted[0, :], 'acute')
    for i in range(pt_sorted.shape[0] - 2):  # other points
        successive_angles[i + 1] = segment_angle(pdiff_sorted[i, :], pdiff_sorted[i + 1, :], 'acute')
    # loop to finish
    # successive_angles[0] = segment_angle(pdiff_sorted[-1,:],pdiff_sorted[0,:],'acute')
    successive_angles = successive_angles.flatten()
    successive_angles *= 180 / np.pi
    corners = np.sort(np.argsort(successive_angles)[-4:])  # find the 4 highest angles
    # Define the various segment and order them so that the 1st is the shallowest |
    # First, cut li_sorted into segments
    segments = []
    for i in range(corners.shape[0] - 1):
        new_segment = {}
        new_segment['nodes'] = li_sorted[corners[i]:corners[i + 1] + 1]
        segments.append(new_segment)
    new_segment = {}
    new_segment['nodes'] = np.append(li_sorted[corners[i + 1]:-1], li_sorted[0:corners[0] + 1])
    segments.append(new_segment)
    # Store vertex coordinates and calculate mean positions for each segment
    zmean = np.empty(shape=[len(segments), 1])
    for i in range(len(segments)):
        segments[i]['vertices'] = np.array(list(pt[idx] for idx in segments[i]['nodes']))  # pt[segments[i]['nodes'], :]
        segments[i]['mean'] = np.mean(segments[i]['vertices'], axis=0)
        zmean[i] = segments[i]['mean'][2]
    # Reorder so that the surface trace is at position 0
    shallowest = np.argmax(zmean)
    dum = np.arange(4)
    dum = np.append(dum, dum)
    dum = dum[shallowest + np.arange(4)]
    # Reorder the list
    segments = [segments[i] for i in dum]
    # if needed, flip order
    test_ends = segments[0]['vertices'][[0, -1], :]
    if np.diff(test_ends, axis=0)[0][1] < 0:  # North first => flip orders
        segments = CFM_flip_segments(segments)
    return segments


def CFM_edges_with_support(surface, Edge0, Edge1):
    # Which unique connection is a surface edge?
    edge_connections = np.where(surface.connections['number'] == 1)
    edge_points = surface.connections['vertices'][edge_connections]
    # order the sequence of points as a loop
    i_now = edge_points[0]
    idx_now = np.where(edge_points == i_now)  # Find instances of i_now
    row_now = idx_now[0][0]  # row of the first instance
    li_sorted = edge_points[row_now, :]  # start the sorted array
    while edge_points.shape[0] > 1:  # iterate
        i_now = li_sorted[-1]  # update latest point index
        edge_points = np.delete(edge_points, row_now, 0)  # remove last recorded segment
        idx_now = np.where(edge_points[:, :] == i_now)
        row_now = idx_now[0][0]
        li_sorted = np.append(li_sorted, edge_points[row_now, 1 - idx_now[1]])
    # print(surface.connections['vertices'][edge_connections])
    # print(li_sorted)
    pt = surface.vertices
    pt_sorted = np.array(list(surface.vertices[i] for i in li_sorted))  # points sorted along the edge
    CORNER = Edge0['vertices'][0, :]
    Distance_square_min = (pt_sorted[:, 0] - CORNER[0]) ** 2 + (pt_sorted[:, 1] - CORNER[1]) ** 2 + (
            pt_sorted[:, 2] - CORNER[2]) ** 2
    CORNER = Edge0['vertices'][-1, :]
    Distance_square_min = np.minimum(Distance_square_min,
                                     (pt_sorted[:, 0] - CORNER[0]) ** 2 + (pt_sorted[:, 1] - CORNER[1]) ** 2 + (
                                             pt_sorted[:, 2] - CORNER[2]) ** 2)
    CORNER = Edge1['vertices'][0, :]
    Distance_square_min = np.minimum(Distance_square_min,
                                     (pt_sorted[:, 0] - CORNER[0]) ** 2 + (pt_sorted[:, 1] - CORNER[1]) ** 2 + (
                                             pt_sorted[:, 2] - CORNER[2]) ** 2)
    CORNER = Edge1['vertices'][-1, :]
    Distance_square_min = np.minimum(Distance_square_min,
                                     (pt_sorted[:, 0] - CORNER[0]) ** 2 + (pt_sorted[:, 1] - CORNER[1]) ** 2 + (
                                             pt_sorted[:, 2] - CORNER[2]) ** 2)
    corners = np.sort(np.argsort(Distance_square_min)[0:4])  # find the 4 highest angles

    # Define the various segment and order them so that the 1st is the shallowest |
    # First, cut li_sorted into segments
    segments = []
    for i in range(corners.shape[0] - 1):
        new_segment = {}
        new_segment['nodes'] = li_sorted[corners[i]:corners[i + 1] + 1]
        segments.append(new_segment)
    new_segment = {}
    new_segment['nodes'] = np.append(li_sorted[corners[i + 1]:-1], li_sorted[0:corners[0] + 1])
    segments.append(new_segment)
    # Store vertex coordinates and calculate mean positions for each segment
    zmean = np.empty(shape=[len(segments), 1])
    for i in range(len(segments)):
        segments[i]['vertices'] = np.array(list(pt[idx] for idx in segments[i]['nodes']))  # pt[segments[i]['nodes'], :]
        segments[i]['mean'] = np.mean(segments[i]['vertices'], axis=0)
        zmean[i] = segments[i]['mean'][2]
    # Reorder so that the surface trace is at position 0
    shallowest = np.argmax(zmean)
    dum = np.arange(4)
    dum = np.append(dum, dum)
    dum = dum[shallowest + np.arange(4)]
    # Reorder the list
    segments = [segments[i] for i in dum]
    # if needed, flip order
    test_ends = segments[0]['vertices'][[0, -1], :]
    if np.diff(test_ends, axis=0)[0][1] < 0:  # North first => flip orders
        segments = CFM_flip_segments(segments)
    return segments


# def CFM_segments(edges):
#     # global pt_sorted, pdiff_sorted, successive_angles, i, corners
#     li = edges.lines
#     pt = edges.points
#
#     # Sort the edges so that they follow a loop | result is li_sorted
#     LIR = li.reshape([int(len(li) / 3), 3])
#     LIR = np.delete(LIR, 0, 1)  # Remove the first column, that should only be 2s
#     # Gotta start somewhere. i_now = 0 is the best
#     i_now = 0
#     idx_now = np.where(LIR[:, :] == i_now)  # Find instances of i_now
#     row_now = idx_now[0][0]  # row of the first instance
#     li_sorted = LIR[row_now, :]  # start the sorted array
#     while LIR.shape[0] > 1:  # iterate
#         i_now = li_sorted[-1]  # update latest point index
#         LIR = np.delete(LIR, row_now, 0)  # remove last recorded segment
#         idx_now = np.where(LIR[:, :] == i_now)
#         row_now = idx_now[0][0]
#         li_sorted = np.append(li_sorted, LIR[row_now, 1 - idx_now[1]])
#     # Calculate the angles between successive segments | result is successive_angles
#     # Find the 4 corners as the 4 largest acute angles
#     pt_sorted = pt[li_sorted, :]  # points sorted along the line
#     pdiff_sorted = np.diff(pt_sorted, axis=0)  # segment length
#     successive_angles = np.zeros([pt_sorted.shape[0] - 1, 1])
#     # First angle, comes from looping points, assuming closed outline
#     successive_angles[0] = segment_angle(pdiff_sorted[-1, :], pdiff_sorted[0, :], 'acute')
#     for i in range(pt_sorted.shape[0] - 2):  # other points
#         successive_angles[i + 1] = segment_angle(pdiff_sorted[i, :], pdiff_sorted[i + 1, :], 'acute')
#     # loop to finish
#     # successive_angles[0] = segment_angle(pdiff_sorted[-1,:],pdiff_sorted[0,:],'acute')
#     successive_angles = successive_angles.flatten()
#     successive_angles *= 180 / np.pi
#     corners = np.sort(np.argsort(successive_angles)[-4:])  # find the 4 highest angles
#     # Define the various segment and order them so that the 1st is the shallowest |
#     # First, cut li_sorted into segments
#     segments = []
#     for i in range(corners.shape[0] - 1):
#         new_segment = {}
#         new_segment['nodes'] = li_sorted[corners[i]:corners[i + 1] + 1]
#         segments.append(new_segment)
#     new_segment = {}
#     new_segment['nodes'] = np.append(li_sorted[corners[i + 1]:-1], li_sorted[0:corners[0] + 1])
#     segments.append(new_segment)
#     # Calculate mean positions
#     zmean = np.empty(shape=[len(segments), 1])
#     for i in range(len(segments)):
#         segments[i]['vertices'] = pt[segments[i]['nodes'], :]
#         segments[i]['mean'] = np.mean(segments[i]['vertices'], axis=0)
#         zmean[i] = segments[i]['mean'][2]
#     # Reorder so that the surface trace is at position 0
#     shallowest = np.argmax(zmean)
#     dum = np.arange(4)
#     dum = np.append(dum, dum)
#     dum = dum[shallowest + np.arange(4)]
#     # Reorder the list
#     segments = [segments[i] for i in dum]
#     # if needed, flip order
#     test_ends = segments[0]['vertices'][[0, -1], :]
#     if np.diff(test_ends, axis=0)[0][1] < 0:  # North first => flip orders
#         segments = CFM_flip_segments(segments)
#
#     return segments

# def CFM_segments_from_support(edges, Edge0, Edge1):
#     # global pt_sorted, pdiff_sorted, successive_angles, i, corners
#     li = edges.lines
#     pt = edges.points
#
#     # Sort the edges so that they follow a loop | result is li_sorted
#     LIR = li.reshape([int(len(li) / 3), 3])
#     LIR = np.delete(LIR, 0, 1)  # Remove the first column, that should only be 2s
#     # Gotta start somewhere. i_now = 0 is the best
#     i_now = 0
#     idx_now = np.where(LIR[:, :] == i_now)  # Find instances of i_now
#     row_now = idx_now[0][0]  # row of the first instance
#     li_sorted = LIR[row_now, :]  # start the sorted array
#     while LIR.shape[0] > 1:  # iterate
#         i_now = li_sorted[-1]  # update latest point index
#         LIR = np.delete(LIR, row_now, 0)  # remove last recorded segment
#         idx_now = np.where(LIR[:, :] == i_now)
#         row_now = idx_now[0][0]
#         li_sorted = np.append(li_sorted, LIR[row_now, 1 - idx_now[1]])
#     # Calculate the angles between successive segments | result is successive_angles
#     # Find the 4 corners as the 4 largest acute angles
#     pt_sorted = pt[li_sorted, :]  # points sorted along the line
#     CORNER = Edge0['vertices'][0, :]
#     Distance_square_min = (pt_sorted[:, 0] - CORNER[0]) ** 2 + (pt_sorted[:, 1] - CORNER[1]) ** 2 + (
#             pt_sorted[:, 2] - CORNER[2]) ** 2
#     CORNER = Edge0['vertices'][-1, :]
#     Distance_square_min = np.minimum(Distance_square_min,
#                                      (pt_sorted[:, 0] - CORNER[0]) ** 2 + (pt_sorted[:, 1] - CORNER[1]) ** 2 + (
#                                              pt_sorted[:, 2] - CORNER[2]) ** 2)
#     CORNER = Edge1['vertices'][0, :]
#     Distance_square_min = np.minimum(Distance_square_min,
#                                      (pt_sorted[:, 0] - CORNER[0]) ** 2 + (pt_sorted[:, 1] - CORNER[1]) ** 2 + (
#                                              pt_sorted[:, 2] - CORNER[2]) ** 2)
#     CORNER = Edge1['vertices'][-1, :]
#     Distance_square_min = np.minimum(Distance_square_min,
#                                      (pt_sorted[:, 0] - CORNER[0]) ** 2 + (pt_sorted[:, 1] - CORNER[1]) ** 2 + (
#                                              pt_sorted[:, 2] - CORNER[2]) ** 2)
#     corners = np.sort(np.argsort(Distance_square_min)[0:4])  # find the 4 highest angles
#     # Define the various segment and order them so that the 1st is the shallowest |
#     # First, cut li_sorted into segments
#     segments = []
#     for i in range(corners.shape[0] - 1):
#         new_segment = {}
#         new_segment['nodes'] = li_sorted[corners[i]:corners[i + 1] + 1]
#         segments.append(new_segment)
#     new_segment = {}
#     new_segment['nodes'] = np.append(li_sorted[corners[i + 1]:-1], li_sorted[0:corners[0] + 1])
#     segments.append(new_segment)
#     # Calculate mean positions
#     zmean = np.empty(shape=[len(segments), 1])
#     for i in range(len(segments)):
#         segments[i]['vertices'] = pt[segments[i]['nodes'], :]
#         segments[i]['mean'] = np.mean(segments[i]['vertices'], axis=0)
#         zmean[i] = segments[i]['mean'][2]
#     # Reorder so that the surface trace is at position 0
#     shallowest = np.argmax(zmean)
#     dum = np.arange(4)
#     dum = np.append(dum, dum)
#     dum = dum[shallowest + np.arange(4)]
#     # Reorder the list
#     segments = [segments[i] for i in dum]
#     segments = [segments[i] for i in dum]
#     # if needed, flip order
#     test_ends = segments[0]['vertices'][[0, -1], :]
#     if np.diff(test_ends, axis=0)[0][1] < 0:  # North first => flip orders
#         segments = CFM_flip_segments(segments)
#
#     return segments


def CFM_flip_segments(SEG):  # Flip the segments so that the first point is to the South
    for i in range(len(SEG)):
        SEG[i]['nodes'] = np.flipud(SEG[i]['nodes'])
        SEG[i]['vertices'] = np.flipud(SEG[i]['vertices'])
    dum = [0, 3, 2, 1]
    SEG = [SEG[i] for i in dum]
    return SEG


def CFM_join(Edge0, Edge1):  # Make a surface supported by two edges
    PT0 = np.array(Edge0['vertices'])
    PT1 = np.array(Edge1['vertices'])
    # Pall = np.vstack([PT0, PT1])
    max_length = 2000
    if (PT0[0, 2] - PT0[-1, 2]) > 0:  # best if increasing; flip if not
        PT0 = np.flipud(PT0)
    if (PT1[0, 2] - PT1[-1, 2]) > 0:  # best if increasing; flip if not
        PT1 = np.flipud(PT1)
    Pall = np.vstack((PT0, PT1))
    nb_dpt = min(PT0.shape[0], PT1.shape[0])
    if PT0.shape[0] == nb_dpt:
        tmp0 = PT0
    else:  # interpolate; remember xp must be increasing
        dpt = np.linspace(PT0[0, 2], PT0[-1, 2], nb_dpt)
        tmp0 = np.transpose(np.vstack((
            np.interp(dpt, PT0[:, 2], PT0[:, 0]),
            np.interp(dpt, PT0[:, 2], PT0[:, 1]),
            dpt)))
    if PT1.shape[0] == nb_dpt:
        tmp1 = PT1
    else:  # interpolate; remember xp must be increasing
        dpt = np.linspace(PT1[0, 2], PT1[-1, 2], nb_dpt)
        tmp1 = np.transpose(np.vstack((
            np.interp(dpt, PT1[:, 2], PT1[:, 0]),
            np.interp(dpt, PT1[:, 2], PT1[:, 1]),
            dpt)))
    for i in range(nb_dpt):
        link = np.vstack((tmp0[i, :], tmp1[i, :]))
        nb_surf = int(np.ceil(np.linalg.norm(np.diff(link, axis=0)) / max_length))
        link = np.linspace(tmp0[i, :], tmp1[i, :], nb_surf)
        Pall = np.vstack((Pall, link[1:-1, :]))
        # Pall = np.vstack((Pall, np.linspace(link[1, :], link[-2, :], nb_surf)))

    points = np.column_stack((Pall[:, 0].ravel(), Pall[:, 1].ravel(), Pall[:, 2].ravel()))
    cloud = pv.PolyData(points)
    surf = cloud.delaunay_2d()
    # Save as CFM_TS
    CFM_surface = CFM_TS(surf.points[:, 0], surf.points[:, 1], surf.points[:, 2], surf.regular_faces,
                         name='SURFACE JOINT')
    # Extract the edges of the tsurf mesh | Uses PyVista
    # edges = surf.extract_feature_edges()
    # CFM_surface.segments = CFM_segments(edges)
    CFM_surface.connections = CFM_connections(CFM_surface.triangles)
    CFM_surface.segments = CFM_edges_with_support(CFM_surface, Edge0, Edge1)
    # test_ends = CFM_surface.segments[0]['vertices'][[0, -1], :]
    # if np.diff(test_ends, axis=0)[0][1] < 0:  # North first => flip orders
    #     CFM_surface.segments = CFM_flip_segments(CFM_surface.segments)
    return CFM_surface


def surf_to_CFM(surf):
    # Initialize CFM object
    CFM_surface = CFM_TS(surf.points[:, 0], surf.points[:, 1], surf.points[:, 2], surf.regular_faces,
                         name='Clipped')
    # Extract the edges of the tsurf mesh | Uses PyVista
    # edges = surf.extract_feature_edges()
    # CFM_surface.segments = CFM_segments(edges)
    CFM_surface.connections = CFM_connections(CFM_surface.triangles)
    CFM_surface.segments = CFM_edges(CFM_surface)

    return CFM_surface


def read_CFM(filename):  # Somehow giving me problems
    print("Processing file {}".format(filename))
    # CFM_surface = read_CFM(filename)
    CFM_surface = CFM_TS(filename)
    CFM_surface.connections = CFM_connections(CFM_surface.triangles)
    CFM_surface.segments = CFM_edges(CFM_surface)

    return CFM_surface


def CFM_Tjunction(SURF, EDGE):
    # Figure which point on EDGE is at the top
    if EDGE['vertices'][0, 2] < EDGE['vertices'][-1, 2]:  # order from bottom upward
        # print('ordered upward')
        point_top = EDGE['vertices'][-1, :]
        point_bottom = EDGE['vertices'][0, :]
    else:
        # print('ordered downward')
        point_top = EDGE['vertices'][0, :]
        point_bottom = EDGE['vertices'][-1, :]
    # find the point on SURF's top segment that is nearest to the top of EDGE
    dx = SURF.segments[0]['vertices'][:, 0] - point_top[0]
    dy = SURF.segments[0]['vertices'][:, 1] - point_top[1]
    dz = SURF.segments[0]['vertices'][:, 2] - point_top[2]
    dist_sq = (dx * dx + dy * dy + dz * dz)  # no need for square root at this stage
    ind_minimum_top = np.argmin(dist_sq)
    pt_grid_min = SURF.segments[0]['vertices'][ind_minimum_top, :]
    point_intersect_top = pt_grid_min
    dp_edge = [[dx[ind_minimum_top], dy[ind_minimum_top], dz[ind_minimum_top]]]
    if ind_minimum_top < SURF.segments[0]['vertices'].shape[0]:  # Try next segment
        pt_grid_end = SURF.segments[0]['vertices'][ind_minimum_top + 1, :]
        dp_segment = pt_grid_end - pt_grid_min
        f = -np.dot(dp_edge, dp_segment) / np.dot(dp_segment, dp_segment)
        if f > 0:
            point_intersect_top = pt_grid_min + f * (pt_grid_end - pt_grid_min)
    if ind_minimum_top > 0:  # Try previous segment
        pt_grid_end = SURF.segments[0]['vertices'][ind_minimum_top - 1, :]
        dp_segment = pt_grid_end - pt_grid_min
        f = -np.dot(dp_edge, dp_segment) / np.dot(dp_segment, dp_segment)
        if f > 0:
            point_intersect_top = pt_grid_min + f * (pt_grid_end - pt_grid_min)
    # Find the normal to the connecting surface
    vect_edge = point_bottom - point_top
    vect_link = point_intersect_top - point_top
    vect_normal = np.cross(vect_edge, vect_link)
    # use pyVista to make two clipped surface
    fault_surface = pv.PolyData(SURF.vertices,
                                np.hstack(
                                    np.concatenate((np.ones([len(SURF.triangles), 1], 'i') * 3, SURF.triangles), 1))
                                )
    clipped = fault_surface.clip(normal=vect_normal, origin=point_top)
    CFM_surface = surf_to_CFM(clipped)  # First clipped fault
    clipped2 = fault_surface.clip(normal=vect_normal, origin=point_top, invert=False)
    CFM_surface2 = surf_to_CFM(clipped2)  # Second clipped fault
    # Which fault is South?
    if CFM_surface.segments[0]['mean'][1] < CFM_surface2.segments[0]['mean'][1]:
        print('First surface is South')
        FAULT_south = CFM_surface
        FAULT_north = CFM_surface2
    else:
        print('First surface is North')
        FAULT_south = CFM_surface2
        FAULT_north = CFM_surface
    # Southern fault's Edge1 should be at the T junction, but check
    iEdge = 1
    dx = FAULT_south.segments[iEdge]['vertices'][:, 0] - point_intersect_top[0]
    dy = FAULT_south.segments[iEdge]['vertices'][:, 1] - point_intersect_top[1]
    dz = FAULT_south.segments[iEdge]['vertices'][:, 2] - point_intersect_top[2]
    dist_sq = (dx * dx + dy * dy + dz * dz)
    print(f"Minimum surface distance: {np.min(dist_sq)}")
    if np.sqrt(np.min(dist_sq)) > 1000:  # too far, it's not the correct edge
        print('Wrong edge')
        iEdge = 3
    LINK_surface = CFM_join(FAULT_south.segments[iEdge], EDGE)

    return FAULT_south, FAULT_north, LINK_surface


def CFM_view(CFM, elevation, azimuth):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for i_file in range(len(CFM)):
        CFM_surface = CFM[i_file]
        ax.plot(CFM_surface.segments[0]['vertices'][0, 0], CFM_surface.segments[0]['vertices'][0, 1],
                CFM_surface.segments[0]['vertices'][0, 2], 'o', label=f"Start {i_file}")
        # ax.plot_trisurf(CFM_surface.x, CFM_surface.y, CFM_surface.z, triangles=CFM_surface.triangles,
        #                 cmap = plt.cm.Spectral, linewidths = 0.2, edgecolors = 'k', alpha = 0.5)
        ax.plot_trisurf(CFM_surface.x, CFM_surface.y, CFM_surface.z, triangles=CFM_surface.triangles,
                        color='none', linewidths=0.2, edgecolors='k')
        for i in range(len(CFM_surface.segments)):
            ax.plot(CFM_surface.segments[i]['vertices'][:, 0], CFM_surface.segments[i]['vertices'][:, 1],
                    CFM_surface.segments[i]['vertices'][:, 2],
                    '.-', label=f"Segment {i_file}-{i}", linewidth=1)
    # ax.quiver(point_top[0], point_top[1], point_top[2], [vect_link[0],vect_edge[0],vect_normal[0]], [vect_link[1],vect_edge[1],vect_normal[1]], [vect_link[2],vect_edge[2],vect_normal[2]],length =  1, normalize=False)
    ax.set_xlabel('Easting')
    ax.set_ylabel('Northing')
    ax.set_zlabel('Vertical')
    ax.legend()
    plt.axis('equal')
    ax.view_init(elev=elevation, azim=azimuth)
    plt.show()
    return fig, ax


def CFM_merge(M0, M1):  # Take two surface and get only one; remove duplicate point
    # Find common edges.
    commonM0 = ()
    commonM1 = ()
    commonD = np.inf
    for i0 in range(4):
        for i1 in range(4):
            V0 = M0.segments[i0]['vertices']
            V1 = M1.segments[i1]['vertices']
            if len(V0) == len(V1):
                D = np.linalg.norm(V0 - V1)
                # print(f"i0={i0}, i1={i1}, D={D}")
                if D < commonD:
                    commonM0 = M0.segments[i0]['nodes']
                    commonM1 = M1.segments[i1]['nodes']
                    commonD = D
                    # print(f"Saving with D={D}")
                    # print(f"commonM0={commonM0}")
                    # print(f"commonM1={commonM1}")
                    # print(f"{V0}\nNEXT\n{V1}")

                D = np.linalg.norm(V0 - np.flipud(V1))
                # print(f"i0={i0}, i1={i1}, flipped, D={D}")
                if D < commonD:
                    commonM0 = M0.segments[i0]['nodes']
                    commonM1 = np.flipud(M1.segments[i1]['nodes'])
                    commonD = D
                    # print(f"Saving with D={D}")
                    # print(f"commonM0={commonM0}")
                    # print(f"commonM1={commonM1}")
                    # print(f"{V0}\nNEXT\n{V1}")
    nb_pt0 = len(M0.vertices)
    nb_pt1 = len(M1.vertices)
    # initialize the merged array
    M2 = copy.deepcopy(M0)
    # put the points together
    M2.vertices.extend(M1.vertices)
    # add the triangles, but first renumber the points of the added triangles
    M2.triangles.extend(list(np.asarray(M1.triangles) + nb_pt0))
    #     M2.x, M2.y, M2.z = zip(*M2.vertices)
    # find the actual indices of the added surface
    sort_order = np.argsort(commonM1)  # [::-1]
    target_index = np.array(commonM0)[sort_order]
    M1_index = np.array(commonM1)[sort_order] + nb_pt0
    # replace indices in triangle definition
    dum = list(range(len(M2.vertices)))  # new indices #someone works only with lists
    for i in range(len(target_index)):
        dum[M1_index[i]] = target_index[i].item()
        for ind in range(M1_index[i] + 1, len(dum)):
            dum[ind] = dum[ind] - 1
    duma = np.array(dum)  # Because lists are confusig
    # Replace in the triangles
    for ind in range(len(M2.triangles)):
        M2.triangles[ind] = duma[M2.triangles[ind]]
    # delete extra points
    M2.vertices = [element for i, element in enumerate(M2.vertices) if i not in M1_index]
    M2.x, M2.y, M2.z = zip(*M2.vertices)

    M2.connections = CFM_connections(M2.triangles)
    M2.segments = CFM_edges(M2)

    return M2
