'''
This module provides search algorithms for finding the path
with the lowest total weight between two points.
'''

import numpy as np


class Node:

    def __init__(self, grid_coord, coord, parent, own_weight, min_weight, ready):

        self.grid_coord = grid_coord
        self.coord = coord
        self.parent = parent
        self.own_weight = own_weight
        self.min_weight = min_weight
        self.ready = ready


class Grid:

    def __init__(self, size):

        self.size = size
        self.grid = [None] * (size * size)

    def add(self, row, col, node):
        self.grid[self.__idx(row, col)] = node

    def at(self, row, col):
        return self.grid[self.__idx(row, col)]

    def __idx(self, row, col):
        return row * self.size + col


def __initializeBFS(map, pa, size):

    # Initialize the necessary data structures:
    #  -> grid
    #  -> queue for the recently explored elements

    # GRID
    grid_size = (size + 1) * 2  # point pa, size pieces to left and right
    grid = Grid(grid_size)  # point pa, size pieces to left and right

    def weight_after_checking_grid_point_is_inside_map(row, col):
        r = row + pa[0] - size
        c = col + pa[1] - size

        if r < 0 or r > map.shape[0] - 1 or c < 0 or c > map.shape[1] - 1:
            return None
        else:
            return map[r, c]

    for row in range(grid_size):
        for col in range(grid_size):

            pm = []  # parameters of node
            pm.append((row, col))  # grid_coord
            pm.append((row + pa[0] - size, col + pa[1] - size))  # coord (on the map)
            pm.append(None)  # parent (not known yet)
            pm.append(weight_after_checking_grid_point_is_inside_map(row, col))  # own_weight
            pm.append(None)   # min_weight (no minimum yet)
            pm.append(False)  # ready

            node = Node(pm[0], pm[1], pm[2], pm[3], pm[4], pm[5])
            grid.add(row, col, node)

    # QUEUE: for recently discovered
    queue = {"queue": [], "min": None}

    return grid, queue


def __discover(grid, node, fast):

    # Discovers the neighbours of a node.
    # Only not ready nodes are taken into account.

    row_shift = [-1, 0, 1]
    col_shift = [-1, 0, 1]

    if node.grid_coord[0] == 0:
        row_shift.remove(-1)
    elif node.grid_coord[0] == grid.size - 1:
        row_shift.remove(1)

    if node.grid_coord[1] == 0:
        col_shift.remove(-1)
    elif node.grid_coord[1] == grid.size - 1:
        col_shift.remove(1)

    neighbors = []

    for rs in row_shift:
        for cs in col_shift:
            i = node.grid_coord[0] + rs
            j = node.grid_coord[1] + cs
            candidate = grid.at(i, j)

            if fast:
                include = candidate.own_weight == 0  # the white pixels are the segmenting lines
            else:
                include = True

            if not candidate.ready and candidate.own_weight is not None and include:
                neighbors.append(candidate)

    return neighbors


def __update(node, nodes, queue):

    # After the new node with the minimum total weight
    # is chosen it discovers its neighbors and update them.

    for n in nodes:

        temp = n.own_weight + node.min_weight
        if (n.min_weight is None) or (n.min_weight > temp):
            if n.min_weight is None:
                queue["queue"].append(n)
            n.min_weight = temp
            n.parent = node

    node.ready = True

    queue["queue"].remove(node)


def bfs(map, pa, pb, size, fast=False):

    # Initialize grid and queue.
    grid, queue = __initializeBFS(map, pa, size)

    # Put the first node (at pa) into the queue.
    grid.at(size, size).min_weight = grid.at(size, size).own_weight
    queue["queue"].append(grid.at(size, size))
    queue["min"] = grid.at(size, size)

    # Start BFS.
    # Continue search until pb becomes ready.

    while not grid.at(pb[0] - pa[0] + size, pb[1] - pa[1] + size).ready and not len(queue["queue"]) == 0:

        # Choose the node with the minimum weighted path.
        min_node = queue["min"] = min(queue["queue"], key=lambda x: x.min_weight)

        neighbors = __discover(grid, min_node, fast)

        __update(min_node, neighbors, queue)

    # Connect the path from pb to pa.

    path = []
    node = grid.at(pb[0] - pa[0] + size, pb[1] - pa[1] + size)  # pb has coordinate on the map not on the grid
    while node is not None:

        path.append(node)
        node = node.parent

    # Convert path to a list with coordinate tuples.
    coord_path = []
    for node in path:
        coord_path.append(node.coord)

    # print("BFS has finished searching for minimum.")

    return coord_path


def test_bfs():

    # Creating an exemplary image
    size = 500
    img = np.ones((size, size))
    for p in range(101):
        img[250 - p, 250] = 0

    # Now try bfs
    path = bfs(img, (250, 250), (150, 250), 105)

    # The expected path.
    expected_path = []
    for p in range(100, -1, -1):
        expected_path.append((250 - p, 250))

    # Compare the two.
    correct = True
    for x, y in zip(path, expected_path):
        if not x == y:
            correct = False
            break

    print(correct)
    print(path[0:10])
    print(expected_path[0:10])

