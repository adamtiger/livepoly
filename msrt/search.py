'''
This module provides search algorithms for finding the path
with the lowest total weight between two points.
'''


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
        self.grid = [0] * (size * size)

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
    for row in range(grid_size):
        for col in range(grid_size):

            pm = []  # parameters of node
            pm.append((row, col))  # grid_coord
            pm.append((row + pa[0] - size, row + pa[1] - size))  # coord (on the map)
            pm.append(None)  # parent (not known yet)
            pm.append(map[pm[1][0], pm[1][1]])  # own_weight
            pm.append(None)  # min_weight (no minimum yet)
            pm.append(False) # ready

            node = Node(pm[0], pm[1], pm[2], pm[3], pm[4], pm[5])
            grid.add(row, col, node)

    # QUEUE: for recently discovered
    queue = {"queue": [], "min": None}

    return grid, queue


def __discover(grid, node):

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
            i = node.grid[0] + rs
            j = node.grid_coord[1] + cs
            candidate = grid.at(i, j)

            if not candidate.ready:
                neighbors.append(candidate)


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
    queue["queue"].remove[node]
    queue["min"] = min(queue["queue"], key= lambda x: x.min_weight)


def bfs(map, pa, pb, size):

    grid, queue = __initializeBFS(map, pa, size)



