class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.explored = False
        self.g_cost = 0
        self.h_cost = 0
        self.parent = None

    def __str__(self):
        return f"node[{self.x}][{self.y}]"

    def __repr__(self):
        return str(self)

    def get_f_cost(self):
        return self.h_cost + self.g_cost

    def get_distance(self, node):
        return abs(self.x - node.x) + abs(self.y - node.y)


class AStar:

    def __init__(self, start_x, start_y, goal_x, goal_y):
        self.grid = list()
        for i in range(10):
            row = list()
            for j in range(10):
                row.append(Node(i, j))
            self.grid.append(row)
        self.start = self.grid[start_x][start_y]
        self.goal = self.grid[goal_x][goal_y]
        self.explored_nodes = list()
        self.unexplored_nodes = list()

    def is_goal(self, node):
        if node.x == self.goal.x and node.y == self.goal.y:
            return True
        else:
            return False

    def get_neighbours(self, node):
        neighbours = []
        for i in [-1, 1]:
            if 0 <= node.x + i < 10:
                neighbours.append(self.grid[node.x + i][node.y])
        for j in [-1, 1]:
            if 0 <= node.y + j < 10:
                neighbours.append(self.grid[node.x][node.y + j])

        return neighbours

    def findPath(self):
        self.explored_nodes = list()
        self.unexplored_nodes = list()
        self.unexplored_nodes.append(self.start)

        while len(self.unexplored_nodes) > 0:
            node = self.unexplored_nodes.pop(0)
            if self.is_goal(node):
                return self.trace_path(node)
            self.explore_node(node)
            self.unexplored_nodes.sort(key=lambda node: node.get_f_cost(), reverse=False)

    def trace_path(self, node):
        path = list()
        path.append(node)
        while node.parent is not None:
            path.append(node.parent)
            node = node.parent
        return path

    def explore_node(self, node):
        if node.explored:
            return
        node.explored = True
        self.explored_nodes.append(node)
        neighbours = self.get_neighbours(node)
        for neighbour in neighbours:
            new_g_cost = node.g_cost + 1
            if new_g_cost < neighbour.g_cost or neighbour not in self.explored_nodes:
                neighbour.g_cost = node.g_cost + 1
                neighbour.h_cost = self.goal.get_distance(neighbour)
                neighbour.parent = node
            if neighbour not in self.unexplored_nodes:
                self.unexplored_nodes.append(neighbour)


astar = AStar(1, 1, 5, 5)
print(astar.findPath())
