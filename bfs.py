import collections

"""
Breadth First Search Algorithum
"""

def bfs(grid, start):
	queue = collections.deque([[start]])
	seen = set([start])
	while queue:
		path = queue.popleft()
#		print(path)
		x, y = path[-1]
		if grid[y][x] == goal:
			return path
		for x2, y2 in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
			if 0 <= x2 < width and 0 <= y2 < height and \
			   grid[y2][x2] != wall1 and grid[y2][x2] != wall2 and (x2, y2) not in seen:
				queue.append(path + [(x2, y2)])
#				print(queue)
				seen.add((x2, y2))
	return({"now way"})

wall1, wall2, clear, goal = "#", "*", ".", "G"
width, height = 10, 5
grid = ["S.........",
		"...#...#*.",
		"..**...#..",
		"##..####..",
		".........G"]
path = bfs(grid, (0, 0))

print("--- Input data ---")
for i in range(len(grid)):
	print(grid[i])
print()

print("--- Step number ---")
print(len(path))
print()

print("--- Step path ---")
print(path)