from pyamaze import maze,COLOR,agent

m=maze(2,2)
m.CreateMaze(theme=COLOR.light)
print(m.maze_map)
# Map of the Maze: A Maze is generated randomly. It is important to know the information of different opened and closed walls of the Maze. That information is available in the attribute maze_map. It is a dictionary with the keys as the cells of the Maze and value as another dictionary with the information of the four walls of that cell in four directions; East, West, North and South.
# m.run()

