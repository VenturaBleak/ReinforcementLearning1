"""
adapted from: https://medium.com/data-science-in-your-pocket/how-to-create-a-custom-openai-gym-environment-with-codes-fb5de015de3c
"""
import gymnasium
from gymnasium import spaces
import numpy as np
import pygame
import time
from pyamaze import maze

class MazeGameEnv(gymnasium.Env):
    def __init__(self, generation_frequency=np.inf, min_maze_size=8, max_maze_size=8):
        super(MazeGameEnv, self).__init__()

        # Set maze generation parameters
        self.generation_frequency = generation_frequency
        self.min_maze_size = min_maze_size
        self.max_maze_size = max_maze_size
        self.reset_counter = 0

        # Initial maze generation
        self._generate_maze()

        # Set facing directions, inital direction at random
        self.facing_directions = ['N', 'S', 'W', 'E']
        self.facing_direction = np.random.choice(self.facing_directions)

        # Set maze map and dimensions
        self.num_rows = max([k[0] for k in self.maze_map.keys()])
        self.num_cols = max([k[1] for k in self.maze_map.keys()])

        # Randomize start and goal positions
        self.start_pos = self._random_position()
        self.goal_pos = self._random_position()

        # Ensure start and goal positions are not the same
        while self.start_pos == self.goal_pos:
            self.goal_pos = self._random_position()

        # respawn positions
        self.current_pos = self.start_pos

        # Save the current vision matrix and vision coordinates after every step
        self.vision_matrix, _ = self._get_vision()  # We only need the matrix here

        # Define action and observation space
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3, 3, 7), dtype=np.int16)

        # boolean to check if the pygame display is initialized
        self.display_initialized = False  # Set to True once the pygame window is set up

        self.steps_since_reset = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.reset_counter += 1

        # Generate a new maze if the counter reaches the generation frequency
        if self.reset_counter % self.generation_frequency == 0:

            self._generate_maze()

            # Randomize start and goal positions
            self.start_pos = self._random_position()
            self.goal_pos = self._random_position()

            # Ensure start and goal positions are not the same
            while self.start_pos == self.goal_pos:
                self.goal_pos = self._random_position()

            # Update the number of rows and columns
            self.num_rows = max([k[0] for k in self.maze_map.keys()])
            self.num_cols = max([k[1] for k in self.maze_map.keys()])

            if self.display_initialized:
                # Shutdown the pygame display
                pygame.display.quit()
                # Update pygame window size
                self.screen = pygame.display.set_mode((self.num_cols * self.cell_size, self.num_rows * self.cell_size))

        self.current_pos = self.start_pos
        self.vision_matrix, _ = self._get_vision()
        self.facing_direction = np.random.choice(self.facing_directions)

        self.steps_since_reset = 0

        return self.vision_matrix, {}

    def _random_position(self):
        """Generate a random position within the maze."""
        x = np.random.randint(1, self.num_rows + 1)
        y = np.random.randint(1, self.num_cols + 1)
        return (x, y)

    def _generate_maze(self):
        size = np.random.randint(self.min_maze_size, self.max_maze_size + 1)
        self.maze_obj = maze(size, size)
        self.maze_obj.CreateMaze(loopPercent=50)
        self.maze_map = self.maze_obj.maze_map
        self.num_cols = size
        self.num_rows = size
        self.goal_pos = (self.num_rows, self.num_cols)

    def step(self, action):
        original_pos = self.current_pos
        new_pos = list(self.current_pos)
        if action == 0:  # Up
            new_pos[0] -= 1
        elif action == 1:  # Down
            new_pos[0] += 1
        elif action == 2:  # Left
            new_pos[1] -= 1
        elif action == 3:  # Right
            new_pos[1] += 1

        # Update facing direction based on the action
        self.facing_direction = self.facing_directions[action]

        # cost of moving
        reward = -1.0/self.num_rows

        # Check validity
        if self._is_valid_position(tuple(new_pos)):
            self.current_pos = tuple(new_pos)
            reward += 0.0  # Neutral reward for a valid move
        else:
            reward -= 1.0/self.num_rows  # Negative reward for an invalid move
            self.current_pos = original_pos  # Reset to original position

        # Goal check
        if self.current_pos == self.goal_pos:
            reward += 1.0  # Large reward for reaching the goal
            terminated = True
        else:
            terminated = False

        # Check for truncation
        self.steps_since_reset += 1  # Increment the step counter
        truncated = self.steps_since_reset > (self.num_rows ** 2)

        if truncated:
            reward -= 1.0

        # Save the current vision matrix and vision coordinates after every step
        self.vision_matrix, _ = self._get_vision()  # We only need the matrix here

        return self.vision_matrix, reward, terminated, truncated, {}

    def _is_valid_position(self, pos):
        if pos not in self.maze_map:
            return False

        move_direction = {
            (-1, 0): 'N',
            (1, 0): 'S',
            (0, -1): 'W',
            (0, 1): 'E'
        }

        dx = pos[0] - self.current_pos[0]
        dy = pos[1] - self.current_pos[1]
        direction = move_direction.get((dx, dy))

        return self.maze_map[self.current_pos][direction] == 1

    def get_valid_actions(self):
        valid_actions = []
        for action in range(self.action_space.n):
            new_pos = list(self.current_pos)
            if action == 0:  # Up
                new_pos[0] -= 1
            elif action == 1:  # Down
                new_pos[0] += 1
            elif action == 2:  # Left
                new_pos[1] -= 1
            elif action == 3:  # Right
                new_pos[1] += 1

            if self._is_valid_position(tuple(new_pos)):
                valid_actions.append(action)

        return valid_actions

    def random_valid_action(self):
        return np.random.choice(self.get_valid_actions())

    def _get_vision(self):
        vision = np.zeros((3, 3, 7), dtype=np.int16)  # 3x3 grid with 7 features per cell
        vision_coordinates = []  # List to store the coordinates of cells in the vision

        # Define vision offsets based on the facing direction
        if self.facing_direction == 'N':
            offsets = [(0, -1), (0, 0), (0, 1),
                       (-1, -1), (-1, 0), (-1, 1),
                       (-2, -1), (-2, 0), (-2, 1)]
        elif self.facing_direction == 'E':
            offsets = [(-1, 0), (0, 0), (1, 0),
                       (-1, 1), (0, 1), (1, 1),
                       (-1, 2), (0, 2), (1, 2)]
        elif self.facing_direction == 'S':
            offsets = [(0, 1), (0, 0), (0, -1),
                       (1, 1), (1, 0), (1, -1),
                       (2, 1), (2, 0), (2, -1)]
        else:  # facing 'W'
            offsets = [(1, 0), (0, 0), (-1, 0),
                       (1, -1), (0, -1), (-1, -1),
                       (1, -2), (0, -2), (-1, -2)]

        for i, (dx, dy) in enumerate(offsets):
            x, y = self.current_pos[0] + dx, self.current_pos[1] + dy
            vision_coordinates.append((x, y))  # Add to vision_coordinates
            vision[i // 3, i % 3] = self._get_cell_info((x, y))

        return vision, vision_coordinates

    def _get_cell_info(self, pos):
        info = np.zeros(7, dtype=np.int8)  # To handle negative values and the inside/outside feature.

        # If the position is outside the maze boundaries, set info to -1
        if pos[0] < 1 or pos[0] > self.num_rows or pos[1] < 1 or pos[1] > self.num_cols:
            return np.array([-1, -1, -1, -1, -1, -1, 0], dtype=np.int8)

        # Wall information
        if pos in self.maze_map:
            for i, direction in enumerate(self.facing_directions):
                info[i] = self.maze_map[pos][direction]

        # Start and goal position information
        info[4] = 1 if pos == self.start_pos else 0
        info[5] = 1 if pos == self.goal_pos else 0

        # Mark this cell as inside
        info[6] = 1

        return info

    def _setup_display(self):
        pygame.init()
        self.cell_size = 40
        self.screen = pygame.display.set_mode((self.num_cols * self.cell_size, self.num_rows * self.cell_size))
        self.display_initialized = True

    def render(self):
        # Initialize the pygame window if it's not set up yet
        if not self.display_initialized:
            self._setup_display()

        self.screen.fill((200, 200, 200))  # Change to light grey
        line_width = 4  # Setting a width for the walls

        for row in range(1, self.num_rows + 1):  # Adjusted to 1-based indexing
            for col in range(1, self.num_cols + 1):  # Adjusted to 1-based indexing
                cell_left = (col - 1) * self.cell_size
                cell_top = (row - 1) * self.cell_size
                cell = (row, col)

                if cell not in self.maze_map:
                    continue  # Skip rendering for this cell if it's not in the maze_map

                # Draw walls
                if self.maze_map[cell]['W'] == 0:
                    pygame.draw.line(self.screen, (0, 0, 0), (cell_left, cell_top),
                                     (cell_left, cell_top + self.cell_size), line_width)
                if self.maze_map[cell]['E'] == 0:
                    pygame.draw.line(self.screen, (0, 0, 0), (cell_left + self.cell_size, cell_top),
                                     (cell_left + self.cell_size, cell_top + self.cell_size), line_width)
                if self.maze_map[cell]['N'] == 0:
                    pygame.draw.line(self.screen, (0, 0, 0), (cell_left, cell_top),
                                     (cell_left + self.cell_size, cell_top), line_width)
                if self.maze_map[cell]['S'] == 0:
                    pygame.draw.line(self.screen, (0, 0, 0), (cell_left, cell_top + self.cell_size),
                                     (cell_left + self.cell_size, cell_top + self.cell_size), line_width)

                # Draw start and goal positions
                cell_margin = 3
                if cell == self.start_pos:
                    pygame.draw.rect(self.screen, (0, 255, 0), (cell_left + cell_margin, cell_top  + cell_margin,
                                                                self.cell_size  - cell_margin, self.cell_size  - cell_margin), 3)
                elif cell == self.goal_pos:
                    pygame.draw.rect(self.screen, (255, 0, 0), (cell_left  + cell_margin, cell_top  + cell_margin,
                                                                self.cell_size - cell_margin, self.cell_size - cell_margin), 3)

                # Draw agent
                if cell == self.current_pos:
                    center_x = int(cell_left + self.cell_size / 2)
                    center_y = int(cell_top + self.cell_size / 2)
                    half_size = int(self.cell_size / 3)

                    if self.facing_direction == 'N':
                        triangle_vertices = [(center_x, center_y - half_size),
                                             (center_x - half_size, center_y + half_size),
                                             (center_x + half_size, center_y + half_size)]
                    elif self.facing_direction == 'E':
                        triangle_vertices = [(center_x + half_size, center_y),
                                             (center_x - half_size, center_y - half_size),
                                             (center_x - half_size, center_y + half_size)]
                    elif self.facing_direction == 'S':
                        triangle_vertices = [(center_x, center_y + half_size),
                                             (center_x - half_size, center_y - half_size),
                                             (center_x + half_size, center_y - half_size)]
                    else:  # 'W'
                        triangle_vertices = [(center_x - half_size, center_y),
                                             (center_x + half_size, center_y - half_size),
                                             (center_x + half_size, center_y + half_size)]

                    pygame.draw.polygon(self.screen, (0, 0, 255), triangle_vertices)

                # Draw agent's vision using the stored vision matrix
                vision_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                vision_color = (255, 165, 0, 5)  # 90% transparent orange
                vision_surface.fill(vision_color)

                _, vision_coordinates = self._get_vision()  # We only need the coordinates here

                for x, y in vision_coordinates:
                    cell_left = (y - 1) * self.cell_size
                    cell_top = (x - 1) * self.cell_size
                    self.screen.blit(vision_surface, (cell_left, cell_top))

                pygame.display.update()

    def describe_state(self):
        """
        Returns a list of lists where each sublist contains the verbal description of the cell.
        """
        descriptions = []
        _, vision_coordinates = self._get_vision()  # Getting the vision coordinates

        for i, row in enumerate(self.vision_matrix):
            row_desc = []
            for j, cell in enumerate(row):
                coord = vision_coordinates[i * 3 + j]  # retrieve the cell's coordinates
                desc = self._describe_cell(cell, coord)
                row_desc.append(desc)
            descriptions.append(row_desc)

        # print descriptions, line by line for each item in the list
        for row in descriptions:
            for cell in row:
                print(cell)
            print("\n")
        time.sleep(100)
        return descriptions

    def _describe_cell(self, cell, coord):
        """
        Returns a human-readable description of a given cell.
        """
        # Determine inside/outside maze
        if cell[6] == 1:
            status = "Inside the maze"
        elif cell[6] == -1:
            status = "Outside of the maze"
        else:
            status = "Unknown status"

        # Walls
        walls = {
            "N": "Wall on top",
            "S": "Wall at the bottom",
            "E": "Wall on the right",
            "W": "Wall on the left"
        }
        wall_desc = [walls[k] for i, k in enumerate(["N", "S", "E", "W"]) if cell[i] == 0]

        # Start/End position
        start_end_desc = []
        if cell[4] == 1:
            start_end_desc.append("Starting position")
        if cell[5] == 1:
            start_end_desc.append("End position")

        return f"Cell at ({coord[0]}, {coord[1]}): {status}. {'; '.join(wall_desc)}. {'; '.join(start_end_desc)}."