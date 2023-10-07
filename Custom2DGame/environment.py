"""""
adapted from: https://medium.com/data-science-in-your-pocket/how-to-create-a-custom-openai-gym-environment-with-codes-fb5de015de3c
"""
import gymnasium
from gymnasium import spaces
import numpy as np
import pygame
import time

class PlainFieldEnv(gymnasium.Env):
    def __init__(self, field_size=6, vision_size=7):
        super(PlainFieldEnv, self).__init__()

        # assert, min field size is vision size * 2 + 1
        # assert field_size >= vision_size * 2 + 1, "field size must be at least vision size * 2 + 1"

        # Field dimensions, including the buffer
        self.buffer_size = int(np.ceil(vision_size / 2.0))
        self.num_rows = 30 + 2 * self.buffer_size
        self.num_cols = 50 + 2 * self.buffer_size
        self.vision_size = vision_size

        # Set facing directions
        self.facing_directions = ['N', 'S', 'W', 'E']

        # Initialize the step counter
        self.steps_since_reset = 0

        # Flag the display as not initialized
        self.display_initialized = False

        # Reward dictionary
        self.rewards = {
            'move': -1.0 / field_size,
            'invalid_move': -2.0 / field_size,
            'goal_reached': 1.0,
            'truncated': -1.0
        }

        # Introducing new cell states
        self.cell_states = {
            'object_layer': {
                'empty': 0,
                'agent': 1,
                'obstacle': 2
            },
            'blocked_layer': {
                'not_blocked': 0,
                'blocked': 1
            },
            'objective_layer': {
                'none': 0,
                'start': 1,
                'goal': 2
            },
            'playing_area': {
                'playing_field': 0,
                'buffer': 1
            }
        }

        # Define action and observation space
        self.action_space = spaces.Discrete(4)  # 4 actions: Up, Down, Left, Right

        # Adjust the observation space for the agent's vision
        max_val = max([max(self.cell_states[layer].values()) for layer in self.cell_states])
        self.observation_space = spaces.Box(low=0, high=max_val, shape=(3, vision_size, vision_size), dtype=np.int8)

        # Initialize the 4D map to zeros
        self.map = np.zeros((4, self.num_rows, self.num_cols), dtype=np.int8)

        # Set a flag to ensure reset is called before using the environment
        self.initialized = False

        # Initialize the 4D map to zeros
        self.map = np.zeros((4, self.num_rows, self.num_cols), dtype=np.int8)

        # Fill the entire map with empty and not blocked by default
        self.map[0, :, :] = self.cell_states['object_layer']['empty']
        self.map[1, :, :] = self.cell_states['blocked_layer']['not_blocked']

        # Define buffer as obstacles and blocked
        # Top buffer
        self.map[0, :self.buffer_size, :] = self.cell_states['object_layer']['obstacle']
        self.map[1, :self.buffer_size, :] = self.cell_states['blocked_layer']['blocked']
        # Bottom buffer
        self.map[0, -self.buffer_size:, :] = self.cell_states['object_layer']['obstacle']
        self.map[1, -self.buffer_size:, :] = self.cell_states['blocked_layer']['blocked']
        # Left buffer
        self.map[0, :, :self.buffer_size] = self.cell_states['object_layer']['obstacle']
        self.map[1, :, :self.buffer_size] = self.cell_states['blocked_layer']['blocked']
        # Right buffer
        self.map[0, :, -self.buffer_size:] = self.cell_states['object_layer']['obstacle']
        self.map[1, :, -self.buffer_size:] = self.cell_states['blocked_layer']['blocked']

        # define everything that is not blocked as playing field
        self.map[3][self.map[1] == self.cell_states['blocked_layer']['not_blocked']] = self.cell_states['playing_area']['playing_field']

        # define everything that is blocked as buffer
        self.map[3][self.map[1] == self.cell_states['blocked_layer']['blocked']] = self.cell_states['playing_area']['buffer']

    def _random_position(self):
        """ returns: random position in the playing field, that is not blocked"""
        valid_positions = list(np.argwhere((self.map[1] == self.cell_states['blocked_layer']['not_blocked']) &
                                           (self.map[3] == self.cell_states['playing_area']['playing_field'])))
        return tuple(valid_positions[np.random.choice(len(valid_positions))])

    def procedural_generation(self):
        # random facing direction
        self.facing_direction = np.random.choice(self.facing_directions)

        if self.initialized:
            # clear old agent position
            self.map[0][self.current_pos[0]][self.current_pos[1]] = self.cell_states['object_layer']['empty']
            self.map[1][self.current_pos[0]][self.current_pos[1]] = self.cell_states['blocked_layer']['not_blocked']

            # clear old start and goal position
            self.map[2][self.start_pos[0]][self.start_pos[1]] = self.cell_states['objective_layer']['none']
            self.map[2][self.goal_pos[0]][self.goal_pos[1]] = self.cell_states['objective_layer']['none']

        # random start and goal position
        self.start_pos = self._random_position()
        self.goal_pos = self._random_position()
        while self.start_pos == self.goal_pos:
            self.goal_pos = self._random_position()

        # set start and goal position in the objective layer
        self.map[2][self.start_pos[0]][self.start_pos[1]] = self.cell_states['objective_layer']['start']
        self.map[2][self.goal_pos[0]][self.goal_pos[1]] = self.cell_states['objective_layer']['goal']

        # spawn agent at start position
        self.map[0][self.start_pos[0]][self.start_pos[1]] = self.cell_states['object_layer']['agent']
        self.map[1][self.start_pos[0]][self.start_pos[1]] = self.cell_states['blocked_layer']['not_blocked']

        # set current position to start position
        self.current_pos = self.start_pos

        # Remember to update the initialized flag at the end
        self.initialized = True

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.procedural_generation()

        # Reset the step counter
        self.steps_since_reset = 0

        # # Validate the generated map
        validator = self.MapValidator(self)
        validator.validate()

        return self._get_vision(), {}

    def step(self, action):
        if not self.initialized:
            raise RuntimeError("You must call reset() before using the environment!")

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
        reward = self.rewards['move']

        # Check boundaries
        if not self._is_valid_position(tuple(new_pos)):
            reward += self.rewards['invalid_move']  # Note: we're adding because the reward is negative
            self.current_pos = original_pos  # Reset to original position
        else:
            # Clear old agent position
            self.map[0][original_pos[0]][original_pos[1]] = self.cell_states['object_layer']['empty']
            self.map[1][original_pos[0]][original_pos[1]] = self.cell_states['blocked_layer']['not_blocked']

            # Update the agent's current position
            self.current_pos = tuple(new_pos)

            # Set new agent position on the map
            self.map[0][new_pos[0]][new_pos[1]] = self.cell_states['object_layer']['agent']
            self.map[1][new_pos[0]][new_pos[1]] = self.cell_states['blocked_layer']['blocked']

        # Goal check
        if self.current_pos == self.goal_pos:
            reward += self.rewards['goal_reached']
            terminated = True
        else:
            terminated = False

        # Check for truncation
        self.steps_since_reset += 1  # Increment the step counter
        truncated = self.steps_since_reset > (self.num_rows ** 2)

        if truncated:
            reward += self.rewards['truncated']

        return self._get_vision(), reward, terminated, truncated, {}

    def _is_valid_position(self, pos):
        # Check if not blocked
        if self.map[1][pos[0]][pos[1]] == self.cell_states['blocked_layer']['blocked']:
            return False

        # Check if not in the buffer
        if self.map[3][pos[0]][pos[1]] == self.cell_states['playing_area']['buffer']:
            return False

        return True

    def _get_vision(self):
        half_vision = self.vision_size // 2
        vision = np.zeros((3, self.vision_size, self.vision_size), dtype=np.int8)
        vision_coordinates = []  # To store coordinates of cells in the agent's vision

        # Determine the boundaries of the vision in the actual field
        start_row = max(0, self.current_pos[0] - half_vision)
        end_row = min(self.num_rows, self.current_pos[0] + half_vision + 1)

        start_col = max(0, self.current_pos[1] - half_vision)
        end_col = min(self.num_cols, self.current_pos[1] + half_vision + 1)

        for x in range(start_row, end_row):
            for y in range(start_col, end_col):
                # Compute the corresponding position in the vision grid
                vx = x - self.current_pos[0] + half_vision
                vy = y - self.current_pos[1] + half_vision

                # Transfer values from the field to the vision grid
                vision[:, vx, vy] = self.map[:3, x, y]

                # Append to vision coordinates
                vision_coordinates.append((x, y))

        return vision, vision_coordinates

    def _setup_display(self):
        pygame.init()
        self.cell_size = 20
        self.screen = pygame.display.set_mode((self.num_cols * self.cell_size+5, self.num_rows * self.cell_size))
        self.display_initialized = True

    def render(self):
        # Initialize the pygame window if it's not set up yet
        if not self.display_initialized:
            self._setup_display()

        self.screen.fill((200, 200, 200))  # Change to light grey
        line_width = 2  # Setting a width for the boundary

        # Draw field boundary
        pygame.draw.rect(self.screen, (0, 0, 0), (0, 0, self.num_cols * self.cell_size, self.num_rows * self.cell_size),
                         line_width)

        cell_margin = 3
        # Use the map to decide how to draw each cell
        for row in range(1, self.num_rows + 1):
            for col in range(1, self.num_cols + 1):
                cell_left = (col - 1) * self.cell_size
                cell_top = (row - 1) * self.cell_size

                # Get states from map layers
                object_state = self.map[0][row - 1][col - 1]
                blocked_state = self.map[1][row - 1][col - 1]
                objective_state = self.map[2][row - 1][col - 1]

                # Render based on object layer
                if object_state == self.cell_states['object_layer']['agent']:
                    # Get center of cell
                    center_x = int(cell_left + self.cell_size / 2)
                    center_y = int(cell_top + self.cell_size / 2)
                    half_size = int(self.cell_size / 4)

                    # Depending on the facing direction, adjust the agent's rectangle vertices
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

                    pygame.draw.polygon(self.screen, (0, 0, 255),triangle_vertices)

                elif object_state == self.cell_states['object_layer']['obstacle']:
                    pygame.draw.rect(self.screen, (0, 0, 0), (cell_left, cell_top, self.cell_size, self.cell_size))

                # Render based on objective layer
                if objective_state == self.cell_states['objective_layer']['start']:
                    pygame.draw.circle(self.screen, (0, 255, 0),
                                       (int(cell_left + self.cell_size / 2), int(cell_top + self.cell_size / 2)),
                                       int(self.cell_size / 4))
                elif objective_state == self.cell_states['objective_layer']['goal']:
                    pygame.draw.circle(self.screen, (255, 0, 0),
                                       (int(cell_left + self.cell_size / 2), int(cell_top + self.cell_size / 2)),
                                       int(self.cell_size / 4))

        # Draw agent's vision using the stored vision matrix
        vision_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        vision_color = (255, 165, 0, 90)  # 30% transparent orange
        vision_surface.fill(vision_color)

        _, vision_coordinates = self._get_vision()  # We only need the coordinates here

        for x, y in vision_coordinates:
            if 1 <= x <= self.num_rows and 1 <= y <= self.num_cols:  # Ensure we are in the boundary
                cell_left = (y) * self.cell_size
                cell_top = (x) * self.cell_size
                self.screen.blit(vision_surface, (cell_left, cell_top))

        pygame.display.update()

    class MapValidator:
        def __init__(self, env):
            self.env = env

        def validate(self):
            self._validate_buffer()
            self._validate_start_goal_positions()

        def _validate_buffer(self):
            # Top buffer
            assert np.all(self.env.map[0, :self.env.buffer_size, :] == self.env.cell_states['object_layer']['obstacle'])
            # Bottom buffer
            assert np.all(
                self.env.map[0, -self.env.buffer_size:, :] == self.env.cell_states['object_layer']['obstacle'])

            non_obstacle_indices = np.argwhere(
                self.env.map[0, :self.env.buffer_size, :] != self.env.cell_states['object_layer']['obstacle'])
            print("Indices of non-obstacle values:", non_obstacle_indices)
            for index in non_obstacle_indices:
                print("Value at index", tuple(index), ":", self.env.map[0, tuple(index)])


            # Left buffer (excluding the cells already covered by top and bottom buffers)
            assert np.all(self.env.map[0, self.env.buffer_size:-self.env.buffer_size, :self.env.buffer_size] ==
                          self.env.cell_states['object_layer']['obstacle'])
            # Right buffer (excluding the cells already covered by top and bottom buffers)
            assert np.all(self.env.map[0, self.env.buffer_size:-self.env.buffer_size, -self.env.buffer_size:] ==
                          self.env.cell_states['object_layer']['obstacle'])

        def _validate_start_goal_positions(self):
            # Check if start and goal positions are within the buffer
            assert self.env.buffer_size <= self.env.start_pos[0] < self.env.num_rows - self.env.buffer_size
            assert self.env.buffer_size <= self.env.start_pos[1] < self.env.num_cols - self.env.buffer_size
            assert self.env.buffer_size <= self.env.goal_pos[0] < self.env.num_rows - self.env.buffer_size
            assert self.env.buffer_size <= self.env.goal_pos[1] < self.env.num_cols - self.env.buffer_size