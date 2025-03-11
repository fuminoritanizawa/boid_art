import pygame
import random
import numpy as np

# ==============================
# CONFIGURATION VARIABLES
# ==============================

FULLSCREEN = True  # Set to False for windowed mode
SCREEN_WIDTH = 1920  # Set width if not fullscreen
SCREEN_HEIGHT = 1080  # Set height if not fullscreen
BACKGROUND_COLOR = (15, 15, 20)  # Dark background for contrast
NUM_BOIDS = 200  # Total number of boids
NUM_TYPES = 7  # Number of distinct flock colors
GRADATION_LEVELS = 4  # Sub-levels of color variation within each type
BOID_SIZE = 30  # Size of the triangular boid
BOID_SPEED = 5  # Max movement speed
PERCEPTION_RADIUS = 120  # How far each boid sees
SEPARATION_RADIUS = 25  # Minimum distance between boids (within the same type)
AVOIDANCE_RADIUS = 100  # Distance where different colors start avoiding each other
COLOR_BLEND_FACTOR = 0.02  # Smooth color blending factor
TURNING_FACTOR = 0.12  # Controls smooth movement (higher = sharper turns)
FPS = 360  # Frame rate

# ==============================
# PYGAME INITIALIZATION
# ==============================

pygame.init()
if FULLSCREEN:
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    SCREEN_WIDTH, SCREEN_HEIGHT = pygame.display.get_surface().get_size()
else:
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

pygame.display.set_caption("Flocking Boids - Cohesive Groups")

# ==============================
# HELPER FUNCTIONS
# ==============================

def generate_palette(n, gradation_levels):
    """
    Generate `n` distinct colors, each with `gradation_levels` variations.
    
    - Ensures smooth color transition within a type.
    - Returns a list of `n * gradation_levels` colors.
    """
    base_colors = [
        (200, 50, 50), (50, 200, 50), (50, 50, 200),
        (200, 200, 50), (200, 50, 200), (50, 200, 200)
    ]
    
    selected_colors = [np.array(base_colors[i % len(base_colors)], dtype=float) for i in range(n)]
    
    # Create smooth gradient variations within each color type
    all_colors = []
    for color in selected_colors:
        for i in range(gradation_levels):
            blend_factor = i / (gradation_levels - 1)
            variation = (255 - color) * blend_factor * 0.3  # Slight brightening effect
            all_colors.append(color + variation)

    return all_colors

# Create the full color palette with gradations
COLOR_PALETTE = generate_palette(NUM_TYPES, GRADATION_LEVELS)

# ==============================
# BOID CLASS
# ==============================

class Boid:
    def __init__(self, x, y, color_index):
        """
        Initialize a Boid (flocking agent).
        
        - Starts with a random velocity.
        - Assigns a color and type based on `color_index`.
        """
        self.position = np.array([x, y], dtype=float)
        angle = random.uniform(0, 2 * np.pi)
        self.velocity = np.array([np.cos(angle), np.sin(angle)]) * BOID_SPEED
        self.acceleration = np.array([0, 0], dtype=float)
        self.max_speed = BOID_SPEED
        self.perception = PERCEPTION_RADIUS
        self.color = COLOR_PALETTE[color_index]
        self.type = color_index // GRADATION_LEVELS  # Ensures gradation colors belong to the same type

    def update(self):
        """
        Update the boid's position and velocity while ensuring smooth movement.
        """
        self.velocity += self.acceleration
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed
        self.position += self.velocity
        self.acceleration *= 0  # Reset acceleration each frame

    def apply_force(self, force):
        """
        Apply a force gradually for smooth curving behavior.
        """
        self.acceleration += force * TURNING_FACTOR

    def edges(self):
        """
        Wrap around edges to create an infinite space effect.
        """
        if self.position[0] > SCREEN_WIDTH: self.position[0] = 0
        if self.position[0] < 0: self.position[0] = SCREEN_WIDTH
        if self.position[1] > SCREEN_HEIGHT: self.position[1] = 0
        if self.position[1] < 0: self.position[1] = SCREEN_HEIGHT

    def flock(self, boids):
        """
        Implements flocking behavior with:
        - Cohesion (moving toward group center)
        - Alignment (matching velocity with neighbors)
        - Separation (avoiding close contacts)
        - Avoidance (different colors avoid each other)
        """
        alignment, cohesion, separation, avoidance = np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2)
        total, avoid_total, different_type_total = 0, 0, 0
        avg_position, avg_velocity = np.zeros(2), np.zeros(2)
        separation_force = np.zeros(2)

        for other in boids:
            if other == self:
                continue

            dist = np.linalg.norm(self.position - other.position)

            # Flocking behavior within the same type
            if self.type == other.type:
                if dist < self.perception:
                    avg_position += other.position
                    avg_velocity += other.velocity
                    total += 1

                    if dist < SEPARATION_RADIUS:
                        separation_force += self.position - other.position
                        avoid_total += 1

            # Avoidance behavior for different types
            else:
                if dist < AVOIDANCE_RADIUS:
                    avoidance += (self.position - other.position) * 0.05  # Move away
                    different_type_total += 1

        if total > 0:
            avg_position /= total
            avg_velocity /= total
            cohesion = (avg_position - self.position) * 0.02  # Move toward the group center
            alignment = (avg_velocity - self.velocity) * 0.05  # Align movement with neighbors

        if avoid_total > 0:
            separation = separation_force * 0.3  # Stronger repulsion for close contacts

        if different_type_total > 0:
            self.apply_force(avoidance)  # Move away from different types

        self.apply_force(alignment + cohesion + separation)

    def draw(self):
        """
        Render the boid as a triangle pointing in its movement direction.
        """
        angle = np.arctan2(self.velocity[1], self.velocity[0])  # Get heading direction
        p1 = self.position + np.array([np.cos(angle), np.sin(angle)]) * BOID_SIZE
        p2 = self.position + np.array([np.cos(angle + 2.5), np.sin(angle + 2.5)]) * (BOID_SIZE / 2)
        p3 = self.position + np.array([np.cos(angle - 2.5), np.sin(angle - 2.5)]) * (BOID_SIZE / 2)
        pygame.draw.polygon(screen, self.color.astype(int), [p1, p2, p3])

# ==============================
# INITIALIZE BOIDS
# ==============================

boids = [Boid(random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT), random.randint(0, NUM_TYPES * GRADATION_LEVELS - 1)) for _ in range(NUM_BOIDS)]

# ==============================
# MAIN LOOP
# ==============================

running = True
clock = pygame.time.Clock()

while running:
    screen.fill(BACKGROUND_COLOR)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    for boid in boids:
        boid.flock(boids)
        boid.update()
        boid.edges()
        boid.draw()

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
