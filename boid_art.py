import pygame
import random
import numpy as np

# ==============================
# CONFIGURATION VARIABLES
# ==============================

FULLSCREEN = True  
SCREEN_WIDTH = 1920  
SCREEN_HEIGHT = 1080  
BACKGROUND_COLOR = (15, 15, 20)  

# Boid settings
NUM_BOIDS = 250  
NUM_TYPES = 7  
GRADATION_LEVELS = 10  
BOID_SPEED = 8  
PERCEPTION_RADIUS = 250  
SEPARATION_RADIUS = 50  
AVOIDANCE_RADIUS = 120  
COLOR_BLEND_FACTOR = 0.02  
TURNING_FACTOR = 0.5  
FPS = 60  

# Body size settings (range and factor selection)
BODY_SIZE_MIN = 10  
BODY_SIZE_MAX = 12  
BODY_SIZE_FACTOR = 2  

# Black hole settings
BLACK_HOLE_RADIUS = 50  
BLACK_HOLE_SPEED = 10  
black_hole_active = True  # NEW: Toggle black hole visibility

# ==============================
# PYGAME INITIALIZATION
# ==============================

pygame.init()
if FULLSCREEN:
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    SCREEN_WIDTH, SCREEN_HEIGHT = pygame.display.get_surface().get_size()
else:
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

pygame.display.set_caption("Flocking Boids - Black Hole & Body Sizes")

# ==============================
# HELPER FUNCTIONS
# ==============================

def generate_palette(n, gradation_levels):
    """Generate `n` distinct colors, each with `gradation_levels` variations."""
    base_colors = [
        (200, 50, 50), (50, 200, 50), (50, 50, 200),
        (200, 200, 50), (200, 50, 200), (50, 200, 200)
    ]
    
    selected_colors = [np.array(base_colors[i % len(base_colors)], dtype=float) for i in range(n)]
    
    all_colors = []
    for color in selected_colors:
        for i in range(gradation_levels):
            blend_factor = i / (gradation_levels - 1)  
            variation = (255 - color) * blend_factor * 0.3  
            all_colors.append(color + variation)

    return all_colors

COLOR_PALETTE = generate_palette(NUM_TYPES, GRADATION_LEVELS)

def generate_body_sizes(min_size, max_size, factor):
    """Generate evenly spaced body sizes based on range and factor."""
    if factor == 1:  
        return [min_size]
    return np.linspace(min_size, max_size, factor, dtype=int).tolist()

BODY_SIZES = generate_body_sizes(BODY_SIZE_MIN, BODY_SIZE_MAX, BODY_SIZE_FACTOR)

# ==============================
# BOID CLASS
# ==============================

class Boid:
    def __init__(self, x, y, color_index):
        """Initialize a Boid with flocking behavior."""
        self.position = np.array([x, y], dtype=np.float64)  
        angle = random.uniform(0, 2 * np.pi)  
        self.velocity = np.array([np.cos(angle), np.sin(angle)]) * BOID_SPEED  
        self.acceleration = np.array([0, 0], dtype=np.float64)  
        self.max_speed = BOID_SPEED  
        self.perception = PERCEPTION_RADIUS  
        self.color = COLOR_PALETTE[color_index]  
        self.type = color_index // GRADATION_LEVELS  
        self.size = random.choice(BODY_SIZES)  
        self.attractiveness = (self.size - min(BODY_SIZES)) / (max(BODY_SIZES) - min(BODY_SIZES) + 1)  

    def update(self):
        """Update boid movement with smooth transitions."""
        self.velocity += self.acceleration  
        speed = np.linalg.norm(self.velocity)  
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed  
        self.position = self.position.astype(np.float64) + self.velocity  
        self.acceleration *= 0  

    def apply_force(self, force):
        """Apply force scaled by boid size for smoother flocking."""
        self.acceleration += force * TURNING_FACTOR * (0.5 + self.attractiveness)

    def flock(self, boids):
        """Implements flocking behavior with cohesion, alignment, and avoidance."""
        alignment, cohesion, separation, avoidance = np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2)
        total, avoid_total, different_type_total = 0, 0, 0
        avg_position, avg_velocity = np.zeros(2), np.zeros(2)
        separation_force = np.zeros(2)

        for other in boids:
            if other == self:
                continue

            dist = np.linalg.norm(self.position - other.position)

            if self.type == other.type:
                if dist < self.perception:
                    avg_position += other.position  
                    avg_velocity += other.velocity  
                    total += 1  

                    if dist < SEPARATION_RADIUS:
                        separation_force += self.position - other.position  
                        avoid_total += 1  

            else:
                if dist < AVOIDANCE_RADIUS:
                    avoidance += (self.position - other.position) * 0.05  
                    different_type_total += 1  

        if total > 0:
            avg_position /= total  
            avg_velocity /= total  

            cohesion = (avg_position - self.position) * 0.02  
            alignment = (avg_velocity - self.velocity) * 0.05  

        if avoid_total > 0:
            separation = separation_force * 0.3  

        if different_type_total > 0:
            self.apply_force(avoidance)  

        self.apply_force(alignment + cohesion + separation)

    def edges(self):
        """Wrap around screen edges for infinite space effect."""
        if self.position[0] > SCREEN_WIDTH: self.position[0] = 0
        if self.position[0] < 0: self.position[0] = SCREEN_WIDTH
        if self.position[1] > SCREEN_HEIGHT: self.position[1] = 0
        if self.position[1] < 0: self.position[1] = SCREEN_HEIGHT

    def draw(self):
        """Render the boid as a triangle with scaled size."""
        angle = np.arctan2(self.velocity[1], self.velocity[0])  
        p1 = self.position + np.array([np.cos(angle), np.sin(angle)]) * self.size  
        p2 = self.position + np.array([np.cos(angle + 2.5), np.sin(angle + 2.5)]) * (self.size / 2)  
        p3 = self.position + np.array([np.cos(angle - 2.5), np.sin(angle - 2.5)]) * (self.size / 2)  
        pygame.draw.polygon(screen, self.color.astype(int), [p1, p2, p3])  

# ==============================
# INITIALIZE BOIDS & BLACK HOLE
# ==============================

boids = [Boid(random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT), random.randint(0, NUM_TYPES * GRADATION_LEVELS - 1)) for _ in range(NUM_BOIDS)]
black_hole_position = np.array([SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2])  

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
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:  
                black_hole_active = not black_hole_active  

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]: black_hole_position[0] -= BLACK_HOLE_SPEED  
    if keys[pygame.K_RIGHT]: black_hole_position[0] += BLACK_HOLE_SPEED  
    if keys[pygame.K_UP]: black_hole_position[1] -= BLACK_HOLE_SPEED  
    if keys[pygame.K_DOWN]: black_hole_position[1] += BLACK_HOLE_SPEED  

    if black_hole_active:
        pygame.draw.circle(screen, (0, 0, 0), black_hole_position.astype(int), BLACK_HOLE_RADIUS)

        for boid in boids:
            if np.linalg.norm(boid.position - black_hole_position) < BLACK_HOLE_RADIUS:
                boid.position = np.array([random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT)])  

    for boid in boids:
        boid.flock(boids)  
        boid.update()
        boid.edges()
        boid.draw()

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
