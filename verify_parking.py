"""
Script pour vérifier visuellement l'environnement de parking
avant de lancer un long entraînement.
"""
import gymnasium as gym
from parking import ParkingEnv
import pygame
import time

# Register
gym.register(
    id="RealisticParking-v0",
    entry_point="parking:ParkingEnv",
    max_episode_steps=2000
)

# Create environment
env = gym.make("RealisticParking-v0", render_mode="human")

print("\n" + "="*70)
print("  VÉRIFICATION DE L'ENVIRONNEMENT")
print("="*70)
print("\nInformations:")
print(f"  Dimensions parking: {env.unwrapped.world_width:.1f}m × {env.unwrapped.world_height:.1f}m")
print(f"  Largeur allée: {env.unwrapped.AISLE_WIDTH}m")
print(f"  Zone circulation: {env.unwrapped.CIRCULATION_ZONE}m")
print(f"  Voitures parkées: {len(env.unwrapped.parked_cars)}")
print(f"  Dimensions voiture: {env.unwrapped.CAR_LENGTH}m × {env.unwrapped.CAR_WIDTH}m")
print("\n" + "="*70)
print("Instructions:")
print("  - Observez la disposition du parking")
print("  - Vérifiez que l'agent (bleu) a de l'espace pour bouger")
print("  - Le spot libre (vert) doit être visible")
print("  - Appuyez sur 'Q' pour quitter")
print("="*70 + "\n")

# Reset and display
obs, info = env.reset()

start_pos = info["start_state"]
target_pos = env.unwrapped.target_spot_center
dist = ((target_pos[0] - start_pos[0])**2 + (target_pos[1] - start_pos[1])**2)**0.5

print(f"Reset effectué:")
print(f"  Position agent: ({start_pos[0]:.1f}, {start_pos[1]:.1f})")
print(f"  Target spot: ({target_pos[0]:.1f}, {target_pos[1]:.1f})")
print(f"  Distance: {dist:.1f}m")
print(f"\nObservation shape: {obs.shape}")
print(f"Action space: {env.action_space}")

# Display for 30 seconds or until quit
clock = pygame.time.Clock()
fps = env.metadata["render_fps"]
quit_loop = False
frames = 0
max_frames = fps * 30  # 30 seconds

print("\nAffichage de l'environnement (30 secondes)...")

while not quit_loop and frames < max_frames:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit_loop = True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            quit_loop = True
    
    env.render()
    clock.tick(fps)
    frames += 1
    
    # Afficher countdown chaque seconde
    if frames % fps == 0:
        remaining = (max_frames - frames) // fps
        print(f"  Temps restant: {remaining}s", end="\r")

env.close()

print("\n\n" + "="*70)
print("✓ Vérification terminée!")
print("="*70)
print("\nSi l'environnement semble correct:")
print("  1. L'agent a assez d'espace pour se déplacer")
print("  2. Les voitures parkées sont visibles")
print("  3. Le spot vert est accessible")
print("\nAlors vous pouvez lancer: python train_parking.py")
print("="*70 + "\n")