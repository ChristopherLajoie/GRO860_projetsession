# --- Fichier de test (ex: test_env.py) ---
# Assure-toi que la classe ParkingEnv est importée ou dans le même fichier
from parking import ParkingEnv
import numpy as np
import pygame # Assure-toi qu'il est importé en haut

def human_render_loop(agent, env, transpose=False, print_step=False):
    obs, info = env.reset()
    clock = pygame.time.Clock()
    fps = env.metadata["render_fps"]

    term = trunc = quit_loop = False
    step = 0
    total_reward = 0

    while not (term or trunc or quit_loop):
        # Vérifier si l'utilisateur veut quitter
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_loop = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                quit_loop = True

        action = agent(obs)
        obs, reward, term, trunc, info = env.step(action)
        
        env.render() # L'appel à render() dessine l'écran

        step += 1
        total_reward += reward

        if print_step:
            print(f"Step {step}, Action {action}, Info {info}")

        clock.tick(fps)
    
    env.close()
    print(f"Boucle terminée. Récompense totale : {total_reward}")


# Agent "Humain" pour piloter avec les flèches
def human_agent(_obs):
    keys = pygame.key.get_pressed()
    
    accel = 0.0
    steer = 0.0
    
    if keys[pygame.K_UP]:
        accel = 2.0 # Accélération max
    elif keys[pygame.K_DOWN]:
        accel = -1.0 # Marche arrière (plus lente)
        
    if keys[pygame.K_LEFT]:
        steer = 0.523 # Tourner à gauche (30 deg)
    elif keys[pygame.K_RIGHT]:
        steer = -0.523 # Tourner à droite (-30 deg)
        
    return np.array([accel, steer], dtype=np.float32)

# --- Exécution du test ---
print("Lancement du test de rendu. Pilotez avec les flèches (HAUT, BAS, GAUCHE, DROITE).")
print("Appuyez sur 'Q' ou fermez la fenêtre pour quitter.")

# IMPORTANT: Initialiser l'environnement en mode "human"
env = ParkingEnv(render_mode="human")

human_render_loop(human_agent, env, print_step=True)