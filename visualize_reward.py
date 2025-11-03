"""
Visualisation simplifiée: contribution de chaque composant par step pour chaque épisode.
"""
import json
import matplotlib.pyplot as plt
from pathlib import Path

def load_latest_log(log_dir="reward_logs"):
    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"❌ Le dossier {log_dir} n'existe pas")
        return None
    
    json_files = list(log_path.glob("rewards_*.json"))
    if not json_files:
        print(f"❌ Aucun fichier de log trouvé dans {log_dir}")
        return None
    
    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    print(f"📂 Chargement de: {latest_file}")
    
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None

def plot_episode_components(episode_data, episode_num, result, output_dir):
    """Plot contribution de chaque composant par step"""
    steps_data = episode_data['steps']
    step_nums = [s['step'] for s in steps_data]
    
    # Collecter tous les composants
    all_components = set()
    for step_data in steps_data:
        all_components.update(step_data['components'].keys())
    
    if not all_components:
        print(f"  ⚠️  Épisode {episode_num}: aucun composant trouvé")
        return
    
    # Créer le graphique
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for comp_name in sorted(all_components):
        values = [step_data['components'].get(comp_name, 0) for step_data in steps_data]
        ax.plot(step_nums, values, label=comp_name, linewidth=2, alpha=0.8, marker='o', markersize=2)
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Reward Component Value', fontsize=12)
    ax.set_title(f'Episode {episode_num} ({result}) - Reward Components per Step', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = output_dir / f"episode_{episode_num:03d}_{result}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    print("\n" + "="*80)
    print("VISUALISATION DES COMPOSANTS DE REWARD")
    print("="*80)
    
    # Charger les données
    data = load_latest_log()
    if data is None:
        return
    
    episodes = data['episodes']
    print(f"\n✅ Chargé {len(episodes)} épisodes")
    
    # Créer le dossier de sortie
    output_dir = Path("reward_logs/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Générer un graphique par épisode
    print("\n📊 Génération des graphiques...")
    for i, episode_data in enumerate(episodes, 1):
        result = episode_data.get('result', 'UNKNOWN')
        plot_episode_components(episode_data, i, result, output_dir)
        print(f"  ✅ Episode {i}/{len(episodes)}: {result}")
    
    print(f"\n✅ Graphiques sauvegardés dans: {output_dir}/")
    print("="*80 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Interrompu")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()