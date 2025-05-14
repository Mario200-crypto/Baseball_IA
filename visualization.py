import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo
import matplotlib.pyplot as plt
import numpy as np
from Baseball_simulator import BaseballSimulator, Player, Pitcher, PitcherRole
import os
import datetime

def ensure_output_dir():
    """Asegura que existe el directorio de salida para los gráficos"""
    output_dir = "baseball_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def plot_runs_distribution(resultados, output_dir, suffix=None):
    """
    Visualiza la distribución de carreras anotadas en las simulaciones.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(resultados['bins'], resultados['distribucion'], alpha=0.7)
    
    # Calcular la mediana
    carreras_array = np.array(resultados['carreras'])
    mediana = np.median(carreras_array)
    
    # Agregar líneas para promedio y mediana
    plt.axvline(resultados['promedio'], color='r', linestyle='--', 
                label=f'Promedio: {resultados["promedio"]:.2f}')
    plt.axvline(mediana, color='g', linestyle='--',
                label=f'Mediana: {mediana:.2f}')
    
    plt.title('Distribución de Carreras Anotadas')
    plt.xlabel('Número de Carreras')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if suffix is None:
        suffix = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
    else:
        suffix = f'_{suffix}'
    
    plt.savefig(os.path.join(output_dir, f'runs_distribution{suffix}.png'))
    plt.close()

def plot_player_stats(player_stats, output_dir, suffix=None):
    """
    Visualiza las estadísticas reales de los jugadores en la alineación.
    """
    names = list(player_stats.keys())
    metrics = ['batting_avg', 'extra_base_pct', 'rbi_per_game', 'strikeout_rate', 'walk_rate', 'clutch_hitting']
    metric_labels = ['Promedio', 'Extra-Base %', 'RBI/Juego', 'Ponches %', 'BB %', 'Clutch %']
    
    # Crear subplots para cada métrica
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [player_stats[name][metric] for name in names]
        ax = axes[idx]
        ax.bar(names, values)
        ax.set_title(label)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if suffix is None:
        suffix = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
    else:
        suffix = f'_{suffix}'
    
    plt.savefig(os.path.join(output_dir, f'player_stats{suffix}.png'))
    plt.close()

def visualize_simulation(simulator, num_simulations=1000, titulo=None):
    """
    Ejecuta y visualiza una simulación completa.
    """
    # Crear directorio para los gráficos
    output_dir = ensure_output_dir()
    
    # Ejecutar simulación
    resultados = simulator.monte_carlo_game_simulation(num_simulations=num_simulations)
    
    # Obtener estadísticas reales
    player_stats = simulator.get_player_statistics()
    
    # Calcular la mediana
    carreras_array = np.array(resultados['carreras'])
    mediana = np.median(carreras_array)
    
    # Mostrar resultados
    print(f"\nResultados de la Simulación ({num_simulations} partidos):")
    print(f"Promedio de carreras: {resultados['promedio']:.2f} ± {resultados['desviacion']:.2f}")
    print(f"Mediana de carreras: {mediana:.2f}")
    print(f"Rango de carreras: {resultados['minimo']} - {resultados['maximo']}")
    
    # Usar título personalizado si se proporciona
    suffix = titulo if titulo else datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
    
    # Generar visualizaciones
    plot_runs_distribution(resultados, output_dir, suffix)
    plot_player_stats(player_stats, output_dir, suffix)
    
    print("\nGráficos guardados en el directorio 'baseball_plots':")
    print(f"- runs_distribution{suffix}.png")
    print(f"- player_stats{suffix}.png")

if __name__ == "__main__":
    # Crear una alineación de ejemplo
    lineup = [
        # Nombre, AVG, SLG, OBP
        Player("Ben Rice", 0.254, 0.558, 0.360),
        Player("Aaron Judge", 0.410, 0.783, 0.500),
        Player("Cody Bellinger", 0.229, 0.440, 0.387),
        Player("Paul Goldschmidt", 0.344, 0.488, 0.394),
        Player("Jasson Domínguez", 0.245, 0.434, 0.331),
        Player("Anthony Volpe", 0.243, 0.364, 0.293),
        Player("Austin Wells", 0.215, 0.479, 0.281),
        Player("Oswaldo Cabrera", 0.243, 0.387, 0.287),
        Player("Jorbit Vivas", 0.217, 0.478, 0.419),
    ]
    # Crear un bullpen completo de lanzadores con roles y resistencia
    bullpen = [
        # Nombre, ERA, WHIP, K/9, Rol, Resistencia
        Pitcher("Bryan Woo", 3.25, 0.92, 8.93, PitcherRole.STARTER, 6.0),
        Pitcher("Gabe Speier", 2.30, 0.96, 10.54, PitcherRole.MIDDLE_RELIEF, 1.0),
        Pitcher("Matt Brash", 0.00, 1.36, 12.27, PitcherRole.MIDDLE_RELIEF, 1.0),
        Pitcher("Andrés Muñoz", 0.00, 0.83, 12.32, PitcherRole.CLOSER, 1.0),
        Pitcher("Carlos Vargas", 4.00, 1.67, 9.75, PitcherRole.MIDDLE_RELIEF, 1.0),
        Pitcher("Casey Legumina", 5.40, 1.00, 6.75, PitcherRole.MIDDLE_RELIEF, 1.0),
    ]
    
    # Crear y ejecutar simulación
    simulator = BaseballSimulator(lineup, bullpen)
    visualize_simulation(simulator, titulo="Yankees vs. Mariners 14 de mayo")