import numpy as np
import random
from typing import List, Tuple, Dict
from copy import deepcopy
from Baseball_simulator import BaseballSimulator, Player, Pitcher, PitcherRole

class BaseballGeneticOptimizer:
    def __init__(self, 
                 lineup: List[Player], 
                 bullpen_realdata_list: List[Tuple[List[Pitcher], Dict]],  # Lista de (bullpen, real_game_data)
                 population_size: int = 40,
                 generations: int = 100,
                 mutation_rate: float = 0.15,
                 crossover_rate: float = 0.8,
                 elitism_count: int = 2):
        self.lineup = lineup
        self.bullpen_realdata_list = bullpen_realdata_list
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.population = self._initialize_population()

    def _initialize_population(self) -> List[List[float]]:
        """Inicializa la población con valores aleatorios para los pesos"""
        population = []
        
        # Añadir valores iniciales conocidos para asegurar buenos puntos de partida
        initial_guesses = [
            # Valores originales + bateadores básicos
            [5.0, 1.5, 9.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            # Variante con énfasis en ERA y OBP
            [7.0, 1.8, 8.0, 2.2, 1.1, 1.5, 0.8, 1.2, 1.3],
            # Variante con énfasis en WHIP y SLG
            [4.0, 2.5, 7.0, 1.8, 0.9, 0.9, 0.7, 1.8, 1.1],
        ]
        
        population.extend(initial_guesses)
        
        for _ in range(self.population_size - len(initial_guesses)):
            # Generar un individuo con valores aleatorios para cada peso
            individual = [
                # Parámetros de pitchers
                random.uniform(1.0, 10.0),     # era_factor_weight (originalmente 5.0)
                random.uniform(0.5, 3.0),      # whip_factor_weight (originalmente 1.5)
                random.uniform(0.5, 2.0),      # k_factor_divisor (originalmente 9.0)
                random.uniform(1.0, 3.0),      # stamina_exponent (originalmente 2.0)
                random.uniform(0.7, 1.3),      # pressure_factor_base (originalmente 1.0)
                
                # Parámetros de bateadores
                random.uniform(0.5, 2.0),      # obp_weight (peso del OBP)
                random.uniform(0.5, 2.0),      # avg_weight (peso del AVG)
                random.uniform(0.5, 2.0),      # slg_weight (peso del SLG)
                random.uniform(0.7, 1.5),      # clutch_factor (factor de presión para bateadores)
            ]
            population.append(individual)
            
        return population
    
    def _fitness(self, individual: List[float]) -> float:
        """
        Calcula la aptitud promediando la similitud de la media simulada vs real para todos los bullpens.
        """
        fitness_scores = []
        for bullpen, real_game_data in self.bullpen_realdata_list:
            simulator = self._create_modified_simulator(individual, bullpen)
            sim_results = simulator.monte_carlo_game_simulation(num_simulations=200)
            fitness = self._calculate_distribution_similarity(sim_results, real_game_data)
            fitness_scores.append(fitness)
        return sum(fitness_scores) / len(fitness_scores)
    
    def _create_modified_simulator(self, individual: List[float], bullpen: List[Pitcher] = None) -> BaseballSimulator:
        """
        Crea un simulador modificado con los pesos del individuo y bullpen específico.
        """
        if bullpen is None:
            bullpen = self.bullpen_realdata_list[0][0]  # Por compatibilidad
        simulator = CustomBaseballSimulator(
            self.lineup, 
            bullpen,
            # Parámetros de pitchers
            era_factor_weight=individual[0],
            whip_factor_weight=individual[1],
            k_factor_divisor=individual[2],
            stamina_exponent=individual[3],
            pressure_factor_base=individual[4],
            
            # Parámetros de bateadores
            obp_weight=individual[5],
            avg_weight=individual[6],
            slg_weight=individual[7],
            batter_clutch_factor=individual[8]
        )
        return simulator
    
    def _calculate_distribution_similarity(self, sim_results: Dict, real_data: Dict) -> float:
        """
        Calcula la similitud SOLO usando la media (promedio) de carreras.
        Una puntuación más alta indica mayor similitud.
        """
        mean_diff = abs(sim_results['promedio'] - real_data['promedio'])
        # Fitness score: 1 / (1 + diferencia absoluta de medias)
        similarity_score = 1.0 / (1.0 + mean_diff)
        return similarity_score
    
    def _select_parents(self, fitnesses: List[float]) -> List[Tuple[int, int]]:
        """Selecciona pares de padres usando selección por torneo"""
        parent_pairs = []
        
        for _ in range(self.population_size // 2):
            # Selección por torneo para el primer padre
            idx1 = self._tournament_selection(fitnesses, tournament_size=4)  # Aumentado de 3 a 4
            
            # Selección por torneo para el segundo padre
            idx2 = self._tournament_selection(fitnesses, tournament_size=4)  # Aumentado de 3 a 4
            
            # Asegurar que los padres sean diferentes
            while idx2 == idx1:
                idx2 = self._tournament_selection(fitnesses, tournament_size=4)
                
            parent_pairs.append((idx1, idx2))
            
        return parent_pairs
    
    def _tournament_selection(self, fitnesses: List[float], tournament_size: int = 4) -> int:
        """Implementa la selección por torneo"""
        # Seleccionar aleatoriamente individuos para el torneo
        tournament_indices = random.sample(range(len(fitnesses)), tournament_size)
        
        # Encontrar el mejor individuo del torneo
        best_idx = tournament_indices[0]
        best_fitness = fitnesses[best_idx]
        
        for idx in tournament_indices[1:]:
            if fitnesses[idx] > best_fitness:
                best_fitness = fitnesses[idx]
                best_idx = idx
                
        return best_idx
    
    def _crossover(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        """Realiza el cruce entre dos padres para producir dos hijos"""
        if random.random() < self.crossover_rate:
            # Implementar cruce uniforme en lugar de punto único
            child1 = []
            child2 = []
            
            for i in range(len(parent1)):
                # Para cada gen, decidir aleatoriamente de qué padre heredar
                if random.random() < 0.5:
                    child1.append(parent1[i])
                    child2.append(parent2[i])
                else:
                    child1.append(parent2[i])
                    child2.append(parent1[i])
            
            # Aplicar interpolación para algunos genes (mezcla de valores)
            if random.random() < 0.3:  # 30% de probabilidad de interpolación
                # Seleccionar un gen aleatorio para interpolación
                gene_idx = random.randint(0, len(parent1) - 1)
                # Crear un valor interpolado
                alpha = random.random()  # Factor de interpolación
                interp_value = alpha * parent1[gene_idx] + (1 - alpha) * parent2[gene_idx]
                # Asignar a uno de los hijos
                child1[gene_idx] = interp_value
            
            return child1, child2
        else:
            # Sin cruce, devolver copias de los padres
            return parent1.copy(), parent2.copy()
    
    def _mutate(self, individual: List[float]) -> List[float]:
        """Aplica mutación a un individuo"""
        mutated = individual.copy()
        
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                # Aplicar mutación según el tipo de peso
                if i == 0:  # era_factor_weight
                    # Mutación adaptativa - mayor precisión cerca del valor actual
                    if random.random() < 0.7:  # Mutación pequeña 70% del tiempo
                        delta = random.uniform(-1.0, 1.0)
                        mutated[i] = max(1.0, min(10.0, mutated[i] + delta))
                    else:  # Mutación grande 30% del tiempo
                        mutated[i] = random.uniform(1.0, 10.0)
                elif i == 1:  # whip_factor_weight
                    if random.random() < 0.7:
                        delta = random.uniform(-0.3, 0.3)
                        mutated[i] = max(0.5, min(3.0, mutated[i] + delta))
                    else:
                        mutated[i] = random.uniform(0.5, 3.0)
                elif i == 2:  # k_factor_divisor
                    if random.random() < 0.7:
                        delta = random.uniform(-0.2, 0.2)
                        mutated[i] = max(0.5, min(2.0, mutated[i] + delta))
                    else:
                        mutated[i] = random.uniform(0.5, 2.0)
                elif i == 3:  # stamina_exponent
                    if random.random() < 0.7:
                        delta = random.uniform(-0.3, 0.3)
                        mutated[i] = max(1.0, min(3.0, mutated[i] + delta))
                    else:
                        mutated[i] = random.uniform(1.0, 3.0)
                elif i == 4:  # pressure_factor_base
                    if random.random() < 0.7:
                        delta = random.uniform(-0.1, 0.1)
                        mutated[i] = max(0.7, min(1.3, mutated[i] + delta))
                    else:
                        mutated[i] = random.uniform(0.7, 1.3)
                elif i == 5:  # obp_weight
                    if random.random() < 0.7:
                        delta = random.uniform(-0.1, 0.1)
                        mutated[i] = max(0.5, min(2.0, mutated[i] + delta))
                    else:
                        mutated[i] = random.uniform(0.5, 2.0)
                elif i == 6:  # avg_weight
                    if random.random() < 0.7:
                        delta = random.uniform(-0.1, 0.1)
                        mutated[i] = max(0.5, min(2.0, mutated[i] + delta))
                    else:
                        mutated[i] = random.uniform(0.5, 2.0)
                elif i == 7:  # slg_weight
                    if random.random() < 0.7:
                        delta = random.uniform(-0.1, 0.1)
                        mutated[i] = max(0.5, min(2.0, mutated[i] + delta))
                    else:
                        mutated[i] = random.uniform(0.5, 2.0)
                elif i == 8:  # clutch_factor
                    if random.random() < 0.7:
                        delta = random.uniform(-0.1, 0.1)
                        mutated[i] = max(0.7, min(1.5, mutated[i] + delta))
                    else:
                        mutated[i] = random.uniform(0.7, 1.5)
                    
        return mutated
    
    def evolve(self) -> Tuple[List[float], float]:
        """Ejecuta el algoritmo genético y devuelve el mejor individuo"""
        best_individual = None
        best_fitness = -float('inf')
        
        # Historial para seguimiento
        fitness_history = []
        best_fitness_history = []
        
        for generation in range(self.generations):
            # Evaluar la aptitud de cada individuo
            fitnesses = [self._fitness(individual) for individual in self.population]
            avg_fitness = sum(fitnesses) / len(fitnesses)
            fitness_history.append(avg_fitness)
            
            # Encontrar el mejor individuo de esta generación
            best_idx = fitnesses.index(max(fitnesses))
            current_best = self.population[best_idx]
            current_best_fitness = fitnesses[best_idx]
            best_fitness_history.append(current_best_fitness)
            
            # Actualizar el mejor individuo global si es necesario
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = current_best.copy()
                
            print(f"Generación {generation + 1}/{self.generations}: " 
                  f"Mejor aptitud = {current_best_fitness:.4f}, "
                  f"Promedio = {avg_fitness:.4f}")
            
            # Detener si se alcanza una aptitud muy alta
            if current_best_fitness > 0.95:
                print(f"Aptitud objetivo alcanzada. Terminando en generación {generation + 1}")
                break
                
            # No evolucionar en la última generación
            if generation == self.generations - 1:
                break
            
            # Implementar elitismo: preservar los mejores individuos
            elite = []
            sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
            for i in range(self.elitism_count):
                if i < len(sorted_indices):
                    elite.append(self.population[sorted_indices[i]].copy())
            
            # Seleccionar padres
            parent_pairs = self._select_parents(fitnesses)
            
            # Crear nueva población mediante cruce y mutación
            new_population = []
            
            for parent1_idx, parent2_idx in parent_pairs:
                # Cruce
                child1, child2 = self._crossover(
                    self.population[parent1_idx],
                    self.population[parent2_idx]
                )
                
                # Mutación
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                # Añadir a la nueva población
                new_population.extend([child1, child2])
            
            # Asegurar que la nueva población tenga el tamaño correcto
            while len(new_population) > self.population_size - len(elite):
                new_population.pop()
                
            # Añadir la élite a la nueva población
            new_population.extend(elite)
                
            # Actualizar la población
            self.population = new_population
            
        # Gráfico opcional de evolución de aptitud
        # self._plot_fitness_history(fitness_history, best_fitness_history)
            
        return best_individual, best_fitness
    
    def _plot_fitness_history(self, fitness_history, best_fitness_history):
        """Gráfico opcional para visualizar la evolución de la aptitud"""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(fitness_history) + 1), fitness_history, label='Promedio')
            plt.plot(range(1, len(best_fitness_history) + 1), best_fitness_history, label='Mejor')
            plt.xlabel('Generación')
            plt.ylabel('Aptitud')
            plt.title('Evolución de la aptitud')
            plt.legend()
            plt.grid(True)
            
            # Guardar o mostrar el gráfico
            plt.savefig('fitness_evolution.png')
            plt.close()
        except ImportError:
            print("Matplotlib no está disponible para generar gráficos.")


# Clase especializada que extiende BaseballSimulator para usar los pesos personalizados
class CustomBaseballSimulator(BaseballSimulator):
    def __init__(self, lineup, bullpen, 
                 # Parámetros de pitchers
                 era_factor_weight=5.0, 
                 whip_factor_weight=1.5, 
                 k_factor_divisor=9.0, 
                 stamina_exponent=2.0, 
                 pressure_factor_base=1.0,
                 
                 # Parámetros de bateadores
                 obp_weight=1.0,
                 avg_weight=1.0,
                 slg_weight=1.0,
                 batter_clutch_factor=1.0):
        super().__init__(lineup, bullpen)
        
        # Parámetros de pitchers
        self.era_factor_weight = era_factor_weight
        self.whip_factor_weight = whip_factor_weight
        self.k_factor_divisor = k_factor_divisor
        self.stamina_exponent = stamina_exponent
        self.pressure_factor_base = pressure_factor_base
        
        # Parámetros de bateadores
        self.obp_weight = obp_weight
        self.avg_weight = avg_weight
        self.slg_weight = slg_weight
        self.batter_clutch_factor = batter_clutch_factor
    
    def _simulate_pitcher_performance(self, pitcher: Pitcher) -> float:
        """Versión modificada que usa pesos optimizados por algoritmo genético"""
        # Factor base basado en ERA (más bajo es mejor)
        era_factor = self.era_factor_weight / pitcher.era if pitcher.era > 0 else 1.0
        
        # Factor de WHIP (más bajo es mejor)
        whip_factor = self.whip_factor_weight / pitcher.whip if pitcher.whip > 0 else 1.0
        
        # Factor de K/9 (más alto es mejor)
        k_factor = pitcher.k_per_9 / self.k_factor_divisor
        
        # Factor de resistencia (exponente personalizable)
        stamina_factor = pitcher.current_stamina ** self.stamina_exponent
        
        # Factor de presión (base personalizable)
        pressure_factor = self.pressure_factor_base
        if self.innings_pitched >= 6:
            pressure_factor *= 0.9
        if self.runs_allowed > 3:
            pressure_factor *= 0.95
        
        return (era_factor * whip_factor * k_factor * stamina_factor * pressure_factor)
    
    def _simulate_at_bat(self, batter: Player, pitcher_factor: float) -> str:
        """
        Versión modificada que usa pesos optimizados para las estadísticas de los bateadores
        """
        # Ajustar probabilidades basado en el rendimiento del pitcher
        # Aplicar pesos personalizados para OBP y AVG
        adjusted_obp = min(batter.obp / pitcher_factor, 0.55)  # Limitar OBP ajustado (más alto)
        adjusted_avg = min(batter.batting_avg / pitcher_factor, 0.42)  # Limitar AVG ajustado (más alto)
        
        # Calcular probabilidad de ponche basada en K/9 del pitcher
        k_prob = min((self.current_pitcher.k_per_9 / 27.0) * pitcher_factor, 0.25)  # Limitar K% (más bajo)
        
        # Calcular probabilidad de base por bolas
        walk_prob = max(min((self.current_pitcher.whip - 1.0) * 0.15, 0.12), 0)
        
        # Probabilidad de hit es la diferencia entre OBP ajustado y BB
        hit_prob = max(adjusted_obp - walk_prob, 0)
        
        # Probabilidad de out es el resto
        out_prob = max(1.0 - (k_prob + walk_prob + hit_prob), 0)
        
        # Factor de clutch para situaciones de presión (corredores en base)
        runners_on_base = sum([False, False, False])  # Simplificado para este ejemplo
        clutch_adjustment = 1.0
        if runners_on_base > 0:
            clutch_adjustment = self.batter_clutch_factor
            # Ajustar probabilidades en situaciones clutch
            hit_prob *= clutch_adjustment
            # Normalizar probabilidades
            total = k_prob + walk_prob + hit_prob + out_prob
            k_prob /= total
            walk_prob /= total
            hit_prob /= total
            out_prob /= total
        
        rand = np.random.random()
        
        if rand < k_prob:
            return 'out'
        elif rand < k_prob + walk_prob:
            return 'walk'
        elif rand < k_prob + walk_prob + hit_prob:
            # Determinar tipo de hit según AVG y SLG
            hit_type = np.random.random()
            # Usar SLG para influir en el tipo de hit
            extra_base_factor = (batter.slugging / batter.batting_avg) * self.slg_weight
            single_prob = 1.0 / extra_base_factor if extra_base_factor > 0 else 0.8
            single_prob = max(0.6, min(single_prob, 0.85))  # Limitar entre 0.6 y 0.85
            
            if hit_type < single_prob:
                return 'single'
            elif hit_type < single_prob + 0.10:
                return 'double'
            elif hit_type < single_prob + 0.13:
                return 'triple'
            else:
                return 'homerun'
        else:
            return 'out'

# Ejemplo de uso
if __name__ == "__main__":
    from Baseball_simulator import Player, Pitcher, PitcherRole
    
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
    
    # Crear varios bullpens rivales y sus resultados reales
    bullpen1 = [
        Pitcher("Bryan Woo", 3.25, 0.92, 8.93, PitcherRole.STARTER, 6.0),
        Pitcher("Gabe Speier", 2.30, 0.96, 10.54, PitcherRole.MIDDLE_RELIEF, 1.0),
        Pitcher("Matt Brash", 0.00, 1.36, 12.27, PitcherRole.MIDDLE_RELIEF, 1.0),
        Pitcher("Andrés Muñoz", 0.00, 0.83, 12.32, PitcherRole.CLOSER, 1.0),
        Pitcher("Carlos Vargas", 4.00, 1.67, 9.75, PitcherRole.MIDDLE_RELIEF, 1.0),
        Pitcher("Casey Legumina", 5.40, 1.00, 6.75, PitcherRole.MIDDLE_RELIEF, 1.0),
    ]
    bullpen2 = [
        Pitcher("Luis Severino", 4.70, 1.32, 6.71, PitcherRole.STARTER, 7.0),
        Pitcher("Mitch Spence", 5.11, 1.58, 9.12, PitcherRole.MIDDLE_RELIEF, 2.0),
        Pitcher("Elvis Alvarado", 10.80, 2.40, 10.80, PitcherRole.MIDDLE_RELIEF, 1.0),
        Pitcher("T.J. McFarland", 5.23, 1.74, 3.60, PitcherRole.MIDDLE_RELIEF, 1.0),
        Pitcher("Hogan Harris", 4.38, 1.38, 10.22, PitcherRole.MIDDLE_RELIEF, 2.0),
    ]
    bullpen3 = [
    # Nombre, ERA, WHIP, K/9, Rol, Resistencia
    Pitcher("Emerson Hancock", 6.91, 1.71, 6.91, PitcherRole.STARTER, 5.0),
    Pitcher("Casey Legumina", 2.08, 1.15, 7.62, PitcherRole.MIDDLE_RELIEF, 1.0),
    Pitcher("Eduard Bazardo", 4.74, 1.21, 5.65, PitcherRole.MIDDLE_RELIEF, 1.0),
    Pitcher("Troy Taylor", 12.00, 2.17, 3.00, PitcherRole.MIDDLE_RELIEF, 1.0),
]

    bullpen4 = [
    Pitcher("Dylan Cease", 4.60, 1.34, 11.11, PitcherRole.STARTER, 6.0),
    Pitcher("Jason Adam", 1.61, 0.94, 11.28, PitcherRole.SETUP, 1.0),
    Pitcher("Robert Suarez", 2.84, 0.79, 9.95, PitcherRole.CLOSER, 1.0),
    Pitcher("Jeremiah Estrada", 2.75, 0.97, 12.43, PitcherRole.MIDDLE_RELIEF, 1.0),
    ]
    bullpen5 = [
        # Nombre, ERA, WHIP, K/9, Rol, Resistencia
    Pitcher("JP Sears", 2.80, 1.00, 7.20, PitcherRole.STARTER, 6.0),
    Pitcher("Justin Sterner", 2.21, 1.08, 11.07, PitcherRole.MIDDLE_RELIEF, 1.0),
    Pitcher("Grant Holman", 0.75, 1.00, 4.50, PitcherRole.MIDDLE_RELIEF, 1.0),
    Pitcher("Tyler Ferguson", 1.96, 1.04, 7.91, PitcherRole.MIDDLE_RELIEF, 1.0),
    Pitcher("Mason Miller", 4.70, 0.98, 17.63, PitcherRole.CLOSER, 1.0),

    ]
    #
    bullpen6 = [
    Pitcher("Michael King", 2.32, 0.99, 10.01, PitcherRole.STARTER, 6.0),
    Pitcher("Adrián Morejón", 3.57, 1.13, 7.50, PitcherRole.MIDDLE_RELIEF, 1.0),
    Pitcher("Wandy Peralta", 7.04, 1.57, 6.46, PitcherRole.MIDDLE_RELIEF, 1.0),
    Pitcher("Ryan Bergert", 0.00, 0.25, 4.50, PitcherRole.MIDDLE_RELIEF, 1.0),
    ]
    #yankees vs rays 1
    bullpen7 = [
    Pitcher("Taj Bradley", 4.24, 1.26, 6.75, PitcherRole.STARTER, 6.0),
    Pitcher("Mason Montgomery", 4.11, 1.11, 11.74, PitcherRole.MIDDLE_RELIEF, 1.0),
    Pitcher("Manuel Rodríguez", 2.29, 0.92, 8.68, PitcherRole.MIDDLE_RELIEF, 1.0),
    Pitcher("Mason Englert", 6.00, 1.50, 9.00, PitcherRole.MIDDLE_RELIEF, 1.0),
    Pitcher("Edwin Uceta", 5.60, 1.53, 8.40, PitcherRole.MIDDLE_RELIEF, 1.0),
    Pitcher("Pete Fairbanks", 2.65, 1.29, 9.00, PitcherRole.CLOSER, 1.0),
    ]
    #yankees vs rays 2
    bullpen8 = [
    Pitcher("Zack Littell", 4.31, 1.12, 5.30, PitcherRole.STARTER, 6.0),
    Pitcher("Garrett Cleavinger", 2.25, 0.75, 10.00, PitcherRole.MIDDLE_RELIEF, 1.0),
    Pitcher("Edwin Uceta", 5.60, 1.53, 8.40, PitcherRole.MIDDLE_RELIEF, 1.0),
    Pitcher("Pete Fairbanks", 2.65, 1.29, 8.47, PitcherRole.CLOSER, 1.0),
    ]
    bullpen9 = [
        #yankees vs rays 3
    Pitcher("Ryan Pepiot", 3.93, 1.29, 8.05, PitcherRole.STARTER, 6.0),
    Pitcher("Mason Montgomery", 4.11, 1.11, 11.74, PitcherRole.MIDDLE_RELIEF, 1.0),
    Pitcher("Eric Orze", 1.10, 1.35, 4.96, PitcherRole.MIDDLE_RELIEF, 1.0),
    Pitcher("Manuel Rodríguez", 2.65, 1.06, 8.00, PitcherRole.MIDDLE_RELIEF, 1.0),
    Pitcher("Garrett Cleavinger", 2.25, 0.75, 10.00, PitcherRole.MIDDLE_RELIEF, 1.0),
    ]
    bullpen10 = [
        #yankees vs baltimore 1
    Pitcher("Cade Povich", 5.55, 1.54, 7.32, PitcherRole.STARTER, 5.0),
    Pitcher("Seranthony Domínguez", 4.26, 1.34, 6.80, PitcherRole.MIDDLE_RELIEF, 1.0),
    Pitcher("Keegan Akin", 3.26, 1.34, 8.40, PitcherRole.MIDDLE_RELIEF, 1.0),
    Pitcher("Yennier Canó", 4.40, 1.40, 6.40, PitcherRole.MIDDLE_RELIEF, 1.0),
    Pitcher("Gregory Soto", 4.85, 1.62, 10.38, PitcherRole.MIDDLE_RELIEF, 1.0),
    Pitcher("Bryan Baker", 2.08, 1.04, 10.38, PitcherRole.MIDDLE_RELIEF, 1.0),
    Pitcher("Félix Bautista", 2.25, 1.00, 11.25, PitcherRole.CLOSER, 1.0),
    ]
    bullpen11 = [
        #yankees vs blue jays 1
    Pitcher("Chris Bassitt", 3.16, 1.27, 9.64, PitcherRole.STARTER, 6.0),
    Pitcher("Brendon Little", 1.86, 1.14, 13.5, PitcherRole.MIDDLE_RELIEF, 1.0),
    Pitcher("Yariel Rodríguez", 4.26, 1.21, 8.83, PitcherRole.MIDDLE_RELIEF, 1.0),
    Pitcher("Chad Green", 3.38, 1.07, 13.43, PitcherRole.MIDDLE_RELIEF, 1.0),
    ]
    bullpen12 = [
        #yankees vs blue jays 2
    Pitcher("Max Fried", 1.11, 0.94, 8.19, PitcherRole.STARTER, 6.0),
    Pitcher("Yerry De Los Santos", 0.00, 1.00, 9.00, PitcherRole.MIDDLE_RELIEF, 1.0),
    Pitcher("Tyler Matzek", 5.40, 3.30, 9.00, PitcherRole.MIDDLE_RELIEF, 1.0),

    ]
    bullpen13 = [
        #yankees vs blue jays 3
    Pitcher("José Berríos", 4.33, 1.42, 8.48, PitcherRole.STARTER, 6.0),
    Pitcher("Brendon Little", 1.86, 1.14, 13.5, PitcherRole.MIDDLE_RELIEF, 1.0),
    Pitcher("Yimi García", 3.71, 1.12, 12.71, PitcherRole.MIDDLE_RELIEF, 1.0),
    Pitcher("Mason Fluharty", 2.16, 0.60, 8.64, PitcherRole.MIDDLE_RELIEF, 1.0),
    Pitcher("Jeff Hoffman", 6.05, 1.19, 14.42, PitcherRole.CLOSER, 1.0),
    ]
    bullpen14 = []
    bullpen15 = []
    real_game_data1 = {'promedio': 1, 'desviacion': 0, 'minimo': 1, 'maximo': 1}
    real_game_data2 = {'promedio': 12, 'desviacion': 0, 'minimo': 12, 'maximo': 12}
    real_game_data3 = {'promedio': 11, 'desviacion': 0, 'minimo': 11, 'maximo': 11}
    real_game_data4 = {'promedio': 4, 'desviacion': 0, 'minimo': 4, 'maximo': 4}
    real_game_data5 = {'promedio': 11, 'desviacion': 0, 'minimo': 11, 'maximo': 11}
    real_game_data6 = {'promedio': 12, 'desviacion': 0, 'minimo': 12, 'maximo': 12}
    real_game_data7 = {'promedio': 5, 'desviacion': 0, 'minimo': 0, 'maximo': 5}
    real_game_data8 = {'promedio': 2, 'desviacion': 0, 'minimo': 2, 'maximo': 2}
    real_game_data9 = {'promedio': 3, 'desviacion': 0, 'minimo': 3, 'maximo': 3}
    real_game_data10 = {'promedio': 4, 'desviacion': 0, 'minimo': 4, 'maximo': 4}
    real_game_data11 = {'promedio': 5, 'desviacion': 0, 'minimo': 5, 'maximo': 5}
    real_game_data12 = {'promedio': 11, 'desviacion': 0, 'minimo': 11, 'maximo': 11}
    real_game_data13 = {'promedio': 2, 'desviacion': 0, 'minimo': 2, 'maximo': 2}
    real_game_data14 = {'promedio': 0, 'desviacion': 0, 'minimo': 0, 'maximo': 0}
    real_game_data15 = {'promedio': 0, 'desviacion': 0, 'minimo': 0, 'maximo': 0}
    bullpen_realdata_list = [
        (bullpen1, real_game_data1),
        (bullpen2, real_game_data2),
        (bullpen3, real_game_data3),
        (bullpen4, real_game_data4),
        (bullpen5, real_game_data5),
        (bullpen6, real_game_data6),
        (bullpen7, real_game_data7),
        (bullpen8, real_game_data8),
        (bullpen9, real_game_data9),
        (bullpen10, real_game_data10),
        (bullpen11, real_game_data11),
        (bullpen12, real_game_data12),
        (bullpen13, real_game_data13),
        (bullpen14, real_game_data14),
        (bullpen15, real_game_data15),
    ]

    # Crear el optimizador con los hiperparámetros mejorados
    optimizer = BaseballGeneticOptimizer(
        lineup=lineup,
        bullpen_realdata_list=bullpen_realdata_list,
        population_size=40,
        generations=25,
        mutation_rate=0.15,
        crossover_rate=0.8,
        elitism_count=2
    )

    # Ejecutar la optimización
    best_weights, best_fitness = optimizer.evolve()

    print("\nMejores pesos encontrados:")
    # Pesos de pitchers
    print("Pesos de pitchers:")
    print(f"ERA Factor Weight: {best_weights[0]:.2f}")
    print(f"WHIP Factor Weight: {best_weights[1]:.2f}")
    print(f"K/9 Factor Divisor: {best_weights[2]:.2f}")
    print(f"Stamina Exponent: {best_weights[3]:.2f}")
    print(f"Pressure Factor Base: {best_weights[4]:.2f}")
    # Pesos de bateadores
    print("\nPesos de bateadores:")
    print(f"OBP Weight: {best_weights[5]:.2f}")
    print(f"AVG Weight: {best_weights[6]:.2f}")
    print(f"SLG Weight: {best_weights[7]:.2f}")
    print(f"Clutch Factor: {best_weights[8]:.2f}")
    print(f"\nFitness Score: {best_fitness:.4f}")

    # Probar los mejores pesos con cada bullpen
    for i, (bullpen, real_data) in enumerate(bullpen_realdata_list):
        optimized_simulator = CustomBaseballSimulator(
            lineup=lineup,
            bullpen=bullpen,
            era_factor_weight=best_weights[0],
            whip_factor_weight=best_weights[1],
            k_factor_divisor=best_weights[2],
            stamina_exponent=best_weights[3],
            pressure_factor_base=best_weights[4],
            obp_weight=best_weights[5],
            avg_weight=best_weights[6],
            slg_weight=best_weights[7],
            batter_clutch_factor=best_weights[8]
        )
        results = optimized_simulator.monte_carlo_game_simulation(num_simulations=1000)
        print(f"\nResultados con pesos optimizados para bullpen {i+1}:")
        print(f"Promedio de carreras simulado: {results['promedio']:.2f} ± {results['desviacion']:.2f}")
        print(f"Promedio real: {real_data['promedio']}")