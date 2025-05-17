import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from enum import Enum

class PitcherRole(Enum):
    STARTER = "Starter"
    MIDDLE_RELIEF = "Middle Relief"
    SETUP = "Setup"
    CLOSER = "Closer"

@dataclass
class Player:
    name: str
    batting_avg: float  # Promedio de bateo
    slugging: float    # Slugging percentage
    obp: float        # On-base percentage

@dataclass
class Pitcher:
    name: str
    era: float        # Earned Run Average
    whip: float       # Walks + Hits per Inning Pitched
    k_per_9: float    # Strikeouts per 9 innings
    role: PitcherRole # Rol del pitcher
    stamina: float    # Resistencia (0-1)
    current_stamina: float = 1.0  # Resistencia actual
    
    def can_pitch(self) -> bool:
        """Determina si el pitcher puede lanzar basado en su resistencia actual"""
        return self.current_stamina > 0.2
    
    def use_stamina(self, innings: float = 1.0):
        """Reduce la resistencia del pitcher"""
        self.current_stamina = max(0, self.current_stamina - (innings / self.stamina))
    
    def rest(self):
        """Recupera resistencia (se llama entre juegos)"""
        self.current_stamina = 1.0

@dataclass
class PlayerStats:
    name: str
    at_bats: int = 0
    hits: int = 0
    extra_base_hits: int = 0
    rbis: int = 0
    strikeouts: int = 0
    walks: int = 0
    runs: int = 0
    runners_on_base: int = 0
    hits_with_runners: int = 0

@dataclass
class PitcherStats:
    name: str
    innings_pitched: float = 0
    earned_runs: int = 0
    hits_allowed: int = 0
    walks_allowed: int = 0
    strikeouts: int = 0
    runners_on_base: int = 0
    runs_with_runners: int = 0
    wins: int = 0
    losses: int = 0

class BaseballSimulator:
    def __init__(self, lineup: List[Player], bullpen: List[Pitcher]):
        self.lineup = lineup
        self.bullpen = bullpen
        self.current_pitcher = None
        self.innings_pitched = 0
        self.runs_allowed = 0
        
        # Inicializar estadísticas
        self.player_stats = {player.name: PlayerStats(name=player.name) for player in lineup}
        self.pitcher_stats = {pitcher.name: PitcherStats(name=pitcher.name) for pitcher in bullpen}
        self.current_game_pitcher = None
        self.batter_index = 0  # Nuevo: índice para el orden al bat
    
    def _select_pitcher(self, inning: int, runs_allowed: int) -> Pitcher:
        """
        Selecciona el siguiente pitcher basado en el inning y las carreras permitidas.
        """
        available_pitchers = [p for p in self.bullpen if getattr(p, 'current_stamina', 1) > 0]
        
        if not available_pitchers:
            # Si no hay pitchers disponibles, mantener el actual y darle 1 de stamina
            if hasattr(self, 'current_pitcher') and self.current_pitcher is not None:
                self.current_pitcher.current_stamina = 1
                return self.current_pitcher
            else:
                raise Exception("No hay pitchers disponibles y no hay pitcher actual")
        
        # Si es el primer inning, usar el abridor
        if inning == 1:
            starters = [p for p in available_pitchers if p.role == PitcherRole.STARTER]
            if starters:
                return starters[0]
        
        # Si es el último inning y hay un cerrador disponible, usarlo
        if inning >= 8:
            closers = [p for p in available_pitchers if p.role == PitcherRole.CLOSER]
            if closers:
                return closers[0]
        
        # Calcular probabilidades de selección basadas en el rol y las carreras permitidas
        probabilities = []
        for pitcher in available_pitchers:
            base_prob = 1.0
            
            # Ajustar probabilidad basado en el rol
            if pitcher.role == PitcherRole.STARTER:
                base_prob *= 0.5 if inning > 5 else 2.0
            elif pitcher.role == PitcherRole.CLOSER:
                base_prob *= 2.0 if inning >= 8 else 0.5
            elif pitcher.role == PitcherRole.SETUP:
                base_prob *= 1.5 if 7 <= inning <= 8 else 0.8
            
            # Ajustar probabilidad basado en las carreras permitidas
            if runs_allowed > 5:
                # Priorizar pitchers con mejor ERA cuando se han permitido muchas carreras
                base_prob *= (5.0 / pitcher.era) if pitcher.era > 0 else 2.0
            
            probabilities.append(base_prob)
        
        # Normalizar probabilidades
        total_prob = sum(probabilities)
        probabilities = [p/total_prob for p in probabilities]
        
        # Seleccionar pitcher basado en probabilidades
        return np.random.choice(available_pitchers, p=probabilities)
    
    def _simulate_pitcher_performance(self, pitcher: Pitcher) -> float:
        """
        Simula el rendimiento del pitcher basado en sus estadísticas.
        Retorna un factor de ajuste para las probabilidades de los bateadores.
        """
        # Factor base basado en ERA (más bajo es mejor)
        era_factor = 4.62 / pitcher.era if pitcher.era > 0 else 1.0
        
        # Factor de WHIP (más bajo es mejor)
        whip_factor = 0.96 / pitcher.whip if pitcher.whip > 0 else 1.0
        
        # Factor de K/9 (más alto es mejor)
        k_factor = pitcher.k_per_9 / 1.52
        
        # Factor de resistencia (afecta más cuando está bajo)
        stamina_factor = pitcher.current_stamina ** 1.32  # Cuadrático para mayor impacto
        
        # Factor de presión (afecta más en situaciones importantes)
        pressure_factor = 0.94
        if self.innings_pitched >= 6:  # Lanzando muchas entradas
            pressure_factor *= 0.9
        if self.runs_allowed > 3:  # Permitiendo muchas carreras
            pressure_factor *= 0.95
        
        return (era_factor * whip_factor * k_factor * stamina_factor * pressure_factor)
    
    def _process_at_bat_result(self, result: str, runs: int, outs: int, bases: List[bool], batter: Player) -> Tuple[int, int, List[bool]]:
        """
        Procesa el resultado de un turno al bat y actualiza el estado del juego y las estadísticas.
        """
        # Actualizar estadísticas del bateador
        stats = self.player_stats[batter.name]
        stats.at_bats += 1
        runners_on_base = sum(bases)
        stats.runners_on_base += runners_on_base

        if result == 'out':
            outs += 1
            stats.strikeouts += 1
        elif result == 'homerun':
            runs += sum(bases) + 1
            stats.hits += 1
            stats.extra_base_hits += 1
            stats.rbis += sum(bases) + 1
            stats.runs += 1
            stats.hits_with_runners += 1
            bases = [False, False, False]
        elif result == 'triple':
            runs += sum(bases)
            stats.hits += 1
            stats.extra_base_hits += 1
            stats.rbis += sum(bases)
            stats.hits_with_runners += 1
            bases = [False, False, True]
        elif result == 'double':
            runs += bases[1] + bases[2]
            stats.hits += 1
            stats.extra_base_hits += 1
            stats.rbis += bases[1] + bases[2]
            stats.hits_with_runners += 1
            bases = [False, True, bases[0]]
        elif result == 'single':
            runs += bases[2]
            stats.hits += 1
            stats.rbis += bases[2]
            stats.hits_with_runners += 1
            bases = [True, bases[0], bases[1]]
        elif result == 'walk':
            if all(bases):
                runs += 1
            stats.walks += 1
            bases = [True] + bases[:-1]
            
        return runs, outs, bases

    def simulate_inning(self, inning: int) -> int:
        """
        Simula una entrada completa y retorna el número de carreras anotadas.
        """
        runs = 0
        outs = 0
        bases = [False, False, False]  # [1B, 2B, 3B]
        
        # Seleccionar pitcher para esta entrada
        self.current_pitcher = self._select_pitcher(inning, self.runs_allowed)
        pitcher_factor = self._simulate_pitcher_performance(self.current_pitcher)
        
        # Actualizar estadísticas del pitcher
        pitcher_stats = self.pitcher_stats[self.current_pitcher.name]
        pitcher_stats.innings_pitched += 1
        
        while outs < 3:
            # Seleccionar el siguiente bateador en orden
            batter = self.lineup[self.batter_index]
            self.batter_index = (self.batter_index + 1) % len(self.lineup)
            
            # Simular el resultado del turno al bat
            result = self._simulate_at_bat(batter, pitcher_factor)
            
            # Procesar el resultado
            runs, outs, bases = self._process_at_bat_result(result, runs, outs, bases, batter)
            
            # Actualizar estadísticas del pitcher
            if result in ['single', 'double', 'triple', 'homerun']:
                pitcher_stats.hits_allowed += 1
            elif result == 'walk':
                pitcher_stats.walks_allowed += 1
            elif result == 'out':
                pitcher_stats.strikeouts += 1
        
        # Actualizar estadísticas del pitcher
        pitcher_stats.earned_runs += runs
        pitcher_stats.runners_on_base += sum(bases)
        
        # Actualizar estadísticas del simulador
        self.current_pitcher.use_stamina(1.0)
        self.runs_allowed += runs
        self.innings_pitched += 1
        
        return runs
    
    def _simulate_at_bat(self, batter: Player, pitcher_factor: float) -> str:
        """
        Simula un turno al bat basado en las estadísticas del bateador y el pitcher.
        """
        # Ajustar probabilidades basado en el rendimiento del pitcher
        adjusted_obp = min(batter.obp / pitcher_factor, 0.55)  # Limitar OBP ajustado (más alto)
        adjusted_avg = min(batter.batting_avg / pitcher_factor, 0.42)  # Limitar AVG ajustado (más alto)

        # Calcular probabilidad de strikeout basada en K/9 del pitcher
        k_prob = min((self.current_pitcher.k_per_9 / 27.0) * pitcher_factor, 0.25)  # Limitar K% (más bajo)

        # Calcular probabilidad de walk basada en WHIP del pitcher
        walk_prob = max(min((self.current_pitcher.whip - 1.0) * 0.15, 0.10), 0)  # Limitar BB% (más bajo y factor reducido)

        # Probabilidad de hit es la diferencia entre OBP y BB
        hit_prob = max(adjusted_obp - walk_prob, 0)
        # Probabilidad de out es el resto
        out_prob = max(1.0 - (k_prob + walk_prob + hit_prob), 0)

        rand = np.random.random()

        if rand < k_prob:
            return 'out'
        elif rand < k_prob + walk_prob:
            return 'walk'
        elif rand < k_prob + walk_prob + hit_prob:
            # Determinar tipo de hit según AVG y SLG
            hit_type = np.random.random()
            # Proporción de sencillos respecto a AVG
            single_prob = min(adjusted_avg / adjusted_obp, 0.85)
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
    
    def simulate_game(self, num_innings: int = 9) -> int:
        """
        Simula un juego completo y retorna el número total de carreras.
        """
        total_runs = 0
        self.runs_allowed = 0
        self.innings_pitched = 0
        
        # Resetear la resistencia de todos los pitchers
        for pitcher in self.bullpen:
            pitcher.rest()
        
        for inning in range(1, num_innings + 1):
            total_runs += self.simulate_inning(inning)
        
        # Actualizar victorias/derrotas
        if total_runs > self.runs_allowed:
            self.pitcher_stats[self.current_pitcher.name].wins += 1
        else:
            self.pitcher_stats[self.current_pitcher.name].losses += 1
        
        return total_runs
    
    def monte_carlo_game_simulation(self, num_simulations: int = 1000, num_innings: int = 9) -> Dict:
        """
        Realiza múltiples simulaciones Monte Carlo del mismo partido.
        
        Args:
            num_simulations: Número de veces a simular el mismo partido
            num_innings: Número de entradas por partido
            
        Returns:
            Diccionario con estadísticas de las simulaciones:
            - 'carreras': Lista de carreras por simulación
            - 'promedio': Promedio de carreras
            - 'desviacion': Desviación estándar de carreras
            - 'maximo': Máximo de carreras
            - 'minimo': Mínimo de carreras
            - 'distribucion': Distribución de carreras (histograma)
        """
        carreras = []
        
        for _ in range(num_simulations):
            # Simular un partido
            runs = self.simulate_game(num_innings)
            carreras.append(runs)
        
        # Calcular estadísticas
        carreras_array = np.array(carreras)
        promedio = np.mean(carreras_array)
        desviacion = np.std(carreras_array)
        maximo = np.max(carreras_array)
        minimo = np.min(carreras_array)
        
        # Calcular distribución (histograma)
        bins = range(minimo, maximo + 2)
        distribucion, _ = np.histogram(carreras_array, bins=bins)
        
        return {
            'carreras': carreras,
            'promedio': promedio,
            'desviacion': desviacion,
            'maximo': maximo,
            'minimo': minimo,
            'distribucion': distribucion,
            'bins': bins[:-1]  # Excluir el último bin para que coincida con la distribución
        }

    def get_player_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Retorna las estadísticas calculadas para cada jugador.
        """
        stats = {}
        for player in self.lineup:
            player_stats = self.player_stats[player.name]
            if player_stats.at_bats > 0:
                stats[player.name] = {
                    'batting_avg': player_stats.hits / player_stats.at_bats,
                    'extra_base_pct': player_stats.extra_base_hits / player_stats.hits if player_stats.hits > 0 else 0,
                    'rbi_per_game': player_stats.rbis / (self.innings_pitched / 9),
                    'strikeout_rate': player_stats.strikeouts / player_stats.at_bats,
                    'walk_rate': player_stats.walks / player_stats.at_bats,
                    'clutch_hitting': player_stats.hits_with_runners / player_stats.runners_on_base if player_stats.runners_on_base > 0 else 0
                }
        return stats

class CustomBaseballSimulator(BaseballSimulator):
    def __init__(self, lineup, bullpen):
        super().__init__(lineup, bullpen)
        self.obp_weight = 1.61
        self.avg_weight = 0.81
        self.slg_weight = 1.41
        self.clutch_factor = 1.35

    def _simulate_at_bat(self, batter: Player, pitcher_factor: float) -> str:
        # Ajustar probabilidades basado en los pesos personalizados
        adjusted_obp = min((batter.obp * self.obp_weight) / pitcher_factor, 0.55)
        adjusted_avg = min((batter.batting_avg * self.avg_weight) / pitcher_factor, 0.42)
        adjusted_slg = min((batter.slugging * self.slg_weight) / pitcher_factor, 1.0)

        # Clutch: si hay corredores en base, aumentar ligeramente la probabilidad de hit
        clutch_multiplier = self.clutch_factor if sum(getattr(self, 'bases', [False, False, False])) > 0 else 1.0
        adjusted_obp *= clutch_multiplier
        adjusted_avg *= clutch_multiplier
        adjusted_slg *= clutch_multiplier

        # Calcular probabilidad de strikeout basada en K/9 del pitcher
        k_prob = min((self.current_pitcher.k_per_9 / 27.0) * pitcher_factor, 0.25)
        # Calcular probabilidad de walk basada en WHIP del pitcher
        walk_prob = max(min((self.current_pitcher.whip - 1.0) * 0.15, 0.10), 0)
        # Probabilidad de hit es la diferencia entre OBP y BB
        hit_prob = max(adjusted_obp - walk_prob, 0)
        # Probabilidad de out es el resto
        out_prob = max(1.0 - (k_prob + walk_prob + hit_prob), 0)

        rand = np.random.random()
        if rand < k_prob:
            return 'out'
        elif rand < k_prob + walk_prob:
            return 'walk'
        elif rand < k_prob + walk_prob + hit_prob:
            # Determinar tipo de hit según AVG y SLG
            hit_type = np.random.random()
            single_prob = min(adjusted_avg / adjusted_obp, 0.85)
            double_prob = min((adjusted_slg - adjusted_avg) / adjusted_slg, 0.10)
            triple_prob = 0.03  # Fijo, poco frecuente
            if hit_type < single_prob:
                return 'single'
            elif hit_type < single_prob + double_prob:
                return 'double'
            elif hit_type < single_prob + double_prob + triple_prob:
                return 'triple'
            else:
                return 'homerun'
        else:
            return 'out'

# Ejemplo de uso
if __name__ == "__main__":
    # Crear una alineación completa de jugadores con estadísticas reales
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
    Pitcher("Luis Severino", 4.70, 1.32, 6.71, PitcherRole.STARTER, 7.0),
    Pitcher("Mitch Spence", 5.11, 1.58, 9.12, PitcherRole.MIDDLE_RELIEF, 2.0),
    Pitcher("Elvis Alvarado", 10.80, 2.40, 10.80, PitcherRole.MIDDLE_RELIEF, 1.0),
    Pitcher("T.J. McFarland", 5.23, 1.74, 3.60, PitcherRole.MIDDLE_RELIEF, 1.0),
    Pitcher("Hogan Harris", 4.38, 1.38, 10.22, PitcherRole.MIDDLE_RELIEF, 2.0),
]

    # Crear el simulador con pesos personalizados de bateadores
    simulator = BaseballSimulator(lineup, bullpen)
    resultados = simulator.monte_carlo_game_simulation(num_simulations=1000, num_innings=9)
    print(f"Promedio de carreras: {resultados['promedio']:.2f} ± {resultados['desviacion']:.2f}")
    print(f"Rango de carreras: {resultados['minimo']} - {resultados['maximo']}")
    for carreras, frecuencia in zip(resultados['bins'], resultados['distribucion']):
        print(f"{carreras} carreras: {frecuencia} veces")

    print("\nSimulación con pesos personalizados de bateadores:")
    custom_simulator = CustomBaseballSimulator(lineup, bullpen)
    resultados_custom = custom_simulator.monte_carlo_game_simulation(num_simulations=1000, num_innings=9)
    print(f"Promedio de carreras: {resultados_custom['promedio']:.2f} ± {resultados_custom['desviacion']:.2f}")
    print(f"Rango de carreras: {resultados_custom['minimo']} - {resultados_custom['maximo']}")
    for carreras, frecuencia in zip(resultados_custom['bins'], resultados_custom['distribucion']):
        print(f"{carreras} carreras: {frecuencia} veces")
