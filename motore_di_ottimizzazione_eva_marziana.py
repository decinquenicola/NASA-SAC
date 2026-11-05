"""
Script per la pianificazione di Marswalk per NASA Space Apps Challenge - Gruppo Morpheus

Cosa fa questo script:
1. Carica 4 layer raster NASA (DEM Topografico, Inerzia Termica, Spettrale/Chimico, Radiazioni)
   oppure genera un terreno marziano sintetico (Mount Sharp Boxwork simulation) per test immediati.
2. Calcola Pendenza, Rugosità e Sky-View Factor (schermatura radiazioni).
3. Costruisce la Mappa Obiettivo e la Mappa Rischio/Costo.
4. Estrae automaticamente i nodi scientifici candidati filtrando le zone pericolose.
5. Calcola i percorsi ottimali come archi tra tutte le coppie di nodi con A* su griglia raster.
6. Risolve l'Orienteering Problem con Google OR-Tools rispettando il budget di Ossigeno/Tempo/Energia.
7. Esporta il risultato in un file GeoJSON ('optimal_eva_mission.geojson') pronto per QGIS o Web App.
"""

# Rivedere codice e commenti
# BACKEND ispirato al metodo consigliato da Marinelli

import heapq
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, uniform_filter

try:
    from skimage.graph import route_through_array
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False

# Opzionale: rasterio per caricare veri GeoTIFF NASA
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def _astar_grid_8conn(
    cost_surface: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
) -> Tuple[List[Tuple[int, int]], float]:
    """Fallback A* 8-connesso geometrico equivalente a skimage.graph.route_through_array."""
    h, w = cost_surface.shape
    sr, sc = start
    er, ec = end
    min_step = float(cost_surface.min())
    SQRT2 = math.sqrt(2.0)

    neighbors = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2),
    ]

    g_score = np.full((h, w), np.inf, dtype=np.float64)
    g_score[sr, sc] = 0.0
    parent_r = np.full((h, w), -1, dtype=np.int32)
    parent_c = np.full((h, w), -1, dtype=np.int32)

    pq = [(0.0, sr, sc)]
    while pq:
        _, r, c = heapq.heappop(pq)
        if r == er and c == ec:
            break
        cur_g = g_score[r, c]
        cur_cost = cost_surface[r, c]
        for dr, dc, dist in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                step_cost = 0.5 * (cur_cost + cost_surface[nr, nc]) * dist
                tentative_g = cur_g + step_cost
                if tentative_g < g_score[nr, nc]:
                    g_score[nr, nc] = tentative_g
                    parent_r[nr, nc] = r
                    parent_c[nr, nc] = c
                    heur = math.hypot(er - nr, ec - nc) * min_step
                    heapq.heappush(pq, (tentative_g + heur, nr, nc))

    path: List[Tuple[int, int]] = []
    cr, cc = er, ec
    while cr != -1 and cc != -1:
        path.append((int(cr), int(cc)))
        if cr == sr and cc == sc:
            break
        pr, pc = int(parent_r[cr, cc]), int(parent_c[cr, cc])
        cr, cc = pr, pc
    path.reverse()
    return path, float(g_score[er, ec])


@dataclass
class MissionConfig:
    """Parametri configurabili dall'astronauta o dalla console dell'Habitat."""
    pixel_size_m: float = 5.0          # Risoluzione spaziale (metri per pixel)
    max_slope_deg: float = 20.0        # Pendenza massima superabile con tuta EVA (gradi)
    min_thermal_inertia: float = 0.22  # Soglia minima inerzia termica (sotto = sabbia troppo soffice)
    base_eva_speed_ms: float = 0.85    # Velocità base astronauta in piano su Marte (m/s)
    eva_budget_minutes: float = 180.0  # Autonomia massima EVA (minuti di O2 / batteria)
    science_time_per_poi_min: float = 15.0  # Tempo speso fermo su ogni Waypoint per campionamento
    num_candidate_pois: int = 18       # Numero di Waypoint candidati estratti dalla Mappa Obiettivo
    num_explorers: int = 1             # 1 = Singolo Astronauta (OP), 2 = Astronauta + Rover / Coppia (TOP)
    # Pesi Mappa Obiettivo (devono sommare a 1.0)
    w_spectral: float = 0.50           # Interesse mineralogico / chimico (es. argille/solfati)
    w_boxwork_geo: float = 0.35        # Strutture geologiche / contatti (es. reticoli boxwork)
    w_thermal_anom: float = 0.15       # Anomalie termiche
    # Peso penalità radiazioni nella Mappa Rischio
    radiation_risk_weight: float = 0.40


class MarsTerrainGenerator:
    """
    Genera o carica i layer geospaziali di Marte.
    Se non sono presenti file GeoTIFF reali, crea un dataset sintetico realistico
    ispirato a Mount Sharp (Gale Crater) per testare la pipeline prima del 13 Novembre.
    """

    @staticmethod
    def load_or_generate(shape: Tuple[int, int] = (200, 200), seed: int = 42) -> Dict[str, np.ndarray]:
        np.random.seed(seed)
        h, w = shape
        y, x = np.mgrid[0:h, 0:w]

        # 1. Topografia (DEM in metri): salita verso Mount Sharp + crateri + creste boxwork
        base_slope = (y / h) * 120.0  # Dislivello di 120m da Nord a Sud
        hills = gaussian_filter(np.random.randn(h, w), sigma=18) * 350.0
        # Aggiungiamo due crateri ripidi (ostacoli naturali da aggirare con A*)
        crater1 = -35.0 * np.exp(-((x - 95)**2 + (y - 90)**2) / (2 * 16**2))
        crater2 = -45.0 * np.exp(-((x - 140)**2 + (y - 135)**2) / (2 * 20**2))
        dem = base_slope + hills + crater1 + crater2

        # 2. Layer Spettrale/Chimico (0..1): concentrazione di minerali idrati / solfati
        spectral = gaussian_filter(np.random.rand(h, w), sigma=8)
        spectral = (spectral - spectral.min()) / (spectral.max() - spectral.min())

        # 3. Layer Morfologico Boxwork / Geologia (0..1): strutture reticolari ad alto interesse
        boxwork = np.abs(np.sin(x / 6.0) * np.cos(y / 6.0)) * gaussian_filter(np.random.rand(h, w), sigma=12)
        boxwork = (boxwork - boxwork.min()) / (boxwork.max() - boxwork.min())

        # 4. Inerzia Termica THEMIS (0..1): valori bassi (<0.22) = trappole di sabbia fine
        thermal_inertia = gaussian_filter(np.random.rand(h, w), sigma=14)
        thermal_inertia = (thermal_inertia - thermal_inertia.min()) / (thermal_inertia.max() - thermal_inertia.min())

        return {
            "dem": dem,
            "spectral": spectral,
            "boxwork": boxwork,
            "thermal_inertia": thermal_inertia
        }

    @staticmethod
    def load_from_geotiffs(dem_path: str, spectral_path: str, boxwork_path: str, thermal_path: str) -> Dict[str, np.ndarray]:
        """Da usare il 13 Novembre con i GeoTIFF ufficiali allineati in QGIS."""
        if not HAS_RASTERIO:
            raise RuntimeError("Installa rasterio ('pip install rasterio') per leggere file GeoTIFF.")

        def read_norm(path: str, normalize: bool = True) -> np.ndarray:
            with rasterio.open(path) as src:
                arr = src.read(1).astype(np.float64)
                arr = np.nan_to_num(arr, nan=np.nanmedian(arr))
                if normalize:
                    mn, mx = arr.min(), arr.max()
                    if mx > mn:
                        arr = (arr - mn) / (mx - mn)
                return arr

        return {
            "dem": read_norm(dem_path, normalize=False),
            "spectral": read_norm(spectral_path, normalize=True),
            "boxwork": read_norm(boxwork_path, normalize=True),
            "thermal_inertia": read_norm(thermal_path, normalize=True),
        }


class MarsGISProcessor:
    """Elabora i raster per generare Mappa Obiettivo, Mappa Rischio e Waypoint candidati."""

    def __init__(self, layers: Dict[str, np.ndarray], config: MissionConfig):
        self.layers = layers
        self.cfg = config
        self.dem = layers["dem"]
        self.h, self.w = self.dem.shape

        # Derivate topografiche e ambientali
        self.slope_deg, self.slope_tan = self._compute_slope()
        self.roughness = self._compute_roughness()
        self.radiation_exposure = self._compute_radiation_exposure()

        # Mappe principali suggerite dal prof di Ricerca Operativa (ma separate per Nodi e Archi!)
        self.safety_mask = self._compute_safety_mask()
        self.objective_map = self._compute_objective_map()
        self.cost_surface_min_per_pixel = self._compute_risk_cost_surface()

    def _compute_slope(self) -> Tuple[np.ndarray, np.ndarray]:
        gy, gx = np.gradient(self.dem, self.cfg.pixel_size_m)
        slope_tan = np.sqrt(gx**2 + gy**2)
        slope_deg = np.degrees(np.arctan(slope_tan))
        return slope_deg, slope_tan

    def _compute_roughness(self) -> np.ndarray:
        """Deviazione standard locale dell'elevazione (ostacoli/massi a piccola scala)."""
        local_mean = uniform_filter(self.dem, size=5)
        local_sq_mean = uniform_filter(self.dem**2, size=5)
        variance = np.maximum(local_sq_mean - local_mean**2, 0.0)
        rough = np.sqrt(variance)
        return (rough - rough.min()) / (rough.max() - rough.min() + 1e-8)

    def _compute_radiation_exposure(self) -> np.ndarray:
        """
        Stima l'esposizione alle radiazioni combinando quota e Sky-View Factor semplificato.
        Creste esposte = radiazione più alta; fondovalle o pareti rocciose = schermatura parziale.
        """
        local_horizon = uniform_filter(self.dem, size=15) - self.dem
        # Se local_horizon > 0 siamo in una conca/canalone protetto dalle pareti circostanti
        shielding = np.clip(local_horizon / 15.0, -1.0, 1.0)
        exposure = 1.0 - (shielding + 1.0) / 2.0
        return exposure

    def _compute_safety_mask(self) -> np.ndarray:
        """Restituisce True dove l'astronauta può camminare in sicurezza."""
        safe_slope = self.slope_deg <= self.cfg.max_slope_deg
        safe_sand = self.layers["thermal_inertia"] >= self.cfg.min_thermal_inertia
        return safe_slope & safe_sand

    def _compute_objective_map(self) -> np.ndarray:
        """Mappa Obiettivo (0..100): combina i layer scientifici e azzera le zone inaccessibili."""
        thermal_anom = np.abs(self.layers["thermal_inertia"] - np.median(self.layers["thermal_inertia"]))
        thermal_anom = thermal_anom / (thermal_anom.max() + 1e-8)

        raw_score = (
            self.cfg.w_spectral * self.layers["spectral"]
            + self.cfg.w_boxwork_geo * self.layers["boxwork"]
            + self.cfg.w_thermal_anom * thermal_anom
        )
        # Applichiamo il filtro di sicurezza (pendenza e sabbia)
        masked_score = np.where(self.safety_mask, raw_score * 100.0, 0.0)
        return masked_score

    def _compute_risk_cost_surface(self) -> np.ndarray:
        """
        Mappa Rischio/Costo: calcola i minuti equivalenti di EVA per attraversare ogni pixel.
        Usa la Tobler's Hiking Function adattata a Marte + penalità sabbia/rugosità/radiazioni.
        """
        # Velocità di Tobler modificata per tuta EVA (m/s)
        tobler_factor = np.exp(-3.5 * np.abs(self.slope_tan + 0.05))

        # Penalità terreno: bassa inerzia termica (sabbia) e alta rugosità rallentano la marcia
        sand_penalty = np.clip(self.layers["thermal_inertia"] / 0.5, 0.35, 1.0)
        rough_penalty = 1.0 - 0.4 * self.roughness
        terrain_factor = sand_penalty * rough_penalty

        speed_ms = np.maximum(self.cfg.base_eva_speed_ms * tobler_factor * terrain_factor, 0.05)

        # Tempo reale di attraversamento del pixel in minuti
        time_min = (self.cfg.pixel_size_m / speed_ms) / 60.0

        # Fattore di rischio radiazioni
        rad_factor = 1.0 + self.cfg.radiation_risk_weight * self.radiation_exposure

        cost_surface = time_min * rad_factor

        # Penalità altissima (muro virtuale per A*) nelle zone non sicure (pendenza > 20° o sabbia mobile)
        cost_surface = np.where(self.safety_mask, cost_surface, 999.0)
        return cost_surface

    def extract_candidate_pois(self, landing_site_rc: Tuple[int, int]) -> List[Dict]:
        """
        Estrae i massimi locali della Mappa Obiettivo per creare i Nodi dell'Orienteering Problem.
        Il nodo 0 è sempre il Lander/Habitat di partenza e arrivo.
        """
        neighborhood_size = 15  # Distanza minima in pixel tra due Waypoint scientifici
        local_max = maximum_filter(self.objective_map, size=neighborhood_size) == self.objective_map
        valid_peaks = local_max & (self.objective_map > 25.0)

        peak_coords = np.argwhere(valid_peaks)
        peak_scores = [self.objective_map[r, c] for r, c in peak_coords]

        # Ordina per valore scientifico decrescente e prendi i migliori K
        sorted_idx = np.argsort(peak_scores)[::-1][: self.cfg.num_candidate_pois]

        nodes = [{
            "id": 0,
            "name": "HABITAT_LANDER",
            "row": int(landing_site_rc[0]),
            "col": int(landing_site_rc[1]),
            "science_score": 0,
            "elevation_m": float(self.dem[landing_site_rc]),
            "slope_deg": float(self.slope_deg[landing_site_rc]),
        }]

        for idx_rank, idx in enumerate(sorted_idx, start=1):
            r, c = peak_coords[idx]
            nodes.append({
                "id": idx_rank,
                "name": f"SCI_POI_{idx_rank:02d}",
                "row": int(r),
                "col": int(c),
                "science_score": int(round(self.objective_map[r, c])),
                "elevation_m": float(self.dem[r, c]),
                "slope_deg": float(self.slope_deg[r, c]),
            })

        return nodes


class MarsEVARouter:
    """
    Combina Micro-Routing (A* su Mappa Rischio) e Macro-Routing (Orienteering Problem con OR-Tools).
    """

    def __init__(self, processor: MarsGISProcessor, nodes: List[Dict], config: MissionConfig):
        self.proc = processor
        self.nodes = nodes
        self.cfg = config
        self.num_nodes = len(nodes)
        self.cost_matrix_min: np.ndarray = np.zeros((self.num_nodes, self.num_nodes))
        self.paths_rc: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

    def compute_all_pairs_astar(self) -> None:
        """Calcola i sentieri sicuri A* tra ogni coppia di Waypoint sulla Mappa Rischio."""
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                start = (self.nodes[i]["row"], self.nodes[i]["col"])
                end = (self.nodes[j]["row"], self.nodes[j]["col"])

                if HAS_SKIMAGE:
                    # route_through_array usa Dijkstra/A* C-ottimizzato su griglia 8-connessa
                    indices, weight = route_through_array(
                        self.proc.cost_surface_min_per_pixel,
                        start=start,
                        end=end,
                        fully_connected=True,
                        geometric=True,
                    )
                else:
                    indices, weight = _astar_grid_8conn(
                        self.proc.cost_surface_min_per_pixel,
                        start=start,
                        end=end,
                    )
                self.cost_matrix_min[i, j] = weight
                self.cost_matrix_min[j, i] = weight
                self.paths_rc[(i, j)] = indices
                self.paths_rc[(j, i)] = list(reversed(indices))

    def solve_orienteering_problem(self) -> Dict:
        """
        Risolve il Team Orienteering Problem (TOP) con Google OR-Tools:
        - Massimizza la somma dei punteggi scientifici dei Waypoint visitati.
        - Rispetta il vincolo di tempo/ossigeno massimo (eva_budget_minutes) per ciascun esploratore.
        """
        self.compute_all_pairs_astar()

        if not HAS_ORTOOLS:
            return self._solve_orienteering_fallback()

        # OR-Tools lavora con interi: convertiamo i minuti in secondi interi
        scale = 60
        max_budget_sec = int(self.cfg.eva_budget_minutes * scale)
        science_service_sec = int(self.cfg.science_time_per_poi_min * scale)

        manager = pywrapcp.RoutingIndexManager(self.num_nodes, self.cfg.num_explorers, 0)
        routing = pywrapcp.RoutingModel(manager)

        def time_callback(from_index: int, to_index: int) -> int:
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            if from_node == to_node:
                return 0
            travel_sec = int(round(self.cost_matrix_min[from_node, to_node] * scale))
            service_sec = science_service_sec if to_node != 0 else 0
            return travel_sec + service_sec

        transit_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Vincolo di autonomia EVA (Ossigeno / Tempo massimo)
        routing.AddDimension(
            transit_callback_index,
            0,               # Nessun tempo di attesa
            max_budget_sec,  # Capacità massima per esploratore (in secondi)
            True,            # Inizia a tempo zero
            "EVA_Time",
        )

        # Trasformazione in Orienteering Problem: tutti i nodi scientifici (1..N-1) sono OPZIONALI,
        # ma se non vengono visitati si paga una penalità proporzionale al loro valore scientifico!
        penalty_multiplier = 100_000
        for node_idx in range(1, self.num_nodes):
             routing_index = manager.NodeToIndex(node_idx)
             penalty = int(self.nodes[node_idx]["science_score"] * penalty_multiplier)
             routing.AddDisjunction([routing_index], penalty)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = 2

        solution = routing.SolveWithParameters(search_parameters)
        if not solution:
            raise RuntimeError("Nessuna soluzione trovata per l'Orienteering Problem.")

        return self._format_solution(manager, routing, solution, scale)

    def _format_solution(self, manager, routing, solution, scale: int) -> Dict:
        routes_output = []
        total_science = 0

        for vehicle_id in range(self.cfg.num_explorers):
            index = routing.Start(vehicle_id)
            visited_nodes = []
            full_raster_path = []
            total_time_sec = 0

            while not routing.IsEnd(index):
                node_id = manager.IndexToNode(index)
                visited_nodes.append(self.nodes[node_id])
                total_science += self.nodes[node_id]["science_score"]

                next_index = solution.Value(routing.NextVar(index))
                next_node_id = manager.IndexToNode(next_index)

                if node_id != next_node_id:
                    segment = self.paths_rc[(node_id, next_node_id)]
                    if full_raster_path:
                        full_raster_path.extend(segment[1:])
                    else:
                        full_raster_path.extend(segment)

                total_time_sec += routing.GetArcCostForVehicle(index, next_index, vehicle_id)
                index = next_index

            # Aggiungi il ritorno al Lander (nodo 0)
            visited_nodes.append(self.nodes[0])

            routes_output.append({
                "explorer_id": vehicle_id + 1,
                "visited_waypoints": visited_nodes,
                "eva_duration_minutes": round(total_time_sec / scale, 2),
                "remaining_o2_minutes": round(self.cfg.eva_budget_minutes - (total_time_sec / scale), 2),
                "detailed_path_rc": full_raster_path,
            })

        return {
            "total_science_score": total_science,
            "eva_budget_minutes": self.cfg.eva_budget_minutes,
            "routes": routes_output,
        }

    def _solve_orienteering_fallback(self) -> Dict:
        """Solutore Orienteering Problem nativo Python (Branch & Bound + 2-opt) per Python >= 3.15."""
        budget = self.cfg.eva_budget_minutes
        service = self.cfg.science_time_per_poi_min
        unvisited = set(range(1, self.num_nodes))
        routes_output = []
        total_science = 0

        for vehicle_id in range(self.cfg.num_explorers):
            best_route = [0, 0]
            best_score = 0
            best_time = 0.0

            def dfs(curr: int, visited: List[int], elapsed: float, score: int):
                nonlocal best_route, best_score, best_time
                ret_time = elapsed + self.cost_matrix_min[curr, 0]
                if score > best_score or (score == best_score and ret_time < best_time):
                    best_score = score
                    best_route = visited + [0]
                    best_time = ret_time

                candidates = sorted(
                    unvisited - set(visited),
                    key=lambda n: self.nodes[n]["science_score"] / (self.cost_matrix_min[curr, n] + service + 1e-6),
                    reverse=True,
                )
                for nxt in candidates[:8]:
                    step_t = self.cost_matrix_min[curr, nxt] + service
                    if elapsed + step_t + self.cost_matrix_min[nxt, 0] <= budget:
                        dfs(nxt, visited + [nxt], elapsed + step_t, score + self.nodes[nxt]["science_score"])

            dfs(0, [0], 0.0, 0)
            for n in best_route[1:-1]:
                unvisited.discard(n)

            total_science += best_score
            visited_nodes = [self.nodes[nid] for nid in best_route]
            full_raster_path: List[Tuple[int, int]] = []
            for a, b in zip(best_route[:-1], best_route[1:]):
                segment = self.paths_rc[(a, b)]
                if full_raster_path:
                    full_raster_path.extend(segment[1:])
                else:
                    full_raster_path.extend(segment)

            routes_output.append({
                "explorer_id": vehicle_id + 1,
                "visited_waypoints": visited_nodes,
                "eva_duration_minutes": round(best_time, 2),
                "remaining_o2_minutes": round(budget - best_time, 2),
                "detailed_path_rc": full_raster_path,
            })

        return {
            "total_science_score": total_science,
            "eva_budget_minutes": self.cfg.eva_budget_minutes,
            "routes": routes_output,
        }

    def export_to_geojson(self, solution_dict: Dict, output_filepath: str = "optimal_eva_mission.geojson") -> None:
        """Esporta Waypoint e Tracciato A* in GeoJSON (coordinate metriche locali o Lon/Lat)."""
        features = []

        # 1. Esporta tutti i Waypoint candidati (visitati e scartati)
        visited_ids = {
            wp["id"]
            for route in solution_dict["routes"]
            for wp in route["visited_waypoints"]
        }

        for node in self.nodes:
            x_m = node["col"] * self.cfg.pixel_size_m
            y_m = (self.proc.h - node["row"]) * self.cfg.pixel_size_m
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [x_m, y_m, node["elevation_m"]]},
                "properties": {
                    **node,
                    "selected_by_orienteering": node["id"] in visited_ids,
                },
            })

        # 2. Esporta le polilinee dei percorsi ottimali A*
        for route in solution_dict["routes"]:
            coords_3d = []
            for r, c in route["detailed_path_rc"]:
                x_m = c * self.cfg.pixel_size_m
                y_m = (self.proc.h - r) * self.cfg.pixel_size_m
                z_m = float(self.proc.dem[r, c])
                coords_3d.append([x_m, y_m, z_m])

            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords_3d},
                "properties": {
                    "explorer_id": route["explorer_id"],
                    "eva_duration_minutes": route["eva_duration_minutes"],
                    "remaining_o2_minutes": route["remaining_o2_minutes"],
                },
            })

        geojson_obj = {"type": "FeatureCollection", "features": features}
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(geojson_obj, f, indent=2)


if __name__ == "__main__":
    print("=== ARES-OP: Inizializzazione Motore EVA Marziana ===")
    cfg = MissionConfig(
        eva_budget_minutes=180.0,  # 3 ore di autonomia EVA
        num_candidate_pois=16,
        num_explorers=1,           # Imposta a 2 per Team Orienteering (Astronauta + Rover)
    )

    # 1. Generazione (o caricamento) dei 4 layer NASA
    layers = MarsTerrainGenerator.load_or_generate(shape=(200, 200))
    processor = MarsGISProcessor(layers, cfg)

    # 2. Posizionamento del Lander in una zona pianeggiante sicura ed estrazione dei POI
    landing_rc = (30, 30)
    candidate_nodes = processor.extract_candidate_pois(landing_site_rc=landing_rc)
    print(f"Estratti {len(candidate_nodes)-1} Waypoint scientifici sicuri dalla Mappa Obiettivo.")

    # 3. Calcolo A* sulla Mappa Rischio + Orienteering Problem con OR-Tools
    router = MarsEVARouter(processor, candidate_nodes, cfg)
    result = router.solve_orienteering_problem()

    # 4. Stampa report missione ed esportazione GeoJSON
    print(f"\n--- RISULTATO OTTIMIZZAZIONE ORIENTEERING PROBLEM ---")
    print(f"Punteggio Scientifico Totale Raccolto: {result['total_science_score']} punti")
    for r in result["routes"]:
        wp_names = " -> ".join(wp["name"] for wp in r["visited_waypoints"])
        print(f"Esploratore #{r['explorer_id']}: {wp_names}")
        print(f"Tempo totale EVA (cammino A* + campionamento): {r['eva_duration_minutes']} / {cfg.eva_budget_minutes} min")
        print(f"Margine Ossigeno residuo: {r['remaining_o2_minutes']} min")

    router.export_to_geojson(result, "optimal_eva_mission.geojson")
    print("\nFile 'optimal_eva_mission.geojson' generato con successo (apribile in QGIS o Streamlit/PyDeck).")