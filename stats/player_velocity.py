import cv2
import numpy as np
from collections import defaultdict
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import measure_distance

class VelocityAnalyzer:
    def __init__(self, fps=30, pixels_per_meter=35, movement_threshold=2.5):
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter  # Calibración más conservadora
        self.movement_threshold = movement_threshold  # Umbral más alto para evitar ruido
        
        # Estadísticas por jugador
        self.velocity_stats = defaultdict(list)
        self.max_velocities = defaultdict(float)
        self.current_velocities = defaultdict(float)
        self.total_distances = defaultdict(float)  # Distancia real acumulada
        self.last_positions = defaultdict(tuple)  # Última posición conocida
        self.velocity_history = defaultdict(lambda: [])  # Para suavizado
        
        # Límites de velocidad realistas para pádel (en píxeles por frame)
        self.max_realistic_velocity_kmh = 25.0  # Velocidad máxima realista en km/h
        self.max_realistic_velocity_px = self.kmh_to_pixels_per_frame(self.max_realistic_velocity_kmh)
        
        # Colores para cada jugador (coincidiendo con el tracker)
        self.trajectory_colors = {
            1: (0, 0, 255),    # Rojo
            2: (0, 255, 0),    # Verde  
            3: (255, 0, 0),    # Azul
            4: (255, 255, 0),  # Amarillo
        }
    
    def calculate_current_velocity(self, player_tracker, virtual_id, current_detection=None):
        if current_detection is None:
            # Usar trayectoria si no hay detección actual
            if (virtual_id not in player_tracker.trajectories or 
                len(player_tracker.trajectories[virtual_id]) < 2):
                return 0.0
            
            trajectory = player_tracker.trajectories[virtual_id]
            prev_pos = trajectory[-2]
            curr_pos = trajectory[-1]
        else:
            # Usar detección actual y posición previa
            if virtual_id not in self.last_positions:
                # Primera detección, guardar posición y retornar 0
                self.last_positions[virtual_id] = current_detection
                return 0.0
            
            prev_pos = self.last_positions[virtual_id]
            curr_pos = current_detection
            
            # Actualizar última posición
            self.last_positions[virtual_id] = current_detection
        
        # Velocidad en píxeles por frame
        velocity_pixels_per_frame = measure_distance(prev_pos, curr_pos)
        
        return velocity_pixels_per_frame
    
    def smooth_velocity(self, virtual_id, velocity, window_size=8):
        # Filtrar valores extremos antes de añadir al historial
        if not self.is_velocity_realistic(velocity):
            # Si la velocidad es irrealista, usar la última velocidad válida o 0
            if self.velocity_history[virtual_id]:
                velocity = self.velocity_history[virtual_id][-1]
            else:
                velocity = 0.0
        
        self.velocity_history[virtual_id].append(velocity)
        
        # Mantener solo los últimos N valores
        if len(self.velocity_history[virtual_id]) > window_size:
            self.velocity_history[virtual_id] = self.velocity_history[virtual_id][-window_size:]
        
        # Calcular mediana en lugar de promedio para ser más robusto ante outliers
        if len(self.velocity_history[virtual_id]) >= 3:
            sorted_velocities = sorted(self.velocity_history[virtual_id])
            median_idx = len(sorted_velocities) // 2
            return sorted_velocities[median_idx]
        else:
            # Si hay muy pocos datos, usar promedio
            return sum(self.velocity_history[virtual_id]) / len(self.velocity_history[virtual_id])

    def pixels_per_frame_to_kmh(self, pixels_per_frame):
        # Convertir a píxeles por segundo
        pixels_per_second = pixels_per_frame * self.fps
        # Convertir a metros por segundo
        meters_per_second = pixels_per_second / self.pixels_per_meter
        # Convertir a km/h
        kmh = meters_per_second * 3.6
        return kmh

    def pixels_to_meters(self, pixels):
        return pixels / self.pixels_per_meter

    def kmh_to_pixels_per_frame(self, kmh):
        meters_per_second = kmh / 3.6
        pixels_per_second = meters_per_second * self.pixels_per_meter
        pixels_per_frame = pixels_per_second / self.fps
        return pixels_per_frame
    
    def is_velocity_realistic(self, velocity_px_per_frame):
        return velocity_px_per_frame <= self.max_realistic_velocity_px

    def update_velocities(self, player_tracker, player_detections, frame_idx):
        # Resetear velocidades actuales para jugadores que no están en este frame
        for virtual_id in list(self.current_velocities.keys()):
            if virtual_id not in player_detections:
                self.current_velocities[virtual_id] = 0.0
                
        for virtual_id, bbox in player_detections.items():
            # Obtener centro del bbox como posición actual
            x1, y1, x2, y2 = bbox
            current_pos = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            # Calcular velocidad actual
            raw_velocity = self.calculate_current_velocity(player_tracker, virtual_id, current_pos)
            
            # Suavizar velocidad
            smoothed_velocity = self.smooth_velocity(virtual_id, raw_velocity)
            
            # Solo considerar movimiento si supera el umbral
            if smoothed_velocity > self.movement_threshold:
                # Actualizar velocidad actual
                self.current_velocities[virtual_id] = smoothed_velocity
                
                # Acumular distancia real (píxeles recorridos)
                self.total_distances[virtual_id] += smoothed_velocity
                
                # Guardar velocidad para estadísticas
                self.velocity_stats[virtual_id].append(smoothed_velocity)
                
                # Actualizar velocidad máxima
                if smoothed_velocity > self.max_velocities[virtual_id]:
                    self.max_velocities[virtual_id] = smoothed_velocity
            else:
                # Si no hay movimiento significativo, velocidad actual es 0
                self.current_velocities[virtual_id] = 0.0

    def get_velocity_stats(self, virtual_id):
        """Obtiene estadísticas de velocidad y distancia de un jugador"""
        current_vel = self.current_velocities.get(virtual_id, 0.0)
        max_vel = self.max_velocities.get(virtual_id, 0.0)
        total_distance_pixels = self.total_distances.get(virtual_id, 0.0)
        total_distance_meters = self.pixels_to_meters(total_distance_pixels)
        
        return {
            'current_velocity_px_per_frame': current_vel,
            'max_velocity': max_vel,
            'current_velocity_kmh': self.pixels_per_frame_to_kmh(current_vel),
            'max_velocity_kmh': self.pixels_per_frame_to_kmh(max_vel),
            'total_distance_meters': total_distance_meters,
            'player_id': virtual_id
        }

    def draw_velocity_panel_simplified(self, frame, player_detections):
        frame_height, frame_width = frame.shape[:2]
        
        panel_x = frame_width - 280  # Panel más compacto
        panel_y = 420  # Más abajo, debajo del panel de la pelota
        
        # Fondo del panel
        panel_width = 270
        panel_height = 30 + len(player_detections) * 60  # Más compacto
        
        # Fondo semitransparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x - 5, panel_y - 25), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Título del panel
        cv2.putText(frame, "VELOCIDADES", (panel_x, panel_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset = panel_y + 30
        
        # Información de cada jugador
        for virtual_id in sorted(player_detections.keys()):
            velocity_stats = self.get_velocity_stats(virtual_id)
            
            if velocity_stats:
                color = self.trajectory_colors.get(virtual_id, (255, 255, 255))
                
                # Nombre del jugador
                player_text = f"Jugador {virtual_id}:"
                cv2.putText(frame, player_text, (panel_x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 18
                
                # Velocidad actual en km/h
                current_vel_kmh = velocity_stats['current_velocity_kmh']
                vel_text = f"  Velocidad: {current_vel_kmh:.1f} km/h"
                cv2.putText(frame, vel_text, (panel_x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 15
                
                # Distancia recorrida en metros
                total_distance_m = velocity_stats['total_distance_meters']
                distance_text = f"  Distancia: {total_distance_m:.1f} m"
                cv2.putText(frame, distance_text, (panel_x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 25

    def print_final_statistics_simplified(self):
        """Imprime estadísticas finales simplificadas"""
        print("\n📊 ESTADÍSTICAS FINALES DE VELOCIDAD:")
        print("="*50)
        
        for player_id in sorted(self.current_velocities.keys()):
            max_vel = self.max_velocities.get(player_id, 0.0)
            max_vel_kmh = self.pixels_per_frame_to_kmh(max_vel)
            total_distance_m = self.pixels_to_meters(self.total_distances.get(player_id, 0.0))
            
            print(f"Jugador {player_id}:")
            print(f"  - Velocidad máxima: {max_vel_kmh:.2f} km/h")
            print(f"  - Distancia total recorrida: {total_distance_m:.2f} metros")
            print(f"  - Frames con movimiento: {len(self.velocity_stats[player_id])}")
            print()
