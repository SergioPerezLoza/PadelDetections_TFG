import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import measure_distance

class BallVelocityAnalyzer:
    def __init__(self, fps=30, pixels_per_meter=35, movement_threshold=1.0):
        """
        Inicializa el analizador de velocidad de la pelota.
        
        Args:
            fps (int): Frames por segundo del video
            pixels_per_meter (float): Píxeles por metro para conversión a unidades reales
            movement_threshold (float): Umbral mínimo de movimiento (píxeles) para considerar movimiento
        """
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter
        self.movement_threshold = movement_threshold
        
        # Estadísticas de la pelota
        self.velocity_history = []  # Para suavizado
        self.current_velocity = 0.0
        self.max_velocity = 0.0
        self.last_position = None
        self.velocity_stats = []  # Todas las velocidades registradas
        
        # Límites de velocidad realistas para pelota de pádel
        self.max_realistic_velocity_kmh = 180.0  # Velocidad máxima realista para pelota de pádel
        self.max_realistic_velocity_px = self.kmh_to_pixels_per_frame(self.max_realistic_velocity_kmh)
        
        # Contador de frames sin detección
        self.frames_without_detection = 0
        self.max_frames_without_detection = 5  # Máximo de frames sin detección antes de reset
        
        # Color para la pelota
        self.ball_color = (0, 255, 255)  # Amarillo para la pelota
    
    def calculate_ball_velocity(self, ball_detection):
        if not ball_detection:  # Si no hay detección
            self.frames_without_detection += 1
            if self.frames_without_detection > self.max_frames_without_detection:
                # Reset si llevamos muchos frames sin detección
                self.last_position = None
                self.current_velocity = 0.0
            return 0.0
        
        # Resetear contador de frames sin detección
        self.frames_without_detection = 0
        
        # Obtener centro de la pelota
        detection = ball_detection[0]  # Tomar la primera detección
        current_pos = detection['center']
        
        if self.last_position is None:
            # Primera detección, guardar posición y retornar 0
            self.last_position = current_pos
            return 0.0
        
        # Calcular velocidad en píxeles por frame
        velocity_pixels_per_frame = measure_distance(self.last_position, current_pos)
        
        # Actualizar última posición
        self.last_position = current_pos
        
        return velocity_pixels_per_frame
    
    def smooth_velocity(self, velocity, window_size=5):
        # Filtrar valores extremos antes de añadir al historial
        if not self.is_velocity_realistic(velocity):
            # Si la velocidad es irrealista, usar la última velocidad válida o 0
            if self.velocity_history:
                velocity = self.velocity_history[-1]
            else:
                velocity = 0.0
        
        self.velocity_history.append(velocity)
        
        # Mantener solo los últimos N valores
        if len(self.velocity_history) > window_size:
            self.velocity_history = self.velocity_history[-window_size:]
        
        # Calcular mediana para ser robusto ante outliers
        if len(self.velocity_history) >= 3:
            sorted_velocities = sorted(self.velocity_history)
            median_idx = len(sorted_velocities) // 2
            return sorted_velocities[median_idx]
        else:
            # Si hay muy pocos datos, usar promedio
            return sum(self.velocity_history) / len(self.velocity_history)
    
    def pixels_per_frame_to_kmh(self, pixels_per_frame):
        """Convierte píxeles por frame a km/h"""
        # Convertir a píxeles por segundo
        pixels_per_second = pixels_per_frame * self.fps
        # Convertir a metros por segundo
        meters_per_second = pixels_per_second / self.pixels_per_meter
        # Convertir a km/h
        kmh = meters_per_second * 3.6
        return kmh
    
    def kmh_to_pixels_per_frame(self, kmh):
        """Convierte km/h a píxeles por frame"""
        meters_per_second = kmh / 3.6
        pixels_per_second = meters_per_second * self.pixels_per_meter
        pixels_per_frame = pixels_per_second / self.fps
        return pixels_per_frame
    
    def is_velocity_realistic(self, velocity_px_per_frame):
        return velocity_px_per_frame <= self.max_realistic_velocity_px
    
    def update_velocity(self, ball_detection, frame_idx):
        # Calcular velocidad actual
        raw_velocity = self.calculate_ball_velocity(ball_detection)
        
        # Suavizar velocidad
        smoothed_velocity = self.smooth_velocity(raw_velocity)
        
        # Solo considerar movimiento si supera el umbral
        if smoothed_velocity > self.movement_threshold:
            # Actualizar velocidad actual
            self.current_velocity = smoothed_velocity
            
            # Guardar velocidad para estadísticas
            self.velocity_stats.append(smoothed_velocity)
            
            # Actualizar velocidad máxima
            if smoothed_velocity > self.max_velocity:
                self.max_velocity = smoothed_velocity
        else:
            # Si no hay movimiento significativo, velocidad actual es 0
            self.current_velocity = 0.0
    
    def get_velocity_stats(self):
        return {
            'current_velocity_px_per_frame': self.current_velocity,
            'max_velocity': self.max_velocity,
            'current_velocity_kmh': self.pixels_per_frame_to_kmh(self.current_velocity),
            'max_velocity_kmh': self.pixels_per_frame_to_kmh(self.max_velocity),
            'avg_velocity_kmh': self.pixels_per_frame_to_kmh(
                sum(self.velocity_stats) / len(self.velocity_stats) if self.velocity_stats else 0
            ),
            'total_measurements': len(self.velocity_stats)
        }
    
    def draw_ball_velocity_panel(self, frame, ball_detection):
        frame_height, frame_width = frame.shape[:2]
        
        # Panel más pequeño en el lado derecho, debajo del minimapa
        panel_x = frame_width - 200  # Panel más estrecho
        panel_y = 350  # Debajo del minimapa (que mide aproximadamente 240px de alto)
        
        # Dimensiones del panel (más pequeño y compacto)
        panel_width = 190  # Más estrecho
        panel_height = 45   # Más bajo
        
        # Fondo semitransparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x - 5, panel_y - 25), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Título del panel (más pequeño)
        cv2.putText(frame, "PELOTA", (panel_x, panel_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.ball_color, 2)
        
        y_offset = panel_y + 25  # Menos espacio vertical
        
        # Obtener velocidad actual
        velocity_stats = self.get_velocity_stats()
        current_vel_kmh = velocity_stats['current_velocity_kmh']
        
        # Estado y velocidad más compacto
        if ball_detection:
            vel_text = f"Velocidad: {current_vel_kmh:.1f} km/h"
            text_color = (0, 255, 0)  # Verde
        else:
            vel_text = "No detectada"
            text_color = (0, 0, 255)  # Rojo
        
        cv2.putText(frame, vel_text, (panel_x, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
    
    def print_final_statistics(self):
        """Imprime estadísticas finales de la pelota"""
        print("\n🎾 ESTADÍSTICAS FINALES DE LA PELOTA:")
        print("="*50)
        
        velocity_stats = self.get_velocity_stats()
        
        print(f"Velocidad máxima: {velocity_stats['max_velocity_kmh']:.2f} km/h")
        print(f"Velocidad promedio: {velocity_stats['avg_velocity_kmh']:.2f} km/h")
        print(f"Total de mediciones: {velocity_stats['total_measurements']}")
        
        print()