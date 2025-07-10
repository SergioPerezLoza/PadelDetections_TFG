import cv2
import numpy as np
from collections import defaultdict, deque

class MinimapVisualizer:
    def __init__(self, court_width=150, court_height=300, history_length=30):
        self.court_width = court_width
        self.court_height = court_height
        self.history_length = history_length
        
        # Historial de posiciones
        self.player_positions_history = defaultdict(lambda: deque(maxlen=history_length))
        self.ball_positions_history = deque(maxlen=history_length)
        
        # Colores
        self.player_colors = {
            1: (0, 0, 255), 2: (0, 255, 0), 3: (255, 0, 0), 
            4: (255, 255, 0)
        }
        self.ball_color = (255, 255, 255)
        
        # Dimensiones de imagen
        self.image_width = 1920
        self.image_height = 1080
        
        # MEJORA: Usar coordenadas fijas y robustas para la cancha de pádel
        # Basado en las dimensiones reales del campo (20x10 metros)
        self.court_bounds = {
            'min_x': -12.0, 'max_x': 12.0,  # Rango más amplio para mejor estabilidad
            'min_y': -7.0, 'max_y': 7.0
        }
        
        # Sistema de calibración mejorado
        self.calibration_data = []
        self.calibration_frames = 50  # Reducido para convergencia más rápida
        self.stats_stabilized = False
        
    def world_to_minimap(self, world_x, world_y):
        """Convierte coordenadas normalizadas del mundo (0-1) a coordenadas del minimapa"""
        world_x = max(0, min(1, world_x))
        world_y = max(0, min(1, world_y))
        minimap_x = int(world_y * (self.court_width - 1))
       # minimap_y = int((1.0 - world_x) * (self.court_height - 1))
        # Invertimos el eje y para corregir la imagen especular vertical
        minimap_y = int(world_x * (self.court_height - 1))
        return max(0, min(self.court_width - 1, minimap_x)), max(0, min(self.court_height - 1, minimap_y))
    
        # minimap_x = int(world_x * (self.court_width - 1))
        # minimap_y = int((1.0 - world_y) * (self.court_height - 1))
        # #minimap_y = int(world_y * (self.court_height - 1))
        # return max(0, min(self.court_width - 1, minimap_y)), max(0, min(self.court_height - 1, minimap_x))

        #return max(0, min(self.court_width - 1, minimap_x)), max(0, min(self.court_height - 1, minimap_y))      
    
    def validate_homography(self, homography_matrix):
        if homography_matrix is None or np.array_equal(homography_matrix, np.eye(3)):
            return False
        try:
            det = np.linalg.det(homography_matrix)
            return abs(det) > 1e-8
        except:
            return False
    
    def fallback_coordinates(self, image_x, image_y):
        """Coordenadas de respaldo mejoradas cuando la homografía falla"""
        norm_x = image_x / self.image_width
        norm_y = image_y / self.image_height
        
        # Mapeo más consistente para pistas horizontales
        # Considerando que la pista está orientada horizontalmente en el video
        
        # Mapear X (horizontal en video) a Y (vertical en minimapa)
        if norm_x < 0.2:
            world_y = norm_x * 2.5  # Lado izquierdo -> parte superior
        elif norm_x > 0.8:
            world_y = 0.5 + (norm_x - 0.8) * 2.5  # Lado derecho -> parte inferior
        else:
            world_y = 0.5  # Centro
        
        # Mapear Y (vertical en video) a X (horizontal en minimapa)
        if norm_y < 0.3:
            world_x = 0.2 + norm_y * 2.0  # Parte superior -> lado derecho
        elif norm_y > 0.7:
            world_x = 0.8 - (norm_y - 0.7) * 2.0  # Parte inferior -> lado izquierdo
        else:
            world_x = 0.5  # Centro
        
        # Clamping más suave
        world_x = max(0.05, min(0.95, world_x))
        world_y = max(0.05, min(0.95, world_y))
        
        return world_x, world_y
    
    
    def image_to_world_coordinates(self, image_x, image_y, homography_matrix):
        """Convierte coordenadas de imagen a coordenadas del mundo normalizadas [0,1]"""
        if not self.validate_homography(homography_matrix):
            return self.fallback_coordinates(image_x, image_y)
        
        try:
            point = np.array([[[float(image_x), float(image_y)]]], dtype=np.float32)
            inv_homography = np.linalg.inv(homography_matrix)
            world_point = cv2.perspectiveTransform(point, inv_homography)
            
            world_x = float(world_point[0][0][0])
            world_y = float(world_point[0][0][1])
            
            # MEJORA: Recopilar datos de calibración sin actualizar estadísticas inmediatamente
            if not self.stats_stabilized and len(self.calibration_data) < self.calibration_frames:
                self.calibration_data.append((world_x, world_y))
                
                # Cuando tenemos suficientes datos, calcular estadísticas robustas
                if len(self.calibration_data) >= self.calibration_frames:
                    self._calculate_robust_bounds()
                    self.stats_stabilized = True
            
            # Normalizar usando bounds fijos y robustos
            world_x = (world_x - self.court_bounds['min_x']) / (self.court_bounds['max_x'] - self.court_bounds['min_x'])
            world_y = (world_y - self.court_bounds['min_y']) / (self.court_bounds['max_y'] - self.court_bounds['min_y'])
            
            # Clamping suave para evitar cortes bruscos
            world_x = max(0.02, min(0.98, world_x))
            world_y = max(0.02, min(0.98, world_y))
            
            return world_x, world_y
            
        except Exception as e:
            print(f"Error en transformación homográfica: {e}")
            return self.fallback_coordinates(image_x, image_y)
    
    def _calculate_robust_bounds(self):
        """Calcula bounds robustos usando percentiles para evitar outliers"""
        if len(self.calibration_data) < 10:
            return
        
        x_coords = [point[0] for point in self.calibration_data]
        y_coords = [point[1] for point in self.calibration_data]
        
        # Usar percentiles para eliminar outliers
        x_min = np.percentile(x_coords, 5)
        x_max = np.percentile(x_coords, 95)
        y_min = np.percentile(y_coords, 5)
        y_max = np.percentile(y_coords, 95)
        
        # Expandir ligeramente para dar margen
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        self.court_bounds['min_x'] = x_min - x_range * 0.1
        self.court_bounds['max_x'] = x_max + x_range * 0.1
        self.court_bounds['min_y'] = y_min - y_range * 0.1
        self.court_bounds['max_y'] = y_max + y_range * 0.1
        
        print(f"Bounds calculados: X[{self.court_bounds['min_x']:.2f}, {self.court_bounds['max_x']:.2f}] Y[{self.court_bounds['min_y']:.2f}, {self.court_bounds['max_y']:.2f}]")
    
    def update_positions(self, player_detections, ball_detections, homography_matrix, frame_idx, frame_shape=None):
        """Actualiza las posiciones de jugadores y pelota"""
        if frame_shape:
            self.image_width, self.image_height = frame_shape[1], frame_shape[0]
        
        # Actualizar jugadores
        if frame_idx < len(player_detections) and player_detections[frame_idx]:
            for player_id, bbox in player_detections[frame_idx].items():
                if len(bbox) >= 4:
                    foot_x = (bbox[0] + bbox[2]) / 2
                    foot_y = bbox[3]  # Parte inferior del bbox
                    
                    world_x, world_y = self.image_to_world_coordinates(foot_x, foot_y, homography_matrix)
                    
                    # Suavizado mejorado para jugadores
                    if len(self.player_positions_history[player_id]) > 0:
                        last_x, last_y = self.player_positions_history[player_id][-1]
                        distance = np.sqrt((world_x - last_x)**2 + (world_y - last_y)**2)
                        
                        # Filtro adaptativo basado en distancia
                        if distance > 0.2:  # Salto muy grande
                            world_x = last_x * 0.8 + world_x * 0.2
                            world_y = last_y * 0.8 + world_y * 0.2
                        elif distance > 0.05:  # Movimiento normal
                            world_x = last_x * 0.3 + world_x * 0.7
                            world_y = last_y * 0.3 + world_y * 0.7
                        # Si distance <= 0.05, usar coordenadas directas
                    
                    self.player_positions_history[player_id].append((world_x, world_y))
        
        # Actualizar pelota con el mismo sistema mejorado
        if frame_idx < len(ball_detections) and ball_detections[frame_idx]:
            ball_data = ball_detections[frame_idx]
            ball_bbox = None
            
            # Manejar el formato específico: [{'bbox': [x1,y1,x2,y2], 'center': (x,y), 'confidence': float}]
            if isinstance(ball_data, list) and len(ball_data) > 0:
                ball_dict = ball_data[0]  # Tomar el primer elemento de la lista
                if isinstance(ball_dict, dict) and 'bbox' in ball_dict:
                    ball_bbox = ball_dict['bbox']
            elif isinstance(ball_data, dict) and 'bbox' in ball_data:
                ball_bbox = ball_data['bbox']
            
            if ball_bbox and len(ball_bbox) >= 4:
                center_x = (ball_bbox[0] + ball_bbox[2]) / 2
                center_y = (ball_bbox[1] + ball_bbox[3]) / 2
                
                world_x, world_y = self.image_to_world_coordinates(center_x, center_y, homography_matrix)
                
                # Suavizado mejorado de la pelota
                if len(self.ball_positions_history) > 0:
                    last_x, last_y = self.ball_positions_history[-1]
                    distance = np.sqrt((world_x - last_x)**2 + (world_y - last_y)**2)
                    
                    # Filtro adaptativo para pelota
                    if distance > 0.4:  # Salto muy grande
                        world_x = last_x * 0.7 + world_x * 0.3
                        world_y = last_y * 0.7 + world_y * 0.3
                    elif distance > 0.1:  # Movimiento normal
                        world_x = last_x * 0.2 + world_x * 0.8
                        world_y = last_y * 0.2 + world_y * 0.8
                    # Si distance <= 0.1, usar coordenadas directas
                
                self.ball_positions_history.append((world_x, world_y))
                
    def create_minimap(self, frame_idx):
        """Crea el minimapa con la cancha de pádel y elementos del juego"""
        minimap = np.zeros((self.court_height, self.court_width, 3), dtype=np.uint8)
        self._draw_court(minimap)
        
        # Trayectoria de la pelota
        if len(self.ball_positions_history) > 1:
            positions = list(self.ball_positions_history)
            for i in range(len(positions) - 1):
                start_pos = self.world_to_minimap(positions[i][0], positions[i][1])
                end_pos = self.world_to_minimap(positions[i + 1][0], positions[i + 1][1])
                
                intensity = (i + 1) / len(positions)
                color = [int(100 * intensity), int(150 * intensity), int(255 * intensity)]
                cv2.line(minimap, start_pos, end_pos, color, max(1, int(3 * intensity)))
        
        # Jugadores
        for player_id, positions in self.player_positions_history.items():
            if positions:
                color = self.player_colors.get(player_id, (255, 255, 255))
                
                # Trayectoria del jugador
                if len(positions) > 1:
                    points = [self.world_to_minimap(pos[0], pos[1]) for pos in positions]
                    for i in range(len(points) - 1):
                        intensity = (i + 1) / len(points)
                        faded_color = [int(c * intensity * 0.8) for c in color]
                        cv2.line(minimap, points[i], points[i + 1], faded_color, max(1, int(2 * intensity)))
                
                # Posición actual del jugador
                current_pos = positions[-1]
                minimap_x, minimap_y = self.world_to_minimap(current_pos[0], current_pos[1])
                cv2.circle(minimap, (minimap_x, minimap_y), 8, color, -1)
                cv2.circle(minimap, (minimap_x, minimap_y), 10, (255, 255, 255), 2)
                cv2.putText(minimap, f'P{player_id}', (minimap_x - 8, minimap_y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Pelota actual
        if self.ball_positions_history:
            current_ball = self.ball_positions_history[-1]
            minimap_x, minimap_y = self.world_to_minimap(current_ball[0], current_ball[1])
            # Pelota con efecto de sombra
            cv2.circle(minimap, (minimap_x, minimap_y), 8, (0, 165, 255), -1)
            cv2.circle(minimap, (minimap_x, minimap_y), 6, (255, 255, 255), -1)
            cv2.circle(minimap, (minimap_x, minimap_y), 8, (0, 0, 0), 2)
        
        return minimap
    
    def _draw_court(self, minimap):
        """Dibuja una cancha de pádel horizontal"""
        line_color = (255, 255, 255)      # Líneas blancas
        net_color = (200, 200, 200)       # Red gris claro
        court_color = (0, 120, 0)         # Verde césped/superficie
        service_color = (255, 255, 255)   # Líneas de servicio blancas
        
        # Fondo de la cancha
        cv2.rectangle(minimap, (0, 0), (self.court_width, self.court_height), court_color, -1)
        
        # Borde exterior de la cancha
        cv2.rectangle(minimap, (2, 2), (self.court_width-2, self.court_height-2), line_color, 2)
        
        # Red central (línea horizontal)
        center_y = self.court_height // 2
        cv2.line(minimap, (2, center_y), (self.court_width-2, center_y), net_color, 4)
        
        # Líneas de servicio - distancia desde la red hacia los fondos
        # Dividimos cada mitad de la cancha en dos partes iguales
        service_line_distance = self.court_height // 3
        
        # Línea de servicio campo superior (desde la red hacia arriba)
        service_y_top = center_y - service_line_distance
        cv2.line(minimap, (2, service_y_top), (self.court_width-2, service_y_top), service_color, 2)
        
        # Línea de servicio campo inferior (desde la red hacia abajo)
        service_y_bottom = center_y + service_line_distance
        cv2.line(minimap, (2, service_y_bottom), (self.court_width-2, service_y_bottom), service_color, 2)
        
        # Línea central de servicio (forma la "T" invertida)
        center_x = self.court_width // 2
        
        # T invertida en campo superior: desde la red hasta la línea de servicio
        cv2.line(minimap, (center_x, center_y), (center_x, service_y_top), service_color, 2)
        
        # T invertida en campo inferior: desde la red hasta la línea de servicio
        cv2.line(minimap, (center_x, center_y), (center_x, service_y_bottom), service_color, 2)
    
    def add_minimap_to_frame(self, frame, minimap, scale=1.0):
        """Añade el minimapa al frame principal"""
        if scale != 1.0:
            new_width = int(minimap.shape[1] * scale)
            new_height = int(minimap.shape[0] * scale)
            minimap = cv2.resize(minimap, (new_width, new_height))
        
        frame_height, frame_width = frame.shape[:2]
        minimap_height, minimap_width = minimap.shape[:2]
        
        margin = 20
        y_offset = margin
        x_offset = frame_width - minimap_width - margin
        
        if (x_offset + minimap_width <= frame_width and y_offset + minimap_height <= frame_height and
            x_offset >= 0 and y_offset >= 0):
            
            frame[y_offset:y_offset+minimap_height, x_offset:x_offset+minimap_width] = minimap
            cv2.rectangle(frame, (x_offset-2, y_offset-2), 
                         (x_offset+minimap_width+1, y_offset+minimap_height+1), 
                         (255, 255, 255), 2)
        
        return frame
    
    def reset_histories(self):
        """Reinicia los historiales de posiciones"""
        self.player_positions_history.clear()
        self.ball_positions_history.clear()
        self.calibration_data.clear()
        self.stats_stabilized = False
        
        # Resetear bounds a valores iniciales
        self.court_bounds = {
            'min_x': -12.0, 'max_x': 12.0,
            'min_y': -7.0, 'max_y': 7.0
        }

    def reset_calibration(self):
        """Reinicia solo la calibración de coordenadas sin borrar historiales"""
        self.calibration_data.clear()
        self.stats_stabilized = False
        
        # Resetear bounds a valores iniciales
        self.court_bounds = {
            'min_x': -12.0, 'max_x': 12.0,
            'min_y': -7.0, 'max_y': 7.0
        }
    
    def get_diagnostic_info(self):
        """Devuelve información de diagnóstico sobre el estado del minimapa"""
        info = {
            'calibration_status': 'Completada' if self.stats_stabilized else f'En progreso ({len(self.calibration_data)}/{self.calibration_frames})',
            'court_bounds': self.court_bounds,
            'player_count': len(self.player_positions_history),
            'ball_history_length': len(self.ball_positions_history),
            'current_player_positions': {}
        }
        
        # Posiciones actuales de jugadores
        for player_id, history in self.player_positions_history.items():
            if history:
                info['current_player_positions'][player_id] = {
                    'world_coords': history[-1],
                    'minimap_coords': self.world_to_minimap(history[-1][0], history[-1][1])
                }
        
        return info
    
    def print_diagnostic_info(self):
        """Imprime información de diagnóstico"""
        info = self.get_diagnostic_info()
        print(f"\n=== DIAGNÓSTICO DEL MINIMAPA ===")
        print(f"Calibración: {info['calibration_status']}")
        print(f"Bounds de la cancha: {info['court_bounds']}")
        print(f"Jugadores detectados: {info['player_count']}")
        print(f"Historial de pelota: {info['ball_history_length']} frames")
        
        for player_id, pos_info in info['current_player_positions'].items():
            print(f"Jugador {player_id}: World {pos_info['world_coords']} -> Minimap {pos_info['minimap_coords']}")
        print("=" * 35)
