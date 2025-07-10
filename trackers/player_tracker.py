from ultralytics import YOLO
import cv2
import pickle
import sys
import numpy as np
from collections import defaultdict
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        # Almacenar las últimas posiciones conocidas de cada jugador
        self.last_positions = {}
        # Contador de frames sin detección
        self.frames_without_detection = defaultdict(int)
        # Máximo de frames que toleramos sin detección antes de reasignar
        self.max_frames_lost = 15
        # Historial de posiciones para predecir movimiento
        self.position_history = defaultdict(list)
        self.max_history = 30  # Aumentado para mejor trayectoria
        # Umbral de distancia para matching
        self.distance_threshold = 140  # 140 cuando se cruzan, 105 cuando hay gente detrás
        
        # CLAVE: Mapeo fijo de IDs originales a IDs virtuales estables
        self.id_mapping = {}  # {yolo_id_actual: id_virtual_fijo}
        self.reverse_mapping = {}  # {id_virtual_fijo: yolo_id_actual}
        self.next_virtual_id = 1
        self.virtual_positions = {}  # Posiciones usando IDs virtuales
        
        # Sistema de trayectorias (mantenido para predicción interna, pero no se dibuja)
        self.trajectories = defaultdict(list)  # Trayectorias completas para cada jugador
        self.trajectory_length = 30 
        self.trajectory_colors = {
            1: (0, 0, 255),    # Rojo
            2: (0, 255, 0),    # Verde  
            3: (255, 0, 0),    # Azul
            4: (255, 255, 0)   # Amarillo
        }

    def choose_and_filter_players(self, court_keypoints, player_detections):
        player_detections_first_frame = player_detections[0]
        chosen_player_ids = self.choose_players(court_keypoints, player_detections_first_frame)
        
        # Crear mapeo inicial estable
        self.create_initial_mapping(chosen_player_ids, player_detections_first_frame)
        
        print(f"Mapeo inicial de jugadores: {self.id_mapping}")
        
        filtered_player_detections = []
        for frame_idx, player_dict in enumerate(player_detections):
            # Convertir detecciones actuales a IDs virtuales estables
            virtual_detections = self.convert_to_virtual_ids(player_dict, frame_idx)
            
            # Asegurar que siempre tengamos los 4 jugadores (usar interpolación si es necesario)
            complete_detections = self.ensure_four_players(virtual_detections, frame_idx)
            
            # Actualizar posiciones conocidas
            self.update_virtual_positions(complete_detections, frame_idx)
            
            filtered_player_detections.append(complete_detections)
            
        return filtered_player_detections

    def create_initial_mapping(self, chosen_player_ids, first_frame_detections):
        """Crea el mapeo inicial entre IDs de YOLO e IDs virtuales fijos"""
        # Ordenar por posición para tener un orden consistente
        positions_and_ids = []
        for player_id in chosen_player_ids:
            if player_id in first_frame_detections:
                bbox = first_frame_detections[player_id]
                center = get_center_of_bbox(bbox)
                positions_and_ids.append((center[0], center[1], player_id))
        
        # Ordenar por posición x, luego y para consistencia
        positions_and_ids.sort(key=lambda x: (x[0], x[1]))
        
        # Asignar IDs virtuales fijos
        for i, (x, y, original_id) in enumerate(positions_and_ids):
            virtual_id = i + 1  # IDs virtuales 1, 2, 3, 4
            self.id_mapping[original_id] = virtual_id
            self.reverse_mapping[virtual_id] = original_id
            
            # Inicializar posiciones virtuales
            bbox = first_frame_detections[original_id]
            self.virtual_positions[virtual_id] = get_center_of_bbox(bbox)
            self.position_history[virtual_id] = [get_center_of_bbox(bbox)]

    def convert_to_virtual_ids(self, current_detections, frame_idx):
        """Convierte las detecciones actuales a IDs virtuales estables"""
        virtual_detections = {}
        
        # Paso 1: Asignar detecciones que ya tienen mapeo conocido
        used_detections = set()
        for yolo_id, bbox in current_detections.items():
            if yolo_id in self.id_mapping:
                virtual_id = self.id_mapping[yolo_id]
                virtual_detections[virtual_id] = bbox
                used_detections.add(yolo_id)
                self.frames_without_detection[virtual_id] = 0

        # Paso 2: Para jugadores virtuales que no aparecieron, intentar reasignar
        missing_virtual_ids = set(range(1, 5)) - set(virtual_detections.keys())
        available_detections = {k: v for k, v in current_detections.items() if k not in used_detections}
        
        for virtual_id in missing_virtual_ids:
            self.frames_without_detection[virtual_id] += 1
            
            if available_detections and virtual_id in self.virtual_positions:
                # Buscar la detección más cercana a la última posición conocida
                best_match = self.find_best_match_for_virtual_id(virtual_id, available_detections)
                
                if best_match is not None:
                    # Actualizar mapeo
                    old_yolo_id = self.reverse_mapping.get(virtual_id)
                    if old_yolo_id and old_yolo_id in self.id_mapping:
                        del self.id_mapping[old_yolo_id]
                    
                    self.id_mapping[best_match] = virtual_id
                    self.reverse_mapping[virtual_id] = best_match
                    
                    virtual_detections[virtual_id] = available_detections[best_match]
                    del available_detections[best_match]
                    self.frames_without_detection[virtual_id] = 0
                    
                    if frame_idx % 100 == 0:
                        print(f"Frame {frame_idx}: Reasignado jugador virtual {virtual_id} a YOLO ID {best_match}")

        return virtual_detections

    def find_best_match_for_virtual_id(self, virtual_id, available_detections):
        """Encuentra la mejor detección para un ID virtual usando trayectoria y proximidad"""
        if virtual_id not in self.virtual_positions:
            return None
        
        # Usar predicción de posición y análisis de trayectoria
        predicted_pos = self.predict_virtual_position(virtual_id)
        if predicted_pos is None:
            predicted_pos = self.virtual_positions[virtual_id]
        
        best_score = float('inf')
        best_match = None
        
        for yolo_id, bbox in available_detections.items():
            current_center = get_center_of_bbox(bbox)
            
            # Factor 1: Distancia a posición predicha
            distance = measure_distance(predicted_pos, current_center)
            
            # Factor 2: Consistencia con trayectoria histórica
            trajectory_score = self.calculate_trajectory_consistency(virtual_id, current_center)
            
            # Factor 3: Velocidad razonable (evitar saltos imposibles)
            velocity_score = self.calculate_velocity_plausibility(virtual_id, current_center)
            
            # Combinar scores (pesos ajustables)
            combined_score = (distance * 0.5) + (trajectory_score * 0.3) + (velocity_score * 0.2)
            
            if combined_score < self.distance_threshold and combined_score < best_score:
                best_score = combined_score
                best_match = yolo_id
        
        return best_match
    
    def calculate_trajectory_consistency(self, virtual_id, new_position):
        """Calcula qué tan consistente es una nueva posición con la trayectoria histórica"""
        if virtual_id not in self.trajectories or len(self.trajectories[virtual_id]) < 3:
            return 0  # Sin penalización si no hay historial suficiente
        
        recent_trajectory = self.trajectories[virtual_id][-5:]  # Últimos 5 puntos
        
        # Calcular dirección promedio de movimiento
        directions = []
        for i in range(1, len(recent_trajectory)):
            prev_pos = recent_trajectory[i-1]
            curr_pos = recent_trajectory[i]
            direction = (curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])
            directions.append(direction)
        
        if not directions:
            return 0
        
        # Dirección promedio
        avg_direction = (
            sum(d[0] for d in directions[-3:]) / min(len(directions), 3),
            sum(d[1] for d in directions[-3:]) / min(len(directions), 3)
        )
        
        # Dirección esperada hacia la nueva posición
        last_pos = recent_trajectory[-1]
        expected_direction = (new_position[0] - last_pos[0], new_position[1] - last_pos[1])
        
        # Calcular diferencia angular (simplificado)
        dot_product = (avg_direction[0] * expected_direction[0] + 
                      avg_direction[1] * expected_direction[1])
        
        # Normalizar y convertir a score (menor es mejor)
        consistency_score = max(0, 100 - abs(dot_product) / 10)
        return consistency_score
    
    def calculate_velocity_plausibility(self, virtual_id, new_position):
        """Calcula si la velocidad hacia la nueva posición es plausible"""
        if virtual_id not in self.virtual_positions:
            return 0
        
        last_pos = self.virtual_positions[virtual_id]
        distance = measure_distance(last_pos, new_position)
        
        # Velocidad máxima plausible por frame (ajustar según tu video)
        max_velocity_per_frame = 50  # píxeles por frame
        
        if distance > max_velocity_per_frame:
            return distance  # Penalizar movimientos demasiado rápidos
        
        return 0  # Sin penalización

    def predict_virtual_position(self, virtual_id):
        """Predice la posición de un jugador virtual basado en su historial"""
        if virtual_id not in self.position_history or len(self.position_history[virtual_id]) < 2:
            return self.virtual_positions.get(virtual_id)
        
        positions = self.position_history[virtual_id]
        
        # Calcular velocidad promedio
        recent_positions = positions[-3:]  # Últimas 3 posiciones
        if len(recent_positions) < 2:
            return positions[-1]
        
        velocities = []
        for i in range(1, len(recent_positions)):
            prev_pos = recent_positions[i-1]
            curr_pos = recent_positions[i]
            vel_x = curr_pos[0] - prev_pos[0]
            vel_y = curr_pos[1] - prev_pos[1]
            velocities.append((vel_x, vel_y))
        
        # Velocidad promedio
        avg_vel_x = sum(v[0] for v in velocities) / len(velocities)
        avg_vel_y = sum(v[1] for v in velocities) / len(velocities)
        
        # Predecir siguiente posición
        last_pos = positions[-1]
        predicted_pos = (
            last_pos[0] + avg_vel_x,
            last_pos[1] + avg_vel_y
        )
        
        return predicted_pos

    def ensure_four_players(self, virtual_detections, frame_idx):
        """Asegura que siempre tengamos 4 jugadores, usando interpolación si es necesario"""
        complete_detections = virtual_detections.copy()
        
        for virtual_id in range(1, 5):
            if virtual_id not in complete_detections:
                # Si hemos perdido el jugador por poco tiempo, usar última posición conocida
                if (virtual_id in self.virtual_positions and 
                    self.frames_without_detection[virtual_id] <= self.max_frames_lost):
                    
                    # Crear bbox estimado basado en última posición conocida
                    last_center = self.virtual_positions[virtual_id]
                    estimated_bbox = [
                        last_center[0] - 30,  # x1
                        last_center[1] - 40,  # y1
                        last_center[0] + 30,  # x2
                        last_center[1] + 40   # y2
                    ]
                    complete_detections[virtual_id] = estimated_bbox
                    
                    if frame_idx % 50 == 0:
                        print(f"Frame {frame_idx}: Interpolando jugador {virtual_id}")
        
        return complete_detections

    def update_virtual_positions(self, virtual_detections, frame_idx):
        """Actualiza posiciones virtuales, historial y trayectorias"""
        for virtual_id, bbox in virtual_detections.items():
            center = get_center_of_bbox(bbox)
            self.virtual_positions[virtual_id] = center
            
            # Actualizar historial de posiciones
            if virtual_id not in self.position_history:
                self.position_history[virtual_id] = []
            
            self.position_history[virtual_id].append(center)
            
            # Mantener solo las últimas N posiciones
            if len(self.position_history[virtual_id]) > self.max_history:
                self.position_history[virtual_id].pop(0)
            
            # Actualizar trayectoria completa
            self.trajectories[virtual_id].append(center)
            
            # Mantener solo los últimos puntos de trayectoria para visualización
            if len(self.trajectories[virtual_id]) > self.trajectory_length:
                self.trajectories[virtual_id].pop(0)

    def choose_players(self, court_keypoints, player_dict):
        """Elige los 4 jugadores más cercanos a los keypoints de la cancha"""
        if len(player_dict) <= 4:
            return list(player_dict.keys())
        
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)
            min_distance = float('inf')
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))
        
        distances.sort(key=lambda x: x[1])
        chosen_players = [distances[i][0] for i in range(min(4, len(distances)))]
        return chosen_players
    
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections
        
        for frame_idx, frame in enumerate(frames):
            if frame_idx % 100 == 0:
                print(f"Detectando frame {frame_idx+1}/{len(frames)}")
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections

    def detect_frame(self, frame):
        # Configuración optimizada del tracker
        results = self.model.track(
            frame, 
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.25,     # Confianza un poco más baja para capturar más detecciones
            iou=0.5,
            max_det=10,
            imgsz=640,
        )[0]

        id_name_dict = results.names
        player_dict = {}
        
        if results.boxes is not None:
            for box in results.boxes:
                if box.id is not None:
                    track_id = int(box.id.tolist()[0])
                    result = box.xyxy.tolist()[0]
                    object_cls_id = box.cls.tolist()[0]
                    object_cls_name = id_name_dict[object_cls_id]
                    confidence = float(box.conf.tolist()[0])
                    
                    if object_cls_name == "person" and confidence > 0.25:
                        player_dict[track_id] = result
                        
        return player_dict
  
    def draw_bboxes(self, video_frames, player_detections):
        out_video_frames = []
        # Colores fijos para cada ID virtual
        colors = self.trajectory_colors
        
        for frame_idx, (frame, player_dict) in enumerate(zip(video_frames, player_detections)):
            # Crear una copia del frame para evitar modificaciones no deseadas
            frame_copy = frame.copy()
            
            # Dibujar cada jugador
            for virtual_id in sorted(player_dict.keys()):
                bbox = player_dict[virtual_id]
                color = colors.get(virtual_id, (255, 255, 255))
                
                x1, y1, x2, y2 = bbox
                
                # Etiqueta del jugador
                label = f"P{virtual_id}"
                
                # Calcular posición de texto que no se superponga
                text_x = int(x1)
                text_y = int(y1 - 15) if y1 > 25 else int(y2 + 25)
                
                # Fondo para el texto para mejor legibilidad
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(frame_copy, 
                             (text_x - 5, text_y - text_size[1] - 5),
                             (text_x + text_size[0] + 5, text_y + 5),
                             (0, 0, 0), -1)
                
                cv2.putText(frame_copy, label, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Bounding box
                cv2.rectangle(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Punto central
                center = get_center_of_bbox(bbox)
                cv2.circle(frame_copy, (int(center[0]), int(center[1])), 4, color, -1)
                
                # Solo mostrar "INTERPOLATED" si es necesario
                if (virtual_id in self.frames_without_detection and 
                    self.frames_without_detection[virtual_id] > 0 and
                    self.frames_without_detection[virtual_id] <= 5):  # Solo primeros 5 frames
                    cv2.putText(frame_copy, "INT", (int(x2 - 30), int(y1 + 15)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            out_video_frames.append(frame_copy)
        return out_video_frames