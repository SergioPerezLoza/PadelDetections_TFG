import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import (read_video, save_video)
from trackers import PlayerTracker, RacketTracker
from trackers.tracknet_ball_tracker import PadelBallTracker
from court_line_detector import CourtLineDetector
from trackers.court_tracker import CourtTracker
from stats import VelocityAnalyzer, BallVelocityAnalyzer  # ACTUALIZADO: Importar BallVelocityAnalyzer también
import cv2
from mini_court import MinimapVisualizer

def main():
    # Donde leo y donde guardo el video
    input_video_path = r"/home/jcperez/Sergio/TFG/src/data/used_images/partido2.mp4"
    output_video_path = r"/home/jcperez/Sergio/TFG/src/data/annotations/output_video2.avi"
    
    # Donde tengo el modelo de la pista (Court Line Detection model)
    court_model_path = r"/home/jcperez/Sergio/TFG/src/models/keypoints_model.pth"
    
    # Modelo de detección de pelota específico para pádel
    padel_ball_model_path = r"/home/jcperez/Sergio/TFG/src/weights/model_best.pt"
    
    # Extraer el nombre base del video para crear pkl únicos
    video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    print(f"Procesando video: {video_name}")
    
    # Crear directorio para los stubs si no existe
    stub_dir = "tracker_stubs"
    os.makedirs(stub_dir, exist_ok=True)
    
    # Crear nombres únicos para los archivos pkl basados en el nombre del video
    court_stub_path = f"{stub_dir}/{video_name}_court_detections.pkl"
    player_stub_path = f"{stub_dir}/{video_name}_player_detections.pkl"
    ball_stub_path = f"{stub_dir}/{video_name}_padel_ball_detections.pkl"
    racket_stub_path = f"{stub_dir}/{video_name}_racket_detections.pkl"
    
    # Todos los frames del video
    video_frames = read_video(input_video_path)
    print(f"Total de frames leídos: {len(video_frames)}")
    
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"FPS del video: {fps}")
    
    # Inicializamos CourtTracker
    court_tracker = CourtTracker()
    
    # Detectar la cancha usando el CourtTracker con pkl único
    homography_matrix, court_detections = court_tracker.detect_frames(
        video_frames, 
        read_from_stub=True, 
        stub_path=court_stub_path
    )
    
    # Obtener keypoints de la pista (primer frame)
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])
    
    # Detectar jugadores con yolov8
    player_tracker = PlayerTracker(model_path='yolov8n')

    # Detectar pelota con modelo específico de pádel
    padel_ball_tracker = PadelBallTracker(
        model_path=padel_ball_model_path,
        device='cuda',
        force_cuda=False
    )
    
    # Detectar raqueta
    racket_tracker = RacketTracker(model_path='/home/jcperez/Sergio/TFG/src/models/best.pt')
    
    # Inicializar analizadores de velocidad
    velocity_analyzer = VelocityAnalyzer(
        fps=fps, 
        pixels_per_meter=35,
        movement_threshold=3.0
    )

    ball_velocity_analyzer = BallVelocityAnalyzer(
        fps=fps,
        pixels_per_meter=35,
        movement_threshold=1.0
    )
    
    # Detectar jugadores, pelota y raqueta (usando stubs)
    player_detections = player_tracker.detect_frames(
        video_frames, 
        read_from_stub=False, 
        stub_path=player_stub_path
    )
    
    ball_detections = padel_ball_tracker.detect_frames(
        video_frames, 
        read_from_stub=True, 
        stub_path=ball_stub_path
    )
       
    racket_detections = racket_tracker.detect_frames(
        video_frames, 
        read_from_stub=True, 
        stub_path=racket_stub_path
    ) 
    
    # Filtrar jugadores más cercanos a los keypoints de la pista
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)
      
    # # Inicializar visualizador del minimapa
    # minimap_visualizer = MinimapVisualizer(
    #     court_width=150,
    #     court_height=300,
    #     history_length=30
    # )
    # Inicializar visualizador del minimapa
    minimap_visualizer = MinimapVisualizer(
        court_width=150,
        court_height=300,
        history_length=30
        # player_detections=player_detections,  # descomentar si se usa jcmini_court
        # ball_detections=ball_detections,
        # homography_matrix=homography_matrix
    )
    
    # Dibujar bounding boxes iniciales
    output_video_frames = player_tracker.draw_bboxes(video_frames.copy(), player_detections)
    output_video_frames = padel_ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    output_video_frames = racket_tracker.draw_bboxes(output_video_frames, racket_detections)
    output_video_frames = court_tracker.draw_bboxes(output_video_frames, court_detections, homography_matrix)
    
    print("Procesando frames y añadiendo minimapa...")
    
    # Procesar cada frame para añadir minimapa y calcular velocidades
    for i, frame in enumerate(output_video_frames):
        
        # Actualizar velocidades de jugadores
        if i < len(player_detections):
            velocity_analyzer.update_velocities(player_tracker, player_detections[i], i)
        
        # Actualizar velocidad de la pelota
        ball_detection_frame = ball_detections[i] if i < len(ball_detections) else []
        ball_velocity_analyzer.update_velocity(ball_detection_frame, i)
        
        # Actualizar posiciones en el minimapa
        minimap_visualizer.update_positions(
            player_detections, 
            ball_detections, 
            homography_matrix, 
            frame_idx=i,
            frame_shape=frame.shape[:2]
        )
        
        # Crear minimapa para este frame
        minimap = minimap_visualizer.create_minimap(frame_idx=i)
        
        # Añadir minimapa al frame
        output_video_frames[i] = minimap_visualizer.add_minimap_to_frame(
            frame, 
            minimap, 
            scale=0.8
        )
        
        # Dibujar panel simplificado de velocidades de jugadores
        if i < len(player_detections):
            velocity_analyzer.draw_velocity_panel_simplified(
                output_video_frames[i], 
                player_detections[i]
            )
        
        # Dibujar panel de velocidad de la pelota debajo
        ball_velocity_analyzer.draw_ball_velocity_panel(
            output_video_frames[i], 
            ball_detection_frame
        )
        
        # Dibujar número de frame
        cv2.putText(output_video_frames[i], f"Frame: {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Guardar video final
    print("Guardando video...")
    save_video(output_video_frames, output_video_path)
    
    # Mostrar estadísticas finales
    velocity_analyzer.print_final_statistics_simplified()
    ball_velocity_analyzer.print_final_statistics()
    
    print(f"✅ Video procesado y guardado en: {output_video_path}")
    print(f"📁 Archivos pkl guardados:")
    print(f"  - {court_stub_path}")
    print(f"  - {player_stub_path}")
    print(f"  - {ball_stub_path}")
    print(f"  - {racket_stub_path}")

if __name__ == '__main__':
    main()

