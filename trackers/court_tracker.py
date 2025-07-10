import cv2
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from utils import import_court, court_borders, get_intersection_1, order_points, homography_scorer

class CourtTracker:
    def __init__(self):
        self.court_reference = import_court()
        self.average_mask = None

    def image_preprocessing(self, frame, k=2, plot=False):
        # Image preprocessing with Kmeans (K = 2)
        Z = frame.reshape((-1, 3))
        Z = np.float32(Z)

        # Define criteria, number of clusters(K) and apply kmeans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = k
        _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Convert back into uint8, and reshape to the original image
        center = np.uint8(center)
        res = center[label.flatten()]
        segmented_image = res.reshape((frame.shape))

        # Get the label for the pixel in the middle (court label)
        middle_pixel_coords = (frame.shape[0] * 2 // 3, frame.shape[1] // 2)
        middle_pixel_index = middle_pixel_coords[0] * frame.shape[1] + middle_pixel_coords[1]
        court_label = label[middle_pixel_index]

        # Create a mask for the court
        mask = np.abs((label == court_label).astype(np.uint8))
        mask = mask.reshape((frame.shape[0], frame.shape[1]))
        cv2.imwrite('mask_no_pre.png', mask*255)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        mask = cv2.blur(mask, (30, 30))
        cv2.circle(mask, (middle_pixel_coords[1], middle_pixel_coords[0]), radius=3, color=(255, 0, 0), thickness=2)

        cv2.imwrite('mask_after_blur.png', mask*255)

        return mask
    
    def compute_average_mask(self, frames, N=5):
        selected_indices = random.sample(range(len(frames)), N)
        mask_sum = None
        for i in selected_indices:
            frame = frames[i]
            mask = self.image_preprocessing(frame, plot=False)
            if mask_sum is None:
                mask_sum = np.zeros_like(mask, dtype=np.float32)
            mask_sum += mask
        threshold = 2* N // 3
        average_mask = (mask_sum > threshold).astype(np.uint8)
        #average_mask = (mask_sum / N).astype(np.uint8)
        self.average_mask = average_mask
        cv2.imwrite('mask_blur.png', average_mask*255)
    
    def contour_detection(self, mask, frame, plot=False):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        cnt = contours[0]
        if plot:    
            frame_copy = frame.copy()
            frame_copy = cv2.drawContours(frame_copy, [cnt], -1, (0, 255, 0), 5)
            cv2.imshow('contours', frame_copy)
            cv2.waitKey(0)

        # Extract coordinates of the contour
        contour_points = cnt[:, 0, :]   
        return contour_points

    def contour_detectionjc(self, mask, frame, plot=True):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        print(len(contours), "contornos encontrados")
        for i, contour in enumerate(contours):
          num_points = len(contour)
          print(f"Contorno {i}: {num_points} puntos")
        if len(contours) > 1:
            inner_contours = sorted(contours[0:], key=cv2.contourArea, reverse=True)
            cnt_inner = inner_contours[0]
            if plot:   
                print("ntro") 
                frame_copy = frame.copy()
                #frame_copy = cv2.drawContours(frame_copy, [inner_contours[-1]], -1, (0, 255, 0), 5)
                frame_copy = cv2.drawContours(frame_copy, inner_contours, -1, (0, 255, 0), 5)
                cv2.imwrite('contours.jpg', frame_copy)
                # cv2.imshow('contours', frame_copy)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
            
            # Extract coordinates of the contour
            contour_points = cnt_inner[:, 0, :]   
            for point in contour_points:
                x, y = point
                cv2.circle(frame, (x, y), radius=2, color=(0, 0, 255), thickness=-1)
            cv2.imwrite("contour_points.jpg", frame)
            return contour_points
        elif len(contours) > 0:
            # Si no hay contornos internos, usar el primero disponible
            cnt = contours[0]
            contour_points = cnt[:, 0, :]
            return contour_points
        else:
            # Si no hay contornos, devolver una matriz vacía
            return np.array([])

    def process_court_contour(self, contour_points, frame, ref_borders, court_reference, plot=False):
        if contour_points.size == 0:
            print("No se detectaron contornos válidos para la cancha")
            # Devolver una matriz de homografía identidad y una detección vacía 
            return np.eye(3), np.zeros_like(frame)
            
        # Find the extreme points to approximate the four sides
        top_points = contour_points[contour_points[:, 1] < frame.shape[0] // 2.7]
        bottom_points = contour_points[contour_points[:, 1] > frame.shape[0] // 1.3]
        
        # Si no hay suficientes puntos, manejar el error
        if len(top_points) == 0 or len(bottom_points) == 0:
            print("No hay suficientes puntos para detectar la cancha")
            return np.eye(3), np.zeros_like(frame)
        
        # Find the extreme points in x from the top and bottom points
        extreme_left_point = bottom_points[np.argmin(bottom_points[:, 0])]
        extreme_right_point = bottom_points[np.argmax(bottom_points[:, 0])]
        extreme_left_point_1 = top_points[np.argmin(top_points[:, 0])]
        extreme_right_point_1 = top_points[np.argmax(top_points[:, 0])]
        cv2.circle(frame, extreme_left_point, radius=2, color=(0, 0, 255), thickness=-1)
        cv2.circle(frame, extreme_right_point, radius=2, color=(0, 0, 255), thickness=-1)
        cv2.circle(frame, extreme_left_point_1, radius=2, color=(0, 0, 255), thickness=-1)
        cv2.circle(frame, extreme_right_point_1, radius=2, color=(0, 0, 255), thickness=-1)

        cv2.imwrite("corners.jpg", frame)
        
        delta_x = 200  # Number of pixels to extend

        # Function to calculate slope
        def calculate_slope(p1, p2):
            if p2[0] - p1[0] == 0:  # Evitar división por cero
                return float('inf')
            return (p2[1] - p1[1]) / (p2[0] - p1[0])

        # Function to extend a line
        def extend_line(p1, p2, delta_x):
            slope = calculate_slope(p1, p2)
            if slope == float('inf'):
                extended_p1 = (p1[0], p1[1] - delta_x)
                extended_p2 = (p2[0], p2[1] + delta_x)
            else:
                extended_p1 = (p1[0] - delta_x, int(p1[1] - slope * delta_x))
                extended_p2 = (p2[0] + delta_x, int(p2[1] + slope * delta_x))
            return extended_p1, extended_p2

        # Draw the lines on the result image
        extended_right_p1, extended_right_p2 = extend_line(extreme_right_point, extreme_right_point_1, -delta_x)
        extended_left_p1, extended_left_p2 = extend_line(extreme_left_point, extreme_left_point_1, delta_x)

        i1 = get_intersection_1(extended_right_p1, extended_right_p2, (0, int(np.min(top_points[:, 1]))), (frame.shape[1], int(np.min(top_points[:, 1]))))
        i2 = get_intersection_1(extended_right_p1, extended_right_p2, (0, int(np.max(bottom_points[:, 1]))), (frame.shape[1], int(np.max(bottom_points[:, 1]))))
        i3 = get_intersection_1(extended_left_p1, extended_left_p2, (0, int(np.min(top_points[:, 1]))), (frame.shape[1], int(np.min(top_points[:, 1]))))
        i4 = get_intersection_1(extended_left_p1, extended_left_p2, (0, int(np.max(bottom_points[:, 1]))), (frame.shape[1], int(np.max(bottom_points[:, 1]))))

        intersections = np.array([i1, i2, i3, i4])
        
        # Verificar que no hay puntos en el infinito
        if np.isinf(intersections).any():
            print("Error: Puntos de intersección en el infinito. Usando matriz de identidad.")
            return np.eye(3), np.zeros_like(frame)
            
        intersections = order_points(intersections)

        try:
            homography_matrix = cv2.getPerspectiveTransform(ref_borders, intersections)
            court = homography_scorer(homography_matrix, court_reference, frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        except Exception as e:
            print(f"Error al calcular la matriz de homografía: {e}")
            return np.eye(3), np.zeros_like(frame)

        if plot:
            plot_court = np.where(court == 1)

            fig, axes = plt.subplots(1, 2, figsize=(15, 15), sharex=True, sharey=True)
            ax = axes.ravel()

            for i in intersections:
                ax[0].scatter(i[0], i[1], color='r', s=30)
            
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                frame_rgb = frame
                
            ax[0].imshow(frame_rgb)

            ax[1].scatter(plot_court[1], plot_court[0], color='g', s=1)
            ax[1].set_xlim(0, frame.shape[1])
            ax[1].set_ylim(0, frame.shape[0])
            ax[1].invert_yaxis()
            ax[1].imshow(frame_rgb)

            plt.tight_layout()
            plt.savefig('court_detection_result.png')  # Guardar como archivo en vez de mostrar
            plt.close()
        return homography_matrix, court

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        if len(frames) == 0:
            print("Error: No hay frames para procesar")
            return np.eye(3), np.zeros((0, 0, 3))
            
        frame = frames[0]
        ref_borders = court_borders(self.court_reference, court_factor=1)

        if read_from_stub and stub_path is not None:
            try:
                with open(stub_path, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, tuple) and len(data) == 2:
                        homography_matrix, court_detections = data
                        return homography_matrix, court_detections
                    else:
                        court_detections = data
                        return np.eye(3), court_detections
            except FileNotFoundError:
                print(f"Archivo stub no encontrado: {stub_path}. Procediendo a detectar la cancha.")
            except Exception as e:
                print(f"Error al cargar el stub: {e}. Procediendo a detectar la cancha.")

        # Image preprocessing (create average mask)
        self.compute_average_mask(frames)
        homography_matrix, court_detections = self.detect_frame(frame, self.average_mask, self.court_reference, ref_borders=ref_borders)
        
        if stub_path is not None:
            try:
                with open(stub_path, 'wb') as f:
                    pickle.dump((homography_matrix, court_detections), f)
            except Exception as e:
                print(f"Error al guardar el stub: {e}")
        
        return homography_matrix, court_detections

    def detect_frame(self, frame, mask, court_reference, ref_borders):
        if mask is None or frame is None:
            print("Error: Máscara o frame nulos")
            return np.eye(3), np.zeros_like(frame)
            
        # Court detection using Contour detection
        try:
            court_outline = self.contour_detectionjc(mask, frame, plot=True)
            
            # Border detection and 4-point homography transformation
            homography_matrix, court_detections = self.process_court_contour(
                court_outline, frame, ref_borders, court_reference, plot=False
            )
            
            return homography_matrix, court_detections
        except Exception as e:
            print(f"Error en la detección de la cancha: {e}")
            return np.eye(3), np.zeros_like(frame)

    def draw_bboxes(self, video_frames, court_detections, homography_matrix=None, output_path=r"/home/jcperez/Sergio/TFG/src/data/annotations/output_video.avi"):
        output_video_frames = []
        
        # Asegurarse de que court_detections no es None
        if court_detections is None:
            print("Error: court_detections es None")
            return video_frames
            
        # Verificar la forma de court_detections
        try:
            if len(court_detections.shape) > 2:
                plot_court = np.where(np.any(court_detections > 0, axis=-1))
            else:
                plot_court = np.where(court_detections == 1)
                
            print(f"Forma de court_detections: {court_detections.shape}")
        except Exception as e:
            print(f"Error al procesar court_detections: {e}")
            return video_frames
        
        # Configurar el VideoWriter para guardar el video resultante
        height, width, _ = video_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec para AVI
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
        
        for i, frame in enumerate(video_frames):
            frame_copy = frame.copy()  # Crear una copia para no modificar el original
            
            # Dibujar la cancha si hay puntos válidos
            try:
                if len(plot_court) > 0 and all(len(coord) > 0 for coord in plot_court):
                    frame_copy[plot_court] = (0, 255, 0)  # Verde en formato BGR
            except Exception as e:
                print(f"Error al dibujar la cancha en el frame {i}: {e}")
            
            # Agregar el frame al video resultante
            out.write(frame_copy)
            output_video_frames.append(frame_copy)
        
        # Liberar recursos
        out.release()
        print(f"Video guardado en: {output_path}")
        
        return output_video_frames