import torch
import torch.nn as nn
import cv2
import numpy as np
from itertools import groupby
from scipy.spatial import distance
from tqdm import tqdm
import pickle
import os

# Definición del modelo BallTrackerNet
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pad=1, stride=1, bias=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x):
        return self.block(x)

class BallTrackerNet(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        self.out_channels = out_channels
        self.conv1 = ConvBlock(in_channels=9, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(in_channels=64, out_channels=128)
        self.conv4 = ConvBlock(in_channels=128, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = ConvBlock(in_channels=128, out_channels=256)
        self.conv6 = ConvBlock(in_channels=256, out_channels=256)
        self.conv7 = ConvBlock(in_channels=256, out_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv8 = ConvBlock(in_channels=256, out_channels=512)
        self.conv9 = ConvBlock(in_channels=512, out_channels=512)
        self.conv10 = ConvBlock(in_channels=512, out_channels=512)
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(in_channels=512, out_channels=256)
        self.conv12 = ConvBlock(in_channels=256, out_channels=256)
        self.conv13 = ConvBlock(in_channels=256, out_channels=256)
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14 = ConvBlock(in_channels=256, out_channels=128)
        self.conv15 = ConvBlock(in_channels=128, out_channels=128)
        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv16 = ConvBlock(in_channels=128, out_channels=64)
        self.conv17 = ConvBlock(in_channels=64, out_channels=64)
        self.conv18 = ConvBlock(in_channels=64, out_channels=self.out_channels)
        self.softmax = nn.Softmax(dim=1)
        self._init_weights()
                  
    def forward(self, x, testing=False): 
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)    
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.ups1(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.ups2(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.ups3(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        out = x.reshape(batch_size, self.out_channels, -1)
        if testing:
            out = self.softmax(out)
        return out                       
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.uniform_(module.weight, -0.05, 0.05)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

# Función de postprocesamiento CORREGIDA
def postprocess(feature_map, original_width, original_height, model_width=640, model_height=360, debug=False):
    """
    Postprocesa el mapa de características y escala correctamente las coordenadas
    """
    feature_map *= 255
    feature_map = feature_map.reshape((model_height, model_width))
    feature_map = feature_map.astype(np.uint8)
    
    if debug:
        print(f"Feature map shape: {feature_map.shape}")
        print(f"Feature map min/max: {feature_map.min()}/{feature_map.max()}")
    
    ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, 
                               param1=50, param2=2, minRadius=2, maxRadius=7)
    
    x, y = None, None
    if circles is not None:
        if len(circles) == 1:
            # Coordenadas en el espacio del modelo (640x360)
            x_model = circles[0][0][0]
            y_model = circles[0][0][1]
            
            # Escalar correctamente al tamaño original
            scale_x = original_width / model_width
            scale_y = original_height / model_height
            
            x = x_model * scale_x
            y = y_model * scale_y
            
            if debug:
                print(f"Model coords: ({x_model:.1f}, {y_model:.1f})")
                print(f"Scale factors: ({scale_x:.3f}, {scale_y:.3f})")
                print(f"Final coords: ({x:.1f}, {y:.1f})")
    
    return x, y

class PadelBallTracker:
    def __init__(self, model_path, device='cuda', force_cuda=True, debug=False):
        self.model_path = model_path
        self.debug = debug
        
        # Configuración de dispositivo
        if force_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA no está disponible pero se requiere force_cuda=True")
            self.device = device
            print(f"Usando CUDA: {self.device}")
        else:
            self.device = device if torch.cuda.is_available() else 'cpu'
            print(f"Usando dispositivo: {self.device}")
            
        self.model = self._load_model()
        
    def _load_model(self):
        """Cargar el modelo entrenado"""
        model = BallTrackerNet()
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        """
        Detectar la pelota en todos los frames del video
        """
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections
        
        print("Detectando pelota en frames...")
        ball_track, dists = self._infer_model(frames)
        ball_track = self._remove_outliers(ball_track, dists)
        
        # Realizar interpolación si es necesario
        ball_track = self._interpolate_track(ball_track)
        
        # Convertir a formato compatible con el resto del código
        ball_detections = self._convert_to_detections_format(ball_track)
        
        if stub_path:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
        
        return ball_detections
    
    def _infer_model(self, frames):
        """Ejecutar el modelo en los frames consecutivos"""
        model_height = 360
        model_width = 640
        dists = [-1] * 2
        ball_track = [(None, None)] * 2
        
        # Obtener dimensiones del video original
        if len(frames) > 0:
            original_height, original_width = frames[0].shape[:2]
            if self.debug:
                print(f"Video original: {original_width}x{original_height}")
                print(f"Modelo: {model_width}x{model_height}")
        
        for num in tqdm(range(2, len(frames)), desc="Procesando frames"):
            # Redimensionar manteniendo aspect ratio si es necesario
            img = cv2.resize(frames[num], (model_width, model_height))
            img_prev = cv2.resize(frames[num-1], (model_width, model_height))
            img_preprev = cv2.resize(frames[num-2], (model_width, model_height))
            
            # Concatenar los 3 frames
            imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
            imgs = imgs.astype(np.float32) / 255.0
            imgs = np.rollaxis(imgs, 2, 0)
            inp = np.expand_dims(imgs, axis=0)

            with torch.no_grad():
                out = self.model(torch.from_numpy(inp).float().to(self.device))
                output = out.argmax(dim=1).detach().cpu().numpy()
                
                # Usar la función de postprocesamiento corregida
                original_height, original_width = frames[num].shape[:2]
                x_pred, y_pred = postprocess(
                    output, 
                    original_width, 
                    original_height, 
                    model_width, 
                    model_height,
                    debug=(self.debug and num == 2)  # Debug solo en el primer frame
                )
                
                ball_track.append((x_pred, y_pred))

            # Calcular distancia para detección de outliers
            if ball_track[-1][0] and ball_track[-2][0]:
                dist = distance.euclidean(ball_track[-1], ball_track[-2])
            else:
                dist = -1
            dists.append(dist)
            
        return ball_track, dists
    
    def _remove_outliers(self, ball_track, dists, max_dist=100):
        """Eliminar outliers de las predicciones del modelo"""
        outliers = list(np.where(np.array(dists) > max_dist)[0])
        for i in outliers:
            if i + 1 < len(dists) and (dists[i+1] > max_dist or dists[i+1] == -1):
                ball_track[i] = (None, None)
            elif i > 0 and dists[i-1] == -1:
                ball_track[i-1] = (None, None)
        return ball_track
    
    def _split_track(self, ball_track, max_gap=4, max_dist_gap=80, min_track=5):
        """Dividir el track de la pelota en subtracks para interpolación"""
        list_det = [0 if x[0] else 1 for x in ball_track]
        groups = [(k, sum(1 for _ in g)) for k, g in groupby(list_det)]

        cursor = 0
        min_value = 0
        result = []
        for i, (k, l) in enumerate(groups):
            if (k == 1) & (i > 0) & (i < len(groups) - 1):
                if cursor > 0 and cursor + l < len(ball_track):
                    dist = distance.euclidean(ball_track[cursor-1], ball_track[cursor+l])
                    if (l >= max_gap) | (dist/l > max_dist_gap):
                        if cursor - min_value > min_track:
                            result.append([min_value, cursor])
                            min_value = cursor + l - 1
            cursor += l
        if len(list_det) - min_value > min_track:
            result.append([min_value, len(list_det)])
        return result
    
    def _interpolation(self, coords):
        """Interpolación de la pelota en un subtrack"""
        def nan_helper(y):
            return np.isnan(y), lambda z: z.nonzero()[0]

        x = np.array([x[0] if x[0] is not None else np.nan for x in coords])
        y = np.array([x[1] if x[1] is not None else np.nan for x in coords])

        nons, yy = nan_helper(x)
        if len(yy(~nons)) > 0:
            x[nons] = np.interp(yy(nons), yy(~nons), x[~nons])
        nans, xx = nan_helper(y)
        if len(xx(~nans)) > 0:
            y[nans] = np.interp(xx(nans), xx(~nans), y[~nans])

        track = [*zip(x, y)]
        return track
    
    def _interpolate_track(self, ball_track):
        """Realizar interpolación completa del track"""
        subtracks = self._split_track(ball_track)
        for r in subtracks:
            ball_subtrack = ball_track[r[0]:r[1]]
            ball_subtrack = self._interpolation(ball_subtrack)
            ball_track[r[0]:r[1]] = ball_subtrack
        return ball_track
    
    def _convert_to_detections_format(self, ball_track):
        """Convertir el track a formato de detecciones compatible"""
        ball_detections = []
        for frame_num, (x, y) in enumerate(ball_track):
            if x is not None and y is not None and not (np.isnan(x) or np.isnan(y)):
                # Crear bbox alrededor del punto detectado
                bbox_size = 15  # Tamaño del bbox alrededor de la pelota
                detection = {
                    'bbox': [
                        max(0, int(x - bbox_size)), 
                        max(0, int(y - bbox_size)), 
                        int(x + bbox_size), 
                        int(y + bbox_size)
                    ],
                    'center': (int(x), int(y)),
                    'confidence': 1.0
                }
                ball_detections.append([detection])
            else:
                ball_detections.append([])  # Frame sin detección
        return ball_detections
    
    def draw_bboxes(self, video_frames, ball_detections, trace=7):
        """Dibujar las detecciones de pelota en los frames con trace mejorado"""
        output_frames = video_frames.copy()
        
        for frame_num in range(len(output_frames)):
            frame = output_frames[frame_num]
            
            # Dibujar trace de la pelota (historial de posiciones)
            trace_points = []
            for i in range(trace):
                trace_frame = frame_num - i
                if trace_frame >= 0 and trace_frame < len(ball_detections):
                    if ball_detections[trace_frame]:  # Si hay detección en ese frame
                        detection = ball_detections[trace_frame][0]
                        center = detection['center']
                        trace_points.append(center)
                    else:
                        break  # Si no hay detección, parar el trace
            
            # Dibujar línea de trace si hay suficientes puntos
            if len(trace_points) > 1:
                for i in range(len(trace_points) - 1):
                    alpha = max(0.3, 1.0 - (i / trace))  # Transparencia decreciente
                    thickness = max(1, 3 - i // 2)
                    color = (0, int(255 * alpha), int(255 * alpha))  # Color amarillo-rojo
                    cv2.line(frame, trace_points[i], trace_points[i + 1], color, thickness)
            
            # Dibujar la detección actual
            if frame_num < len(ball_detections) and ball_detections[frame_num]:
                detection = ball_detections[frame_num][0]
                center = detection['center']
                x, y = center
                
                # Círculo principal de la pelota
                cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)  # Relleno rojo
                cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)  # Borde blanco
                
                # Opcional: dibujar bbox
                if self.debug:
                    bbox = detection['bbox']
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
            
            output_frames[frame_num] = frame
            
        return output_frames