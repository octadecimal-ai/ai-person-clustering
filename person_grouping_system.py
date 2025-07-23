"""
Kompletny system grupowania osób na zdjęciach
Wykorzystuje AutoGen + ensemble modeli YOLO + zaawansowane przetwarzanie
Gotowy do uruchomienia - wystarczy zainstalować dependencies i podać ścieżki

Instalacja wymaganych bibliotek:
pip install ultralytics opencv-python face-recognition mediapipe scikit-learn torch torchvision
pip install insightface matplotlib seaborn tqdm pillow numpy pandas
pip install pyautogen onnx onnxruntime-gpu

Autor: System AI do grupowania osób
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
from typing import List, Dict, Tuple, Optional, Union
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict
from datetime import datetime
import warnings
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Główne biblioteki AI/ML
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from tqdm import tqdm

# YOLO i detekcja
from ultralytics import YOLO
import ultralytics.utils.plotting as plotting

# Rozpoznawanie twarzy
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("face_recognition nie jest dostępne - będzie używane tylko InsightFace")

try:
    import insightface
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("InsightFace nie jest dostępne - będzie używane tylko face_recognition")

# Analiza pozy
import mediapipe as mp

# Clustering i analiza
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# AutoGen (opcjonalne - jeśli chcesz używać LLM)
try:
    import autogen
    from autogen import ConversableAgent
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    print("AutoGen nie jest dostępne - system będzie działać bez LLM")

# Wyłącz ostrzeżenia
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

@dataclass
class PersonFeatures:
    """Kompletna reprezentacja osoby z wszystkimi cechami"""
    image_path: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float = 0.0
    detection_source: str = "unknown"
    
    # Cechy wizualne
    face_encoding: Optional[np.ndarray] = None
    face_landmarks: Optional[Dict] = None
    body_pose: Optional[np.ndarray] = None
    clothing_features: Optional[np.ndarray] = None
    color_histogram: Optional[np.ndarray] = None
    
    # Cechy geometryczne
    body_shape: Optional[Dict] = None
    aspect_ratio: float = 0.0
    relative_position: Optional[Tuple[float, float]] = None
    area_ratio: float = 0.0
    
    # Metadane
    image_metadata: Optional[Dict] = None
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict:
        """Konwersja do słownika (dla JSON)"""
        result = asdict(self)
        # Usuń numpy arrays (nie da się serializować do JSON)
        for key in ['face_encoding', 'body_pose', 'clothing_features', 'color_histogram']:
            if result[key] is not None:
                result[key] = f"<numpy_array_shape_{getattr(self, key).shape}>"
        return result

class EnsembleYOLODetector:
    """Ensemble detektor wykorzystujący różne modele YOLO"""
    
    def __init__(self, device: str = "auto"):
        self.device = self._setup_device(device)
        self.models = {}
        self.model_weights = {}
        self.logger = logging.getLogger(__name__)
        
        # Dostępne modele YOLO (od najdokładniejszego do najszybszego)
        self.available_models = {
            'yolov8x': {'weight': 1.0, 'conf_threshold': 0.25, 'imgsz': 1280},
            'yolov8l': {'weight': 0.9, 'conf_threshold': 0.3, 'imgsz': 1280}, 
            'yolov8m': {'weight': 0.8, 'conf_threshold': 0.35, 'imgsz': 1280},
            'yolov8s': {'weight': 0.7, 'conf_threshold': 0.4, 'imgsz': 1280},
            'yolov8n': {'weight': 0.6, 'conf_threshold': 0.45, 'imgsz': 640},
        }
        
        self._initialize_models()
    
    def _setup_device(self, device: str) -> str:
        """Automatyczna konfiguracja urządzenia"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"  # Apple Silicon
            else:
                return "cpu"
        return device
    
    def _initialize_models(self):
        """Inicjalizacja wszystkich dostępnych modeli YOLO"""
        self.logger.info(f"Inicjalizacja modeli YOLO na urządzeniu: {self.device}")
        
        for model_name, config in self.available_models.items():
            try:
                model_path = f"{model_name}.pt"
                self.logger.info(f"Ładowanie {model_name}...")
                
                model = YOLO(model_path)
                model.to(self.device)
                
                self.models[model_name] = model
                self.model_weights[model_name] = config['weight']
                
                self.logger.info(f"✓ {model_name} załadowany pomyślnie")
                
            except Exception as e:
                self.logger.warning(f"Nie udało się załadować {model_name}: {e}")
                continue
        
        if not self.models:
            raise RuntimeError("Nie udało się załadować żadnego modelu YOLO!")
        
        self.logger.info(f"Załadowano {len(self.models)} modeli YOLO")
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocessing obrazu dla lepszej detekcji"""
        
        # Oryginał
        original = image.copy()
        
        # Ulepszona wersja
        enhanced = image.copy()
        
        # 1. Poprawa kontrastu (CLAHE)
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # 2. Lekkie wyostrzenie
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # 3. Redukcja szumu
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return original, enhanced
    
    def detect_single_model(self, image: np.ndarray, model_name: str, 
                           confidence_override: Optional[float] = None) -> List[Dict]:
        """Detekcja pojedynczym modelem"""
        
        if model_name not in self.models:
            return []
        
        model = self.models[model_name]
        config = self.available_models[model_name]
        conf = confidence_override or config['conf_threshold']
        
        try:
            results = model(
                image,
                conf=conf,
                iou=0.5,
                classes=[0],  # Tylko osoby
                device=self.device,
                verbose=False,
                imgsz=config['imgsz'],
                augment=True,
                agnostic_nms=False,
                max_det=100
            )
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        conf_score = float(box.conf.cpu().numpy())
                        
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': conf_score,
                            'model': model_name,
                            'weight': config['weight']
                        })
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Błąd detekcji w modelu {model_name}: {e}")
            return []
    
    def ensemble_detect(self, image_path: str, use_preprocessing: bool = True) -> List[PersonFeatures]:
        """Główna funkcja detekcji ensemble"""
        
        start_time = time.time()
        
        # Wczytaj obraz
        image = cv2.imread(image_path)
        if image is None:
            self.logger.error(f"Nie można wczytać obrazu: {image_path}")
            return []
        
        h, w = image.shape[:2]
        all_detections = []
        
        # Preprocessing
        if use_preprocessing:
            original, enhanced = self.preprocess_image(image)
            images_to_process = [("original", original), ("enhanced", enhanced)]
        else:
            images_to_process = [("original", image)]
        
        # Detekcja wszystkimi modelami na wszystkich wersjach obrazu
        for img_type, img in images_to_process:
            for model_name in self.models.keys():
                detections = self.detect_single_model(img, model_name)
                
                for det in detections:
                    det['image_type'] = img_type
                    all_detections.append(det)
        
        # Zaawansowany NMS z wagami
        final_detections = self._weighted_nms(all_detections, iou_threshold=0.6)
        
        # Konwersja do PersonFeatures
        persons = []
        for det in final_detections:
            person = PersonFeatures(
                image_path=image_path,
                bbox=det['bbox'],
                confidence=det['confidence'],
                detection_source=f"ensemble_{det['model']}",
                aspect_ratio=self._calculate_aspect_ratio(det['bbox']),
                relative_position=self._calculate_relative_position(det['bbox'], w, h),
                area_ratio=self._calculate_area_ratio(det['bbox'], w, h),
                processing_time=time.time() - start_time
            )
            persons.append(person)
        
        self.logger.info(f"Wykryto {len(persons)} osób w {image_path} (czas: {time.time()-start_time:.2f}s)")
        return persons
    
    def _weighted_nms(self, detections: List[Dict], iou_threshold: float = 0.6) -> List[Dict]:
        """Zaawansowany NMS z uwzględnieniem wag modeli"""
        
        if not detections:
            return []
        
        # Sortuj według confidence * weight
        detections.sort(key=lambda x: x['confidence'] * x['weight'], reverse=True)
        
        final_detections = []
        
        while detections:
            # Weź najlepszy
            best = detections.pop(0)
            final_detections.append(best)
            
            # Usuń nakładające się
            remaining = []
            for det in detections:
                if self._calculate_iou(best['bbox'], det['bbox']) < iou_threshold:
                    remaining.append(det)
                else:
                    # Możemy zaktualizować confidence na podstawie ensemble
                    if det['confidence'] * det['weight'] > best['confidence'] * best['weight'] * 0.8:
                        # Jeśli inne wykrycie jest podobnie dobre, uśrednij
                        weight_sum = best['weight'] + det['weight']
                        best['confidence'] = (
                            best['confidence'] * best['weight'] + 
                            det['confidence'] * det['weight']
                        ) / weight_sum
            
            detections = remaining
        
        return final_detections
    
    def _calculate_iou(self, box1: Tuple[int, int, int, int], 
                      box2: Tuple[int, int, int, int]) -> float:
        """Obliczenie IoU (Intersection over Union)"""
        
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Przecięcie
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Suma obszarów
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_aspect_ratio(self, bbox: Tuple[int, int, int, int]) -> float:
        """Obliczenie aspect ratio bbox"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        return height / width if width > 0 else 0.0
    
    def _calculate_relative_position(self, bbox: Tuple[int, int, int, int], 
                                   img_width: int, img_height: int) -> Tuple[float, float]:
        """Obliczenie względnej pozycji w obrazie"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0 / img_width
        center_y = (y1 + y2) / 2.0 / img_height
        return (center_x, center_y)
    
    def _calculate_area_ratio(self, bbox: Tuple[int, int, int, int], 
                            img_width: int, img_height: int) -> float:
        """Obliczenie stosunku powierzchni bbox do obrazu"""
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        img_area = img_width * img_height
        return bbox_area / img_area if img_area > 0 else 0.0

class AdvancedFeatureExtractor:
    """Zaawansowany ekstraktor cech wizualnych"""
    
    def __init__(self, device: str = "auto"):
        self.device = device
        self.logger = logging.getLogger(__name__)
        self._initialize_models()
    
    def _initialize_models(self):
        """Inicjalizacja wszystkich modeli do ekstrakcji cech"""
        
        # 1. Modele rozpoznawania twarzy
        self.face_models = {}
        
        if INSIGHTFACE_AVAILABLE:
            try:
                self.face_models['insightface'] = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
                self.face_models['insightface'].prepare(ctx_id=0 if self.device == 'cuda' else -1)
                self.logger.info("✓ InsightFace załadowany")
            except Exception as e:
                self.logger.warning(f"InsightFace błąd: {e}")
        
        # 2. MediaPipe do analizy pozy
        mp_pose = mp.solutions.pose
        self.pose_model = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # 3. Model do analizy ubrań (ResNet50)
        self.clothing_model = models.resnet50(pretrained=True)
        self.clothing_model.fc = torch.nn.Identity()
        self.clothing_model.eval()
        
        if self.device == 'cuda' and torch.cuda.is_available():
            self.clothing_model = self.clothing_model.cuda()
        
        self.clothing_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 4. Model do segmentacji (opcjonalnie)
        try:
            self.segmentation_model = YOLO('yolov8n-seg.pt')
            self.logger.info("✓ Model segmentacji załadowany")
        except:
            self.segmentation_model = None
            self.logger.warning("Model segmentacji niedostępny")
        
        self.logger.info("Modele cech załadowane pomyślnie")
    
    def extract_face_features(self, image: np.ndarray, person: PersonFeatures) -> bool:
        """Ekstrakcja cech twarzy"""
        
        x1, y1, x2, y2 = person.bbox
        person_crop = image[y1:y2, x1:x2]
        
        if person_crop.size == 0:
            return False
        
        # InsightFace
        if 'insightface' in self.face_models:
            try:
                faces = self.face_models['insightface'].get(person_crop)
                if faces:
                    face = faces[0]
                    person.face_encoding = face.embedding
                    
                    # Dodatkowe cechy twarzy
                    person.face_landmarks = {
                        'age': face.age if hasattr(face, 'age') else None,
                        'gender': face.gender if hasattr(face, 'gender') else None,
                        'landmarks': face.kps.tolist() if hasattr(face, 'kps') else None
                    }
                    return True
            except Exception as e:
                self.logger.debug(f"InsightFace błąd: {e}")
        
        # face_recognition jako fallback
        if FACE_RECOGNITION_AVAILABLE:
            try:
                face_encodings = face_recognition.face_encodings(person_crop)
                if face_encodings:
                    person.face_encoding = face_encodings[0]
                    
                    # Landmarks
                    landmarks = face_recognition.face_landmarks(person_crop)
                    if landmarks:
                        person.face_landmarks = landmarks[0]
                    return True
            except Exception as e:
                self.logger.debug(f"face_recognition błąd: {e}")
        
        return False
    
    def extract_pose_features(self, image: np.ndarray, person: PersonFeatures) -> bool:
        """Ekstrakcja cech pozy ciała"""
        
        x1, y1, x2, y2 = person.bbox
        person_crop = image[y1:y2, x1:x2]
        
        if person_crop.size == 0:
            return False
        
        try:
            # Konwersja do RGB
            rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            
            results = self.pose_model.process(rgb_crop)
            
            if results.pose_landmarks:
                # Ekstrakcja punktów kluczowych
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.extend([
                        landmark.x, landmark.y, landmark.z, 
                        landmark.visibility
                    ])
                
                person.body_pose = np.array(landmarks)
                return True
        except Exception as e:
            self.logger.debug(f"Pose estimation błąd: {e}")
        
        return False
    
    def extract_clothing_features(self, image: np.ndarray, person: PersonFeatures) -> bool:
        """Ekstrakcja cech ubrań"""
        
        x1, y1, x2, y2 = person.bbox
        person_crop = image[y1:y2, x1:x2]
        
        if person_crop.size == 0:
            return False
        
        try:
            # Przygotowanie obrazu
            pil_image = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
            input_tensor = self.clothing_transform(pil_image).unsqueeze(0)
            
            if self.device == 'cuda' and torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
            
            # Ekstrakcja cech
            with torch.no_grad():
                features = self.clothing_model(input_tensor)
                person.clothing_features = features.cpu().numpy().flatten()
            
            return True
        except Exception as e:
            self.logger.debug(f"Clothing features błąd: {e}")
            return False
    
    def extract_color_features(self, image: np.ndarray, person: PersonFeatures) -> bool:
        """Ekstrakcja histogramu kolorów"""
        
        x1, y1, x2, y2 = person.bbox
        person_crop = image[y1:y2, x1:x2]
        
        if person_crop.size == 0:
            return False
        
        try:
            # Histogram w przestrzeni HSV
            hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
            
            # Oblicz histogramy dla każdego kanału
            hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [60], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [60], [0, 256])
            
            # Połącz histogramy
            color_hist = np.concatenate([
                hist_h.flatten(), 
                hist_s.flatten(), 
                hist_v.flatten()
            ])
            
            # Normalizacja
            person.color_histogram = color_hist / (color_hist.sum() + 1e-7)
            
            return True
        except Exception as e:
            self.logger.debug(f"Color histogram błąd: {e}")
            return False
    
    def extract_body_shape_features(self, image: np.ndarray, person: PersonFeatures) -> bool:
        """Ekstrakcja cech kształtu ciała"""
        
        x1, y1, x2, y2 = person.bbox
        
        try:
            # Podstawowe wymiary
            width = x2 - x1
            height = y2 - y1
            
            # Proporcje
            aspect_ratio = height / width if width > 0 else 0
            
            # Pozycja w obrazie
            img_height, img_width = image.shape[:2]
            center_x = (x1 + x2) / 2.0 / img_width
            center_y = (y1 + y2) / 2.0 / img_height
            bottom_y = y2 / img_height
            
            # Powierzchnia
            area = width * height
            area_ratio = area / (img_width * img_height)
            
            person.body_shape = {
                'width': width,
                'height': height,
                'aspect_ratio': aspect_ratio,
                'center_x': center_x,
                'center_y': center_y,
                'bottom_y': bottom_y,
                'area': area,
                'area_ratio': area_ratio
            }
            
            return True
        except Exception as e:
            self.logger.debug(f"Body shape błąd: {e}")
            return False
    
    def extract_all_features(self, image: np.ndarray, person: PersonFeatures) -> PersonFeatures:
        """Ekstrakcja wszystkich cech dla osoby"""
        
        start_time = time.time()
        
        # Próbuj wszystkie ekstraktory
        self.extract_face_features(image, person)
        self.extract_pose_features(image, person)
        self.extract_clothing_features(image, person)
        self.extract_color_features(image, person)
        self.extract_body_shape_features(image, person)
        
        person.processing_time += time.time() - start_time
        
        return person

class IntelligentClustering:
    """Inteligentny system klastrowania z adaptacyjnymi parametrami"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Domyślne parametry
        self.default_params = {
            'similarity_threshold': 0.7,
            'min_cluster_size': 2,
            'eps': 0.4,
            'clustering_method': 'adaptive',
            'use_face_weight': 2.0,
            'use_pose_weight': 1.0,
            'use_clothing_weight': 1.5,
            'use_color_weight': 0.8,
            'use_shape_weight': 0.5
        }
        
        self.default_params.update(self.config)
    
    def create_feature_matrix(self, persons: List[PersonFeatures]) -> Tuple[np.ndarray, List[str]]:
        """Tworzenie macierzy cech z inteligentnym ważeniem"""
        
        features_list = []
        feature_names = []
        
        # Analizuj dostępne cechy
        has_faces = sum(1 for p in persons if p.face_encoding is not None)
        has_poses = sum(1 for p in persons if p.body_pose is not None) 
        has_clothing = sum(1 for p in persons if p.clothing_features is not None)
        has_colors = sum(1 for p in persons if p.color_histogram is not None)
        
        face_ratio = has_faces / len(persons)
        pose_ratio = has_poses / len(persons)
        clothing_ratio = has_clothing / len(persons)
        color_ratio = has_colors / len(persons)
        
        self.logger.info(f"Dostępność cech: Face={face_ratio:.1%}, Pose={pose_ratio:.1%}, "
                        f"Clothing={clothing_ratio:.1%}, Color={color_ratio:.1%}")
        
        for person in persons:
            person_features = []
            
            # 1. Cechy twarzy (najważniejsze jeśli dostępne)
            if person.face_encoding is not None:
                face_features = person.face_encoding * self.default_params['use_face_weight']
                person_features.extend(face_features)
                if not feature_names or 'face' not in feature_names:
                    feature_names.extend([f'face_{i}' for i in range(len(face_features))])
            elif face_ratio > 0.3:  # Jeśli inne mają twarze, uzupełnij zerami
                face_dim = 512  # Standardowy rozmiar
                person_features.extend([0.0] * face_dim)
                if not feature_names or 'face' not in feature_names:
                    feature_names.extend([f'face_{i}' for i in range(face_dim)])
            
            # 2. Cechy pozy
            if person.body_pose is not None:
                pose_features = person.body_pose * self.default_params['use_pose_weight']
                person_features.extend(pose_features)
                if 'pose' not in str(feature_names):
                    feature_names.extend([f'pose_{i}' for i in range(len(pose_features))])
            elif pose_ratio > 0.3:
                pose_dim = 132
                person_features.extend([0.0] * pose_dim)
                if 'pose' not in str(feature_names):
                    feature_names.extend([f'pose_{i}' for i in range(pose_dim)])
            
            # 3. Cechy ubrań
            if person.clothing_features is not None:
                clothing_features = person.clothing_features * self.default_params['use_clothing_weight']
                person_features.extend(clothing_features)
                if 'clothing' not in str(feature_names):
                    feature_names.extend([f'clothing_{i}' for i in range(len(clothing_features))])
            elif clothing_ratio > 0.3:
                clothing_dim = 2048
                person_features.extend([0.0] * clothing_dim)
                if 'clothing' not in str(feature_names):
                    feature_names.extend([f'clothing_{i}' for i in range(clothing_dim)])
            
            # 4. Histogram kolorów
            if person.color_histogram is not None:
                color_features = person.color_histogram * self.default_params['use_color_weight']
                person_features.extend(color_features)
                if 'color' not in str(feature_names):
                    feature_names.extend([f'color_{i}' for i in range(len(color_features))])
            elif color_ratio > 0.3:
                color_dim = 170  # 50+60+60 z HSV histogramu
                person_features.extend([0.0] * color_dim)
                if 'color' not in str(feature_names):
                    feature_names.extend([f'color_{i}' for i in range(color_dim)])
            
            # 5. Cechy kształtu ciała
            if person.body_shape is not None:
                shape_values = [
                    person.body_shape.get('aspect_ratio', 0),
                    person.body_shape.get('area_ratio', 0),
                    person.body_shape.get('center_x', 0),
                    person.body_shape.get('center_y', 0),
                    person.body_shape.get('bottom_y', 0)
                ]
                shape_features = np.array(shape_values) * self.default_params['use_shape_weight']
                person_features.extend(shape_features)
                if 'shape' not in str(feature_names):
                    feature_names.extend(['shape_aspect', 'shape_area', 'shape_cx', 'shape_cy', 'shape_bottom'])
            else:
                person_features.extend([0.0] * 5)
                if 'shape' not in str(feature_names):
                    feature_names.extend(['shape_aspect', 'shape_area', 'shape_cx', 'shape_cy', 'shape_bottom'])
            
            features_list.append(person_features)
        
        feature_matrix = np.array(features_list)
        self.logger.info(f"Utworzono macierz cech: {feature_matrix.shape}")
        
        return feature_matrix, feature_names
    
    def adaptive_clustering(self, persons: List[PersonFeatures]) -> Dict[int, List[PersonFeatures]]:
        """Adaptacyjne klastrowanie z automatycznym doborem parametrów"""
        
        if len(persons) < 2:
            return {0: persons}
        
        # Tworzenie macierzy cech
        feature_matrix, feature_names = self.create_feature_matrix(persons)
        
        # Normalizacja
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(feature_matrix)
        
        # Automatyczny dobór parametrów na podstawie danych
        optimal_params = self._estimate_optimal_parameters(normalized_features, persons)
        
        # Próbuj różne metody klastrowania
        clustering_results = {}
        
        # 1. DBSCAN z optymalnymi parametrami
        try:
            dbscan = DBSCAN(
                eps=optimal_params['eps'],
                min_samples=optimal_params['min_samples'],
                metric='cosine'
            )
            dbscan_labels = dbscan.fit_predict(normalized_features)
            clustering_results['dbscan'] = self._create_clusters_from_labels(persons, dbscan_labels)
            
        except Exception as e:
            self.logger.warning(f"DBSCAN błąd: {e}")
        
        # 2. Agglomerative Clustering
        try:
            n_clusters = min(len(persons) // 2, optimal_params.get('estimated_clusters', 5))
            if n_clusters >= 2:
                agg = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage='average',
                    metric='cosine'
                )
                agg_labels = agg.fit_predict(normalized_features)
                clustering_results['agglomerative'] = self._create_clusters_from_labels(persons, agg_labels)
                
        except Exception as e:
            self.logger.warning(f"Agglomerative błąd: {e}")
        
        # 3. Własny algorytm oparty na podobieństwie twarzy
        if any(p.face_encoding is not None for p in persons):
            try:
                face_clusters = self._face_similarity_clustering(persons)
                clustering_results['face_similarity'] = face_clusters
            except Exception as e:
                self.logger.warning(f"Face similarity błąd: {e}")
        
        # Wybierz najlepszy wynik
        best_clustering = self._select_best_clustering(clustering_results, persons)
        
        return best_clustering
    
    def _estimate_optimal_parameters(self, features: np.ndarray, persons: List[PersonFeatures]) -> Dict:
        """Estymacja optymalnych parametrów klastrowania"""
        
        # Analiza rozkładu odległości
        distances = euclidean_distances(features)
        
        # Usuń przekątną (odległości do siebie)
        mask = np.ones(distances.shape, dtype=bool)
        np.fill_diagonal(mask, False)
        distance_values = distances[mask]
        
        # Statystyki odległości
        mean_dist = np.mean(distance_values)
        std_dist = np.std(distance_values)
        median_dist = np.median(distance_values)
        
        # Estymacja eps dla DBSCAN
        eps_candidates = [
            median_dist * 0.7,
            median_dist * 0.8, 
            median_dist * 0.9,
            mean_dist * 0.6,
            mean_dist * 0.7
        ]
        
        # Estymacja liczby klastrów (heurystyka)
        estimated_clusters = min(
            max(2, len(persons) // 5),  # Co najmniej 2, maksymalnie co piąta osoba
            len(persons) // 2           # Ale nie więcej niż połowa
        )
        
        # Parametry oparte na dostępności cech
        face_available = sum(1 for p in persons if p.face_encoding is not None) / len(persons)
        
        if face_available > 0.7:
            # Dużo twarzy - możemy być bardziej precyzyjni
            min_samples = 2
            eps = min(eps_candidates)
        elif face_available > 0.3:
            # Średnio twarzy - balansuj
            min_samples = 2
            eps = sorted(eps_candidates)[len(eps_candidates)//2]
        else:
            # Mało twarzy - bardziej liberalne parametry
            min_samples = 2
            eps = max(eps_candidates)
        
        return {
            'eps': eps,
            'min_samples': min_samples,
            'estimated_clusters': estimated_clusters,
            'mean_distance': mean_dist,
            'median_distance': median_dist
        }
    
    def _create_clusters_from_labels(self, persons: List[PersonFeatures], labels: np.ndarray) -> Dict[int, List[PersonFeatures]]:
        """Tworzenie klastrów z etykiet"""
        
        clusters = defaultdict(list)
        for person, label in zip(persons, labels):
            clusters[int(label)].append(person)
        
        return dict(clusters)
    
    def _face_similarity_clustering(self, persons: List[PersonFeatures]) -> Dict[int, List[PersonFeatures]]:
        """Klastrowanie oparte na podobieństwie twarzy"""
        
        # Filtruj osoby z twarzami
        persons_with_faces = [p for p in persons if p.face_encoding is not None]
        persons_without_faces = [p for p in persons if p.face_encoding is None]
        
        if len(persons_with_faces) < 2:
            # Jeśli mało twarzy, dodaj wszystkich do jednego klastra
            return {0: persons}
        
        # Macierz podobieństwa twarzy
        face_encodings = [p.face_encoding for p in persons_with_faces]
        similarities = cosine_similarity(face_encodings)
        
        # Próg podobieństwa dla twarzy (wysoki, bo twarze są najważniejsze)
        face_threshold = 0.8
        
        # Grupy podobnych twarzy
        clusters = {}
        cluster_id = 0
        assigned = set()
        
        for i, person_i in enumerate(persons_with_faces):
            if i in assigned:
                continue
                
            # Nowy klaster
            cluster = [person_i]
            assigned.add(i)
            
            # Znajdź podobne twarze
            for j, person_j in enumerate(persons_with_faces):
                if j in assigned or i == j:
                    continue
                    
                if similarities[i][j] > face_threshold:
                    cluster.append(person_j)
                    assigned.add(j)
            
            clusters[cluster_id] = cluster
            cluster_id += 1
        
        # Osoby bez twarzy przypisz do najbardziej podobnych klastrów
        # (na podstawie innych cech)
        if persons_without_faces:
            for person in persons_without_faces:
                best_cluster = self._find_best_cluster_for_person(person, clusters)
                if best_cluster is not None:
                    clusters[best_cluster].append(person)
                else:
                    # Nowy klaster dla osób bez rozpoznanych twarzy
                    clusters[cluster_id] = [person]
                    cluster_id += 1
        
        return clusters
    
    def _find_best_cluster_for_person(self, person: PersonFeatures, 
                                    clusters: Dict[int, List[PersonFeatures]]) -> Optional[int]:
        """Znajdź najlepszy klaster dla osoby bez twarzy"""
        
        if not clusters:
            return None
        
        best_cluster = None
        best_similarity = -1
        
        for cluster_id, cluster_persons in clusters.items():
            # Oblicz średnie podobieństwo do klastra
            similarities = []
            
            for cluster_person in cluster_persons:
                sim = self._calculate_person_similarity(person, cluster_person)
                if sim is not None:
                    similarities.append(sim)
            
            if similarities:
                avg_similarity = np.mean(similarities)
                if avg_similarity > best_similarity and avg_similarity > 0.5:
                    best_similarity = avg_similarity
                    best_cluster = cluster_id
        
        return best_cluster
    
    def _calculate_person_similarity(self, person1: PersonFeatures, 
                                   person2: PersonFeatures) -> Optional[float]:
        """Oblicz podobieństwo między dwiema osobami"""
        
        similarities = []
        
        # Podobieństwo pozy
        if person1.body_pose is not None and person2.body_pose is not None:
            pose_sim = cosine_similarity([person1.body_pose], [person2.body_pose])[0][0]
            similarities.append(('pose', pose_sim, 1.0))
        
        # Podobieństwo ubrań
        if person1.clothing_features is not None and person2.clothing_features is not None:
            clothing_sim = cosine_similarity([person1.clothing_features], [person2.clothing_features])[0][0]
            similarities.append(('clothing', clothing_sim, 1.5))
        
        # Podobieństwo kolorów
        if person1.color_histogram is not None and person2.color_histogram is not None:
            color_sim = cosine_similarity([person1.color_histogram], [person2.color_histogram])[0][0]
            similarities.append(('color', color_sim, 0.8))
        
        # Podobieństwo kształtu
        if person1.body_shape is not None and person2.body_shape is not None:
            shape1 = [person1.body_shape.get(k, 0) for k in ['aspect_ratio', 'area_ratio']]
            shape2 = [person2.body_shape.get(k, 0) for k in ['aspect_ratio', 'area_ratio']]
            if all(v != 0 for v in shape1 + shape2):
                shape_sim = cosine_similarity([shape1], [shape2])[0][0]
                similarities.append(('shape', shape_sim, 0.5))
        
        if not similarities:
            return None
        
        # Ważona średnia
        weighted_sum = sum(sim * weight for _, sim, weight in similarities)
        total_weight = sum(weight for _, _, weight in similarities)
        
        return weighted_sum / total_weight if total_weight > 0 else None
    
    def _select_best_clustering(self, results: Dict[str, Dict], persons: List[PersonFeatures]) -> Dict[int, List[PersonFeatures]]:
        """Wybierz najlepszy wynik klastrowania"""
        
        if not results:
            return {0: persons}
        
        best_method = None
        best_score = -1
        
        for method, clusters in results.items():
            score = self._evaluate_clustering(clusters, persons)
            self.logger.info(f"Metoda {method}: score={score:.3f}, clusters={len(clusters)}")
            
            if score > best_score:
                best_score = score
                best_method = method
        
        if best_method:
            self.logger.info(f"Wybrano metodę: {best_method} (score: {best_score:.3f})")
            return results[best_method]
        else:
            return {0: persons}
    
    def _evaluate_clustering(self, clusters: Dict[int, List[PersonFeatures]], 
                           persons: List[PersonFeatures]) -> float:
        """Ocena jakości klastrowania"""
        
        if not clusters:
            return 0.0
        
        scores = []
        
        # 1. Penalizuj za zbyt małe klastry (szum)
        noise_clusters = sum(1 for cluster in clusters.values() if len(cluster) == 1)
        noise_penalty = noise_clusters / len(clusters) if clusters else 0
        
        # 2. Oceń spójność wewnętrzną klastrów
        intra_cluster_similarities = []
        for cluster in clusters.values():
            if len(cluster) > 1:
                cluster_similarities = []
                for i in range(len(cluster)):
                    for j in range(i+1, len(cluster)):
                        sim = self._calculate_person_similarity(cluster[i], cluster[j])
                        if sim is not None:
                            cluster_similarities.append(sim)
                
                if cluster_similarities:
                    intra_cluster_similarities.append(np.mean(cluster_similarities))
        
        avg_intra_similarity = np.mean(intra_cluster_similarities) if intra_cluster_similarities else 0
        
        # 3. Oceń różnorodność między klastrami
        inter_cluster_distances = []
        cluster_list = list(clusters.values())
        for i in range(len(cluster_list)):
            for j in range(i+1, len(cluster_list)):
                # Średnia odległość między klastrami
                distances = []
                for person1 in cluster_list[i][:3]:  # Tylko pierwsze 3 osoby dla wydajności
                    for person2 in cluster_list[j][:3]:
                        sim = self._calculate_person_similarity(person1, person2)
                        if sim is not None:
                            distances.append(1 - sim)  # Odległość = 1 - podobieństwo
                
                if distances:
                    inter_cluster_distances.append(np.mean(distances))
        
        avg_inter_distance = np.mean(inter_cluster_distances) if inter_cluster_distances else 0
        
        # Ostateczna ocena
        final_score = (
            avg_intra_similarity * 0.4 +     # Spójność wewnętrzna
            avg_inter_distance * 0.3 +       # Separacja między klastrami  
            (1 - noise_penalty) * 0.3        # Mniej szumu = lepiej
        )
        
        return max(0, final_score)


class ComprehensivePersonGroupingSystem:
    """Główny system integrujący wszystkie komponenty"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = self._setup_logger()
        self.config = self._load_config(config_path)
        
        # Inicjalizacja komponentów
        self.detector = EnsembleYOLODetector(device=self.config.get('device', 'auto'))
        self.feature_extractor = AdvancedFeatureExtractor(device=self.config.get('device', 'auto'))
        self.clusterer = IntelligentClustering(self.config)
        
        # AutoGen (opcjonalnie)
        if AUTOGEN_AVAILABLE and self.config.get('use_autogen', False):
            self._setup_autogen()
        
        self.logger.info("System inicjalizowany pomyślnie")
    
    def _setup_logger(self) -> logging.Logger:
        """Konfiguracja systemu logowania"""
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        logger = logging.getLogger(__name__)
        
        # Dodaj handler do pliku
        file_handler = logging.FileHandler('person_grouping.log')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
        
        return logger
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Ładowanie konfiguracji"""
        
        default_config = {
            'device': 'auto',
            'use_autogen': False,
            'batch_size': 4,
            'num_workers': min(4, mp.cpu_count()),
            'similarity_threshold': 0.7,
            'min_cluster_size': 2,
            'save_intermediate_results': True,
            'output_format': 'comprehensive',  # 'simple' lub 'comprehensive'
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                self.logger.info(f"Załadowano konfigurację z {config_path}")
            except Exception as e:
                self.logger.warning(f"Błąd ładowania konfiguracji: {e}")
        
        return default_config
    
    def _setup_autogen(self):
        """Konfiguracja AutoGen (opcjonalna)"""
        
        try:
            # Przykładowa konfiguracja - dostosuj według potrzeb
            llm_config = {
                "config_list": [{
                    "model": "gpt-4",
                    "api_key": os.getenv("OPENAI_API_KEY", "sk-dummy")
                }]
            }
            
            self.analyzer_agent = ConversableAgent(
                name="image_analyzer",
                system_message="""Jesteś ekspertem od analizy wizualnej osób na zdjęciach.
                Pomagasz w ocenie jakości grupowania osób na podstawie cech wizualnych.""",
                llm_config=llm_config,
                human_input_mode="NEVER"
            )
            
            self.logger.info("AutoGen skonfigurowany")
            
        except Exception as e:
            self.logger.warning(f"AutoGen niedostępny: {e}")
            self.analyzer_agent = None
    
    def process_single_image(self, image_path: str) -> List[PersonFeatures]:
        """Przetwarzanie pojedynczego obrazu"""
        
        start_time = time.time()
        self.logger.info(f"Przetwarzanie: {os.path.basename(image_path)}")
        
        # 1. Detekcja osób
        persons = self.detector.ensemble_detect(image_path)
        
        if not persons:
            self.logger.warning(f"Nie wykryto osób w {image_path}")
            return []
        
        # 2. Ekstrakcja cech
        image = cv2.imread(image_path)
        if image is None:
            self.logger.error(f"Nie można wczytać {image_path}")
            return []
        
        # Dodaj metadane obrazu
        img_metadata = {
            'file_size': os.path.getsize(image_path),
            'dimensions': (image.shape[1], image.shape[0]),
            'channels': image.shape[2] if len(image.shape) > 2 else 1,
            'modification_time': os.path.getmtime(image_path)
        }
        
        # Przetwarzaj każdą osobę
        processed_persons = []
        for person in persons:
            person.image_metadata = img_metadata
            enhanced_person = self.feature_extractor.extract_all_features(image, person)
            processed_persons.append(enhanced_person)
        
        total_time = time.time() - start_time
        self.logger.info(f"Przetworzono {len(processed_persons)} osób w {total_time:.2f}s")
        
        return processed_persons
    
    def process_image_batch(self, image_paths: List[str], 
                          use_multiprocessing: bool = True) -> List[PersonFeatures]:
        """Przetwarzanie wsadowe obrazów"""
        
        all_persons = []
        
        if use_multiprocessing and len(image_paths) > 1:
            # Przetwarzanie równoległe
            max_workers = min(self.config['num_workers'], len(image_paths))
            self.logger.info(f"Przetwarzanie równoległe: {max_workers} workerów")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self.process_single_image, path) for path in image_paths]
                
                for future in tqdm(futures, desc="Przetwarzanie obrazów"):
                    try:
                        persons = future.result()
                        all_persons.extend(persons)
                    except Exception as e:
                        self.logger.error(f"Błąd przetwarzania: {e}")
        else:
            # Przetwarzanie sekwencyjne
            for image_path in tqdm(image_paths, desc="Przetwarzanie obrazów"):
                try:
                    persons = self.process_single_image(image_path)
                    all_persons.extend(persons)
                except Exception as e:
                    self.logger.error(f"Błąd przetwarzania {image_path}: {e}")
        
        self.logger.info(f"Przetworzono łącznie {len(all_persons)} osób z {len(image_paths)} obrazów")
        return all_persons
    
    def run_full_pipeline(self, image_directory: str, 
                         output_directory: str) -> Dict[int, List[PersonFeatures]]:
        """Uruchomienie pełnego pipeline'u grupowania"""
        
        self.logger.info("=== ROZPOCZĘCIE PIPELINE'U GRUPOWANIA OSÓB ===")
        start_time = time.time()
        
        # 1. Znajdź obrazy
        image_paths = self._find_images(image_directory)
        if not image_paths:
            raise ValueError(f"Nie znaleziono obrazów w {image_directory}")
        
        self.logger.info(f"Znaleziono {len(image_paths)} obrazów")
        
        # 2. Przetwarzanie obrazów
        all_persons = self.process_image_batch(image_paths)
        
        if not all_persons:
            raise ValueError("Nie wykryto żadnych osób na obrazach")
        
        # 3. Klastrowanie
        self.logger.info("Rozpoczynam klastrowanie...")
        clusters = self.clusterer.adaptive_clustering(all_persons)
        
        # 4. Postprocessing klastrów
        refined_clusters = self._refine_clusters(clusters)
        
        # 5. Zapisywanie wyników
        os.makedirs(output_directory, exist_ok=True)
        self._save_comprehensive_results(refined_clusters, output_directory, image_directory)
        
        # 6. Generowanie raportów i wizualizacji
        self._generate_analysis_report(refined_clusters, output_directory)
        self._create_visualizations(refined_clusters, output_directory)
        
        total_time = time.time() - start_time
        self.logger.info(f"=== PIPELINE ZAKOŃCZONY W {total_time:.1f}s ===")
        self.logger.info(f"Wyniki w: {output_directory}")
        
        self._print_summary(refined_clusters, total_time)
        
        return refined_clusters
    
    def _find_images(self, directory: str) -> List[str]:
        """Znajdź wszystkie obrazy w katalogu"""
        
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_paths = []
        
        directory_path = Path(directory)
        
        for ext in extensions:
            image_paths.extend(directory_path.glob(f'*{ext}'))
            image_paths.extend(directory_path.glob(f'*{ext.upper()}'))
            # Rekurencyjnie w podkatalogach
            image_paths.extend(directory_path.glob(f'**/*{ext}'))
            image_paths.extend(directory_path.glob(f'**/*{ext.upper()}'))
        
        # Usuń duplikaty i konwertuj do stringów
        unique_paths = list(set(str(path) for path in image_paths))
        unique_paths.sort()
        
        return unique_paths
    
    def _refine_clusters(self, clusters: Dict[int, List[PersonFeatures]]) -> Dict[int, List[PersonFeatures]]:
        """Rafinacja klastrów - łączenie podobnych, podział dużych"""
        
        # Usuń klastry z jedną osobą jeśli są inne opcje
        single_person_clusters = [cid for cid, persons in clusters.items() if len(persons) == 1]
        
        if len(single_person_clusters) > len(clusters) // 2:
            # Jeśli za dużo pojedynczych klastrów, spróbuj je przypisać
            for cluster_id in single_person_clusters[:]:
                person = clusters[cluster_id][0]
                best_match = self._find_best_cluster_match(person, clusters, exclude_id=cluster_id)
                
                if best_match is not None:
                    clusters[best_match].append(person)
                    del clusters[cluster_id]
        
        # Przenumeruj klastry
        refined_clusters = {}
        for new_id, (old_id, persons) in enumerate(clusters.items()):
            refined_clusters[new_id] = persons
        
        return refined_clusters
    
    def _find_best_cluster_match(self, person: PersonFeatures, 
                               clusters: Dict[int, List[PersonFeatures]], 
                               exclude_id: int) -> Optional[int]:
        """Znajdź najlepszy klaster dla danej osoby"""
        
        best_cluster = None
        best_similarity = 0.4  # Minimum threshold
        
        for cluster_id, cluster_persons in clusters.items():
            if cluster_id == exclude_id or len(cluster_persons) == 0:
                continue
            
            # Oblicz średnie podobieństwo do klastra
            similarities = []
            for cluster_person in cluster_persons:
                sim = self.clusterer._calculate_person_similarity(person, cluster_person)
                if sim is not None:
                    similarities.append(sim)
            
            if similarities:
                avg_sim = np.mean(similarities)
                if avg_sim > best_similarity:
                    best_similarity = avg_sim
                    best_cluster = cluster_id
        
        return best_cluster
    
    def _save_comprehensive_results(self, clusters: Dict[int, List[PersonFeatures]], 
                                  output_dir: str, source_dir: str):
        """Zapisz kompletne wyniki"""
        
        # 1. Podstawowy JSON z informacjami o klastrach
        cluster_summary = {}
        for cluster_id, persons in clusters.items():
            cluster_summary[cluster_id] = {
                'person_count': len(persons),
                'images': list(set(p.image_path for p in persons)),
                'avg_confidence': float(np.mean([p.confidence for p in persons])),
                'detection_sources': list(set(p.detection_source for p in persons)),
                'features_available': {
                    'faces': sum(1 for p in persons if p.face_encoding is not None),
                    'poses': sum(1 for p in persons if p.body_pose is not None),
                    'clothing': sum(1 for p in persons if p.clothing_features is not None),
                    'colors': sum(1 for p in persons if p.color_histogram is not None)
                }
            }
        
        with open(os.path.join(output_dir, 'cluster_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(cluster_summary, f, indent=2, ensure_ascii=False)
        
        # 2. Szczegółowe dane (pickle)
        with open(os.path.join(output_dir, 'detailed_results.pkl'), 'wb') as f:
            pickle.dump(clusters, f)
        
        # 3. CSV dla łatwej analizy
        self._save_results_csv(clusters, output_dir)
        
        # 4. Organizacja plików obrazów
        self._organize_cluster_images(clusters, output_dir, source_dir)
        
        self.logger.info("Wyniki zapisane pomyślnie")
    
    def _save_results_csv(self, clusters: Dict[int, List[PersonFeatures]], output_dir: str):
        """Zapisz wyniki w formacie CSV"""
        
        data = []
        for cluster_id, persons in clusters.items():
            for person in persons:
                