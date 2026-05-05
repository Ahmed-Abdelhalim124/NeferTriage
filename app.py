import gradio as gr
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import tempfile
from collections import deque
from ultralytics import YOLO
import mediapipe as mp
import os

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ------------------------------
# Simple Fall Detection Model
# ------------------------------
class FallDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(FallDetector, self).__init__()
        base = models.efficientnet_b0(weights=None)
        self.feature_extractor = base.features
        self.pool = base.avgpool
        in_features = base.classifier[1].in_features
        self.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):  # (B, T, 3, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        feats = self.feature_extractor(x)
        feats = self.pool(feats)
        feats = torch.flatten(feats, 1)
        feats = feats.view(B, T, -1).mean(dim=1)
        out = self.fc(feats)
        return out

# ------------------------------
# Simple Fall Detection System 
# ------------------------------
class SimpleFallDetectionSystem:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the fall detection model
        self.model = FallDetector(num_classes=2).to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print("✅ Loaded simple fall detection model weights successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not load fall model weights: {e}")
        self.model.eval()
        
        # Initialize MediaPipe for pose detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.3
        )
        
        # Classes and transforms
        self.classes = ["NO FALL", "FALL"]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def detect_sitting_position(self, frame):
        """Detect if the person in the frame is sitting (from your code)"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(frame_rgb)
        
        if not pose_results.pose_landmarks:
            return False, 0.0
        
        landmarks = pose_results.pose_landmarks.landmark
        
        # Get key landmarks
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        
        sitting_indicators = []
        
        # 1. Check torso upright (shoulders above hips)
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_center_y = (left_hip.y + right_hip.y) / 2
        torso_upright = shoulder_center_y < hip_center_y
        sitting_indicators.append(torso_upright)
        
        # 2. Check knee angle (bent knees indicate sitting)
        def calculate_angle(p1, p2, p3):
            v1 = np.array([p1.x - p2.x, p1.y - p2.y])
            v2 = np.array([p3.x - p2.x, p3.y - p2.y])
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle) * 180 / np.pi
            return angle
        
        try:
            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
            avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
            
            # Sitting typically has knee angles between 70-110 degrees
            knee_bent_sitting = 70 <= avg_knee_angle <= 110
            sitting_indicators.append(knee_bent_sitting)
        except:
            sitting_indicators.append(False)
        
        # 3. Check head position relative to hips
        head_above_hips = nose.y < hip_center_y - 0.1
        sitting_indicators.append(head_above_hips)
        
        # 4. Check if legs are visible and positioned correctly for sitting
        legs_visible = (left_knee.visibility > 0.5 and right_knee.visibility > 0.5)
        sitting_indicators.append(legs_visible)
        
        # 5. Check hip height (sitting people have hips in middle region)
        hip_in_middle = 0.3 <= hip_center_y <= 0.8
        sitting_indicators.append(hip_in_middle)
        
        # Calculate sitting confidence
        sitting_score = sum(sitting_indicators) / len(sitting_indicators)
        is_sitting = sitting_score >= 0.6  
        
        return is_sitting, sitting_score
    
    def analyze_fall_characteristics(self, frame):
        """Analyze frame for fall characteristics using pose and geometry (from your code)"""
        height, width = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Pose analysis
        pose_results = self.pose.process(frame_rgb)
        fall_indicators = []
        
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            
            # Get key points
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
            
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
            hip_center_y = (left_hip.y + right_hip.y) / 2
            
            # 1. Check if person is horizontal (major fall indicator)
            torso_horizontal = abs(shoulder_center_y - hip_center_y) < 0.1
            fall_indicators.append(torso_horizontal)
            
            # 2. Check if head is at same level as body (lying down)
            head_body_level = abs(nose.y - hip_center_y) < 0.2
            fall_indicators.append(head_body_level)
            
            # 3. Check if person is in lower part of frame
            in_lower_frame = hip_center_y > 0.7
            fall_indicators.append(in_lower_frame)
            
            # 4. Check body orientation (side view indicates fall)
            shoulder_spread = abs(left_shoulder.x - right_shoulder.x)
            hip_spread = abs(left_hip.x - right_hip.x)
            side_orientation = shoulder_spread < 0.15 or hip_spread < 0.15
            fall_indicators.append(side_orientation)
        else:
            # No pose detected, use basic geometric analysis
            fall_indicators = [False] * 4
        
        fall_score = sum(fall_indicators) / len(fall_indicators) if fall_indicators else 0.0
        return fall_score
    
    def process_frame(self, frame):
        """Process a single frame with enhanced logic (from your code)"""
        # Step 1: Check if person is sitting (if so, not a fall)
        is_sitting, sitting_confidence = self.detect_sitting_position(frame)
        
        if is_sitting and sitting_confidence > 0.6:
            return {
                'prediction': 0,  # NO FALL
                'confidence': 1.0 - sitting_confidence,
                'reason': 'SITTING_DETECTED',
                'ml_prediction': None,
                'ml_confidence': None,
                'sitting_score': sitting_confidence
            }
        
        # Step 2: Get ML model prediction
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_t.unsqueeze(0))  # (1,1,C,H,W)
            probs = F.softmax(outputs, dim=1)
            ml_pred = torch.argmax(probs, 1).item()
            ml_confidence = probs[0][ml_pred].item()
        
        # Step 3: Analyze fall characteristics
        fall_score = self.analyze_fall_characteristics(frame)
        
        # Step 4: Combined decision making
        if ml_pred == 1:  # ML model predicts fall
            if fall_score < 0.3:  # But pose analysis suggests no fall
                # Likely false positive, reduce confidence
                final_pred = 0
                final_confidence = 1.0 - ml_confidence
                reason = 'ML_FALL_BUT_POSE_NORMAL'
            else:
                # Both agree it's a fall
                final_pred = 1
                final_confidence = (ml_confidence + fall_score) / 2
                reason = 'ML_AND_POSE_AGREE_FALL'
        else:  # ML model predicts no fall
            if fall_score > 0.6:  # But pose suggests fall
                final_pred = 1
                final_confidence = fall_score
                reason = 'POSE_DETECTED_FALL'
            else:
                # Both agree no fall
                final_pred = 0
                final_confidence = 1.0 - ml_confidence
                reason = 'NORMAL'
        
        return {
            'prediction': final_pred,
            'confidence': final_confidence,
            'reason': reason,
            'ml_prediction': ml_pred,
            'ml_confidence': ml_confidence,
            'fall_score': fall_score,
            'sitting_score': sitting_confidence if is_sitting else 0.0
        }

# ------------------------------
# Chest Pain Detection System 
# ------------------------------
class ChestPainDetectionSystem:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Chest Pain Detection Model
        self.chest_pain_model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.chest_pain_model.classifier[1] = torch.nn.Linear(
            self.chest_pain_model.classifier[1].in_features, 2
        )
        try:
            self.chest_pain_model.load_state_dict(torch.load(model_path, map_location=self.device))
            print("✅ Loaded chest pain detection model weights successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not load chest pain model weights: {e}")
        self.chest_pain_model = self.chest_pain_model.to(self.device)
        self.chest_pain_model.eval()
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.4
        )
        
        # Classes and transforms
        self.chest_pain_classes = ["chest_pain", "normal"]
        self.chest_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def detect_hands_near_chest(self, frame, bbox, hand_sensitivity=0.8):
        """Enhanced hand detection for chest pain analysis (PRESERVED EXACTLY)"""
        x1, y1, x2, y2 = bbox
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            return False, []

        person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

        # Pose detection for chest area estimation
        pose_results = self.pose.process(person_rgb)
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            shoulder_left = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            shoulder_right = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            chest_center_x = int((shoulder_left.x + shoulder_right.x)/2 * (x2 - x1))
            chest_center_y = int((shoulder_left.y + shoulder_right.y)/2 * (y2 - y1))
            shoulder_dist = int(np.abs(shoulder_right.x - shoulder_left.x) * (x2 - x1))
            chest_width = int(shoulder_dist * 1.2)
            chest_height = int((y2 - y1) * 0.5)
        else:
            person_height = y2 - y1
            person_width = x2 - x1
            chest_center_x = person_width // 2
            chest_center_y = int(person_height * 0.4)
            chest_width = int(person_width * 0.5)
            chest_height = int(person_height * 0.5)

        chest_width = max(chest_width, 1)
        chest_height = max(chest_height, 1)

        # Hand detection
        hand_results = self.hands.process(person_rgb)
        if not hand_results.multi_hand_landmarks:
            return False, []

        person_height = y2 - y1
        person_width = x2 - x1
        detected_hands = []

        for hand_landmarks in hand_results.multi_hand_landmarks:
            in_chest_count = 0
            for lm in hand_landmarks.landmark:
                hx = int(lm.x * person_width)
                hy = int(lm.y * person_height)
                # Check if hand is in chest ellipse
                if ((hx - chest_center_x)**2) / (chest_width/2)**2 + ((hy - chest_center_y)**2) / (chest_height/2)**2 <= 1:
                    in_chest_count += 1
            
            if in_chest_count >= 10:  # At least 10 landmarks in chest area
                hand_x = np.mean([lm.x for lm in hand_landmarks.landmark]) * person_width
                hand_y = np.mean([lm.y for lm in hand_landmarks.landmark]) * person_height
                distance = np.sqrt((hand_x - chest_center_x)**2 + (hand_y - chest_center_y)**2)
                normalized_distance = distance / max(person_width, person_height)
                
                if normalized_distance < hand_sensitivity * 0.5:
                    detected_hands.append({
                        'position': (hand_x, hand_y),
                        'distance': normalized_distance,
                        'landmarks': [(lm.x * person_width, lm.y * person_height) for lm in hand_landmarks.landmark]
                    })

        return len(detected_hands) > 0, detected_hands

    def detect_chest_pain(self, person_crop):
        """Chest pain detection using trained model (PRESERVED EXACTLY)"""
        if person_crop.size == 0:
            return "normal", 0.0
        
        person_crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(person_crop_rgb)
        img_tensor = self.chest_transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.chest_pain_model(img_tensor)
            probs = F.softmax(outputs, dim=1)
            pred = torch.argmax(probs, 1).item()
            confidence = probs[0][pred].item()
        
        return self.chest_pain_classes[pred], confidence

# ------------------------------
# Combined Detection System
# ------------------------------
class CombinedDetectionSystem:
    def __init__(self, fall_model_path, chest_pain_model_path):
        # Initialize both systems
        self.fall_detector = SimpleFallDetectionSystem(fall_model_path)
        self.chest_pain_detector = ChestPainDetectionSystem(chest_pain_model_path)
        
        # Initialize YOLO for person detection
        self.person_detector = YOLO('yolov8n.pt')

    def detect_persons(self, frame, confidence_threshold=0.5):
        """Detect persons in frame"""
        results = self.person_detector(frame, verbose=False, conf=confidence_threshold)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    if int(box.cls) == 0:  # Person class
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf.cpu().numpy()[0]
                        area = (x2 - x1) * (y2 - y1)
                        
                        if area > 1000:  # Minimum area threshold
                            detections.append({
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'confidence': conf
                            })
        
        return detections

    def extract_person_crop(self, frame, bbox, padding=0.1):
        """Extract person crop with padding"""
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        pad_w, pad_h = int(w * padding), int(h * padding)
        
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(frame.shape[1], x2 + pad_w)
        y2 = min(frame.shape[0], y2 + pad_h)
        
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            return crop
        return np.zeros((224, 224, 3), dtype=np.uint8)

# ------------------------------
# Unified Video Processing Function 
# ------------------------------
def process_unified_video_simple(video_path, fall_confidence=0.7, chest_pain_confidence=0.7, 
                                hand_sensitivity=0.8, person_confidence=0.5):
    
    # Initialize the combined system
    fall_model_path = "/kaggle/input/fall_detector2/pytorch/default/1/fall_detector.pth"
    chest_pain_model_path = "/kaggle/input/chestpain2/pytorch/default/1/best_chestpain_model2.pth"
    
    detector = CombinedDetectionSystem(fall_model_path, chest_pain_model_path)
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Output video
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp_out.name, fourcc, fps, (width, height))
    
    # Simple tracking variables (only for chest pain)
    chest_pain_trackers = {}
    frame_count = 0
    persistence_frames_chest = 3
    persistence_frames_fall = 5
    
    # Fall detection buffer for temporal consistency (FULL FRAME)
    fall_prediction_buffer = deque(maxlen=persistence_frames_fall)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        annotated_frame = frame.copy()
        
        # ===== FULL FRAME FALL DETECTION  =====
        fall_result = detector.fall_detector.process_frame(frame)  # FULL FRAME
        fall_prediction_buffer.append(fall_result['prediction'])
        
        # Temporal smoothing for falls - require consistent predictions
        recent_falls = sum(fall_prediction_buffer)
        is_fall_consistent = recent_falls >= 3  
        
        # Final fall decision for the entire frame
        fall_alert = (fall_result['prediction'] == 1 and 
                     fall_result['confidence'] > fall_confidence and 
                     is_fall_consistent)
        
        # Detect persons for chest pain detection only
        persons = detector.detect_persons(frame, person_confidence)
        
        # Process each detected person for chest pain only
        chest_pain_alerts = []
        for i, person in enumerate(persons):
            person_id = f"person_{i}"
            bbox = person['bbox']
            x1, y1, x2, y2 = bbox
            
            # Extract person crop for chest pain analysis
            person_crop = detector.extract_person_crop(frame, bbox, padding=0.2)
            
            # ===== CHEST PAIN DETECTION (WITH SIMPLE TRACKING) =====
            chest_pain_alert = False
            
            # Only check chest pain if no fall detected (not fallen)
            if not fall_alert and fall_result['reason']:
                # Initialize chest pain tracker if new person
                if person_id not in chest_pain_trackers:
                    chest_pain_trackers[person_id] = {
                        'buffer': deque(maxlen=persistence_frames_chest),
                        'last_seen': frame_count
                    }
                
                tracker = chest_pain_trackers[person_id]
                tracker['last_seen'] = frame_count
                
                # Check for hands near chest
                hands_detected, hand_info = detector.chest_pain_detector.detect_hands_near_chest(
                    frame, bbox, hand_sensitivity)
                
                if hands_detected:
                    chest_pred, chest_conf = detector.chest_pain_detector.detect_chest_pain(person_crop)
                    if chest_pred == "chest_pain" and chest_conf >= chest_pain_confidence:
                        tracker['buffer'].append(1)
                    else:
                        tracker['buffer'].append(0)
                else:
                    tracker['buffer'].append(0)
                
                chest_pain_alert = sum(tracker['buffer']) >= persistence_frames_chest
                
            chest_pain_alerts.append((person_id, bbox, chest_pain_alert))
        
        # ===== DETERMINE FINAL ALERT AND VISUALIZATION =====
        # Check if any person has chest pain alert
        any_chest_pain = any(alert for _, _, alert in chest_pain_alerts)
        
        # Determine overall frame status
        if any_chest_pain and fall_alert:
            # Critical: both fall and chest pain
            color = (0, 0, 255)  # Red
            label = "🚨 CRITICAL: CHEST PAIN + FALL! 🚨"
            alert_message = "EMERGENCY: Person has fallen and chest pain detected! Immediate assistance required!"
            thickness = 4
        elif fall_alert:
            # Fall detected in frame
            color = (0, 0, 255)  # Red  
            label = f"FALL DETECTED ({fall_result['confidence']*100:.1f}%)"
            alert_message = "EMERGENCY: Fall detected! Immediate assistance required!"
            thickness = 3
        elif any_chest_pain:
            # Chest pain detected
            color = (0, 255, 255)  # Yellow
            label = "🫀 CHEST PAIN ALERT!"
            alert_message = "WARNING: Possible chest pain detected! Medical attention may be needed!"
            thickness = 3
        elif fall_result['reason'] == 'SITTING_DETECTED':
            # Person sitting
            color = (0, 255, 0)  # Green
            sitting_score = fall_result.get('sitting_score', 0.0)
            label = f"SITTING ({sitting_score*100:.1f}%)"
            alert_message = ""
            thickness = 2
        else:
            # Normal state
            color = (0, 255, 0)  # Green
            conf = 1 - fall_result['confidence'] if fall_result['prediction'] == 1 else fall_result['confidence']
            label = f"NO FALL ({conf*100:.1f}%)"
            alert_message = ""
            thickness = 2
        
        # Draw main frame status (full frame border and main label)
        cv2.rectangle(annotated_frame, (10, 10), (width-10, 60), color, -1)
        cv2.putText(annotated_frame, label, (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        
        # Draw individual person bounding boxes for chest pain
        for person_id, bbox, chest_pain_alert in chest_pain_alerts:
            x1, y1, x2, y2 = bbox
            
            if chest_pain_alert:
                # Highlight person with chest pain
                person_color = (0, 255, 255)  # Yellow
                person_label = f"CHEST PAIN - {person_id}"
                person_thickness = 3
            else:
                # Normal person
                person_color = (0, 255, 0)  # Green
                person_label = f"Normal - {person_id}"
                person_thickness = 2
            
            # Draw person bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), person_color, person_thickness)
            
            # Draw person label
            person_label_size = cv2.getTextSize(person_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1-30), (x1 + person_label_size[0] + 10, y1), person_color, -1)
            cv2.putText(annotated_frame, person_label, (x1+5, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show fall detection details
        details = f"Fall:{fall_result['confidence']:.2f} Sit:{fall_result.get('sitting_score', 0):.2f}"
        cv2.putText(annotated_frame, details, (20, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show ML prediction details if available
        if fall_result.get('ml_prediction') is not None:
            ml_text = f"ML: {detector.fall_detector.classes[fall_result['ml_prediction']]} ({fall_result['ml_confidence']*100:.1f}%)"
            cv2.putText(annotated_frame, ml_text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show pose analysis
        if 'fall_score' in fall_result:
            pose_text = f"Pose Score: {fall_result['fall_score']*100:.1f}%"
            cv2.putText(annotated_frame, pose_text, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show reason
        reason_text = f"Reason: {fall_result['reason']}"
        cv2.putText(annotated_frame, reason_text, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add alert message if there's an alert
        if alert_message:
            # Flash effect for critical alerts
            if frame_count % 10 < 5:  # Flash every 10 frames
                cv2.rectangle(annotated_frame, (5, 5), (width-5, height-5), color, 8)
            
            # Display alert message at bottom of screen
            alert_y_pos = height - 40
            cv2.rectangle(annotated_frame, (10, alert_y_pos - 25), (width - 10, alert_y_pos + 5), color, -1)
            cv2.putText(annotated_frame, alert_message, (15, alert_y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Clean up old chest pain trackers
        chest_pain_trackers = {
            pid: tracker for pid, tracker in chest_pain_trackers.items() 
            if frame_count - tracker['last_seen'] < 30
        }
        
        # Add frame information
        info_text = f"Frame: {frame_count} | Persons: {len(persons)} | Active Trackers: {len(chest_pain_trackers)}"
        cv2.putText(annotated_frame, info_text, (10, height-30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Add timestamp
        timestamp = f"Time: {frame_count/fps:.1f}s"
        cv2.putText(annotated_frame, timestamp, (10, height-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        out.write(annotated_frame)
    
    cap.release()
    out.release()
    return tmp_out.name

# ------------------------------
# Gradio Interface
# ------------------------------
with gr.Blocks(title="Simplified Fall & Chest Pain Detection System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚨 Simplified Fall & Chest Pain Detection System

    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📹 Video Input")
            video_input = gr.Video(label="Upload Video for Analysis")
            
            gr.Markdown("### ⚙️ Detection Parameters")
            
            with gr.Row():
                fall_confidence = gr.Slider(
                    minimum=0.1, maximum=0.95, value=0.7, step=0.05,
                    label="🤕 Fall Detection Confidence",
                    info="Higher = stricter fall detection (recommended: 0.7)"
                )
                chest_pain_confidence = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.7, step=0.05,
                    label="🫀 Chest Pain Confidence", 
                    info="Higher = stricter chest pain detection"
                )
            
            with gr.Row():
                hand_sensitivity = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.8, step=0.05,
                    label="✋ Hand Detection Sensitivity",
                    info="Higher = more sensitive hand detection"
                )
                person_confidence = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.6, step=0.05,
                    label="👤 Person Detection Confidence",
                    info="Higher = stricter person detection"
                )
            
            gr.Markdown("### 🚀 Process Video")
            process_btn = gr.Button("🔍 Analyze Video with Simplified Detection", variant="primary", size="lg")
        
        with gr.Column():
            gr.Markdown("### 📺 Processed Output")
            video_output = gr.Video(label="Analyzed Video with Simplified Detection Results")
    
    # Connect the processing function
    process_btn.click(
        fn=process_unified_video_simple,
        inputs=[video_input, fall_confidence, chest_pain_confidence, 
                hand_sensitivity, person_confidence],
        outputs=video_output,
        show_progress=True
    )
    
    
    if __name__ == "__main__":

        demo.launch(debug=True, share=True)

