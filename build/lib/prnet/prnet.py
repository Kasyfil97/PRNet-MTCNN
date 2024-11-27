import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BoundingBox:
    """Data class to store face bounding box information."""
    left: int
    top: int
    right: int
    bottom: int
    
    @classmethod
    def from_detection(cls, detection: Dict[str, Any]) -> 'BoundingBox':
        """Create BoundingBox from detection dictionary."""
        box = detection['box']
        return cls(
            left=box[0],
            top=box[1],
            right=box[0] + box[2],
            bottom=box[1] + box[3]
        )

class ImageProcessingError(Exception):
    """Custom exception for image processing errors."""
    pass

class PRNFeatures:
    """
    Class for processing facial features using PRN (Position Regression Network).
    
    This class provides methods for facial landmark detection, face alignment,
    and depth map generation using PRN.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize PRN Features processor.
        
        Args:
            model_path: Optional path to PRN model file. If None, uses default.
        
        Raises:
            RuntimeError: If PRN initialization fails.
        """
        try:
            from .api import PRN
            from .utils.render_app import get_depth_image
            self.get_depth_image = get_depth_image
            self.prn = PRN(model_path)
            logger.info("PRN Features initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PRN: {str(e)}")
            raise RuntimeError(f"PRN initialization failed: {str(e)}")

    def validate_image(self, image: np.ndarray) -> None:
        """
        Validate input image.
        
        Args:
            image: Input image as numpy array
            
        Raises:
            ImageProcessingError: If image is invalid
        """
        if image is None or not isinstance(image, np.ndarray):
            raise ImageProcessingError("Invalid image input")
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ImageProcessingError("Image must be 3-channel BGR")
        if not image.size:
            raise ImageProcessingError("Empty image")

    def get_landmark(self, image: np.ndarray) -> np.ndarray:
        """
        Get facial landmarks from image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            numpy.ndarray: Array of landmark coordinates
            
        Raises:
            ImageProcessingError: If landmark detection fails
        """
        try:
            self.validate_image(image)
            pos = self.prn.process(image)
            if pos is None:
                raise ImageProcessingError("Failed to process image with PRN")
            
            landmarks = self.prn.get_landmarks(pos)
            if landmarks is None:
                raise ImageProcessingError("No landmarks detected")
                
            return landmarks
            
        except Exception as e:
            logger.error(f"Landmark detection failed: {str(e)}")
            raise ImageProcessingError(f"Landmark detection failed: {str(e)}")

    def face_crop(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Crop face region from image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            numpy.ndarray or None: Cropped face image if face detected, None otherwise
            
        Raises:
            ImageProcessingError: If image processing fails
        """
        try:
            self.validate_image(image)
            detections = self.prn.face_detect(image)
            
            if not detections:
                logger.warning("No face detected in image")
                return None
                
            box = BoundingBox.from_detection(detections[0])
            img_crop = image[box.top:box.bottom, box.left:box.right].copy()
            
            if img_crop.size == 0:
                raise ImageProcessingError("Invalid crop region")
                
            return img_crop
            
        except Exception as e:
            logger.error(f"Face cropping failed: {str(e)}")
            raise ImageProcessingError(f"Face cropping failed: {str(e)}")

    @staticmethod
    def align_face(
        image: np.ndarray,
        landmarks: np.ndarray,
        desired_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Align face based on facial landmarks.
        
        Args:
            image: Input image as numpy array
            landmarks: Array of landmark coordinates (68, 3)
            desired_size: Desired output image size (width, height)
            
        Returns:
            numpy.ndarray: Aligned face image
            
        Raises:
            ImageProcessingError: If alignment fails
        """
        try:
            if landmarks.shape != (68, 3):
                raise ImageProcessingError("Invalid landmarks shape")
                
            landmarks_2d = landmarks[:, :2].astype(np.float32)
            
            # Get eye coordinates
            left_eye = landmarks_2d[36:42]
            right_eye = landmarks_2d[42:48]
            
            # Calculate eye centers
            left_eye_center = left_eye.mean(axis=0)
            right_eye_center = right_eye.mean(axis=0)
            
            # Calculate angle for alignment
            dy = right_eye_center[1] - left_eye_center[1]
            dx = right_eye_center[0] - left_eye_center[0]
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Calculate center point between eyes
            eye_center = (
                int((left_eye_center[0] + right_eye_center[0]) / 2),
                int((left_eye_center[1] + right_eye_center[1]) / 2)
            )
            
            # Calculate scaling factor
            desired_eye_distance = desired_size[0] * 0.3
            eye_distance = np.sqrt((dx ** 2) + (dy ** 2))
            scale = desired_eye_distance / eye_distance
            
            # Create transformation matrix
            rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale)
            
            # Adjust translation
            tx = desired_size[0] * 0.5
            ty = desired_size[1] * 0.35
            rotation_matrix[0, 2] += (tx - eye_center[0])
            rotation_matrix[1, 2] += (ty - eye_center[1])
            
            # Apply transformation
            aligned_face = cv2.warpAffine(
                image,
                rotation_matrix,
                desired_size,
                flags=cv2.INTER_CUBIC
            )
            
            return aligned_face
            
        except Exception as e:
            logger.error(f"Face alignment failed: {str(e)}")
            raise ImageProcessingError(f"Face alignment failed: {str(e)}")

    def face_alignment(
        self,
        image: np.ndarray,
        desired_size: Tuple[int, int] = (256, 256)
    ) -> np.ndarray:
        """
        Detect landmarks and align face in one step.
        
        Args:
            image: Input image as numpy array
            desired_size: Desired output image size (width, height)
            
        Returns:
            numpy.ndarray: Aligned face image
            
        Raises:
            ImageProcessingError: If alignment process fails
        """
        try:
            landmarks = self.get_landmark(image)
            aligned_face = self.align_face(image, landmarks, desired_size)
            return aligned_face
        except Exception as e:
            logger.error(f"Face alignment pipeline failed: {str(e)}")
            raise ImageProcessingError(f"Face alignment pipeline failed: {str(e)}")

    def get_depth_map(
        self,
        image: np.ndarray,
        shape: int = 450
    ) -> np.ndarray:
        """
        Generate depth map from face image.
        
        Args:
            image: Input image as numpy array
            shape: Output shape of depth map
            
        Returns:
            numpy.ndarray: Generated depth map
            
        Raises:
            ImageProcessingError: If depth map generation fails
        """
        try:
            self.validate_image(image)
            pos = self.prn.process(image)
            if pos is None:
                raise ImageProcessingError("Failed to process image for depth map")
                
            vertices = self.prn.get_vertices(pos)
            if vertices is None:
                raise ImageProcessingError("Failed to get vertices")
                
            depth = self.get_depth_image(
                vertices,
                self.prn.triangles,
                shape,
                shape
            )
            return depth
            
        except Exception as e:
            logger.error(f"Depth map generation failed: {str(e)}")
            raise ImageProcessingError(f"Depth map generation failed: {str(e)}")