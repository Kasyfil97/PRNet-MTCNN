import numpy as np
import os
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict, Any
from dataclasses import dataclass
from skimage.io import imread
from skimage.transform import estimate_transform, warp
import logging
from mtcnn.mtcnn import MTCNN
from prnet.network.model import PRNet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ImageInfo:
    """Data class for face bounding box or keypoint information."""
    points: np.ndarray
    is_keypoints: bool = False

    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Calculate bounds from points."""
        if self.is_keypoints:
            kpt = self.points
            if kpt.shape[0] > 3:
                kpt = kpt.T
            left = float(np.min(kpt[0, :]))
            right = float(np.max(kpt[0, :]))
            top = float(np.min(kpt[1, :]))
            bottom = float(np.max(kpt[1, :]))
        else:
            bbox = self.points
            left, right, top, bottom = [float(x) for x in bbox]
        return left, right, top, bottom

class PRN:
    """Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network."""

    def __init__(self, is_mtcnn: bool = True, prefix: Optional[Union[str, Path]] = None):
        """
        Initialize PRN model and load necessary weights and data.
        
        Args:
            is_mtcnn: Whether to use MTCNN for face detection
            prefix: Path to model weights and UV data
            
        Raises:
            FileNotFoundError: If required model files are not found
            RuntimeError: If initialization fails
        """
        try:
            self.resolution_inp = 256
            self.resolution_op = 256

            # Initialize face detector
            if is_mtcnn:
                self.face_detector = MTCNN()
                logger.info("MTCNN face detector initialized")

            # Set up paths
            if not prefix:
                prefix = Path(os.path.dirname(os.path.abspath(__file__)))
            else:
                prefix = Path(prefix)

            # Initialize PRN model
            self.pos_predictor = PRNet(self.resolution_inp, self.resolution_op)
            prn_path = prefix / 'Weights' / 'net-data' / '256_256_resfcn256_weight'
            if not prn_path.exists():
                raise FileNotFoundError("PRN trained model not found. Please download it first.")
            self.pos_predictor.restore(str(prn_path))
            logger.info("PRN model loaded successfully")

            # Load UV data
            self._load_uv_data(prefix)
            self.uv_coords = self.generate_uv_coords()
            logger.info("UV data loaded successfully")

        except Exception as e:
            logger.error(f"PRN initialization failed: {str(e)}")
            raise RuntimeError(f"Failed to initialize PRN: {str(e)}")

    def _load_uv_data(self, prefix: Path) -> None:
        """Load UV mapping data from files."""
        try:
            # Load landmark indices
            uv_kpt_path = prefix / 'Weights' / 'uv-data' / 'uv_kpt_ind.txt'
            self.uv_kpt_ind = np.loadtxt(str(uv_kpt_path)).astype(np.int32)

            # Load face indices
            face_ind_path = prefix / 'Weights' / 'uv-data' / 'face_ind.txt'
            self.face_ind = np.loadtxt(str(face_ind_path)).astype(np.int32)

            # Load triangles
            triangles_path = prefix / 'Weights' / 'uv-data' / 'triangles.txt'
            self.triangles = np.loadtxt(str(triangles_path)).astype(np.int32)

        except Exception as e:
            logger.error(f"Failed to load UV data: {str(e)}")
            raise

    def generate_uv_coords(self) -> np.ndarray:
        """Generate UV coordinates for the face model."""
        try:
            resolution = self.resolution_op
            uv_coords = np.meshgrid(range(resolution), range(resolution))
            uv_coords = np.transpose(np.array(uv_coords), [1, 2, 0])
            uv_coords = np.reshape(uv_coords, [resolution**2, -1])
            uv_coords = uv_coords[self.face_ind, :]
            uv_coords = np.hstack((uv_coords[:,:2], np.zeros([uv_coords.shape[0], 1])))
            return uv_coords
        except Exception as e:
            logger.error(f"Failed to generate UV coordinates: {str(e)}")
            raise

    def face_detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in the image using MTCNN.
        
        Args:
            image: Input image array
            
        Returns:
            List of detected face information dictionaries
        """
        try:
            return self.face_detector.detect_faces(image)
        except Exception as e:
            logger.error(f"Face detection failed: {str(e)}")
            raise

    def net_forward(self, image: np.ndarray) -> np.ndarray:
        """
        Process image through the PRN network.
        
        Args:
            image: Normalized image array (256,256,3), value range: 0~1
            
        Returns:
            3D position map array (256, 256, 3)
        """
        try:
            return self.pos_predictor.predict(image)
        except Exception as e:
            logger.error(f"Network forward pass failed: {str(e)}")
            raise

    def process(
        self,
        input_img: Union[str, np.ndarray],
        image_info: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Process image with cropping operation.
        
        Args:
            input_img: Image path or array (h,w,3), value range: 1~255
            image_info: Optional bounding box or keypoint information
            
        Returns:
            3D position map (256, 256, 3) or None if processing fails
        """
        try:
            # Load and validate image
            if isinstance(input_img, str):
                try:
                    image = imread(input_img)
                except IOError:
                    logger.error(f"Error opening file: {input_img}")
                    return None
            else:
                image = input_img

            # Handle grayscale images
            if image.ndim < 3:
                image = np.tile(image[:,:,np.newaxis], [1,1,3])

            # Get face bounds
            if image_info is not None:
                info = ImageInfo(image_info, np.max(image_info.shape) > 4)
                left, right, top, bottom = info.get_bounds()
                old_size = (right - left + bottom - top) / 2
                center = np.array([
                    right - (right - left) / 2.0,
                    bottom - (bottom - top) / 2.0
                ])
                size = int(old_size * 1.6)
            else:
                detected_faces = self.face_detect(image)
                if not detected_faces:
                    logger.warning("No face detected in image")
                    return None

                d = detected_faces[0]
                old_size = (d['box'][2] + d['box'][3]) / 2
                center = np.array([
                    d['box'][0] + d['box'][2] / 2.0,
                    d['box'][1] + d['box'][3] / 2.0
                ])
                size = int(old_size * 1.58)

            # Prepare transformation points
            src_pts = np.array([
                [center[0]-size/2, center[1]-size/2],
                [center[0]-size/2, center[1]+size/2],
                [center[0]+size/2, center[1]-size/2]
            ])
            dst_pts = np.array([
                [0, 0],
                [0, self.resolution_inp - 1],
                [self.resolution_inp - 1, 0]
            ])

            # Apply transformations
            tform = estimate_transform('similarity', src_pts, dst_pts)
            image = image / 255.
            cropped_image = warp(
                image,
                tform.inverse,
                output_shape=(self.resolution_inp, self.resolution_inp)
            )

            # Get position map
            cropped_pos = self.net_forward(cropped_image)

            # Restore original coordinates
            cropped_vertices = np.reshape(cropped_pos, [-1, 3]).T
            z = cropped_vertices[2,:].copy() / tform.params[0,0]
            cropped_vertices[2,:] = 1
            vertices = np.dot(np.linalg.inv(tform.params), cropped_vertices)
            vertices = np.vstack((vertices[:2,:], z))
            pos = np.reshape(vertices.T, [self.resolution_op, self.resolution_op, 3])

            return pos

        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            return None

    def get_landmarks(self, pos: np.ndarray) -> np.ndarray:
        """
        Extract 68 3D landmarks from position map.
        
        Args:
            pos: 3D position map (256, 256, 3)
            
        Returns:
            68 3D landmarks (68, 3)
        """
        try:
            return pos[self.uv_kpt_ind[1,:], self.uv_kpt_ind[0,:], :]
        except Exception as e:
            logger.error(f"Landmark extraction failed: {str(e)}")
            raise

    def get_vertices(self, pos: np.ndarray) -> np.ndarray:
        """
        Extract vertices from position map.
        
        Args:
            pos: 3D position map (256, 256, 3)
            
        Returns:
            Vertices point cloud (~40K, 3)
        """
        try:
            all_vertices = np.reshape(pos, [self.resolution_op**2, -1])
            return all_vertices[self.face_ind, :]
        except Exception as e:
            logger.error(f"Vertex extraction failed: {str(e)}")
            raise

    def get_colors_from_texture(self, texture: np.ndarray) -> np.ndarray:
        """
        Get vertex colors from texture map.
        
        Args:
            texture: Texture map (256, 256, 3)
            
        Returns:
            Vertex colors (~45128, 3)
        """
        try:
            all_colors = np.reshape(texture, [self.resolution_op**2, -1])
            return all_colors[self.face_ind, :]
        except Exception as e:
            logger.error(f"Color extraction from texture failed: {str(e)}")
            raise

    def get_colors(self, image: np.ndarray, vertices: np.ndarray) -> np.ndarray:
        """
        Get vertex colors from original image.
        
        Args:
            image: Original image
            vertices: 3D vertices
            
        Returns:
            Vertex colors
        """
        try:
            h, w, _ = image.shape
            vertices = vertices.copy()
            vertices[:,0] = np.minimum(np.maximum(vertices[:,0], 0), w - 1)
            vertices[:,1] = np.minimum(np.maximum(vertices[:,1], 0), h - 1)
            ind = np.round(vertices).astype(np.int32)
            return image[ind[:,1], ind[:,0], :]
        except Exception as e:
            logger.error(f"Color extraction from image failed: {str(e)}")
            raise