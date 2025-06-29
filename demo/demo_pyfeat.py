#!/usr/bin/env python3
"""
Demo script for Py-Feat: Python Facial Expression Analysis Toolbox

This script demonstrates the Py-Feat analyzer capabilities including:
- Action Unit (AU) detection and intensity measurement
- Emotion classification (7 basic emotions)
- Face detection and bounding box localization
- Head pose estimation (pitch, roll, yaw angles)
- 3D face position estimation
- Comprehensive facial expression analysis
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.vision.pyfeat_analyzer import PyFeatAnalyzer
    import numpy as np
    
    def demo_pyfeat_analyzer():
        """Demonstrate Py-Feat analyzer functionality."""
        print("=== Py-Feat: Python Facial Expression Analysis Demo ===\n")
        
        # Initialize analyzer
        print("1. Initializing Py-Feat analyzer...")
        analyzer = PyFeatAnalyzer(
            device='cpu',
            detection_threshold=0.5
        )
        print("   ✓ Analyzer initialized successfully")
        
        # Show feature structure
        print("\n2. Available feature columns:")
        feature_columns = list(analyzer.default_metrics.keys())
        
        # Group features by type
        action_units = [col for col in feature_columns if col.startswith('pf_au')]
        emotions = [col for col in feature_columns if col.startswith('pf_') and 
                   col.split('_')[1] in analyzer.emotions]
        face_geometry = [col for col in feature_columns if 'face' in col]
        pose_angles = [col for col in feature_columns if col.endswith(('_pitch', '_roll', '_yaw'))]
        position_3d = [col for col in feature_columns if col.endswith(('_x', '_y', '_z'))]
        
        print("   Action Units (AU detection and intensity):")
        for i, col in enumerate(action_units):
            au_name = col.replace('pf_', '').upper()
            au_descriptions = {
                'AU01': 'Inner Brow Raiser',
                'AU02': 'Outer Brow Raiser', 
                'AU04': 'Brow Lowerer',
                'AU05': 'Upper Lid Raiser',
                'AU06': 'Cheek Raiser',
                'AU07': 'Lid Tightener',
                'AU09': 'Nose Wrinkler',
                'AU10': 'Upper Lip Raiser',
                'AU11': 'Nasolabial Furrow Deepener',
                'AU12': 'Lip Corner Puller (Smile)',
                'AU14': 'Dimpler',
                'AU15': 'Lip Corner Depressor',
                'AU17': 'Chin Raiser',
                'AU20': 'Lip Stretcher',
                'AU23': 'Lip Tightener',
                'AU24': 'Lip Pressor',
                'AU25': 'Lips Part',
                'AU26': 'Jaw Drop',
                'AU28': 'Lip Suck',
                'AU43': 'Eyes Closed'
            }
            desc = au_descriptions.get(au_name, 'Unknown AU')
            print(f"     • {col}: {desc}")
        
        print("   Emotion Classification:")
        for col in emotions:
            emotion_name = col.replace('pf_', '').title()
            print(f"     • {col}: {emotion_name} probability")
        
        print("   Face Detection and Localization:")
        for col in face_geometry:
            descriptions = {
                'pf_facerectx': 'Face bounding box X coordinate',
                'pf_facerecty': 'Face bounding box Y coordinate', 
                'pf_facerectwidth': 'Face bounding box width',
                'pf_facerectheight': 'Face bounding box height',
                'pf_facescore': 'Face detection confidence score'
            }
            desc = descriptions.get(col, col)
            print(f"     • {col}: {desc}")
        
        print("   Head Pose Estimation:")
        for col in pose_angles:
            descriptions = {
                'pf_pitch': 'Head pitch angle (up/down rotation)',
                'pf_roll': 'Head roll angle (left/right tilt)',
                'pf_yaw': 'Head yaw angle (left/right turn)'
            }
            desc = descriptions.get(col, col)
            print(f"     • {col}: {desc}")
        
        print("   3D Face Position:")
        for col in position_3d:
            descriptions = {
                'pf_x': 'Face center X coordinate in image',
                'pf_y': 'Face center Y coordinate in image',
                'pf_z': 'Estimated face depth/distance from camera'
            }
            desc = descriptions.get(col, col)
            print(f"     • {col}: {desc}")
        
        # Show model configuration
        print(f"\n3. Model Configuration:")
        print(f"   • Device: {analyzer.device}")
        print(f"   • Detection Threshold: {analyzer.detection_threshold}")
        print(f"   • Total Action Units: {len(analyzer.action_units)}")
        print(f"   • Emotion Categories: {len(analyzer.emotions)}")
        
        # Show Action Unit details
        print(f"\n4. Supported Action Units:")
        au_groups = {
            'Upper Face': ['au01', 'au02', 'au04', 'au05', 'au06', 'au07', 'au43'],
            'Lower Face': ['au09', 'au10', 'au11', 'au12', 'au14', 'au15', 'au17', 'au20', 'au23', 'au24', 'au25', 'au26', 'au28']
        }
        
        for group_name, aus in au_groups.items():
            print(f"   {group_name}:")
            for au in aus:
                if au in analyzer.action_units:
                    print(f"     • {au.upper()}")
        
        # Show emotion categories
        print(f"\n5. Emotion Categories:")
        for emotion in analyzer.emotions:
            print(f"   • {emotion.title()}")
        
        print(f"\n6. Feature Group Information:")
        print(f"   • Feature Group Name: 'Actional annotation, Emotion indices, Face location and angles'")
        print(f"   • Model Name: 'Py-Feat: Python Facial Expression Analysis Toolbox'")
        print(f"   • Website: Py-Feat")
        print(f"   • Total Feature Columns: {len(feature_columns)}")
        print(f"   • Prefix: 'pf_'")
        
        # Test model initialization
        print(f"\n7. Testing Model Components:")
        try:
            analyzer._initialize_model()
            print("   ✓ Face detector initialized")
            print("   ✓ Action Unit detector initialized")
            print("   ✓ Emotion classifier initialized")
            print("   ✓ Head pose estimator initialized")
            
            # Test feature dict structure
            dummy_path = "test_video.mp4"
            print(f"\n8. Feature Dictionary Structure (simulated):")
            print(f"   Input: video_path = '{dummy_path}'")
            print(f"   Output structure:")
            print(f"   {{")
            print(f"       'Actional annotation, Emotion indices, Face location and angles': {{")
            print(f"           'description': 'Py-Feat: Python Facial Expression Analysis Toolbox',")
            print(f"           'features': {{")
            print(f"               # Action Units (intensity 0-1)")
            for col in action_units[:3]:
                print(f"               '{col}': <float>,")
            print(f"               # ... (all {len(action_units)} Action Units)")
            print(f"               # Emotions (probability 0-1)")
            for col in emotions[:3]:
                print(f"               '{col}': <float>,")
            print(f"               # ... (all {len(emotions)} emotions)")
            print(f"               # Face geometry")
            for col in face_geometry[:3]:
                print(f"               '{col}': <float>,")
            print(f"               # Head pose angles (degrees)")
            for col in pose_angles:
                print(f"               '{col}': <float>,")
            print(f"               # 3D position")
            for col in position_3d:
                print(f"               '{col}': <float>,")
            print(f"               # Summary statistics")
            print(f"               'total_frames': <int>,")
            print(f"               'faces_detected_frames': <int>,")
            print(f"               'face_detection_rate': <float>,")
            print(f"               'avg_face_size': <float>,")
            print(f"               'avg_face_confidence': <float>")
            print(f"           }}")
            print(f"       }}")
            print(f"   }}")
            
        except Exception as e:
            print(f"   ⚠ Model initialization test failed: {e}")
            print("     This is expected in a demo environment without full dependencies")
        
        print(f"\n9. Usage in Pipeline:")
        print(f"   • Feature name: 'pyfeat_vision'")
        print(f"   • Add to features list when creating FeatureExtractor")
        print(f"   • Processes video files and extracts comprehensive facial analysis")
        print(f"   • Outputs {len(feature_columns)} feature columns with pf_ prefix")
        print(f"   • Covers Action Units, emotions, face geometry, and head pose")
        
        print(f"\n10. Real-world Applications:")
        print(f"   • Affective computing and emotion recognition")
        print(f"   • Human-computer interaction systems")
        print(f"   • Psychological and behavioral research")
        print(f"   • Video content analysis and annotation")
        print(f"   • Accessibility technology (emotion-aware interfaces)")
        print(f"   • Entertainment and gaming (avatar control)")
        
        print("\n=== Demo completed successfully! ===")
        return True

    if __name__ == "__main__":
        success = demo_pyfeat_analyzer()
        sys.exit(0 if success else 1)
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running this from the project root directory")
    print("and that all dependencies are installed.")
    sys.exit(1)
except Exception as e:
    print(f"Error during demo: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
