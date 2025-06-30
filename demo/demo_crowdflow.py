#!/usr/bin/env python3
"""
Demo script for CrowdFlow (Optical flow fields, Person trajectories, Tracking accuracy) integration.

This script demonstrates:
1. Foreground/background separation for crowd analysis
2. Short-term optical flow metrics (EPE, R²) for static/dynamic regions
3. Long-term tracking accuracy with multiple interpolation methods
4. Person trajectory analysis with motion patterns
5. Pipeline integration and feature extraction

Usage:
    python demo/demo_crowdflow.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vision.crowdflow_analyzer import CrowdFlowAnalyzer
from src.pipeline import MultimodalPipeline


def create_crowd_test_video(output_path: str, duration: float = 3.0, fps: int = 30):
    """
    Create a test video with simulated crowd movement for demonstration.
    
    Args:
        output_path: Path to save the test video
        duration: Duration of the video in seconds
        fps: Frames per second
    """
    print(f"Creating test video with crowd movement: {output_path}")
    
    # Video properties
    width, height = 640, 480
    total_frames = int(duration * fps)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create multiple moving objects to simulate a crowd
    num_people = 8
    people_positions = np.random.rand(num_people, 2) * np.array([width, height])
    people_velocities = (np.random.rand(num_people, 2) - 0.5) * 4
    people_colors = [(int(np.random.rand() * 255), int(np.random.rand() * 255), int(np.random.rand() * 255)) 
                     for _ in range(num_people)]
    
    for frame_idx in range(total_frames):
        # Create background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 50
        
        # Add some static background elements
        cv2.rectangle(frame, (50, 50), (150, 100), (100, 100, 100), -1)
        cv2.rectangle(frame, (width-150, height-100), (width-50, height-50), (80, 80, 80), -1)
        
        # Add dynamic crowd elements
        for i in range(num_people):
            # Update positions
            people_positions[i] += people_velocities[i]
            
            # Bounce off walls
            if people_positions[i][0] <= 20 or people_positions[i][0] >= width - 20:
                people_velocities[i][0] *= -1
            if people_positions[i][1] <= 20 or people_positions[i][1] >= height - 20:
                people_velocities[i][1] *= -1
            
            # Keep within bounds
            people_positions[i][0] = np.clip(people_positions[i][0], 20, width - 20)
            people_positions[i][1] = np.clip(people_positions[i][1], 20, height - 20)
            
            # Draw person as circle
            center = (int(people_positions[i][0]), int(people_positions[i][1]))
            cv2.circle(frame, center, 15, people_colors[i], -1)
            
            # Add motion blur effect
            if frame_idx > 0:
                prev_center = (int(people_positions[i][0] - people_velocities[i][0]), 
                              int(people_positions[i][1] - people_velocities[i][1]))
                cv2.line(frame, prev_center, center, people_colors[i], 8)
        
        # Add timestamp
        cv2.putText(frame, f"Frame {frame_idx:03d}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Test video created successfully: {output_path}")


def demo_direct_analyzer():
    """Demonstrate direct usage of CrowdFlowAnalyzer."""
    print("\\n" + "="*50)
    print("DEMO 1: Direct CrowdFlow Analyzer Usage")
    print("="*50)
    
    # Create test video
    test_video_path = "test_crowd_video.mp4"
    create_crowd_test_video(test_video_path, duration=2.0)
    
    try:
        # Initialize analyzer
        analyzer = CrowdFlowAnalyzer(device='cpu')
        
        # Analyze video
        print(f"\\nAnalyzing crowd video: {test_video_path}")
        results = analyzer.analyze_video(test_video_path)
        
        # Display results
        print(f"\\nCrowdFlow Analysis Results:")
        print(f"- Video path: {results.get('video_path', 'N/A')}")
        print(f"- Frames analyzed: {results.get('total_frames_analyzed', 0)}")
        print(f"- Analysis type: {results.get('analysis_type', 'N/A')}")
        
        # Short-term optical flow metrics
        print(f"\\nShort-term Optical Flow Metrics:")
        print(f"- Foreground static EPE: {results.get('of_fg_static_epe_st', 0.0):.3f}")
        print(f"- Foreground dynamic EPE: {results.get('of_fg_dynamic_epe_st', 0.0):.3f}")
        print(f"- Background static EPE: {results.get('of_bg_static_epe_st', 0.0):.3f}")
        print(f"- Background dynamic EPE: {results.get('of_bg_dynamic_epe_st', 0.0):.3f}")
        print(f"- Average EPE: {results.get('of_avg_epe_st', 0.0):.3f}")
        print(f"- Average R²: {results.get('of_avg_r2_st', 0.0):.3f}")
        print(f"- Time length: {results.get('of_time_length_st', 0.0):.1f}")
        
        # Tracking accuracy metrics
        print(f"\\nTracking Accuracy Metrics:")
        print(f"- IM01 tracking accuracy: {results.get('of_ta_IM01', 0.0):.3f}")
        print(f"- IM01 dynamic tracking: {results.get('of_ta_IM01_Dyn', 0.0):.3f}")
        print(f"- IM02 tracking accuracy: {results.get('of_ta_IM02', 0.0):.3f}")
        print(f"- IM03 tracking accuracy: {results.get('of_ta_IM03', 0.0):.3f}")
        print(f"- Average tracking accuracy: {results.get('of_ta_average', 0.0):.3f}")
        
        # Person trajectory metrics
        print(f"\\nPerson Trajectory Metrics:")
        print(f"- IM01 person trajectory: {results.get('of_pt_IM01', 0.0):.3f}")
        print(f"- IM01 dynamic trajectory: {results.get('of_pt_IM01_Dyn', 0.0):.3f}")
        print(f"- IM02 person trajectory: {results.get('of_pt_IM02', 0.0):.3f}")
        print(f"- IM03 person trajectory: {results.get('of_pt_IM03', 0.0):.3f}")
        print(f"- Average person trajectory: {results.get('of_pt_average', 0.0):.3f}")
        
        print(f"\\nDirect analyzer demo completed successfully!")
        
    except Exception as e:
        print(f"Error in direct analyzer demo: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if os.path.exists(test_video_path):
            os.remove(test_video_path)


def demo_pipeline_integration():
    """Demonstrate CrowdFlow integration with the multimodal pipeline."""
    print("\\n" + "="*50)
    print("DEMO 2: CrowdFlow Pipeline Integration")
    print("="*50)
    
    # Create test video
    test_video_path = "test_crowd_pipeline.mp4"
    create_crowd_test_video(test_video_path, duration=2.0)
    
    try:
        # Initialize pipeline with only CrowdFlow feature
        pipeline = MultimodalPipeline(
            features=["crowdflow_vision"],
            device='cpu'
        )
        
        # Process video
        print(f"\\nProcessing video through pipeline: {test_video_path}")
        results = pipeline.process_file(test_video_path)
        
        # Display results
        print(f"\\nPipeline Results:")
        print(f"- Total feature groups: {len(results)}")
        
        # Check if CrowdFlow results are present
        crowd_group = "Optical flow fields, Person trajectories, Tracking accuracy"
        if crowd_group in results:
            crowd_results = results[crowd_group]
            print(f"\\nCrowdFlow Feature Group: {crowd_group}")
            print(f"- Description: {crowd_results.get('description', 'N/A')}")
            
            features = crowd_results.get('features', {})
            print(f"- Number of features: {len(features)}")
            
            # Display key metrics
            if features:
                print(f"\\nKey CrowdFlow Metrics:")
                print(f"- Foreground avg EPE: {features.get('of_fg_avg_epe_st', 0.0):.3f}")
                print(f"- Background avg EPE: {features.get('of_bg_avg_epe_st', 0.0):.3f}")
                print(f"- Overall avg EPE: {features.get('of_avg_epe_st', 0.0):.3f}")
                print(f"- Avg tracking accuracy: {features.get('of_ta_average', 0.0):.3f}")
                print(f"- Avg person trajectory: {features.get('of_pt_average', 0.0):.3f}")
        else:
            print(f"WARNING: CrowdFlow feature group not found in results")
            print(f"Available groups: {list(results.keys())}")
        
        print(f"\\nPipeline integration demo completed successfully!")
        
    except Exception as e:
        print(f"Error in pipeline integration demo: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if os.path.exists(test_video_path):
            os.remove(test_video_path)


def demo_multi_feature_analysis():
    """Demonstrate CrowdFlow analysis alongside other vision features."""
    print("\\n" + "="*50)
    print("DEMO 3: CrowdFlow Multi-Feature Analysis")
    print("="*50)
    
    # Create test video
    test_video_path = "test_crowd_multi.mp4"
    create_crowd_test_video(test_video_path, duration=2.0)
    
    try:
        # Initialize pipeline with CrowdFlow and basic optical flow
        pipeline = MultimodalPipeline(
            features=["crowdflow_vision", "optical_flow_vision"],
            device='cpu'
        )
        
        # Process video
        print(f"\\nProcessing video with multiple vision features: {test_video_path}")
        results = pipeline.process_file(test_video_path)
        
        # Display results comparison
        print(f"\\nMulti-Feature Analysis Results:")
        print(f"- Total feature groups: {len(results)}")
        
        # Compare CrowdFlow vs basic optical flow
        crowd_group = "Optical flow fields, Person trajectories, Tracking accuracy"
        basic_flow_group = "Movement and estimation of motion"
        
        if crowd_group in results and basic_flow_group in results:
            print(f"\\nFeature Comparison:")
            
            crowd_features = results[crowd_group].get('features', {})
            basic_features = results[basic_flow_group].get('features', {})
            
            print(f"\\nCrowdFlow Features ({len(crowd_features)} total):")
            print(f"- Short-term metrics: EPE, R², time length")
            print(f"- Tracking accuracy: 5 interpolation methods")
            print(f"- Person trajectories: 5 interpolation methods") 
            print(f"- Foreground/background separation")
            
            print(f"\\nBasic Optical Flow Features ({len(basic_features)} total):")
            print(f"- Motion magnitude and direction")
            print(f"- Sparse and dense flow visualizations")
            print(f"- Basic motion detection")
            
            print(f"\\nCrowdFlow provides more detailed analysis for:")
            print(f"- Crowd behavior understanding")
            print(f"- Person tracking in complex scenes")
            print(f"- Foreground/background motion separation")
            print(f"- Long-term trajectory analysis")
        
        print(f"\\nMulti-feature analysis completed successfully!")
        
    except Exception as e:
        print(f"Error in multi-feature analysis demo: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if os.path.exists(test_video_path):
            os.remove(test_video_path)


def main():
    """Run all CrowdFlow demonstrations."""
    print("CrowdFlow (Optical flow fields, Person trajectories, Tracking accuracy) Demonstration")
    print("This demo showcases advanced crowd analysis with foreground/background separation")
    print("and comprehensive tracking metrics for visual crowd analysis applications.")
    
    try:
        # Run all demos
        demo_direct_analyzer()
        demo_pipeline_integration()
        demo_multi_feature_analysis()
        
        print("\\n" + "="*50)
        print("ALL CROWDFLOW DEMOS COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("\\nCrowdFlow provides comprehensive optical flow analysis including:")
        print("- Foreground/background separation for crowd scenes")
        print("- Short-term optical flow metrics (EPE, R²) for static/dynamic regions")
        print("- Long-term tracking accuracy with 5 interpolation methods")
        print("- Person trajectory analysis with motion pattern detection")
        print("- Integration with the multimodal pipeline for comprehensive analysis")
        
    except Exception as e:
        print(f"\\nERROR in CrowdFlow demo: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
