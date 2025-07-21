#!/usr/bin/env python3
"""
Demo script for LaneGCN autonomous driving motion forecasting.

This script demonstrates the LaneGCN analyzer for learning lane graph representations
and motion forecasting in autonomous driving scenarios.

Features:
- Autonomous driving motion forecasting with graph convolution networks
- Lane graph representation learning and trajectory prediction
- Actor-lane and lane-lane interaction modeling
- Single-mode (K=1) and multi-mode (K=6) trajectory prediction metrics
- Quantitative evaluation with ADE, FDE, and Miss Rate metrics

Example usage:
    python demo/demo_lanegcn.py --video_path path/to/driving_video.mp4
    python demo/demo_lanegcn.py --video_path path/to/traffic_scene.mp4 --output_dir output/lanegcn_demo
"""

import sys
import argparse
from pathlib import Path

# Add the src directory to Python path for importing modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.vision.lanegcn_analyzer import LaneGCNAnalyzer


def main():
    parser = argparse.ArgumentParser(description="Demo for LaneGCN autonomous driving motion forecasting")
    parser.add_argument("--video_path", type=str, required=True, 
                       help="Path to the input video file (driving/traffic scene)")
    parser.add_argument("--output_dir", type=str, default="output/lanegcn_demo",
                       help="Directory to save output files")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                       help="Device to run the model on")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚗 LaneGCN Autonomous Driving Motion Forecasting Demo")
    print("=" * 60)
    print(f"📹 Video: {args.video_path}")
    print(f"💾 Output: {args.output_dir}")
    print(f"🖥️  Device: {args.device}")
    print()
    
    try:
        # Initialize LaneGCN analyzer
        print("🔧 Initializing LaneGCN analyzer...")
        analyzer = LaneGCNAnalyzer(device=args.device)
        
        # Extract features
        print("🚀 Processing video for autonomous driving motion forecasting...")
        features = analyzer.get_feature_dict(args.video_path)
        
        print("\n📊 LaneGCN Motion Forecasting Results:")
        print("-" * 50)
        
        # Display K=1 metrics (single-mode prediction)
        print("\n🎯 Single-Mode Trajectory Prediction (K=1):")
        if "GCN_min_ade_k1" in features:
            print(f"  • Minimum ADE (K=1): {features['GCN_min_ade_k1']:.4f}")
        if "GCN_min_fde_k1" in features:
            print(f"  • Minimum FDE (K=1): {features['GCN_min_fde_k1']:.4f}")
        if "GCN_MR_k1" in features:
            print(f"  • Miss Rate (K=1): {features['GCN_MR_k1']:.4f}")
        
        # Display K=6 metrics (multi-mode prediction)
        print("\n🎯 Multi-Mode Trajectory Prediction (K=6):")
        if "GCN_min_ade_k6" in features:
            print(f"  • Minimum ADE (K=6): {features['GCN_min_ade_k6']:.4f}")
        if "GCN_min_fde_k6" in features:
            print(f"  • Minimum FDE (K=6): {features['GCN_min_fde_k6']:.4f}")
        if "GCN_MR_k6" in features:
            print(f"  • Miss Rate (K=6): {features['GCN_MR_k6']:.4f}")
        
        # Feature interpretation
        print("\n📖 Metric Interpretation:")
        print("  • ADE (Average Displacement Error): Average L2 distance between predicted and ground truth positions")
        print("  • FDE (Final Displacement Error): L2 distance at the final predicted time step")
        print("  • Miss Rate (MR): Percentage of predictions with FDE > 2.0 meters")
        print("  • K=1: Single most likely trajectory prediction")
        print("  • K=6: Best of 6 predicted trajectory modes")
        print("  • Lower values indicate better prediction accuracy")
        
        # Lane graph analysis
        print(f"\n🛣️  Lane Graph Analysis:")
        print("  • Lane topology understanding and road structure modeling")
        print("  • Actor-lane interaction analysis for context-aware prediction")
        print("  • Lane-lane relationship modeling for traffic flow understanding")
        print("  • Graph convolution networks for spatial-temporal reasoning")
        
        # Save results
        import json
        output_file = output_dir / "lanegcn_results.json"
        with open(output_file, 'w') as f:
            json.dump(features, f, indent=2)
        print(f"\n💾 Results saved to: {output_file}")
        
        print(f"\n✅ LaneGCN motion forecasting analysis completed!")
        print(f"📈 Total features extracted: {len(features)}")
        
        # Performance summary
        if all(key in features for key in ["GCN_min_ade_k1", "GCN_min_fde_k1", "GCN_MR_k1"]):
            print(f"\n🏆 Performance Summary:")
            print(f"  • Single-mode accuracy: ADE={features['GCN_min_ade_k1']:.3f}, FDE={features['GCN_min_fde_k1']:.3f}")
            if features.get('GCN_MR_k1', 1.0) < 0.2:
                print(f"  • 🟢 Excellent prediction accuracy (Miss Rate < 20%)")
            elif features.get('GCN_MR_k1', 1.0) < 0.5:
                print(f"  • 🟡 Good prediction accuracy (Miss Rate < 50%)")
            else:
                print(f"  • 🔴 Challenging scenario (Miss Rate ≥ 50%)")
        
    except Exception as e:
        print(f"❌ Error during LaneGCN analysis: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
