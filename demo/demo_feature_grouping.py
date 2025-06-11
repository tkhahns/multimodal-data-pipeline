#!/usr/bin/env python3
"""
Demo script showing the new feature grouping functionality.
"""

from ..src.pipeline import MultimodalPipeline
import json

def main():
    print("🎯 Feature Grouping by Model Categories Demo")
    print("=" * 55)
    
    # Create pipeline with all features
    pipeline = MultimodalPipeline(
        output_dir='output/feature_grouping_demo',
        features=['basic_audio', 'librosa_spectral', 'opensmile', 'speech_emotion', 'heinsen_sentiment', 'whisperx_transcription', 'deberta_text'],
        device='cpu'
    )
    
    print("📹 Processing video file with complete feature extraction...")
    results = pipeline.process_directory('data/', is_video=True)
    
    # Load the grouped JSON structure
    with open('output/feature_grouping_demo/pipeline_features.json', 'r') as f:
        grouped_json = json.load(f)
    
    print("\n🗂️  GROUPED FEATURE STRUCTURE")
    print("=" * 40)
    
    for filename, file_data in grouped_json.items():
        print(f"\n📁 File: {filename}")
        print("-" * 30)
        
        total_features = 0
        group_count = 0
        
        for group_name, group_data in file_data.items():
            if isinstance(group_data, dict) and 'Feature' in group_data:
                feature_count = len(group_data.get('features', {}))
                total_features += feature_count
                group_count += 1
                
                print(f"\n📊 Feature Category: \"{group_data['Feature']}\"")
                print(f"   🏷️  Model: {group_data['Model']}")
                print(f"   🔢 Feature Count: {feature_count}")
                
                # Show a few example features
                sample_features = list(group_data.get('features', {}).keys())[:5]
                print(f"   📝 Sample Features:")
                for feature in sample_features:
                    print(f"      • {feature}")
                if feature_count > 5:
                    print(f"      ... and {feature_count - 5} more")
            elif group_name == 'metadata':
                print(f"\n📋 Metadata: File information and extraction details")
        
        print(f"\n📈 Summary: {group_count} feature groups, {total_features} total features")
    
    print("\n✅ VERIFICATION AGAINST SPECIFICATION")
    print("=" * 40)
    
    # Expected groups from your specification table
    expected_groups = {
        "Audio volume": "OpenCV",
        "Change in audio volume": "OpenCV", 
        "Average audio pitch": "OpenCV",
        "Change in audio pitch": "OpenCV",
        "Speech emotion/emotional speech classification": "Speech Emotion Recognition",
        "Time-Accurate Speech Transcription": "WhisperX: Time-Accurate Speech Transcription of Long-Form Audio",
        "Spectral Features, Pitch, Rhythm": "Librosa",
        "Speech feature extraction": "openSMILE",
        "Sentiment Analysis": "AnAlgorithm for Routing Vectors in Sequences",
        "Disentangled Attention Mechanism & Enhanced Mask Decoder": "DEBERTA"
    }
    
    # Check each expected group
    file_data = list(grouped_json.values())[0]  # Get first file's data
    found_groups = {}
    
    for group_name, group_data in file_data.items():
        if isinstance(group_data, dict) and 'Feature' in group_data:
            found_groups[group_data['Feature']] = group_data['Model']
    
    print("Expected vs Found Groups:")
    all_found = True
    for expected_feature, expected_model in expected_groups.items():
        if expected_feature in found_groups:
            found_model = found_groups[expected_feature]
            status = "✅" if found_model == expected_model else "⚠️"
            print(f"  {status} {expected_feature}")
            if found_model != expected_model:
                print(f"      Expected: {expected_model}")
                print(f"      Found: {found_model}")
        else:
            print(f"  ❌ Missing: {expected_feature}")
            all_found = False
    
    # Check for unexpected groups
    unexpected = set(found_groups.keys()) - set(expected_groups.keys())
    if unexpected:
        print(f"\n⚠️  Additional groups found: {unexpected}")
    
    if all_found and not unexpected:
        print(f"\n🎉 SUCCESS: All feature groups match specification perfectly!")
    else:
        print(f"\n⚠️  Some discrepancies found (see above)")
    
    print(f"\n📊 FINAL STATS")
    print(f"   • Feature groups: {len(found_groups)}")
    print(f"   • Total features: {sum(len(gd.get('features', {})) for gd in file_data.values() if isinstance(gd, dict) and 'features' in gd)}")
    print(f"   • Specification compliance: {'✅ Perfect' if all_found and not unexpected else '⚠️ Partial'}")

if __name__ == "__main__":
    main()
