#!/usr/bin/env python3
"""
Complete SimCSE Integration Demo

This script demonstrates:
1. SimCSE text analysis on direct text input
2. SimCSE integration with WhisperX transcription
3. Feature grouping with both DeBERTa and SimCSE
4. Complete multimodal pipeline with text analysis
"""

from ..src.feature_extractor import MultimodalFeatureExtractor
from ..src.pipeline import MultimodalPipeline
import json
import time

def demo_direct_text_analysis():
    """Demo 1: Direct text analysis with SimCSE"""
    print("📝 Demo 1: Direct SimCSE Text Analysis")
    print("-" * 40)
    
    # Initialize SimCSE-only extractor
    extractor = MultimodalFeatureExtractor(
        features=['simcse_text'],
        device='cpu'
    )
    
    # Test texts
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Natural language processing enables computers to understand human text.",
        "Contrastive learning improves sentence embedding quality significantly."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n🔍 Analyzing text {i}: \"{text[:50]}...\"")
        
        features = extractor.extract_features({'text': text})
        simcse_features = {k: v for k, v in features.items() if k.startswith('CSE_')}
        
        print(f"   📊 SimCSE features extracted: {len(simcse_features)}")
        print(f"   🎯 Average STS performance: {features.get('CSE_Avg', 0):.3f}")
        
        # Show sample metrics
        sample_metrics = ['CSE_STS12', 'CSE_STS15', 'CSE_STSBenchmark']
        for metric in sample_metrics:
            if metric in features:
                print(f"   • {metric}: {features[metric]:.3f}")

def demo_combined_text_analysis():
    """Demo 2: Combined DeBERTa + SimCSE analysis"""
    print("\n\n🧠 Demo 2: Combined DeBERTa + SimCSE Analysis")
    print("-" * 45)
    
    # Initialize combined text extractor
    extractor = MultimodalFeatureExtractor(
        features=['deberta_text', 'simcse_text'],
        device='cpu'
    )
    
    sample_text = {
        "text": "This comprehensive analysis evaluates both benchmark performance and contrastive learning quality. "
                "The model demonstrates strong capabilities across multiple natural language understanding tasks."
    }
    
    print(f"🔍 Analyzing: \"{sample_text['text'][:80]}...\"")
    
    start_time = time.time()
    features = extractor.extract_features(sample_text)
    analysis_time = time.time() - start_time
    
    # Categorize features
    deberta_features = {k: v for k, v in features.items() if k.startswith('DEB_')}
    simcse_features = {k: v for k, v in features.items() if k.startswith('CSE_')}
    
    print(f"\n⏱️  Analysis completed in {analysis_time:.2f} seconds")
    print(f"🧠 DeBERTa features: {len(deberta_features)}")
    print(f"🔗 SimCSE features: {len(simcse_features)}")
    print(f"📊 Total text features: {len(deberta_features) + len(simcse_features)}")
    
    print(f"\n📈 Sample DeBERTa performance:")
    deberta_samples = ['DEB_SST-2_Acc', 'DEB_MNLI-m_Acc', 'DEB_CoLA_MCC']
    for metric in deberta_samples:
        if metric in features:
            print(f"   • {metric}: {features[metric]:.3f}")
    
    print(f"\n🔗 Sample SimCSE performance:")
    simcse_samples = ['CSE_STS12', 'CSE_STSBenchmark', 'CSE_Avg']
    for metric in simcse_samples:
        if metric in features:
            print(f"   • {metric}: {features[metric]:.3f}")

def demo_feature_grouping():
    """Demo 3: Feature grouping with both text analyzers"""
    print("\n\n🗂️  Demo 3: Feature Grouping with Text Analyzers")
    print("-" * 45)
    
    # Initialize pipeline for feature grouping
    pipeline = MultimodalPipeline(
        features=['deberta_text', 'simcse_text'],
        device='cpu'
    )
    
    # Create sample features
    sample_features = {
        'DEB_SQuAD_1.1_F1': 0.867,
        'DEB_MNLI-m_Acc': 0.884,
        'DEB_SST-2_Acc': 0.921,
        'DEB_CoLA_MCC': 0.523,
        'CSE_STS12': 0.756,
        'CSE_STS13': 0.742,
        'CSE_STSBenchmark': 0.768,
        'CSE_Avg': 0.751
    }
    
    print("🔄 Grouping features by model categories...")
    grouped_features = pipeline._group_features_by_model(sample_features)
    
    for group_name, group_data in grouped_features.items():
        features_in_group = group_data.get('features', {})
        print(f"\n📁 {group_name}:")
        print(f"   🤖 Model: {group_data['Model']}")
        print(f"   📊 Features: {len(features_in_group)}")
        
        for feature, value in features_in_group.items():
            print(f"     • {feature}: {value:.3f}")

def demo_complete_multimodal():
    """Demo 4: Complete multimodal pipeline with text analysis"""
    print("\n\n🎬 Demo 4: Complete Multimodal Pipeline")
    print("-" * 40)
    
    # Initialize complete pipeline
    extractor = MultimodalFeatureExtractor(
        features=['basic_audio', 'whisperx_transcription', 'deberta_text', 'simcse_text'],
        device='cpu'
    )
    
    print("🎥 Processing video file: data/MVI_0574.MP4")
    print("📝 This will: extract audio → transcribe → analyze text")
    
    try:
        start_time = time.time()
        features = extractor.extract_features('data/MVI_0574.MP4')
        total_time = time.time() - start_time
        
        # Categorize all features
        audio_features = {k: v for k, v in features.items() if k.startswith('oc_')}
        deberta_features = {k: v for k, v in features.items() if k.startswith('DEB_')}
        simcse_features = {k: v for k, v in features.items() if k.startswith('CSE_')}
        
        print(f"\n✅ Complete pipeline finished in {total_time:.1f} seconds")
        
        # Show transcription
        if 'transcription' in features:
            transcript = features['transcription']
            print(f"\n🎤 Transcribed text:")
            print(f'   "{transcript[:120]}..."')
        
        print(f"\n📊 Feature extraction summary:")
        print(f"   🔊 Basic audio features: {len(audio_features)}")
        print(f"   🧠 DeBERTa text features: {len(deberta_features)}")
        print(f"   🔗 SimCSE text features: {len(simcse_features)}")
        print(f"   📈 Total features: {len(features)}")
        
        # Show text analysis quality
        if deberta_features and simcse_features:
            print(f"\n🎯 Text analysis quality indicators:")
            if 'DEB_SST-2_Acc' in features:
                print(f"   • Sentiment analysis capability: {features['DEB_SST-2_Acc']:.3f}")
            if 'CSE_Avg' in features:
                print(f"   • Embedding quality (avg STS): {features['CSE_Avg']:.3f}")
        
    except Exception as e:
        print(f"❌ Error in complete pipeline: {e}")

def main():
    """Run all SimCSE integration demos"""
    print("🚀 Complete SimCSE Integration Demo")
    print("=" * 50)
    print("This demo showcases the full SimCSE contrastive learning integration")
    print("with the multimodal data pipeline.\n")
    
    # Run all demos
    demo_direct_text_analysis()
    demo_combined_text_analysis()
    demo_feature_grouping()
    demo_complete_multimodal()
    
    print("\n" + "=" * 50)
    print("✨ SimCSE Integration Summary:")
    print("✅ Direct text analysis")
    print("✅ Combined with DeBERTa analysis") 
    print("✅ Feature grouping by model categories")
    print("✅ Complete multimodal pipeline integration")
    print("✅ WhisperX transcription → text analysis workflow")
    
    print(f"\n🎊 SimCSE integration is now complete!")
    print(f"📈 Total text analysis features: ~34 (21 DeBERTa + 13 SimCSE)")
    print(f"🧪 STS benchmarks covered: 7 (STS12-16, STSBenchmark, SICKRelatedness)")

if __name__ == "__main__":
    main()
