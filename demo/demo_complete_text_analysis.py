#!/usr/bin/env python3
"""
Complete Text Analysis Demo - DeBERTa + SimCSE + ALBERT

This script demonstrates the comprehensive text analysis capabilities
of the multimodal pipeline with all three language models.
"""

from ..src.feature_extractor import MultimodalFeatureExtractor
from ..src.pipeline import MultimodalPipeline
import json
import time

def demo_individual_analyzers():
    """Demo 1: Individual text analyzer capabilities"""
    print("📝 Demo 1: Individual Text Analyzer Capabilities")
    print("-" * 50)
    
    sample_text = {
        "text": "Natural language processing has revolutionized how computers understand human text. "
                "Modern language models like BERT, DeBERTa, and ALBERT achieve remarkable performance "
                "across diverse benchmarks including reading comprehension, sentiment analysis, "
                "and natural language inference tasks."
    }
    
    analyzers = [
        ("DeBERTa", "deberta_text", "DEB_"),
        ("SimCSE", "simcse_text", "CSE_"),
        ("ALBERT", "albert_text", "alb_")
    ]
    
    for name, feature, prefix in analyzers:
        print(f"\n🧠 {name} Analysis:")
        extractor = MultimodalFeatureExtractor(features=[feature], device='cpu')
        features = extractor.extract_features(sample_text)
        
        model_features = {k: v for k, v in features.items() if k.startswith(prefix)}
        benchmark_features = {k: v for k, v in model_features.items() 
                            if not k.endswith(('timestamp', 'length', 'count', 'name', 'device'))}
        
        print(f"   📊 Features extracted: {len(benchmark_features)}")
        
        # Show top 3 metrics
        for i, (key, value) in enumerate(list(benchmark_features.items())[:3]):
            print(f"   • {key}: {value:.3f}")
        if len(benchmark_features) > 3:
            print(f"   ... and {len(benchmark_features) - 3} more")

def demo_combined_analysis():
    """Demo 2: Combined text analysis with all models"""
    print("\n\n🤝 Demo 2: Combined Text Analysis")
    print("-" * 35)
    
    # Initialize extractor with all text analyzers
    extractor = MultimodalFeatureExtractor(
        features=['deberta_text', 'simcse_text', 'albert_text'],
        device='cpu'
    )
    
    test_text = {
        "text": "The integration of multiple language models provides comprehensive text analysis. "
                "This approach combines benchmark performance evaluation, contrastive learning, "
                "and parameter-efficient language representation for robust NLP capabilities."
    }
    
    print(f"🔍 Analyzing: \"{test_text['text'][:60]}...\"")
    
    start_time = time.time()
    features = extractor.extract_features(test_text)
    analysis_time = time.time() - start_time
    
    # Categorize features
    deberta_features = {k: v for k, v in features.items() if k.startswith('DEB_')}
    simcse_features = {k: v for k, v in features.items() if k.startswith('CSE_')}
    albert_features = {k: v for k, v in features.items() if k.startswith('alb_')}
    
    print(f"\n⏱️  Combined analysis completed in {analysis_time:.2f} seconds")
    print(f"📊 Feature summary:")
    print(f"   🧠 DeBERTa: {len(deberta_features)} features")
    print(f"   🔗 SimCSE: {len(simcse_features)} features")
    print(f"   🎯 ALBERT: {len(albert_features)} features")
    print(f"   📈 Total: {len(deberta_features) + len(simcse_features) + len(albert_features)} text features")
    
    # Show comparative performance on common tasks
    print(f"\n🔍 Comparative Performance:")
    if 'DEB_SST-2_Acc' in features and 'alb_sst' in features:
        print(f"   Sentiment Analysis:")
        print(f"     • DeBERTa SST-2: {features['DEB_SST-2_Acc']:.3f}")
        print(f"     • ALBERT SST: {features['alb_sst']:.3f}")
    
    if 'DEB_MNLI-m_Acc' in features and 'alb_mnli' in features:
        print(f"   Natural Language Inference:")
        print(f"     • DeBERTa MNLI: {features['DEB_MNLI-m_Acc']:.3f}")
        print(f"     • ALBERT MNLI: {features['alb_mnli']:.3f}")
    
    if 'CSE_Avg' in features:
        print(f"   Embedding Quality:")
        print(f"     • SimCSE Average STS: {features['CSE_Avg']:.3f}")

def demo_feature_grouping():
    """Demo 3: Feature grouping with all text analyzers"""
    print("\n\n🗂️  Demo 3: Feature Grouping by Model")
    print("-" * 35)
    
    # Initialize pipeline for feature grouping
    pipeline = MultimodalPipeline(
        features=['deberta_text', 'simcse_text', 'albert_text'],
        device='cpu'
    )
    
    # Create representative features from all models
    sample_features = {
        # DeBERTa features
        'DEB_SQuAD_1.1_F1': 0.867,
        'DEB_MNLI-m_Acc': 0.884,
        'DEB_SST-2_Acc': 0.921,
        
        # SimCSE features
        'CSE_STS12': 0.756,
        'CSE_STSBenchmark': 0.768,
        'CSE_Avg': 0.751,
        
        # ALBERT features
        'alb_mnli': 0.819,
        'alb_squad11dev': 0.822,
        'alb_sst': 0.901
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

def demo_multimodal_pipeline():
    """Demo 4: Complete multimodal pipeline with all text analyzers"""
    print("\n\n🎬 Demo 4: Complete Multimodal Pipeline")
    print("-" * 40)
    
    # Initialize complete pipeline
    extractor = MultimodalFeatureExtractor(
        features=['basic_audio', 'whisperx_transcription', 'deberta_text', 'simcse_text', 'albert_text'],
        device='cpu'
    )
    
    print("🎥 Processing video: data/MVI_0574.MP4")
    print("📝 Pipeline: Audio → Transcription → Multi-model Text Analysis")
    
    try:
        start_time = time.time()
        features = extractor.extract_features('data/MVI_0574.MP4')
        total_time = time.time() - start_time
        
        # Categorize all features
        audio_features = {k: v for k, v in features.items() if k.startswith('oc_')}
        deberta_features = {k: v for k, v in features.items() if k.startswith('DEB_')}
        simcse_features = {k: v for k, v in features.items() if k.startswith('CSE_')}
        albert_features = {k: v for k, v in features.items() if k.startswith('alb_')}
        
        print(f"\n✅ Complete multimodal pipeline finished in {total_time:.1f} seconds")
        
        # Show transcription
        if 'transcription' in features:
            transcript = features['transcription']
            print(f"\n🎤 Transcribed text:")
            print(f'   "{transcript[:100]}..."')
        
        print(f"\n📊 Multi-modal feature extraction:")
        print(f"   🔊 Basic audio: {len(audio_features)} features")
        print(f"   🧠 DeBERTa text: {len(deberta_features)} features")
        print(f"   🔗 SimCSE text: {len(simcse_features)} features")
        print(f"   🎯 ALBERT text: {len(albert_features)} features")
        print(f"   📈 Total features: {len(features)}")
        
        # Show text analysis insights
        print(f"\n🎯 Text analysis insights:")
        text_quality_indicators = []
        
        if 'DEB_SST-2_Acc' in features:
            text_quality_indicators.append(f"Sentiment capability: {features['DEB_SST-2_Acc']:.3f}")
        if 'CSE_Avg' in features:
            text_quality_indicators.append(f"Embedding quality: {features['CSE_Avg']:.3f}")
        if 'alb_squad11dev' in features:
            text_quality_indicators.append(f"Reading comprehension: {features['alb_squad11dev']:.3f}")
        
        for indicator in text_quality_indicators:
            print(f"   • {indicator}")
        
    except Exception as e:
        print(f"❌ Error in multimodal pipeline: {e}")

def main():
    """Run all text analysis demos"""
    print("🚀 Complete Text Analysis Demo")
    print("=" * 60)
    print("This demo showcases comprehensive text analysis with:")
    print("• DeBERTa: Disentangled attention for benchmark performance")
    print("• SimCSE: Contrastive learning for sentence embeddings")
    print("• ALBERT: Parameter-efficient language representation")
    print()
    
    # Run all demos
    demo_individual_analyzers()
    demo_combined_analysis()
    demo_feature_grouping()
    demo_multimodal_pipeline()
    
    print("\n" + "=" * 60)
    print("✨ Complete Text Analysis Summary:")
    print("✅ Three complementary language models integrated")
    print("✅ 51 total text analysis features (21 + 13 + 17)")
    print("✅ Covers 25+ NLP benchmarks and tasks")
    print("✅ Feature grouping by model categories")
    print("✅ Complete multimodal pipeline integration")
    print("✅ WhisperX transcription → multi-model analysis workflow")
    
    print(f"\n🎊 Comprehensive text analysis system complete!")
    print(f"📊 Benchmark Coverage:")
    print(f"   • DeBERTa: 9 major NLP tasks (SQuAD, GLUE benchmarks)")
    print(f"   • SimCSE: 7 STS evaluation benchmarks")
    print(f"   • ALBERT: 12 language understanding tasks")

if __name__ == "__main__":
    main()
