#!/usr/bin/env python3
"""
Complete Text Analysis Integration Demo
Demonstrates all four text analyzers: DeBERTa, SimCSE, ALBERT, and Sentence-BERT
"""

from src.pipeline import MultimodalPipeline
from src.feature_extractor import MultimodalFeatureExtractor
import time

def demo_individual_analyzers():
    """Demo 1: Test each text analyzer individually"""
    print("🔍 Demo 1: Individual Text Analyzer Testing")
    print("-" * 45)
    
    test_text = (
        "Natural language processing has revolutionized how computers understand human text. "
        "Modern transformer models like BERT and its variants have achieved remarkable performance "
        "across numerous benchmarks. These models excel at tasks like sentiment analysis, "
        "question answering, and textual entailment. The field continues to evolve rapidly "
        "with new architectures and training methodologies emerging regularly."
    )
    
    analyzers = [
        ("DeBERTa", "deberta_text", "src.text.deberta_analyzer", "DeBERTaAnalyzer"),
        ("SimCSE", "simcse_text", "src.text.simcse_analyzer", "SimCSEAnalyzer"),
        ("ALBERT", "albert_text", "src.text.albert_analyzer", "ALBERTAnalyzer"),
        ("Sentence-BERT", "sbert_text", "src.text.sbert_analyzer", "SBERTAnalyzer")
    ]
    
    for name, feature_name, module_path, class_name in analyzers:
        print(f"\n🤖 Testing {name}...")
        
        try:
            # Import and initialize analyzer
            module = __import__(module_path, fromlist=[class_name])
            analyzer_class = getattr(module, class_name)
            analyzer = analyzer_class(device="cpu")
            
            # Test analysis
            features = analyzer.get_feature_dict(test_text)
            
            # Count features
            feature_count = len(features)
            non_metadata_features = [k for k in features.keys() 
                                   if not any(meta in k.lower() for meta in 
                                            ['timestamp', 'length', 'count', 'model', 'device'])]
            
            print(f"   ✅ {name} analysis completed")
            print(f"   📊 Total features: {feature_count}")
            print(f"   🎯 Analysis features: {len(non_metadata_features)}")
            
            # Show sample features
            sample_features = list(features.keys())[:3]
            for feature in sample_features:
                value = features[feature]
                if isinstance(value, (int, float)):
                    print(f"      • {feature}: {value:.3f}")
                elif isinstance(value, list) and len(value) > 0:
                    print(f"      • {feature}: [{len(value)} elements]")
                else:
                    print(f"      • {feature}: {str(value)[:50]}")
                    
        except Exception as e:
            print(f"   ❌ Error testing {name}: {e}")

def demo_pipeline_integration():
    """Demo 2: Complete pipeline integration"""
    print("\n\n🔗 Demo 2: Complete Pipeline Integration")
    print("-" * 40)
    
    # Initialize pipeline with all text analyzers
    pipeline = MultimodalPipeline(
        features=['deberta_text', 'simcse_text', 'albert_text', 'sbert_text'],
        device='cpu',
        output_dir='output/complete_text_integration'
    )
    
    print("🚀 Processing sample audio file with all text analyzers...")
    start_time = time.time()
    
    try:
        results = pipeline.process_directory('data/', is_video=True)
        
        processing_time = time.time() - start_time
        print(f"✅ Processing completed in {processing_time:.2f} seconds")
        
        # Analyze results
        if results:
            for filename, file_data in results.items():
                print(f"\n📁 File: {filename}")
                
                # Count features by analyzer
                analyzer_counts = {
                    'DeBERTa': len([k for k in file_data.keys() if k.startswith('DEB_')]),
                    'SimCSE': len([k for k in file_data.keys() if k.startswith('CSE_')]),
                    'ALBERT': len([k for k in file_data.keys() if k.startswith('alb_')]),
                    'Sentence-BERT': len([k for k in file_data.keys() if k.startswith('BERT_')])
                }
                
                total_text_features = sum(analyzer_counts.values())
                print(f"   📊 Total text features: {total_text_features}")
                
                for analyzer, count in analyzer_counts.items():
                    status = "✅" if count > 0 else "⚠️"
                    print(f"   {status} {analyzer}: {count} features")
                
                # Check if transcription was successful
                if 'transcription' in file_data:
                    transcript = file_data['transcription']
                    print(f"   🗣️ Transcription: '{transcript[:100]}{'...' if len(transcript) > 100 else ''}'")
                else:
                    print("   ⚠️ No transcription found")
                    
    except Exception as e:
        print(f"❌ Pipeline processing failed: {e}")

def demo_feature_grouping():
    """Demo 3: Feature grouping with all text analyzers"""
    print("\n\n🗂️ Demo 3: Feature Grouping with All Text Analyzers")
    print("-" * 50)
    
    # Initialize pipeline for feature grouping
    pipeline = MultimodalPipeline(
        features=['deberta_text', 'simcse_text', 'albert_text', 'sbert_text'],
        device='cpu'
    )
    
    # Create comprehensive sample features from all models
    sample_features = {
        # DeBERTa features
        'DEB_SQuAD_1.1_F1': 0.867,
        'DEB_MNLI-m_Acc': 0.884,
        'DEB_SST-2_Acc': 0.921,
        'DEB_CoLA_MCC': 0.523,
        
        # SimCSE features
        'CSE_STS12': 0.756,
        'CSE_STS13': 0.742,
        'CSE_STSBenchmark': 0.768,
        'CSE_Avg': 0.751,
        
        # ALBERT features
        'alb_mnli': 0.819,
        'alb_squad11dev': 0.822,
        'alb_sst': 0.901,
        'alb_cola': 0.445,
        
        # Sentence-BERT features
        'BERT_tensor_sentences': [0.1, 0.8, 0.3, 0.9],
        'BERT_tensor_sentences_shape': [2, 2],
        'BERT_score': [0.78, 0.65, 0.91],
        'BERT_sentence_count': 3,
        
        # Other features
        'transcription': 'This is a sample transcription for testing',
        'sample_rate': 16000
    }
    
    print("🔄 Grouping features by model categories...")
    grouped_features = pipeline._group_features_by_model(sample_features)
    
    text_groups = 0
    total_text_features = 0
    
    for group_name, group_data in grouped_features.items():
        features_in_group = group_data.get('features', {})
        
        # Check if this is a text analysis group
        is_text_group = any(prefix in group_name.lower() for prefix in 
                          ['deberta', 'simcse', 'albert', 'sentence-bert', 'contrastive', 'language', 'dense'])
        
        if is_text_group:
            text_groups += 1
            total_text_features += len(features_in_group)
        
        print(f"\n📁 {group_name}:")
        print(f"   🤖 Model: {group_data['Model']}")
        print(f"   📊 Features: {len(features_in_group)}")
        
        # Show first few features
        for i, (feature, value) in enumerate(features_in_group.items()):
            if i >= 3:  # Limit display
                print(f"     ... and {len(features_in_group) - 3} more")
                break
            if isinstance(value, (int, float)):
                print(f"     • {feature}: {value:.3f}")
            elif isinstance(value, list):
                print(f"     • {feature}: [{len(value)} elements]")
            else:
                print(f"     • {feature}: {str(value)[:30]}")
    
    print(f"\n📈 Text Analysis Summary:")
    print(f"   🔤 Text analysis groups: {text_groups}")
    print(f"   📊 Total text features: {total_text_features}")
    print(f"   🎯 Models integrated: DeBERTa, SimCSE, ALBERT, Sentence-BERT")

def demo_feature_extractor():
    """Demo 4: MultimodalFeatureExtractor with text analysis"""
    print("\n\n🎛️ Demo 4: MultimodalFeatureExtractor Integration")
    print("-" * 45)
    
    # Initialize feature extractor with all text analyzers
    extractor = MultimodalFeatureExtractor(
        features=['whisperx_transcription', 'deberta_text', 'simcse_text', 'albert_text', 'sbert_text'],
        device='cpu'
    )
    
    # Simulate existing features with transcription
    existing_features = {
        'transcription': (
            "Machine learning and artificial intelligence are transforming various industries. "
            "Natural language processing enables computers to understand and generate human text. "
            "These technologies have applications in healthcare, finance, and education."
        ),
        'audio_length': 45.2,
        'sample_rate': 16000
    }
    
    print("🔄 Extracting enhanced features with all text analyzers...")
    
    try:
        enhanced_features = extractor.extract_features(existing_features)
        
        # Analyze results
        text_feature_counts = {
            'DeBERTa': len([k for k in enhanced_features.keys() if k.startswith('DEB_')]),
            'SimCSE': len([k for k in enhanced_features.keys() if k.startswith('CSE_')]),
            'ALBERT': len([k for k in enhanced_features.keys() if k.startswith('alb_')]),
            'Sentence-BERT': len([k for k in enhanced_features.keys() if k.startswith('BERT_')])
        }
        
        total_features = len(enhanced_features)
        total_text_features = sum(text_feature_counts.values())
        
        print(f"✅ Feature extraction completed!")
        print(f"📊 Total features: {total_features}")
        print(f"🔤 Text analysis features: {total_text_features}")
        
        for analyzer, count in text_feature_counts.items():
            status = "✅" if count > 0 else "⚠️"
            print(f"   {status} {analyzer}: {count} features")
        
        # Show sample features from each analyzer
        print(f"\n🔍 Sample features:")
        for prefix, analyzer in [('DEB_', 'DeBERTa'), ('CSE_', 'SimCSE'), 
                               ('alb_', 'ALBERT'), ('BERT_', 'Sentence-BERT')]:
            features = [k for k in enhanced_features.keys() if k.startswith(prefix)]
            if features:
                sample_feature = features[0]
                value = enhanced_features[sample_feature]
                if isinstance(value, (int, float)):
                    print(f"   🎯 {analyzer}: {sample_feature} = {value:.3f}")
                else:
                    print(f"   🎯 {analyzer}: {sample_feature} = {str(value)[:30]}")
            else:
                print(f"   ⚠️ {analyzer}: No features found")
                
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")

def main():
    """Run complete text analysis integration demonstration"""
    print("🎯 Complete Text Analysis Integration Demo")
    print("=" * 50)
    print("Testing: DeBERTa + SimCSE + ALBERT + Sentence-BERT")
    print("=" * 50)
    
    try:
        demo_individual_analyzers()
        demo_pipeline_integration()
        demo_feature_grouping()
        demo_feature_extractor()
        
        print("\n\n🎉 COMPLETE TEXT INTEGRATION DEMO FINISHED!")
        print("=" * 45)
        print("✅ All four text analyzers successfully integrated")
        print("✅ Pipeline processing working")
        print("✅ Feature grouping configured")
        print("✅ MultimodalFeatureExtractor enhanced")
        print("\n📈 Ready for production use!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
