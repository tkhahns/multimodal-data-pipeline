"""
OpenSMILE feature extraction for comprehensive audio analysis.
Extracts Low-Level Descriptors (LLDs) and Functionals using the openSMILE toolkit.
"""
import numpy as np
import pandas as pd
import opensmile
from pathlib import Path
from typing import Dict, Any, List, Optional


class OpenSMILEFeatureExtractor:
    """Extract comprehensive audio features using OpenSMILE."""
    
    def __init__(self, feature_set: str = 'ComParE_2016', sampling_rate: int = 16000):
        """
        Initialize the OpenSMILE feature extractor.
        
        Args:
            feature_set: OpenSMILE feature set to use
                - 'ComParE_2016': Computational Paralinguistics Challenge 2016 feature set
                - 'eGeMAPSv02': extended Geneva Minimalistic Acoustic Parameter Set v02
                - 'GeMAPSv01b': Geneva Minimalistic Acoustic Parameter Set v01b
            sampling_rate: Target sampling rate for audio processing
        """
        self.feature_set = feature_set
        self.sampling_rate = sampling_rate
        
        # Initialize OpenSMILE with the specified feature set
        try:
            self.smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.ComParE_2016,
                feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
            )
            
            # Also initialize for functionals
            self.smile_functionals = opensmile.Smile(
                feature_set=opensmile.FeatureSet.ComParE_2016,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
        except Exception as e:
            print(f"Warning: OpenSMILE initialization failed: {e}")
            self.smile = None
            self.smile_functionals = None
    
    def extract_low_level_descriptors(self, audio_path: str) -> Dict[str, np.ndarray]:
        """
        Extract Low-Level Descriptors (LLDs) from audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dict containing LLD features as time series
        """
        if self.smile is None:
            raise RuntimeError("OpenSMILE not properly initialized")
        
        try:
            # Extract LLDs
            lld_features = self.smile.process_file(audio_path)
            
            # Convert to dictionary format with proper naming
            features = {}
            
            # Map OpenSMILE column names to our standardized names
            column_mapping = {
                # Energy & Voice Quality
                'pcm_RMSenergy_sma': 'osm_pcm_RMSenergy_sma',
                'pcm_zcr_sma': 'osm_zcr_sma',
                'F0final_sma': 'osm_F0final_sma',
                'voicingFinalUnclipped_sma': 'osm_voicingFinalUnclipped_sma',
                'jitterLocal_sma': 'osm_jitterLocal_sma',
                'jitterDDP_sma': 'osm_jitterDDP_sma',
                'shimmerLocal_sma': 'osm_shimmerLocal_sma',
                'logHNR_sma': 'osm_logHNR_sma',
                
                # Spectral Features with correct names
                'pcm_fftMag_spectralFlux_sma': 'osm_spectralFlux_sma',
                'pcm_fftMag_spectralRollOff25.0_sma': 'osm_spectralRollOff25_sma',
                'pcm_fftMag_spectralRollOff50.0_sma': 'osm_spectralRollOff50_sma',
                'pcm_fftMag_spectralRollOff75.0_sma': 'osm_spectralRollOff75_sma',
                'pcm_fftMag_spectralRollOff90.0_sma': 'osm_spectralRollOff90_sma',
                'pcm_fftMag_spectralCentroid_sma': 'osm_spectralCentroid_sma',
                'pcm_fftMag_spectralEntropy_sma': 'osm_spectralEntropy_sma',
                'pcm_fftMag_spectralVariance_sma': 'osm_spectralVariance_sma',
                'pcm_fftMag_spectralSkewness_sma': 'osm_spectralSkewness_sma',
                'pcm_fftMag_spectralKurtosis_sma': 'osm_spectralKurtosis_sma',
                'pcm_fftMag_spectralSlope_sma': 'osm_spectralSlope_sma',
                'pcm_fftMag_spectralDecrease_sma': 'osm_spectralDecrease_sma',
                
                # MFCCs with correct format
                'mfcc_sma[1]': 'osm_mfcc1_sma',
                'mfcc_sma[2]': 'osm_mfcc2_sma',
                'mfcc_sma[3]': 'osm_mfcc3_sma',
                'mfcc_sma[4]': 'osm_mfcc4_sma',
                'mfcc_sma[5]': 'osm_mfcc5_sma',
                'mfcc_sma[6]': 'osm_mfcc6_sma',
                'mfcc_sma[7]': 'osm_mfcc7_sma',
                'mfcc_sma[8]': 'osm_mfcc8_sma',
                'mfcc_sma[9]': 'osm_mfcc9_sma',
                'mfcc_sma[10]': 'osm_mfcc10_sma',
                'mfcc_sma[11]': 'osm_mfcc11_sma',
                'mfcc_sma[12]': 'osm_mfcc12_sma',
                
                # Auditory spectrum features
                'audspec_lengthL1norm_sma': 'osm_audspec_lengthL1norm_sma',
                'audspecRasta_lengthL1norm_sma': 'osm_audspecRasta_lengthL1norm_sma',
                
                # Auditory spectrum filter bank features (first 10 filters)
                'audSpec_Rfilt_sma[0]': 'osm_audSpec_Rfilt_0',
                'audSpec_Rfilt_sma[1]': 'osm_audSpec_Rfilt_1',
                'audSpec_Rfilt_sma[2]': 'osm_audSpec_Rfilt_2',
                'audSpec_Rfilt_sma[3]': 'osm_audSpec_Rfilt_3',
                'audSpec_Rfilt_sma[4]': 'osm_audSpec_Rfilt_4',
                'audSpec_Rfilt_sma[5]': 'osm_audSpec_Rfilt_5',
                'audSpec_Rfilt_sma[6]': 'osm_audSpec_Rfilt_6',
                'audSpec_Rfilt_sma[7]': 'osm_audSpec_Rfilt_7',
                'audSpec_Rfilt_sma[8]': 'osm_audSpec_Rfilt_8',
                'audSpec_Rfilt_sma[9]': 'osm_audSpec_Rfilt_9',
            }
            
            # Extract features with standardized names
            features = {}
            
            # Process all OpenSMILE columns and create standardized names
            for col in lld_features.columns:
                # Create standardized feature name
                standard_name = f"osm_{col}"
                # Clean up special characters in the name
                standard_name = standard_name.replace('[', '_').replace(']', '').replace('.', '_').replace('-', '_')
                
                # Extract the feature values
                features[standard_name] = lld_features[col].values
            
            return features
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract LLD features: {e}")
    
    def extract_functionals(self, audio_path: str) -> Dict[str, float]:
        """
        Extract Functional features (statistical summaries) from audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dict containing functional features as single values
        """
        if self.smile_functionals is None:
            raise RuntimeError("OpenSMILE functionals not properly initialized")
        
        try:
            # Extract functionals
            functional_features = self.smile_functionals.process_file(audio_path)
            
            # Convert to dictionary with standardized names
            features = {}
            
            # Define functional statistics we want to extract
            functional_stats = [
                'mean', 'stddev', 'skewness', 'kurtosis',
                'percentile1.0', 'percentile5.0', 'percentile25.0', 
                'percentile50.0', 'percentile75.0', 'percentile95.0', 'percentile99.0',
                'min', 'max', 'minPos', 'maxPos', 'range',
                'quartile1', 'quartile3', 'interquartileRange',
                'linregc1', 'linregc2', 'linregerr'
            ]
            
            # Extract functional features
            for col in functional_features.columns:
                for stat in functional_stats:
                    if stat in col.lower():
                        # Create standardized feature name
                        feature_name = f"osm_{stat}"
                        if feature_name not in features:
                            features[feature_name] = []
                        features[feature_name].append(functional_features[col].iloc[0])
            
            # Convert lists to mean values if multiple features contribute to the same stat
            for key, value in features.items():
                if isinstance(value, list):
                    features[key] = np.mean(value) if value else 0.0
            
            return features
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract functional features: {e}")
    
    def calculate_custom_functionals(self, lld_features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calculate custom functional statistics from LLD features.
        
        Args:
            lld_features: Dictionary of LLD features
            
        Returns:
            Dict containing calculated functional statistics
        """
        functionals = {}
        
        # Calculate functionals for each LLD feature
        for feature_name, feature_values in lld_features.items():
            if len(feature_values) == 0:
                continue
                
            # Remove NaN and infinite values
            clean_values = feature_values[np.isfinite(feature_values)]
            
            if len(clean_values) == 0:
                continue
            
            # Basic statistics
            functionals[f"{feature_name}_mean"] = np.mean(clean_values)
            functionals[f"{feature_name}_stddev"] = np.std(clean_values)
            functionals[f"{feature_name}_min"] = np.min(clean_values)
            functionals[f"{feature_name}_max"] = np.max(clean_values)
            functionals[f"{feature_name}_range"] = np.max(clean_values) - np.min(clean_values)
            
            # Percentiles
            percentiles = [1, 5, 25, 50, 75, 95, 99]
            for p in percentiles:
                functionals[f"{feature_name}_percentile{p}"] = np.percentile(clean_values, p)
            
            # Quartiles
            functionals[f"{feature_name}_quartile1"] = np.percentile(clean_values, 25)
            functionals[f"{feature_name}_quartile3"] = np.percentile(clean_values, 75)
            functionals[f"{feature_name}_interquartileRange"] = (
                np.percentile(clean_values, 75) - np.percentile(clean_values, 25)
            )
            
            # Moments
            if len(clean_values) > 1:
                from scipy import stats
                functionals[f"{feature_name}_skewness"] = stats.skew(clean_values)
                functionals[f"{feature_name}_kurtosis"] = stats.kurtosis(clean_values)
            
            # Positions of min/max
            functionals[f"{feature_name}_minPos"] = np.argmin(clean_values) / len(clean_values)
            functionals[f"{feature_name}_maxPos"] = np.argmax(clean_values) / len(clean_values)
            
            # Linear regression coefficients
            if len(clean_values) > 2:
                x = np.arange(len(clean_values))
                slope, intercept = np.polyfit(x, clean_values, 1)
                residuals = clean_values - (slope * x + intercept)
                
                functionals[f"{feature_name}_linregc1"] = slope
                functionals[f"{feature_name}_linregc2"] = intercept
                functionals[f"{feature_name}_linregerr"] = np.sqrt(np.mean(residuals**2))
        
        return functionals
    
    def extract_all_features(self, audio_path: str) -> Dict[str, Any]:
        """
        Extract both LLD and functional features from an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dict containing all extracted features
        """
        features = {}
        
        try:
            # Extract Low-Level Descriptors
            lld_features = self.extract_low_level_descriptors(audio_path)
            features.update(lld_features)
            
            # Calculate custom functionals from LLDs
            custom_functionals = self.calculate_custom_functionals(lld_features)
            features.update(custom_functionals)
            
            # Try to extract official OpenSMILE functionals
            try:
                opensmile_functionals = self.extract_functionals(audio_path)
                # Prefix with 'osm_official_' to distinguish from custom functionals
                for key, value in opensmile_functionals.items():
                    features[f"osm_official_{key}"] = value
            except Exception as e:
                print(f"Warning: Could not extract OpenSMILE functionals: {e}")
            
            # Add metadata
            features['opensmile_feature_set'] = self.feature_set
            features['opensmile_sampling_rate'] = self.sampling_rate
            features['num_frames'] = len(next(iter(lld_features.values()))) if lld_features else 0
            
        except Exception as e:
            print(f"Error extracting OpenSMILE features: {e}")
            # Return empty features on error
            features = {'error': str(e)}
        
        return features
    
    def get_feature_dict(self, audio_path: str) -> Dict[str, Any]:
        """
        Extract features and format them for the pipeline.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dict: Dictionary of extracted features
        """
        return self.extract_all_features(audio_path)
    
    def get_available_features(self) -> List[str]:
        """
        Get list of available feature names.
        
        Returns:
            List of feature names that will be extracted
        """
        # Get the actual column names from OpenSMILE
        if self.smile is None:
            return []
            
        try:
            # Use a dummy file to get column names
            # For now, return the expected feature structure
            base_features = [
                'osm_F0final_sma',
                'osm_voicingFinalUnclipped_sma', 
                'osm_jitterLocal_sma',
                'osm_jitterDDP_sma',
                'osm_shimmerLocal_sma',
                'osm_logHNR_sma',
                'osm_audspec_lengthL1norm_sma',
                'osm_audspecRasta_lengthL1norm_sma',
                'osm_pcm_RMSenergy_sma',
                'osm_pcm_zcr_sma',
                'osm_pcm_fftMag_fband250_650_sma',
                'osm_pcm_fftMag_fband1000_4000_sma',
                'osm_pcm_fftMag_spectralRollOff25_0_sma',
                'osm_pcm_fftMag_spectralRollOff50_0_sma',
                'osm_pcm_fftMag_spectralRollOff75_0_sma',
                'osm_pcm_fftMag_spectralRollOff90_0_sma',
                'osm_pcm_fftMag_spectralFlux_sma',
                'osm_pcm_fftMag_spectralCentroid_sma',
                'osm_pcm_fftMag_spectralEntropy_sma',
                'osm_pcm_fftMag_spectralVariance_sma',
                'osm_pcm_fftMag_spectralSkewness_sma',
                'osm_pcm_fftMag_spectralKurtosis_sma',
                'osm_pcm_fftMag_spectralSlope_sma',
                'osm_pcm_fftMag_psySharpness_sma',
                'osm_pcm_fftMag_spectralHarmonicity_sma',
            ]
            
            # Add MFCC features
            for i in range(1, 13):
                base_features.append(f'osm_mfcc_sma_{i}')
            
            # Add auditory spectrum filter bank features
            for i in range(26):
                base_features.append(f'osm_audSpec_Rfilt_sma_{i}')
            
            functional_stats = [
                'mean', 'stddev', 'skewness', 'kurtosis',
                'percentile1', 'percentile5', 'percentile25', 'percentile50', 'percentile75', 'percentile95', 'percentile99',
                'min', 'max', 'minPos', 'maxPos', 'range',
                'quartile1', 'quartile3', 'interquartileRange',
                'linregc1', 'linregc2', 'linregerr'
            ]
            
            # Generate functional feature names for each LLD
            all_features = base_features.copy()
            for lld in base_features:
                for stat in functional_stats:
                    all_features.append(f"{lld}_{stat}")
            
            return all_features
            
        except Exception as e:
            print(f"Warning: Could not determine available features: {e}")
            return []
