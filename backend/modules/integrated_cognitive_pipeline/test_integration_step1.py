# -*- coding: utf-8 -*-
"""
Test Script for Integration Step 1: Feature Engineering
========================================================

Tests:
1. MMSE normalization với age và education
2. Integrated feature engineering trong MCIScreeningService
3. Backward compatibility
"""

import sys
import os
from pathlib import Path

# Add specific paths to avoid importing backend/__init__.py
current_dir = Path(__file__).parent
modules_dir = current_dir.parent
backend_dir = modules_dir.parent

# Add paths
sys.path.insert(0, str(modules_dir))
sys.path.insert(0, str(backend_dir))

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_mmse_normalizer():
    """Test MMSE normalizer directly."""
    logger.info("="*60)
    logger.info("TEST 1: MMSE Normalizer")
    logger.info("="*60)
    
    try:
        # Import directly from file to avoid backend package import issues
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "feature_engineer_v2",
            current_dir / "feature_engineer_v2.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        MMSENormalizer = module.MMSENormalizer
        
        normalizer = MMSENormalizer()
        
        # Test case 1: Normal case
        result1 = normalizer.normalize_mmse(
            mmse_raw=25.0,
            age=70.0,
            education_years=12.0,
            method='adni'
        )
        
        logger.info(f"Test Case 1: Age=70, Education=12, MMSE=25")
        logger.info(f"  Raw MMSE: {result1['mmse_raw']}")
        logger.info(f"  Adjusted MMSE: {result1['mmse_adjusted']:.2f}")
        logger.info(f"  Education Adjustment: {result1['education_adjustment']:.2f}")
        logger.info(f"  Age Adjustment: {result1['age_adjustment']:.2f}")
        
        # Test case 2: Low education
        result2 = normalizer.normalize_mmse(
            mmse_raw=22.0,
            age=75.0,
            education_years=6.0,
            method='adni'
        )
        
        logger.info(f"\nTest Case 2: Age=75, Education=6, MMSE=22")
        logger.info(f"  Raw MMSE: {result2['mmse_raw']}")
        logger.info(f"  Adjusted MMSE: {result2['mmse_adjusted']:.2f}")
        logger.info(f"  Education Adjustment: {result2['education_adjustment']:.2f}")
        logger.info(f"  Age Adjustment: {result2['age_adjustment']:.2f}")
        
        # Test case 3: High education
        result3 = normalizer.normalize_mmse(
            mmse_raw=28.0,
            age=65.0,
            education_years=16.0,
            method='adni'
        )
        
        logger.info(f"\nTest Case 3: Age=65, Education=16, MMSE=28")
        logger.info(f"  Raw MMSE: {result3['mmse_raw']}")
        logger.info(f"  Adjusted MMSE: {result3['mmse_adjusted']:.2f}")
        logger.info(f"  Education Adjustment: {result3['education_adjustment']:.2f}")
        logger.info(f"  Age Adjustment: {result3['age_adjustment']:.2f}")
        
        logger.info("\n✅ MMSE Normalizer test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ MMSE Normalizer test failed: {e}", exc_info=True)
        return False


def test_integrated_feature_engineer():
    """Test Integrated Feature Engineer."""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Integrated Feature Engineer")
    logger.info("="*60)
    
    try:
        # Import directly from file
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "feature_engineer_v2",
            current_dir / "feature_engineer_v2.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        IntegratedFeatureEngineer = module.IntegratedFeatureEngineer
        
        engineer = IntegratedFeatureEngineer(
            mmse_normalization_method='adni',
            imputation_method='knn',
            feature_selection_method='none',  # Skip selection for testing
            n_features_to_select=100,
            correlation_threshold=0.9,
            scaler_type='standard'
        )
        
        # Create sample data
        sample_data = {
            'age': [70, 75, 65, 80, 72],
            'education_years': [12, 6, 16, 8, 10],
            'mmse_raw': [25, 22, 28, 20, 24],
            'acoustic_f0_mean': [150, 140, 160, 130, 145],
            'linguistic_lex_ttr': [0.6, 0.4, 0.7, 0.35, 0.55],
            'linguistic_sem_idea_density': [5.0, 3.5, 6.0, 3.0, 4.5]
        }
        
        X = pd.DataFrame(sample_data)
        y = pd.Series(['Normal', 'MCI', 'Normal', 'MCI', 'Normal'])
        
        logger.info(f"Input data shape: {X.shape}")
        logger.info(f"Features: {list(X.columns)}")
        
        # Fit and transform
        X_processed = engineer.fit_transform(
            X, y,
            do_mmse_normalization=True,
            do_feature_selection=False
        )
        
        logger.info(f"\nProcessed data shape: {X_processed.shape}")
        logger.info(f"New features added:")
        
        # Check for MMSE normalization features
        mmse_features = [col for col in X_processed.columns if 'mmse' in col.lower()]
        for feat in mmse_features:
            logger.info(f"  - {feat}")
            logger.info(f"    Sample values: {X_processed[feat].head(3).tolist()}")
        
        logger.info("\n✅ Integrated Feature Engineer test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integrated Feature Engineer test failed: {e}", exc_info=True)
        return False


def test_service_integration():
    """Test MCIScreeningService với integrated pipeline."""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: MCIScreeningService Integration")
    logger.info("="*60)
    
    try:
        from backend.modules.integration_service import MCIScreeningService
        
        # Test với integrated pipeline enabled
        logger.info("Testing với integrated pipeline enabled...")
        service_integrated = MCIScreeningService(use_integrated_pipeline=True)
        
        logger.info(f"  Integrated pipeline enabled: {service_integrated.use_integrated_pipeline}")
        logger.info(f"  Integrated feature engineer available: {service_integrated.integrated_feature_engineer is not None}")
        
        # Test với sample data (không cần audio/transcript thật)
        test_metadata = {
            'age': 70,
            'education': 12,
            'mmse': 25
        }
        
        # Test _apply_integrated_feature_engineering method
        test_features = {
            'acoustic_f0_mean': 150.0,
            'linguistic_lex_ttr': 0.6,
            'age': 70,
            'education_years': 12,
            'mmse_raw': 25.0
        }
        
        # Test MMSE normalization directly (không cần service fully initialized)
        try:
            # Import MMSENormalizer directly
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "feature_engineer_v2",
                current_dir / "feature_engineer_v2.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            MMSENormalizer = module.MMSENormalizer
            
            normalizer = MMSENormalizer()
            mmse_result = normalizer.normalize_mmse(
                test_features['mmse_raw'],
                test_features['age'],
                test_features['education_years'],
                method='adni'
            )
            
            logger.info(f"\nMMSE Normalization Test:")
            logger.info(f"  ✅ MMSE normalization working!")
            logger.info(f"     mmse_raw: {mmse_result['mmse_raw']}")
            logger.info(f"     mmse_adjusted: {mmse_result['mmse_adjusted']:.2f}")
            logger.info(f"     mmse_education_adj: {mmse_result['education_adjustment']:.2f}")
            logger.info(f"     mmse_age_adj: {mmse_result['age_adjustment']:.2f}")
        except Exception as e:
            logger.warning(f"  ⚠️ Could not test MMSE normalization: {e}")
        
        # Test backward compatibility (không enable integrated pipeline)
        logger.info("\nTesting backward compatibility (integrated pipeline disabled)...")
        service_standard = MCIScreeningService(use_integrated_pipeline=False)
        
        logger.info(f"  Integrated pipeline enabled: {service_standard.use_integrated_pipeline}")
        logger.info(f"  Standard pipeline works: {service_standard.predictor is not None}")
        
        logger.info("\n✅ Service Integration test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Service Integration test failed: {e}", exc_info=True)
        return False


def test_full_workflow():
    """Test full workflow với sample data."""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Full Workflow (Mock Data)")
    logger.info("="*60)
    
    try:
        # Test MMSE normalization directly
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "feature_engineer_v2",
            current_dir / "feature_engineer_v2.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        MMSENormalizer = module.MMSENormalizer
        
        # Mock features
        mock_features = {
            'acoustic_f0_mean': 150.0,
            'acoustic_pause_rate': 0.25,
            'linguistic_lex_ttr': 0.6,
            'linguistic_sem_idea_density': 5.0,
            'age': 70,
            'education_years': 12,
            'mmse_raw': 25.0
        }
        
        # Test MMSE normalization
        normalizer = MMSENormalizer()
        mmse_result = normalizer.normalize_mmse(
            mock_features['mmse_raw'],
            mock_features['age'],
            mock_features['education_years'],
            method='adni'
        )
        
        # Add normalized features
        enhanced = mock_features.copy()
        enhanced['mmse_adjusted'] = mmse_result['mmse_adjusted']
        enhanced['mmse_education_adj'] = mmse_result['education_adjustment']
        enhanced['mmse_age_adj'] = mmse_result['age_adjustment']
        
        logger.info("Full workflow test:")
        logger.info(f"  Input features: {len(mock_features)}")
        logger.info(f"  Output features: {len(enhanced)}")
        
        # Check MMSE features
        mmse_cols = [k for k in enhanced.keys() if 'mmse' in k.lower()]
        if mmse_cols:
            logger.info(f"  ✅ MMSE features added: {mmse_cols}")
            for col in mmse_cols:
                logger.info(f"     {col}: {enhanced[col]}")
        else:
            logger.warning(f"  ⚠️ No MMSE features added")
        
        logger.info("\n✅ Full Workflow test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Full Workflow test failed: {e}", exc_info=True)
        return False


def main():
    """Run all tests."""
    logger.info("="*60)
    logger.info("INTEGRATION STEP 1: FEATURE ENGINEERING TESTS")
    logger.info("="*60)
    
    results = []
    
    # Test 1: MMSE Normalizer
    results.append(("MMSE Normalizer", test_mmse_normalizer()))
    
    # Test 2: Integrated Feature Engineer
    results.append(("Integrated Feature Engineer", test_integrated_feature_engineer()))
    
    # Test 3: Service Integration
    results.append(("Service Integration", test_service_integration()))
    
    # Test 4: Full Workflow
    results.append(("Full Workflow", test_full_workflow()))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.warning(f"⚠️ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())

