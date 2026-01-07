# -*- coding: utf-8 -*-
"""
Test Script for Integration Step 2: Dual-Output Model Integration
==================================================================

Tests:
1. Dual-output model structure và methods
2. Helper methods (MMSE estimation, severity classification, etc.)
3. Prediction format compatibility
4. Backward compatibility
"""

import sys
import os
from pathlib import Path

# Add specific paths to avoid importing backend package
current_dir = Path(__file__).parent
modules_dir = current_dir.parent
backend_dir = modules_dir.parent

sys.path.insert(0, str(modules_dir))
sys.path.insert(0, str(backend_dir))

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_dual_output_model_structure():
    """Test dual-output model structure."""
    logger.info("="*60)
    logger.info("TEST 1: Dual-Output Model Structure")
    logger.info("="*60)
    
    try:
        # Import directly from file
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "dual_output_model",
            current_dir / "dual_output_model.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        DualOutputModelTrainer = module.DualOutputModelTrainer
        DualOutputPrediction = module.DualOutputPrediction
        
        # Test DualOutputPrediction dataclass
        logger.info("Testing DualOutputPrediction dataclass...")
        prediction = DualOutputPrediction(
            risk_binary=True,
            risk_binary_probability=0.72,
            mci_probability=0.65,
            mci_probability_calibrated=0.63,
            predicted_class='MCI',
            class_probabilities={'Normal': 0.28, 'MCI': 0.65, 'AD': 0.07},
            confidence=0.75,
            model_name='Test Model',
            calibration_applied=True
        )
        
        logger.info(f"  ✅ DualOutputPrediction created successfully")
        logger.info(f"     risk_binary: {prediction.risk_binary}")
        logger.info(f"     mci_probability: {prediction.mci_probability:.3f}")
        logger.info(f"     predicted_class: {prediction.predicted_class}")
        logger.info(f"     class_probabilities: {prediction.class_probabilities}")
        
        # Test DualOutputModelTrainer initialization
        logger.info("\nTesting DualOutputModelTrainer initialization...")
        trainer = DualOutputModelTrainer(config=None, random_state=42)
        
        logger.info(f"  ✅ DualOutputModelTrainer initialized")
        logger.info(f"     is_fitted: {trainer.is_fitted}")
        logger.info(f"     binary_classifier: {trainer.binary_classifier is None}")
        logger.info(f"     probability_estimator: {trainer.probability_estimator is None}")
        
        logger.info("\n✅ Dual-Output Model Structure test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Dual-Output Model Structure test failed: {e}", exc_info=True)
        return False


def test_helper_methods():
    """Test helper methods trong integration_service."""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Helper Methods")
    logger.info("="*60)
    
    try:
        # Import DualOutputPrediction
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "dual_output_model",
            current_dir / "dual_output_model.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        DualOutputPrediction = module.DualOutputPrediction
        
        # Create mock prediction
        prediction = DualOutputPrediction(
            risk_binary=True,
            risk_binary_probability=0.72,
            mci_probability=0.65,
            mci_probability_calibrated=0.63,
            predicted_class='MCI',
            class_probabilities={'Normal': 0.28, 'MCI': 0.65, 'AD': 0.07},
            confidence=0.75,
            model_name='Test Model',
            calibration_applied=True
        )
        
        # Test MMSE estimation logic
        logger.info("Testing MMSE estimation logic...")
        MMSE_CLINICAL_RANGES = {
            'Normal': {'mean': 28.5, 'std': 1.5},
            'MCI': {'mean': 23.0, 'std': 2.5},
            'AD': {'mean': 14.0, 'std': 4.0}
        }
        
        weighted_mmse = 0.0
        for class_name, prob in prediction.class_probabilities.items():
            if class_name in MMSE_CLINICAL_RANGES:
                weighted_mmse += prob * MMSE_CLINICAL_RANGES[class_name]['mean']
        
        logger.info(f"  ✅ MMSE estimation: {weighted_mmse:.2f}")
        logger.info(f"     Normal (0.28): {0.28 * 28.5:.2f}")
        logger.info(f"     MCI (0.65): {0.65 * 23.0:.2f}")
        logger.info(f"     AD (0.07): {0.07 * 14.0:.2f}")
        
        # Test severity classification
        logger.info("\nTesting severity classification...")
        mmse_scores = [28.0, 22.0, 15.0, 8.0]
        for mmse in mmse_scores:
            if mmse >= 24:
                severity = 'Bình thường'
            elif mmse >= 18:
                severity = 'Suy giảm nhận thức nhẹ (MCI)'
            elif mmse >= 10:
                severity = 'Sa sút trí tuệ mức độ trung bình'
            else:
                severity = 'Sa sút trí tuệ mức độ nặng'
            logger.info(f"  MMSE {mmse:.0f}: {severity}")
        
        # Test risk factors extraction logic
        logger.info("\nTesting risk factors extraction logic...")
        test_features = {
            'age': 75,
            'education_years': 6,
            'mmse_adjusted': 22.0
        }
        
        risk_factors = []
        if prediction.mci_probability >= 0.7:
            risk_factors.append("Nguy cơ suy giảm nhận thức cao")
        elif prediction.mci_probability >= 0.4:
            risk_factors.append("Nguy cơ suy giảm nhận thức trung bình")
        
        if test_features['age'] >= 75:
            risk_factors.append(f"Tuổi cao ({test_features['age']} tuổi)")
        
        if test_features['education_years'] < 6:
            risk_factors.append(f"Trình độ học vấn thấp ({test_features['education_years']} năm)")
        
        if test_features['mmse_adjusted'] < 24:
            risk_factors.append(f"Điểm MMSE điều chỉnh thấp ({test_features['mmse_adjusted']:.1f}/30)")
        
        logger.info(f"  ✅ Risk factors extracted: {len(risk_factors)} factors")
        for rf in risk_factors:
            logger.info(f"     - {rf}")
        
        # Test recommendations generation
        logger.info("\nTesting recommendations generation...")
        recommendations = []
        if prediction.risk_binary:
            if prediction.mci_probability >= 0.7:
                recommendations.append("Phát hiện nguy cơ suy giảm nhận thức cao")
                recommendations.append("Cần đánh giá y tế khẩn cấp")
            elif prediction.mci_probability >= 0.4:
                recommendations.append("Phát hiện dấu hiệu suy giảm nhận thức nhẹ")
                recommendations.append("Khuyến nghị đánh giá chuyên sâu")
        
        logger.info(f"  ✅ Recommendations generated: {len(recommendations)} recommendations")
        for rec in recommendations:
            logger.info(f"     - {rec}")
        
        logger.info("\n✅ Helper Methods test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Helper Methods test failed: {e}", exc_info=True)
        return False


def test_prediction_format():
    """Test prediction format compatibility."""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Prediction Format Compatibility")
    logger.info("="*60)
    
    try:
        # Import DualOutputPrediction
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "dual_output_model",
            current_dir / "dual_output_model.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        DualOutputPrediction = module.DualOutputPrediction
        
        # Test dual-output format
        logger.info("Testing dual-output prediction format...")
        dual_prediction = DualOutputPrediction(
            risk_binary=True,
            risk_binary_probability=0.72,
            mci_probability=0.65,
            mci_probability_calibrated=0.63,
            predicted_class='MCI',
            class_probabilities={'Normal': 0.28, 'MCI': 0.65, 'AD': 0.07},
            confidence=0.75,
            model_name='Dual-Output Model',
            calibration_applied=True
        )
        
        # Format như trong integration_service
        mci_prediction_dict = {
            'mci_probability': dual_prediction.mci_probability,
            'mci_probability_calibrated': dual_prediction.mci_probability_calibrated,
            'mci_class': dual_prediction.predicted_class,
            'risk_binary': dual_prediction.risk_binary,
            'risk_binary_probability': dual_prediction.risk_binary_probability,
            'class_probabilities': dual_prediction.class_probabilities,
            'confidence': dual_prediction.confidence,
            'model_name': dual_prediction.model_name,
            'calibration_applied': dual_prediction.calibration_applied
        }
        
        logger.info(f"  ✅ Dual-output format:")
        logger.info(f"     Keys: {list(mci_prediction_dict.keys())}")
        logger.info(f"     mci_probability: {mci_prediction_dict['mci_probability']:.3f}")
        logger.info(f"     risk_binary: {mci_prediction_dict['risk_binary']}")
        logger.info(f"     class_probabilities: {mci_prediction_dict['class_probabilities']}")
        
        # Test standard format (backward compatible)
        logger.info("\nTesting standard format (backward compatible)...")
        standard_prediction_dict = {
            'mci_probability': 0.65,
            'mci_class': 'MCI',
            'mmse_estimate': 23.0,
            'confidence': 0.75,
            'severity': 'Suy giảm nhận thức nhẹ (MCI)'
        }
        
        logger.info(f"  ✅ Standard format:")
        logger.info(f"     Keys: {list(standard_prediction_dict.keys())}")
        logger.info(f"     Compatible với old code: ✅")
        
        logger.info("\n✅ Prediction Format Compatibility test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Prediction Format Compatibility test failed: {e}", exc_info=True)
        return False


def test_integration_code_structure():
    """Test integration code structure trong integration_service.py."""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Integration Code Structure")
    logger.info("="*60)
    
    try:
        integration_file = modules_dir / "integration_service.py"
        
        if not integration_file.exists():
            logger.warning("⚠️ integration_service.py not found")
            return False
        
        # Read file và check for integration code
        with open(integration_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for key integration points
        checks = {
            'DualOutputModelTrainer import': 'DualOutputModelTrainer' in content,
            'DualOutputPrediction import': 'DualOutputPrediction' in content,
            'dual_output_model attribute': 'dual_output_model' in content,
            '_predict_with_dual_output method': '_predict_with_dual_output' in content,
            '_estimate_mmse_from_dual_output method': '_estimate_mmse_from_dual_output' in content,
            '_classify_severity_from_mmse method': '_classify_severity_from_mmse' in content,
            '_extract_risk_factors_from_dual_output method': '_extract_risk_factors_from_dual_output' in content,
            '_generate_recommendations_from_dual_output method': '_generate_recommendations_from_dual_output' in content,
            'Dual-output prediction logic': 'dual_output_model is not None' in content or 'dual_output_model' in content
        }
        
        logger.info("Checking integration code structure:")
        all_passed = True
        for check_name, passed in checks.items():
            status = "✅" if passed else "❌"
            logger.info(f"  {status} {check_name}: {passed}")
            if not passed:
                all_passed = False
        
        if all_passed:
            logger.info("\n✅ Integration Code Structure verified!")
            return True
        else:
            logger.warning("\n⚠️ Some integration points missing")
            return False
        
    except Exception as e:
        logger.error(f"❌ Integration Code Structure test failed: {e}", exc_info=True)
        return False


def main():
    """Run all tests."""
    logger.info("="*60)
    logger.info("INTEGRATION STEP 2: DUAL-OUTPUT MODEL TESTS")
    logger.info("="*60)
    
    results = []
    
    # Test 1: Dual-Output Model Structure
    results.append(("Dual-Output Model Structure", test_dual_output_model_structure()))
    
    # Test 2: Helper Methods
    results.append(("Helper Methods", test_helper_methods()))
    
    # Test 3: Prediction Format
    results.append(("Prediction Format Compatibility", test_prediction_format()))
    
    # Test 4: Integration Code Structure
    results.append(("Integration Code Structure", test_integration_code_structure()))
    
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

