#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple sanity check script for factor_ut and graph_build integration
Avoids circular import issues by testing components separately

Run from FYP_salm directory: python sim/test.py
"""
import numpy as np
import math
import logging
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("SimpleSanityCheck")

def test_factor_ut_import():
    """Test factor_ut module import and basic functionality"""
    logger.info("🧪 Testing factor_ut import...")
    
    try:
        # Import factor_ut components one by one
        from algorithm.frontend.factor_ut import PriorFactor
        logger.info("  ✅ PriorFactor imported successfully")
        
        from algorithm.frontend.factor_ut import OdometryFactor
        logger.info("  ✅ OdometryFactor imported successfully")
        
        from algorithm.frontend.factor_ut import BearingRangeUTFactor
        logger.info("  ✅ BearingRangeUTFactor imported successfully")
        
        from algorithm.frontend.factor_ut import wrap_angle
        logger.info("  ✅ wrap_angle imported successfully")
        
        # Test basic functionality
        angle = wrap_angle(3 * math.pi)
        expected = -math.pi
        assert abs(angle - expected) < 1e-10, f"wrap_angle failed: {angle} != {expected}"
        logger.info("  ✅ wrap_angle function working correctly")
        
        return True, {
            'PriorFactor': PriorFactor,
            'OdometryFactor': OdometryFactor, 
            'BearingRangeUTFactor': BearingRangeUTFactor,
            'wrap_angle': wrap_angle
        }
        
    except ImportError as e:
        logger.error("  ❌ factor_ut import failed: %s", e)
        return False, {}
    except Exception as e:
        logger.error("  ❌ factor_ut test failed: %s", e)
        return False, {}

def test_factor_creation(factor_classes):
    """Test creating individual factors"""
    logger.info("🔧 Testing factor creation...")
    
    try:
        PriorFactor = factor_classes['PriorFactor']
        OdometryFactor = factor_classes['OdometryFactor']
        BearingRangeUTFactor = factor_classes['BearingRangeUTFactor']
        
        # Test PriorFactor creation
        prior = PriorFactor("x0_0", np.array([0, 0, 0]), np.array([0.1, 0.1, 0.05]))
        logger.info("  ✅ PriorFactor created successfully")
        
        # Test OdometryFactor creation
        odom = OdometryFactor("x0_0", "x0_1", np.array([1.0, 0.0, 0.0]), np.array([0.1, 0.1, 0.05]))
        logger.info("  ✅ OdometryFactor created successfully")
        
        # Test BearingRangeUTFactor creation
        R = np.diag([math.radians(2)**2, 0.1**2])
        br_factor = BearingRangeUTFactor(
            "x0_0", "l_0", 
            np.array([math.pi/4, math.sqrt(2)]),
            R,
            mode="gbp"
        )
        logger.info("  ✅ BearingRangeUTFactor created successfully")
        
        return True, [prior, odom, br_factor]
        
    except Exception as e:
        logger.error("  ❌ Factor creation failed: %s", e)
        return False, []

def test_factor_linearization(factors):
    """Test factor linearization"""
    logger.info("📐 Testing factor linearization...")
    
    try:
        prior, odom, br_factor = factors
        
        # Test PriorFactor linearization
        test_mu = {"x0_0": np.array([0.1, 0.1, 0.1])}
        test_cov = {"x0_0": np.eye(3) * 0.1}
        result = prior.linearize(test_mu, test_cov)
        assert "x0_0" in result, "Prior linearization missing variable"
        logger.info("  ✅ PriorFactor linearization successful")
        
        # Test OdometryFactor linearization
        test_mu = {"x0_0": np.array([0, 0, 0]), "x0_1": np.array([1, 0, 0])}
        test_cov = {"x0_0": np.eye(3) * 0.1, "x0_1": np.eye(3) * 0.1}
        result = odom.linearize(test_mu, test_cov)
        assert "x0_0" in result and "x0_1" in result, "Odometry linearization missing variables"
        logger.info("  ✅ OdometryFactor linearization successful")
        
        # Test BearingRangeUTFactor linearization
        test_mu = {"x0_0": np.array([0, 0, 0]), "l_0": np.array([1, 1])}
        test_cov = {
            "x0_0": np.eye(3) * 0.1, 
            "l_0": np.eye(2) * 0.1,
            ("x0_0", "l_0"): np.zeros((3, 2))
        }
        result = br_factor.linearize(test_mu, test_cov)
        assert "x0_0" in result and "l_0" in result, "BearingRange linearization missing variables"
        logger.info("  ✅ BearingRangeUTFactor linearization successful")
        
        return True
        
    except Exception as e:
        logger.error("  ❌ Factor linearization failed: %s", e)
        import traceback
        traceback.print_exc()
        return False

def test_simple_graph_build():
    """Test a very simple graph build without imports"""
    logger.info("🏗️  Testing simple manual graph building...")
    
    try:
        # Import only what we need, when we need it
        from algorithm.frontend.factor_ut import PriorFactor, OdometryFactor
        
        # Create a simple manual factor graph
        factors = []
        variables = {}
        
        # Simple 2-pose trajectory
        pose0 = np.array([0.0, 0.0, 0.0])
        pose1 = np.array([1.0, 0.0, 0.0])
        
        variables["x0_0"] = pose0
        variables["x0_1"] = pose1
        
        # Add prior factor
        prior_factor = PriorFactor("x0_0", pose0, np.array([0.1, 0.1, 0.05]))
        factors.append(prior_factor)
        
        # Add odometry factor
        delta = np.array([1.0, 0.0, 0.0])  # 1m forward
        odom_factor = OdometryFactor("x0_0", "x0_1", delta, np.array([0.1, 0.1, 0.05]))
        factors.append(odom_factor)
        
        logger.info("  Manual graph created:")
        logger.info("    Variables: %d", len(variables))
        logger.info("    Factors: %d", len(factors))
        logger.info("  ✅ Simple manual graph building successful")
        
        return True
        
    except Exception as e:
        logger.error("  ❌ Simple graph building failed: %s", e)
        import traceback
        traceback.print_exc()
        return False

def test_graph_build_import():
    """Test graph_build import separately"""
    logger.info("🏭 Testing graph_build import...")
    
    try:
        # Import graph_build AFTER factor_ut is fully loaded
        from sim.graph_build import GBPGraphBuilder
        logger.info("  ✅ EnhancedGBPGraphBuilder imported successfully")
        
        # Try to create builder
        builder = GBPGraphBuilder()
        logger.info("  ✅ EnhancedGBPGraphBuilder created successfully")
        
        return True, builder
        
    except ImportError as e:
        logger.error("  ❌ graph_build import failed: %s", e)
        return False, None
    except Exception as e:
        logger.error("  ❌ graph_build test failed: %s", e)
        return False, None

def test_integrated_build(builder):
    """Test integrated graph building"""
    logger.info("🚀 Testing integrated graph building...")
    
    try:
        # Create very simple test data
        robot_trajectory = np.array([
            [0.0, 0.0, 0.0],  # Start
            [1.0, 0.0, 0.0],  # 1m east
            [1.0, 1.0, math.pi/2]  # 1m north, turn 90 degrees
        ])
        
        landmarks = np.array([
            [0.5, 0.5],  # Simple landmark
            [1.5, 0.5]   # Another landmark
        ])
        
        # Create simple measurements
        measurements = [
            {
                "type": "bearing_range",
                "robot": 0,
                "time": 0,
                "id": 0,
                "bearing": math.atan2(0.5, 0.5),  # bearing to landmark 0
                "range": math.hypot(0.5, 0.5),   # distance to landmark 0
                "bearing_range": [math.atan2(0.5, 0.5), math.hypot(0.5, 0.5)]
            }
        ]
        
        # Try to build graph
        factors, variables = builder.build_single_robot(robot_trajectory, landmarks, measurements)
        
        logger.info("  Integrated build results:")
        logger.info("    Variables: %d", len(variables))
        logger.info("    Factors: %d", len(factors))
        
        # Basic checks
        assert len(factors) > 0, "No factors created"
        assert len(variables) > 0, "No variables created"
        
        # Check variable types
        pose_vars = [k for k in variables if k.startswith('x')]
        landmark_vars = [k for k in variables if k.startswith('l')]
        
        logger.info("    Pose variables: %d", len(pose_vars))
        logger.info("    Landmark variables: %d", len(landmark_vars))
        
        logger.info("  ✅ Integrated graph building successful")
        return True
        
    except Exception as e:
        logger.error("  ❌ Integrated build failed: %s", e)
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests in order"""
    logger.info("🚀 Starting Simple Sanity Check")
    logger.info("Testing factor_ut and graph_build integration")
    logger.info("=" * 50)
    
    test_results = {}
    
    # Test 1: factor_ut import and basic functionality
    success, factor_classes = test_factor_ut_import()
    test_results["factor_ut_import"] = success
    if not success:
        logger.error("💥 Cannot continue without factor_ut")
        return False
    
    # Test 2: Factor creation
    success, factors = test_factor_creation(factor_classes)
    test_results["factor_creation"] = success
    if not success:
        logger.error("💥 Cannot continue without factor creation")
        return False
    
    # Test 3: Factor linearization
    success = test_factor_linearization(factors)
    test_results["factor_linearization"] = success
    
    # Test 4: Simple manual graph building
    success = test_simple_graph_build()
    test_results["simple_graph_build"] = success
    
    # Test 5: graph_build import (separately)
    success, builder = test_graph_build_import()
    test_results["graph_build_import"] = success
    
    # Test 6: Integrated building (only if graph_build import succeeded)
    if success and builder:
        success = test_integrated_build(builder)
        test_results["integrated_build"] = success
    else:
        logger.info("⚠️  Skipping integrated build test (graph_build not available)")
        test_results["integrated_build"] = None
    
    # Summary
    logger.info("=" * 50)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = 0
    total = 0
    
    for test_name, result in test_results.items():
        if result is None:
            status = "⏭️  SKIP"
        elif result:
            status = "✅ PASS"
            passed += 1
            total += 1
        else:
            status = "❌ FAIL"
            total += 1
        
        logger.info("  %-20s: %s", test_name, status)
    
    logger.info("-" * 30)
    if total > 0:
        logger.info("Overall: %d/%d tests passed (%.1f%%)", passed, total, (passed/total)*100)
        
        if passed == total:
            logger.info("🎉 ALL TESTS PASSED!")
            logger.info("The basic factor_ut functionality works correctly!")
            if test_results.get("integrated_build"):
                logger.info("Integration with graph_build also works!")
            return True
        else:
            logger.error("💥 Some tests failed.")
            return False
    else:
        logger.error("💥 No tests could be run.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("💥 Unexpected error: %s", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)