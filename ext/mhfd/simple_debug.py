#!/usr/bin/env python
# coding: utf-8
"""
Simple debug script for autoscaler functionality testing
Tests basic autoscaler functionality without heavy dependencies
"""

import logging
import time
import random
from typing import Dict, List

# Import autoscaler and simulation components
from ext.mhfd.scaling.base_autoscaler import BaseAutoscaler
from ext.mhfd.autoscaler import create_heterogeneous_edge_autoscaler

# Import main simulation components
from ext.raith21.main import (
    SimulationConfig, setup_devices_and_topology, setup_oracles, 
    setup_scheduler_config, create_benchmark, setup_environment
)

# Core simulation imports
from sim.core import Environment
from sim.metrics import Metrics
from sim.logging import RuntimeLogger, SimulatedClock

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleAutoscalerTester:
    """Simple test suite for autoscaler functionality"""
    
    def __init__(self):
        self.test_results = {}
        
    def setup_test_environment(self):
        """Set up a test environment matching the main simulation structure"""
        logger.info("🔧 Setting up test environment using main simulation structure...")
        
        # Create configuration with minimal settings for testing
        config = SimulationConfig()
        config.num_devices = 5  # Very minimal for testing
        config.duration = 50
        config.total_rps = 25
        config.scenario = "custom"
        config.custom_counts = {
            "resnet50-inference": 1,  # Minimal for testing
            "speech-inference": 1,
            "resnet50-preprocessing": 0,
            "resnet50-training": 0,
            "python-pi": 0,
            "fio": 0,
        }
        
        # Set seeds for reproducible testing
        random.seed(config.random_seed)
        
        try:
            # Generate minimal infrastructure using main script functions
            devices, ether_nodes, topology, storage_index = setup_devices_and_topology(config)
            fet_oracle, resource_oracle, power_oracle = setup_oracles()
            
            # Configure scheduler
            sched_params = setup_scheduler_config(config, fet_oracle, resource_oracle)
            
            # Create benchmark
            benchmark = create_benchmark(config)
            
            # Setup environment using the main script's function
            env = setup_environment(topology, storage_index, sched_params, fet_oracle, resource_oracle, power_oracle, config)
            
            logger.info(f"✅ Test environment setup complete:")
            logger.info(f"   - Devices: {len(devices)}")
            logger.info(f"   - Topology nodes: {len(topology.nodes)}")
            logger.info(f"   - Benchmark duration: {config.duration}s")
            logger.info(f"   - Total RPS: {config.total_rps}")
            
            return env, benchmark, power_oracle, config
            
        except Exception as e:
            logger.error(f"❌ Failed to setup test environment: {e}")
            raise
    
    def test_autoscaler_instantiation(self, env, power_oracle, config):
        """Test if autoscaler can be created properly"""
        logger.info("🧪 Testing autoscaler instantiation...")
        
        test_results = {
            'test_name': 'autoscaler_instantiation',
            'success': False,
            'error': None,
            'autoscaler_type': None
        }
        
        try:
            # Create autoscaler instance using the main script's function
            autoscaler = create_heterogeneous_edge_autoscaler(
                env, env.faas, power_oracle, strategy=config.scaling_strategy
            )
            
            test_results['autoscaler_type'] = type(autoscaler).__name__
            test_results['success'] = True
            logger.info(f"✅ Autoscaler instantiation PASSED: {test_results['autoscaler_type']}")
            
            return autoscaler, test_results
            
        except Exception as e:
            test_results['error'] = str(e)
            logger.error(f"❌ Autoscaler instantiation ERROR: {e}")
            return None, test_results
    
    def test_basic_methods(self, autoscaler: BaseAutoscaler, deployment_name: str = "resnet50-inference"):
        """Test if basic autoscaler methods work"""
        logger.info(f"🔍 Testing basic methods for {deployment_name}...")
        
        test_results = {
            'test_name': 'basic_methods',
            'deployment': deployment_name,
            'methods_tested': {},
            'success': False,
            'error': None
        }
        
        try:
            # Test get_current_load method
            try:
                load = autoscaler.get_current_load(deployment_name)
                test_results['methods_tested']['get_current_load'] = {
                    'success': True,
                    'result': f"{load:.2f} RPS"
                }
                logger.info(f"  ✅ get_current_load: {load:.2f} RPS")
            except Exception as e:
                test_results['methods_tested']['get_current_load'] = {
                    'success': False,
                    'error': str(e)
                }
                logger.warning(f"  ❌ get_current_load failed: {e}")
            
            # Test get_average_response_time method
            try:
                response_time = autoscaler.get_average_response_time(deployment_name)
                test_results['methods_tested']['get_average_response_time'] = {
                    'success': True,
                    'result': f"{response_time:.1f} ms"
                }
                logger.info(f"  ✅ get_average_response_time: {response_time:.1f} ms")
            except Exception as e:
                test_results['methods_tested']['get_average_response_time'] = {
                    'success': False,
                    'error': str(e)
                }
                logger.warning(f"  ❌ get_average_response_time failed: {e}")
            
            # Test environment access
            try:
                env_check = hasattr(autoscaler, 'env') and autoscaler.env is not None
                test_results['methods_tested']['environment_access'] = {
                    'success': env_check,
                    'result': f"Environment accessible: {env_check}"
                }
                logger.info(f"  ✅ Environment access: {env_check}")
            except Exception as e:
                test_results['methods_tested']['environment_access'] = {
                    'success': False,
                    'error': str(e)
                }
                logger.warning(f"  ❌ Environment access failed: {e}")
            
            # Check success rate
            successful_methods = sum(1 for m in test_results['methods_tested'].values() if m['success'])
            total_methods = len(test_results['methods_tested'])
            
            if successful_methods >= 2:  # At least 2 methods should work
                test_results['success'] = True
                logger.info(f"✅ Basic methods test PASSED: {successful_methods}/{total_methods} methods working")
            else:
                logger.warning(f"❌ Basic methods test FAILED: Only {successful_methods}/{total_methods} methods working")
                
        except Exception as e:
            test_results['error'] = str(e)
            logger.error(f"❌ Basic methods test ERROR: {e}")
            
        self.test_results['basic_methods'] = test_results
        return test_results
    
    def test_topology_access(self, autoscaler: BaseAutoscaler):
        """Test if autoscaler can access topology information"""
        logger.info("🌐 Testing topology access...")
        
        test_results = {
            'test_name': 'topology_access',
            'success': False,
            'error': None,
            'topology_info': {}
        }
        
        try:
            # Check if topology is accessible
            if hasattr(autoscaler.env, 'topology') and autoscaler.env.topology:
                topology = autoscaler.env.topology
                
                # Count different types of nodes
                total_nodes = len(topology.nodes)
                compute_nodes = []
                infrastructure_nodes = []
                
                for node_name, node in topology.nodes.items():
                    name_str = str(node_name)
                    if any(keyword in name_str.lower() for keyword in ['link', 'switch', 'shared', 'registry']):
                        infrastructure_nodes.append(node_name)
                    else:
                        compute_nodes.append(node_name)
                
                test_results['topology_info'] = {
                    'total_nodes': total_nodes,
                    'compute_nodes': len(compute_nodes),
                    'infrastructure_nodes': len(infrastructure_nodes),
                    'sample_compute_nodes': [str(n) for n in compute_nodes[:3]]  # First 3
                }
                
                if len(compute_nodes) > 0:
                    test_results['success'] = True
                    logger.info(f"✅ Topology access PASSED:")
                    logger.info(f"   Total nodes: {total_nodes}")
                    logger.info(f"   Compute nodes: {len(compute_nodes)}")
                    logger.info(f"   Infrastructure nodes: {len(infrastructure_nodes)}")
                    logger.info(f"   Sample compute nodes: {test_results['topology_info']['sample_compute_nodes']}")
                else:
                    logger.warning("❌ Topology access FAILED: No compute nodes found")
            else:
                logger.warning("❌ Topology access FAILED: No topology available")
                
        except Exception as e:
            test_results['error'] = str(e)
            logger.error(f"❌ Topology access ERROR: {e}")
            
        self.test_results['topology_access'] = test_results
        return test_results
    
    def run_all_tests(self):
        """Run all autoscaler tests"""
        logger.info("🧪 Starting simple autoscaler tests...")
        
        try:
            # Set up test environment
            env, benchmark, power_oracle, config = self.setup_test_environment()
            
            # Test autoscaler instantiation
            autoscaler, instantiation_result = self.test_autoscaler_instantiation(env, power_oracle, config)
            self.test_results['autoscaler_instantiation'] = instantiation_result
            
            if autoscaler:
                # Run additional tests
                self.test_basic_methods(autoscaler)
                self.test_topology_access(autoscaler)
            else:
                logger.error("❌ Cannot run further tests without autoscaler instance")
            
            # Generate summary report
            self.generate_test_report()
            
        except Exception as e:
            logger.error(f"❌ Test suite failed: {e}")
            return {'error': str(e)}
        
        return self.test_results
    
    def generate_test_report(self):
        """Generate a simple test report"""
        logger.info("\n" + "="*50)
        logger.info("📋 SIMPLE AUTOSCALER TEST SUMMARY")
        logger.info("="*50)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result.get('success', False))
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {total_tests - passed_tests}")
        logger.info(f"Success Rate: {passed_tests/total_tests:.1%}" if total_tests > 0 else "Success Rate: 0%")
        
        logger.info("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
            logger.info(f"  {status} {test_name}")
            if result.get('error'):
                logger.info(f"    Error: {result['error']}")

def main():
    """Main test runner"""
    tester = SimpleAutoscalerTester()
    results = tester.run_all_tests()
    
    # Return exit code based on test results
    if 'error' in results:
        return 1
    
    passed_tests = sum(1 for result in results.values() if result.get('success', False))
    total_tests = len(results)
    
    if passed_tests == total_tests and total_tests > 0:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.warning(f"⚠️ {total_tests - passed_tests} tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())
