import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import time

# Import your autoscaler and simulation components
from ext.mhfd.scaling.base_autoscaler import BaseAutoscaler
from ext.mhfd.scaling.strategies.standard_first_fit_packer import StandardFirstFitBinPacker
from ext.raith21.main import SimulationConfig, create_benchmark, create_topology, setup_scheduler_config
from sim.core import Environment
from sim.metrics import Metrics
from sim.logging import RuntimeLogger, SimulatedClock

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AutoscalerTester:
    """Comprehensive test suite for autoscaler functionality"""
    
    def __init__(self):
        self.test_results = {}
        self.simulation_data = {}
        
    def setup_test_environment(self):
        """Set up a minimal test environment"""
        logger.info("🔧 Setting up test environment...")
        
        config = SimulationConfig()
        config.num_devices = 50  # Smaller for testing
        config.duration = 120    # 2 minutes
        config.total_rps = 100   # Moderate load
        
        # Create topology and benchmark
        topology = create_topology(config)
        benchmark = create_benchmark(config)
        
        # Create environment
        env = Environment()
        env.metrics = Metrics(env, log=RuntimeLogger(SimulatedClock(env)))
        env.topology = topology
        
        # Simple FaaS system setup for testing
        from sim.faas.system import DefaultFaasSystem
        env.faas = DefaultFaasSystem(env, scale_by_requests=True)
        
        # Power oracle (mock for testing)
        class MockPowerOracle:
            def predict_power(self, *args, **kwargs):
                return 10.0  # Mock power consumption
        
        power_oracle = MockPowerOracle()
        
        return env, benchmark, power_oracle, config
    
    def test_load_tracking(self, autoscaler: BaseAutoscaler, deployment_name: str = "resnet50-inference"):
        """Test if load tracking works correctly"""
        logger.info(f"📊 Testing load tracking for {deployment_name}...")
        
        test_results = {
            'test_name': 'load_tracking',
            'deployment': deployment_name,
            'measurements': [],
            'success': False,
            'error': None
        }
        
        try:
            # Simulate multiple load measurements over time
            for i in range(5):
                # Advance simulation time
                autoscaler.env.now = i * 5.0  # Every 5 seconds
                
                # Mock some invocations
                if not hasattr(autoscaler.env.metrics, 'invocations'):
                    autoscaler.env.metrics.invocations = {}
                
                # Simulate increasing invocations
                autoscaler.env.metrics.invocations[deployment_name] = i * 10
                
                # Get load measurement
                load = autoscaler.get_current_load(deployment_name)
                
                measurement = {
                    'time': autoscaler.env.now,
                    'total_invocations': autoscaler.env.metrics.invocations[deployment_name],
                    'measured_load_rps': load,
                    'expected_load_rps': 2.0 if i > 0 else 0.0  # Should be 10/5 = 2.0 RPS
                }
                
                test_results['measurements'].append(measurement)
                logger.debug(f"  Time {autoscaler.env.now}s: Load = {load:.2f} RPS")
            
            # Validate results
            non_zero_loads = [m for m in test_results['measurements'] if m['measured_load_rps'] > 0]
            if len(non_zero_loads) >= 2:  # Should have load after first measurement
                test_results['success'] = True
                logger.info("✅ Load tracking test PASSED")
            else:
                logger.warning("❌ Load tracking test FAILED - No meaningful load detected")
                
        except Exception as e:
            test_results['error'] = str(e)
            logger.error(f"❌ Load tracking test ERROR: {e}")
            
        self.test_results['load_tracking'] = test_results
        return test_results
    
    def test_response_time_tracking(self, autoscaler: BaseAutoscaler, deployment_name: str = "resnet50-inference"):
        """Test if response time tracking works correctly"""
        logger.info(f"⏱️ Testing response time tracking for {deployment_name}...")
        
        test_results = {
            'test_name': 'response_time_tracking',
            'deployment': deployment_name,
            'measurements': [],
            'success': False,
            'error': None
        }
        
        try:
            # Create mock FET data
            current_time = 100.0
            autoscaler.env.now = current_time
            
            # Create mock FETs DataFrame
            mock_fets_data = []
            for i in range(10):  # 10 mock invocations
                start_time = current_time - 5.0 + i * 0.5  # In last 5 seconds
                execution_time = 0.1 + np.random.normal(0, 0.02)  # ~100ms ± 20ms
                wait_time = 0.05 + np.random.normal(0, 0.01)     # ~50ms ± 10ms
                
                mock_fets_data.append({
                    't_fet_start': start_time,
                    't_fet_end': start_time + execution_time,
                    't_wait_start': start_time,
                    't_wait_end': start_time + wait_time,
                    'function_name': deployment_name.replace('_', '-'),
                    'node': f'test_node_{i % 3}',
                    'replica_id': f'replica_{i % 2}',
                    'request_id': i
                })
            
            # Mock the extract_dataframe method
            mock_fets_df = pd.DataFrame(mock_fets_data)
            
            def mock_extract_dataframe(metric_type):
                if metric_type == "fets":
                    return mock_fets_df
                return pd.DataFrame()
            
            autoscaler.env.metrics.extract_dataframe = mock_extract_dataframe
            
            # Test response time measurement
            response_time = autoscaler.get_average_response_time(deployment_name)
            detailed_metrics = autoscaler.get_detailed_metrics(deployment_name)
            
            test_results['measurements'] = [{
                'avg_response_time': response_time,
                'detailed_metrics': detailed_metrics,
                'expected_range_ms': [120, 180],  # Expected ~150ms total
                'sample_count': len(mock_fets_data)
            }]
            
            # Validate results
            if 100 <= response_time <= 200:  # Reasonable response time range
                test_results['success'] = True
                logger.info(f"✅ Response time tracking test PASSED: {response_time:.1f}ms")
                if detailed_metrics:
                    logger.info(f"   Wait time: {detailed_metrics.get('avg_wait_time', 0):.1f}ms")
                    logger.info(f"   Execution time: {detailed_metrics.get('avg_execution_time', 0):.1f}ms")
            else:
                logger.warning(f"❌ Response time tracking test FAILED - Unexpected value: {response_time:.1f}ms")
                
        except Exception as e:
            test_results['error'] = str(e)
            logger.error(f"❌ Response time tracking test ERROR: {e}")
            
        self.test_results['response_time_tracking'] = test_results
        return test_results
    
    def test_scaling_decisions(self, autoscaler: BaseAutoscaler, deployment_name: str = "resnet50-inference"):
        """Test if scaling decisions are made correctly"""
        logger.info(f"🎯 Testing scaling decisions for {deployment_name}...")
        
        test_results = {
            'test_name': 'scaling_decisions',
            'deployment': deployment_name,
            'decisions': [],
            'success': False,
            'error': None
        }
        
        try:
            # Test scenarios: [current_replicas, load_rps, response_time_ms, expected_decision]
            test_scenarios = [
                (1, 5.0, 200.0, "no_action"),      # Low load, good response time
                (1, 60.0, 300.0, "scale_up"),      # High load, ok response time  
                (1, 30.0, 1500.0, "scale_up"),     # Medium load, high response time
                (5, 2.0, 100.0, "scale_down"),     # Low load, good response time, many replicas
                (10, 40.0, 800.0, "no_action"),    # Medium load, medium response time
            ]
            
            for current_replicas, load_rps, response_time_ms, expected in test_scenarios:
                # Test the decision logic directly
                decision = autoscaler.make_scaling_decision(
                    deployment_name, current_replicas, load_rps, response_time_ms
                )
                
                scenario_result = {
                    'current_replicas': current_replicas,
                    'load_rps': load_rps,
                    'response_time_ms': response_time_ms,
                    'expected_decision': expected,
                    'actual_decision': decision,
                    'correct': decision == expected
                }
                
                test_results['decisions'].append(scenario_result)
                
                status = "✅" if decision == expected else "❌"
                logger.info(f"  {status} Replicas: {current_replicas}, Load: {load_rps} RPS, "
                           f"Response: {response_time_ms}ms → {decision} (expected: {expected})")
            
            # Check success rate
            correct_decisions = sum(1 for d in test_results['decisions'] if d['correct'])
            success_rate = correct_decisions / len(test_scenarios)
            
            if success_rate >= 0.8:  # 80% success rate
                test_results['success'] = True
                logger.info(f"✅ Scaling decisions test PASSED: {success_rate:.1%} correct")
            else:
                logger.warning(f"❌ Scaling decisions test FAILED: Only {success_rate:.1%} correct")
                
        except Exception as e:
            test_results['error'] = str(e)
            logger.error(f"❌ Scaling decisions test ERROR: {e}")
            
        self.test_results['scaling_decisions'] = test_results
        return test_results
    
    def test_node_selection(self, autoscaler: BaseAutoscaler, deployment_name: str = "resnet50-inference"):
        """Test if node selection works"""
        logger.info(f"🖥️ Testing node selection for {deployment_name}...")
        
        test_results = {
            'test_name': 'node_selection',
            'deployment': deployment_name,
            'selections': [],
            'success': False,
            'error': None
        }
        
        try:
            # Test node selection multiple times
            for i in range(5):
                selected_node = autoscaler.select_node_for_scaling(deployment_name)
                
                selection_result = {
                    'attempt': i + 1,
                    'selected_node': selected_node.name if selected_node else None,
                    'node_type': autoscaler.extract_node_type(selected_node.name) if selected_node else None,
                    'success': selected_node is not None
                }
                
                test_results['selections'].append(selection_result)
                
                status = "✅" if selected_node else "❌"
                logger.info(f"  {status} Attempt {i+1}: Selected {selection_result['selected_node']} "
                           f"(type: {selection_result['node_type']})")
            
            # Check if any nodes were selected
            successful_selections = sum(1 for s in test_results['selections'] if s['success'])
            
            if successful_selections > 0:
                test_results['success'] = True
                logger.info(f"✅ Node selection test PASSED: {successful_selections}/5 successful")
            else:
                logger.warning("❌ Node selection test FAILED: No nodes selected")
                
        except Exception as e:
            test_results['error'] = str(e)
            logger.error(f"❌ Node selection test ERROR: {e}")
            
        self.test_results['node_selection'] = test_results
        return test_results
    
    def test_detailed_metrics_dataframe(self, autoscaler: BaseAutoscaler):
        """Test if detailed metrics can be converted to DataFrame"""
        logger.info("📊 Testing detailed metrics DataFrame conversion...")
        
        test_results = {
            'test_name': 'detailed_metrics_dataframe',
            'success': False,
            'error': None,
            'dataframe_info': {}
        }
        
        try:
            # First, populate some detailed metrics by calling get_average_response_time
            # (this would normally happen during autoscaling)
            
            # Mock some detailed metrics data
            if not hasattr(autoscaler, '_detailed_metrics'):
                autoscaler._detailed_metrics = {}
            
            autoscaler._detailed_metrics['resnet50-inference'] = {
                'avg_response_time': 150.0,
                'avg_execution_time': 100.0,
                'avg_wait_time': 50.0,
                'wait_percentage': 33.3,
                'high_wait_count': 2,
                'sample_count': 10,
                'timestamp': 100.0
            }
            
            autoscaler._detailed_metrics['speech-inference'] = {
                'avg_response_time': 80.0,
                'avg_execution_time': 60.0,
                'avg_wait_time': 20.0,
                'wait_percentage': 25.0,
                'high_wait_count': 0,
                'sample_count': 8,
                'timestamp': 100.0
            }
            
            # Test DataFrame conversion (you'll need to add these methods)
            df = autoscaler.get_detailed_metrics_df()
            
            test_results['dataframe_info'] = {
                'shape': df.shape if not df.empty else (0, 0),
                'columns': list(df.columns) if not df.empty else [],
                'deployments': list(df['deployment_name'].unique()) if 'deployment_name' in df.columns else []
            }
            
            if not df.empty and len(df) >= 2:
                test_results['success'] = True
                logger.info(f"✅ DataFrame test PASSED: {df.shape[0]} rows, {df.shape[1]} columns")
                logger.info(f"   Columns: {list(df.columns)}")
            else:
                logger.warning(f"❌ DataFrame test FAILED: Empty or insufficient data")
                
        except Exception as e:
            test_results['error'] = str(e)
            logger.error(f"❌ DataFrame test ERROR: {e}")
            
        self.test_results['detailed_metrics_dataframe'] = test_results
        return test_results
    
    def run_all_tests(self):
        """Run all autoscaler tests"""
        logger.info("🧪 Starting comprehensive autoscaler tests...")
        
        # Set up test environment
        env, benchmark, power_oracle, config = self.setup_test_environment()
        
        # Create autoscaler instance
        autoscaler = StandardFirstFitPacker(env, env.faas, power_oracle)
        
        # Run tests
        tests = [
            self.test_load_tracking,
            self.test_response_time_tracking,
            self.test_scaling_decisions,
            self.test_node_selection,
            self.test_detailed_metrics_dataframe
        ]
        
        for test_func in tests:
            try:
                test_func(autoscaler)
            except Exception as e:
                logger.error(f"❌ Test {test_func.__name__} failed with error: {e}")
        
        # Generate summary report
        self.generate_test_report()
        
        return self.test_results
    
    def generate_test_report(self):
        """Generate a comprehensive test report"""
        logger.info("\n" + "="*60)
        logger.info("📋 AUTOSCALER TEST SUMMARY REPORT")
        logger.info("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result.get('success', False))
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {total_tests - passed_tests}")
        logger.info(f"Success Rate: {passed_tests/total_tests:.1%}")
        
        logger.info("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
            logger.info(f"  {status} {test_name}")
            if result.get('error'):
                logger.info(f"    Error: {result['error']}")
        
        # Generate visual plot if possible
        try:
            self.plot_test_results()
        except Exception as e:
            logger.warning(f"Could not generate plots: {e}")
    
    def plot_test_results(self):
        """Create visual test results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Test success rate pie chart
        passed = sum(1 for r in self.test_results.values() if r.get('success', False))
        failed = len(self.test_results) - passed
        
        axes[0,0].pie([passed, failed], labels=['Passed', 'Failed'], 
                     colors=['green', 'red'], autopct='%1.1f%%')
        axes[0,0].set_title('Test Success Rate')
        
        # Load tracking data if available
        if 'load_tracking' in self.test_results:
            load_data = self.test_results['load_tracking'].get('measurements', [])
            if load_data:
                times = [m['time'] for m in load_data]
                loads = [m['measured_load_rps'] for m in load_data]
                axes[0,1].plot(times, loads, 'b-o')
                axes[0,1].set_title('Load Tracking Over Time')
                axes[0,1].set_xlabel('Time (s)')
                axes[0,1].set_ylabel('Load (RPS)')
        
        # Response time data if available
        if 'response_time_tracking' in self.test_results:
            rt_data = self.test_results['response_time_tracking'].get('measurements', [])
            if rt_data and rt_data[0].get('detailed_metrics'):
                metrics = rt_data[0]['detailed_metrics']
                components = ['avg_execution_time', 'avg_wait_time']
                values = [metrics.get(comp, 0) for comp in components]
                axes[1,0].bar(components, values, color=['blue', 'orange'])
                axes[1,0].set_title('Response Time Components')
                axes[1,0].set_ylabel('Time (ms)')
        
        # Scaling decisions accuracy
        if 'scaling_decisions' in self.test_results:
            decisions_data = self.test_results['scaling_decisions'].get('decisions', [])
            if decisions_data:
                correct = sum(1 for d in decisions_data if d['correct'])
                incorrect = len(decisions_data) - correct
                axes[1,1].bar(['Correct', 'Incorrect'], [correct, incorrect], 
                             color=['green', 'red'])
                axes[1,1].set_title('Scaling Decision Accuracy')
                axes[1,1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig('autoscaler_test_results.png', dpi=300, bbox_inches='tight')
        logger.info("📊 Test results plot saved as 'autoscaler_test_results.png'")
        plt.show()

def main():
    """Main test runner"""
    tester = AutoscalerTester()
    results = tester.run_all_tests()
    
    # Return exit code based on test results
    passed_tests = sum(1 for result in results.values() if result.get('success', False))
    total_tests = len(results)
    
    if passed_tests == total_tests:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.warning(f"⚠️ {total_tests - passed_tests} tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())