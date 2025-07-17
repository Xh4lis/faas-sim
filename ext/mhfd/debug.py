#!/usr/bin/env python3
"""
Simple debug for ResourceState without circular imports
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from sim.resource import ResourceUtilization, ResourceState


def debug_resource_state_methods():
    """Debug ResourceState methods and find the missing functionality"""
    
    print("🔧 DEBUG: ResourceState Methods Analysis")
    print("=" * 60)
    
    # Create a ResourceState directly
    resource_state = ResourceState()
    print(f"✅ Created ResourceState: {resource_state}")
    
    # Check available methods
    all_methods = [method for method in dir(resource_state) if not method.startswith('_')]
    print(f"\n📋 Available methods: {all_methods}")
    
    # Test each method to understand signatures
    print(f"\n🔍 Method signatures:")
    
    for method_name in all_methods:
        method = getattr(resource_state, method_name)
        if callable(method):
            try:
                import inspect
                sig = inspect.signature(method)
                print(f"  {method_name}{sig}")
            except:
                print(f"  {method_name}(...)")
    
    # Test put_resource method (the closest we have)
    print(f"\n🧪 Testing put_resource method:")
    
    # Create mock replica
    from unittest.mock import Mock
    mock_replica = Mock()
    mock_replica.node.name = "test_node"
    mock_replica.function.name = "test_function"
    
    # Try different ways to add resource data
    try:
        # Method 1: put_resource - this exists
        print("  Testing put_resource(replica, 'cpu', 0.25)...")
        resource_state.put_resource(mock_replica, 'cpu', 0.25)
        print("  ✅ put_resource worked!")
        
        # Method 2: Try multiple resources
        resource_state.put_resource(mock_replica, 'memory', 0.15)
        resource_state.put_resource(mock_replica, 'gpu', 0.05)
        print("  ✅ Multiple put_resource calls worked!")
        
        # Method 3: Test retrieval
        result = resource_state.list_resource_utilization("test_node")
        print(f"  📊 Retrieval result: {result}")
        print(f"  📊 Result type: {type(result)}")
        
        if result:
            print("  ✅ SUCCESS: ResourceState contains data!")
            for i, (replica, resource_util) in enumerate(result):
                print(f"    Entry {i}: replica={replica}, util={resource_util}")
                if resource_util and hasattr(resource_util, 'list_resources'):
                    resources = resource_util.list_resources()
                    print(f"    Resources: {resources}")
        else:
            print("  ❌ PROBLEM: No data retrieved")
        
    except Exception as e:
        print(f"  ❌ Error with put_resource: {e}")
    
    # Test get_resource_utilization method
    print(f"\n🧪 Testing get_resource_utilization:")
    try:
        util_result = resource_state.get_resource_utilization(mock_replica)
        print(f"  Result: {util_result}")
        if util_result and hasattr(util_result, 'list_resources'):
            resources = util_result.list_resources()
            print(f"  Resources: {resources}")
    except Exception as e:
        print(f"  Error: {e}")


def test_resource_utilization_creation():
    """Test creating ResourceUtilization objects manually"""
    
    print(f"\n\n🧪 TEST: Manual ResourceUtilization Creation")
    print("=" * 60)
    
    # Create ResourceUtilization
    util = ResourceUtilization()
    print(f"✅ Created ResourceUtilization: {util}")
    
    # Add resources
    util.put_resource('cpu', 0.30)
    util.put_resource('memory', 0.20)
    util.put_resource('gpu', 0.10)
    
    # Test retrieval
    resources = util.list_resources()
    print(f"📊 Resources: {resources}")
    
    # Test individual access
    print(f"🔍 Individual access:")
    print(f"  CPU: {util.get_resource('cpu')}")
    print(f"  Memory: {util.get_resource('memory')}")
    print(f"  GPU: {util.get_resource('gpu')}")
    
    return util


if __name__ == "__main__":
    print("🚀 Simplified ResourceState Debug Analysis")
    
    # Test ResourceUtilization first
    util = test_resource_utilization_creation()
    
    # Test ResourceState methods
    debug_resource_state_methods()
    
    print("\n\n✅ Simplified debug analysis complete!")