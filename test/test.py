import numpy as np
import sys
import traceback

def pause_on_error():
    print("\n" + "=" * 60 )
    input("Press Enter to continue after error...")
    

try:
    from ann_engine.core import Tensor, Parameter
    from ann_engine.layers import ReLU
except ImportError as e:
    print(f"Falied to Import lbraries : {e}")
    pause_on_error()
    sys.exit(1)
    
def run_test(test_name, test_func):
    print(f"\n" + "=" * 50)
    print(f"Running test : {test_name}")
    print(f"\n" + "=" * 50)
    
    try:
        result = test_func()
        print("\n Test Passed : {test_name}")
        return True, result
    except AssertionError as e:
        print(f"\n test failed (Assertion Error): {test_name}")
        print(f" {e}")
        print("\n" + "-" * 50)
        traceback.print_exc()
        pause_on_error()
        return False , None
    
    
    except Exception as e:
        print('\n Test  Failed (Exception): {test_name}')
        print(f" {e}")
        print("\n" + "-" * 50)
        traceback.print_exc()
        pause_on_error()
        return False , None
    

        
        