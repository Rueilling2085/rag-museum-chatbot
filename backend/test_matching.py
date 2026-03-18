# test_matching.py
import sys
import os

# 將目前目錄加入 path 才能 import museum_rag_core
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from museum_rag_core import find_best_artifact_match, GLOBAL_ARTIFACTS
    
    print(f"Loaded {len(GLOBAL_ARTIFACTS)} artifacts.")
    
    test_cases = [
        "提梁壺",
        "提樑壺",
        "青瓷",
        "海棠式茶壺",
        "琺瑯蓋碗"
    ]
    
    for case in test_cases:
        print(f"\nTesting: '{case}'")
        matches = find_best_artifact_match(case)
        if matches:
            for i, (name, score) in enumerate(matches[:3]):
                print(f"  {i+1}. {name} (Score: {score:.4f})")
        else:
            print("  No matches found.")
            
except Exception as e:
    print(f"Error during testing: {e}")
    import traceback
    traceback.print_exc()
