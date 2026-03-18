
import os
import sys

# Add current directory to path so we can import museum_rag_core
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from museum_rag_core import rag_answer
    print("Successfully imported museum_rag_core")
except ImportError as e:
    print(f"Failed to import museum_rag_core: {e}")
    sys.exit(1)

def test_persistence():
    # Simulate a follow-up question that caused issues
    question = "為何為鎖的形狀？"
    artifact_name = "玉鎖形佩"
    
    print(f"\nTesting with question: '{question}' and locked artifact: '{artifact_name}'")
    
    result = rag_answer(question=question, artifact_name=artifact_name)
    
    print(f"Answer: {result['answer'][:100]}...")
    print(f"Used Artifact: {result['artifact_name']}")
    
    # Check if any sources from other artifacts leaked in
    other_artifacts = [s for s in result['sources'] if s['title'] != artifact_name]
    
    if result['artifact_name'] == artifact_name and not other_artifacts:
        print("\n✅ Verification SUCCESS: Focus maintained on the locked artifact.")
    else:
        print("\n❌ Verification FAILED: Focus shifted or mixed results found.")
        if other_artifacts:
            print(f"Leaked artifacts: {[s['title'] for s in other_artifacts]}")

if __name__ == "__main__":
    # Ensure OPENAI_API_KEY is available or at least we can run the test
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not found. LLM part will be skipped, but retrieval logic can still be tested.")
    
    test_persistence()
