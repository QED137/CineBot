#!/usr/bin/env python3
"""
Test all query types to verify the RAG system works correctly
"""
import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_query(query_text, description):
    """Test a single query"""
    print(f"\n{'='*70}")
    print(f"TEST: {description}")
    print(f"Query: '{query_text}'")
    print(f"{'='*70}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/chat",
            json={"message": query_text},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get('response', 'No response')
            movies = data.get('movies', [])
            
            print(f"\n✓ SUCCESS!")
            print(f"\nResponse:\n{answer[:500]}...")  # First 500 chars
            print(f"\nMovies found: {len(movies)}")
            if movies:
                print("\nTop 3 movies:")
                for i, movie in enumerate(movies[:3], 1):
                    print(f"  {i}. {movie.get('title', 'Unknown')} - {movie.get('overview', '')[:80]}...")
            return True
        else:
            print(f"\n✗ FAILED! Status: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        return False

def main():
    """Run all tests"""
    
    print("\n" + "="*70)
    print("CINEBOT RAG SYSTEM TEST SUITE")
    print("="*70)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        print(f"\n✓ Server is running on {BASE_URL}")
    except:
        print(f"\n✗ ERROR: Server is not running on {BASE_URL}")
        print("Please start the server with: python3 app.py")
        return
    
    results = {}
    
    # Test 1: Genre Query
    results['genre'] = test_query(
        "tell me about romantic movies",
        "Genre Query - Should use HAS_GENRE relationships"
    )
    
    # Test 2: Factual Query - Director
    results['director'] = test_query(
        "who directed titanic",
        "Factual Query - Should generate Cypher for director"
    )
    
    # Test 3: Factual Query - Actor
    results['actor'] = test_query(
        "who acted in inception",
        "Factual Query - Should generate Cypher for actors"
    )
    
    # Test 4: Semantic Search
    results['semantic'] = test_query(
        "movies about love and sacrifice",
        "Semantic Search - Should use text embeddings"
    )
    
    # Test 5: Another Genre Query
    results['genre2'] = test_query(
        "show me action movies",
        "Genre Query - Action genre"
    )
    
    # Test 6: Descriptive Query
    results['description'] = test_query(
        "beautiful love story with happy ending",
        "Descriptive Query - Should use embeddings and genre"
    )
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for test_name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print("="*70 + "\n")
    
    if passed == total:
        print("🎉 All tests passed! Your CineBot is working perfectly!")
    else:
        print("⚠ Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()
