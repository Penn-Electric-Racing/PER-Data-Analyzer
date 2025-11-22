"""
Example usage of the PER CAN Search Engine.

Demonstrates how to use the search engine directly (without MCP).
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from per_mcp_server.can_search import CANSearch

# Load environment variables
load_dotenv()

def main():
    print("=== PER CAN Search Engine Demo ===\n")
    
    # Initialize search engine
    # Option 1: Use S3 (credentials from .env)
    search = CANSearch(use_s3=True)
    
    # Option 2: Use local file
    # search = CANSearch(can_defines_path='/path/to/GeneratedCanIds.xml', use_s3=False)
    
    if not search.indexed:
        print("❌ Failed to load CAN definitions")
        return
    
    print(f"✓ Loaded {len(search.variables)} CAN variables\n")
    
    # Example 1: Search for battery voltage
    print("Example 1: Search for 'battery voltage'")
    print("-" * 50)
    results = search.search("battery voltage", limit=5)
    for result in results:
        print(f"\nScore: {result['score']}/100")
        print(result['formatted'])
    
    # Example 2: Get specific variable by path
    print("\n\nExample 2: Get variable by path")
    print("-" * 50)
    var = search.get_by_path("pdu.sensors.batCurrent")
    if var:
        print(var.format_result())
    else:
        print("Variable not found")
    
    # Example 3: List all devices
    print("\n\nExample 3: List all devices")
    print("-" * 50)
    devices = search.list_devices()
    for device in devices:
        print(f"- {device['name']} (ID: {device['id']}): {device['count']} variables")
    
    # Example 4: Get all variables for PDU
    print("\n\nExample 4: Get PDU variables (first 10)")
    print("-" * 50)
    pdu_vars = search.get_by_device("pdu")
    for var in pdu_vars[:10]:
        print(f"- {var.path} ({var.type})")
    print(f"... and {len(pdu_vars) - 10} more")
    
    # Example 5: Fuzzy search with lower threshold
    print("\n\nExample 5: Fuzzy search for 'temp' (min_score=40)")
    print("-" * 50)
    results = search.search("temp", limit=10, min_score=40)
    for result in results:
        print(f"- {result['variable']['path']} (Score: {result['score']})")
    
    # Example 6: Search for wheel speeds
    print("\n\nExample 6: Search for 'wheel speed'")
    print("-" * 50)
    results = search.search("wheel speed", limit=5)
    for result in results:
        var = result['variable']
        print(f"- {var['path']}: {var['description']} (Score: {result['score']})")


if __name__ == "__main__":
    main()
