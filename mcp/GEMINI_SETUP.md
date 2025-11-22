# Using MCP Server with Gemini

This guide shows you how to use your PER MCP server with Google's Gemini.

## Prerequisites

1. **Google AI Studio API Key**: Get one from https://aistudio.google.com/app/apikey
2. **MCP Server Setup**: Your MCP server installed and configured
3. **S3 Access**: Credentials configured in `.env` file

## Option 1: Use with Claude Desktop (Easiest)

Claude Desktop has built-in MCP support and works great with your server:

### 1. Install Dependencies

```bash
cd /Users/ericcao/Documents/GitHub/car-data-server/mcp
pip install -e .
```

### 2. Configure Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "per-telemetry": {
      "command": "python",
      "args": ["-m", "per_mcp_server"],
      "cwd": "/Users/ericcao/Documents/GitHub/car-data-server/mcp",
      "env": {
        "S3_BUCKET": "your-bucket-name",
        "S3_KEY": "your-access-key",
        "S3_SECRET": "your-secret-key",
        "S3_ENDPOINT": "https://your-endpoint.com"
      }
    }
  }
}
```

### 3. Restart Claude Desktop

Your MCP server tools will now be available in Claude!

Try asking:
- "Search for battery voltage variables"
- "What CAN devices are available?"
- "Show me all motor temperature sensors"

---

## Option 2: Use with Gemini via MCP Client

You can use Gemini to interact with your MCP server using the MCP client library:

### 1. Install MCP Client

```bash
pip install mcp anthropic google-generativeai
```

### 2. Create Gemini Client Script

Create `gemini_client.py`:

```python
import asyncio
import os
from google import generativeai as genai
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-pro')

async def main():
    # Connect to MCP server
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "per_mcp_server"],
        cwd="/Users/ericcao/Documents/GitHub/car-data-server/mcp",
        env={
            "S3_BUCKET": os.getenv("S3_BUCKET"),
            "S3_KEY": os.getenv("S3_KEY"),
            "S3_SECRET": os.getenv("S3_SECRET"),
            "S3_ENDPOINT": os.getenv("S3_ENDPOINT"),
        }
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            
            # List available tools
            tools = await session.list_tools()
            print("Available MCP Tools:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")
            
            # Use Gemini to interact with MCP tools
            while True:
                user_input = input("\nYou: ")
                if user_input.lower() in ['exit', 'quit']:
                    break
                
                # You would typically:
                # 1. Use Gemini to understand the user's intent
                # 2. Determine which MCP tool to call
                # 3. Call the tool via session.call_tool()
                # 4. Send results back to Gemini for natural response
                
                response = model.generate_content(user_input)
                print(f"\nGemini: {response.text}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. Set Environment Variables

```bash
export GEMINI_API_KEY="your-gemini-api-key"
export S3_BUCKET="your-bucket-name"
export S3_KEY="your-access-key"
export S3_SECRET="your-secret-key"
export S3_ENDPOINT="https://your-endpoint.com"
```

### 4. Run the Client

```bash
python gemini_client.py
```

---

## Option 3: Direct MCP Server Testing

Test your MCP server directly without Gemini:

### 1. Start the Server

```bash
cd /Users/ericcao/Documents/GitHub/car-data-server/mcp
python -m per_mcp_server
```

### 2. Test with MCP Inspector

Install and run the MCP inspector:

```bash
npx @modelcontextprotocol/inspector python -m per_mcp_server
```

This opens a web UI where you can:
- See all available tools
- Test tool calls
- View responses
- Debug your MCP server

---

## Quick Start (Recommended)

The **easiest way** to use your MCP server is with **Claude Desktop**:

1. Make sure your `.env` file has S3 credentials:
   ```bash
   cd /Users/ericcao/Documents/GitHub/car-data-server/mcp
   cat .env  # Check credentials are set
   ```

2. Install the MCP server:
   ```bash
   pip install -e .
   ```

3. Test it works:
   ```bash
   python -m per_mcp_server
   # Should start without errors
   # Press Ctrl+C to stop
   ```

4. Configure Claude Desktop (see Option 1 above)

5. Start asking Claude about your CAN data!
   - "What battery sensors do we have?"
   - "Find all temperature variables"
   - "Show me motor controller variables"

---

## Troubleshooting

### Server won't start
- Check S3 credentials in `.env`
- Verify GeneratedCanIds.xml exists in S3 bucket
- Run `pip install -e .` to install dependencies

### No tools available
- Make sure S3 credentials are in Claude Desktop config
- Check server logs for errors
- Verify GeneratedCanIds.xml downloaded successfully

### Search returns no results
- Confirm GeneratedCanIds.xml is properly formatted
- Try lowering min_score parameter
- Check that variables were indexed (look for log message)

---

## Example Queries

Once connected to Claude Desktop or Gemini:

**Search queries:**
- "Find battery voltage variables"
- "What temperature sensors are on the AMS?"
- "Show me all motor torque variables"
- "List wheel speed sensors"

**Direct access:**
- "Get the variable at path pdu.sensors.batCurrent"
- "Show me all variables for the PCM device"
- "List all CAN devices"

**Analysis:**
- "What's the difference between batCurrent and pwrCurrent?"
- "Explain the AMS fault signals"
- "What motor variables can I monitor?"
