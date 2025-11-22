# PER Telemetry MCP Server

Model Context Protocol (MCP) server for Penn Electric Racing telemetry data and CAN variable search.

## Features

- **AI-Powered Search**: Uses Gemini for intelligent, semantic search of CAN variables
- **Natural Language Understanding**: Ask questions like "show me battery temperature sensors"
- **S3 Integration**: Automatically fetches GeneratedCanIds.xml from your S3 bucket
- **Smart Fallback**: Falls back to keyword search if Gemini is unavailable
- **MCP Integration**: Works with Claude Desktop, custom Gemini clients, and other MCP tools

## Installation

```bash
cd mcp
pip install -e .
```

## Configuration

Create a `.env` file in the `mcp/` directory:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# Gemini API Key (required for AI search)
# Get yours from: https://aistudio.google.com/app/apikey
GEMINI_API_KEY=your-gemini-api-key-here

# S3 Configuration (required for fetching GeneratedCanIds.xml)
S3_BUCKET=your-bucket-name
S3_KEY=your-access-key
S3_SECRET=your-secret-key
S3_ENDPOINT=https://your-endpoint.com

# Optional: Use local file instead of S3
#CAN_DEFINES_PATH=/path/to/GeneratedCanIds.xml

# Optional: Logging level
LOG_LEVEL=INFO
```

## Usage

### Quick Start with Claude Desktop or Gemini

**📖 See [GEMINI_SETUP.md](GEMINI_SETUP.md) for complete setup instructions!**

The easiest way to use this MCP server is with Claude Desktop:

1. Install: `pip install -e .`
2. Configure Claude Desktop (see [GEMINI_SETUP.md](GEMINI_SETUP.md))
3. Start asking about your CAN data!

### Running Standalone

```bash
python -m per_mcp_server
```

### Using with Claude Desktop

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "per-telemetry": {
      "command": "python",
      "args": ["-m", "per_mcp_server"],
      "cwd": "/path/to/car-data-server/mcp",
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

## Available Tools

The server provides these MCP tools:

### 1. `search_can_variables`
Search for CAN variables using natural language.

**Examples:**
- "battery voltage"
- "motor temperature"
- "wheel speed rear left"
- "accumulator current"

**Response includes:**
- Match score (0-100)
- Variable path (e.g., `pdu.sensors.batCurrent`)
- Device information
- Type, units, description
- CAN frequency

### 2. `get_can_variable`
Get a specific variable by its full path.

**Example:**
```
path: "pdu.sensors.batCurrent"
```

### 3. `list_device_variables`
List all variables for a specific device.

**Example:**
```
device: "pdu"
```

### 4. `list_can_devices`
List all CAN devices with variable counts.

### 5. `decode_can_message`
Decode a raw CAN message into human-readable format.

**Example:**
```
can_id: "0x123"
data: "01020304"
```

### 6. `get_telemetry`
Get current telemetry data from the car.

## How It Works

### Search Algorithm

Instead of using AI embeddings, we use **rapidfuzz** for fuzzy string matching:

1. **Indexing**: Parse GeneratedCanIds.xml and extract all variables into memory
2. **Searchable Text**: Each variable is converted to searchable text containing:
   - Name
   - Identifier
   - Path
   - Description
   - Device
   - Units
   - Type
3. **Fuzzy Matching**: Use WRatio algorithm to find best matches
4. **Ranking**: Return top N results with match scores

This approach provides:
- ✅ Instant results (no API calls)
- ✅ Deterministic matching (same query = same results)
- ✅ Explainable scores
- ✅ Works offline
- ✅ Zero cost

### S3 Integration

The server automatically fetches `GeneratedCanIds.xml` from your S3 bucket on startup:

1. Uses boto3 to connect to S3
2. Downloads GeneratedCanIds.xml to a temporary file
3. Parses and indexes all variables
4. Ready to serve search requests

You can also use a local file by setting `CAN_DEFINES_PATH` in your `.env`.

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black .
ruff check .
```

## Architecture

```
mcp/
├── per_mcp_server/
│   ├── __init__.py
│   ├── __main__.py          # Entry point
│   ├── server.py            # MCP server implementation
│   ├── can_search.py        # Fast search engine
│   ├── can_decoder.py       # CAN message decoder
│   └── telemetry.py         # Telemetry manager
├── tests/
│   ├── test_server.py
│   ├── test_can_search.py
│   ├── test_can_decoder.py
│   └── test_telemetry.py
├── pyproject.toml
├── .env.example
└── README.md
```

## Troubleshooting

### "CAN definitions not available"

Make sure either:
1. S3 credentials are correct in `.env` and GeneratedCanIds.xml exists in your bucket
2. Or set `CAN_DEFINES_PATH` to a local GeneratedCanIds.xml file

### Search returns no results

Try lowering the `min_score` parameter (default is 60):

```python
search_can_variables(query="battery", min_score=40)
```

### S3 connection fails

Check:
- S3 credentials are correct
- S3 endpoint URL is correct
- GeneratedCanIds.xml exists in the bucket with that exact name
- Network connectivity to S3 endpoint

## License

MIT
