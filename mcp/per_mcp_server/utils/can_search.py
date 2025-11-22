"""
CAN Variable Search Engine

AI-powered semantic search for CAN variables from GeneratedCanIds.xml.
Uses Gemini for intelligent natural language understanding.
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from lxml import etree
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("google-generativeai not installed. Install with: pip install google-generativeai")

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class CANVariable:
    """Represents a single CAN variable with all metadata."""
    
    identifier: str
    name: str
    device: str
    device_id: int
    can_id: str  # CAN ID like "607"
    type: str
    path: str  # Full path like "pdu.sensors.batCurrent"
    
    # Optional fields
    description: str = ""
    units: str = ""
    default: str = ""
    min_value: str = ""
    max_value: str = ""
    can_frequency: str = ""
    array_dimensions: str = ""
    enum_values: str = ""
    
    # Hierarchy information
    parent_struct: str = ""
    is_enum: bool = False
    is_struct: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "identifier": self.identifier,
            "name": self.name,
            "device": self.device,
            "device_id": self.device_id,
            "can_id": self.can_id,
            "type": self.type,
            "path": self.path,
            "description": self.description,
            "units": self.units,
            "default": self.default,
            "min": self.min_value,
            "max": self.max_value,
            "can_frequency": self.can_frequency,
            "array_dimensions": self.array_dimensions,
            "enum_values": self.enum_values,
            "parent_struct": self.parent_struct,
            "is_enum": self.is_enum,
            "is_struct": self.is_struct,
        }
    
    def to_text(self) -> str:
        """Convert to human-readable text for matching."""
        parts = [
            self.name,
            self.identifier,
            self.path,
            self.description,
            self.device,
            self.units,
            self.type,
        ]
        return " ".join(p for p in parts if p)
    
    def format_result(self) -> str:
        """Format as a readable search result."""
        result = f"**{self.name}** (`{self.path}`)\n"
        result += f"  - CAN ID: {self.can_id}\n"
        result += f"  - Device: {self.device} (ID: {self.device_id})\n"
        result += f"  - Type: {self.type}"
        if self.units:
            result += f" ({self.units})"
        result += "\n"
        if self.description:
            result += f"  - Description: {self.description}\n"
        if self.enum_values:
            result += f"  - Values: {self.enum_values}\n"
        if self.can_frequency:
            result += f"  - CAN Frequency: {self.can_frequency}\n"
        return result


class CANDefinesParser:
    """Parser for GeneratedCanIds.xml file."""
    
    def __init__(self, xml_path: str):
        """
        Initialize parser.
        
        Args:
            xml_path: Path to GeneratedCanIds.xml
        """
        self.xml_path = xml_path
        self.tree = etree.parse(xml_path)
        self.root = self.tree.getroot()
    
    def parse_all(self) -> List[CANVariable]:
        """Parse all variables from the XML file."""
        variables = []
        
        # GeneratedCanIds structure: <CanIds> -> <CanId> -> <Value>
        for can_id_elem in self.root.findall('CanId'):
            can_id = can_id_elem.get('Id', '')
            device_id = int(can_id_elem.get('DeviceId', 0))
            can_frequency = can_id_elem.get('CanFrequency', '')
            
            # Parse each Value in this CanId
            for value_elem in can_id_elem.findall('Value'):
                var = self._parse_value(value_elem, can_id, device_id, can_frequency)
                if var:
                    variables.append(var)
        
        return variables
    
    def _parse_value(
        self,
        elem,
        can_id: str,
        device_id: int,
        can_frequency: str
    ) -> Optional[CANVariable]:
        """Parse a Value element from GeneratedCanIds.xml."""
        access_string = elem.get('AccessString', '')
        if not access_string:
            return None
        
        # Parse device name from AccessString (e.g., "ams.dcdc.temp" -> device is "ams")
        parts = access_string.split('.')
        if len(parts) < 1:
            return None
        
        device_identifier = parts[0]
        
        # Map device identifiers to full names (you might want to expand this)
        device_name_map = {
            'ams': 'Accumulator Management System',
            'pdu': 'Power Distribution Unit',
            'pcm': 'Powertrain Control Module',
            'moc': 'Motor Controller',
            'ludwig': 'Ludwig',
            'playground': 'Playground',
            'playgroundrpi': 'Playground Rpi',
        }
        device_name = device_name_map.get(device_identifier, device_identifier.upper())
        
        # Check if this is an array element (e.g., "cellV[0]")
        identifier = parts[-1] if len(parts) > 0 else ''
        # Remove array index from identifier if present
        if '[' in identifier:
            identifier = identifier.split('[')[0]
        
        return CANVariable(
            identifier=identifier,
            name=elem.get('Name', ''),
            device=device_name,
            device_id=device_id,
            can_id=can_id,
            type=elem.get('Type', ''),
            path=access_string,
            description=elem.get('Description', ''),
            units=elem.get('Unit', ''),  # Note: 'Unit' not 'Units' in GeneratedCanIds
            default=elem.get('Default', ''),
            min_value='',  # Not in GeneratedCanIds
            max_value='',  # Not in GeneratedCanIds
            can_frequency=elem.get('CanFrequency', can_frequency),
            array_dimensions=elem.get('ArrayDimensions', ''),
            enum_values=elem.get('EnumValues', ''),
            parent_struct='.'.join(parts[1:-1]) if len(parts) > 2 else '',
            is_enum=bool(elem.get('EnumValues')),
            is_struct=False,
        )


class CANSearch:
    """
    AI-powered semantic search engine for CAN variables.
    Uses Gemini for intelligent natural language understanding.
    """
    
    def __init__(
        self,
        can_defines_path: Optional[str] = None,
        use_s3: bool = True,
        use_gemini: bool = True,
    ):
        """
        Initialize the search engine.
        
        Args:
            can_defines_path: Path to local GeneratedCanIds.xml (optional)
            use_s3: Whether to try fetching from S3 if local path not available (default: True)
            use_gemini: Whether to use Gemini for AI-powered search (default: True)
        """
        self.variables: List[CANVariable] = []
        self.indexed = False
        self.use_gemini = use_gemini and GEMINI_AVAILABLE
        self.gemini_model = None
        
        # Configure Gemini if available
        if self.use_gemini:
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if gemini_api_key:
                try:
                    genai.configure(api_key=gemini_api_key)
                    self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
                    logger.info("Gemini AI search enabled")
                except Exception as e:
                    logger.warning(f"Failed to configure Gemini: {e}")
                    self.use_gemini = False
            else:
                logger.warning("GEMINI_API_KEY not set. AI search disabled.")
                self.use_gemini = False
        
        # Try to get XML file
        xml_path = self._get_xml_file(can_defines_path, use_s3)
        
        if xml_path:
            self.index_definitions(xml_path)
    
    def _get_xml_file(
        self,
        local_path: Optional[str],
        use_s3: bool,
    ) -> Optional[str]:
        """Get GeneratedCanIds.xml from local path or S3."""
        
        # Try local path first
        if local_path and Path(local_path).exists():
            logger.info(f"Using local GeneratedCanIds.xml: {local_path}")
            return local_path
        
        # Try S3 if enabled
        if use_s3:
            return self._download_from_s3()
        
        logger.warning("No GeneratedCanIds.xml file available")
        return None
    
    def _download_from_s3(self) -> Optional[str]:
        """
        Download GeneratedCanIds.xml from S3 to a temporary file.
        Uses the same S3 client pattern as gateway/gateway.py.
        """
        s3_bucket = os.getenv("S3_BUCKET")
        s3_key = os.getenv("S3_KEY")
        s3_secret = os.getenv("S3_SECRET")
        s3_endpoint = os.getenv("S3_ENDPOINT")
        
        if not all([s3_bucket, s3_key, s3_secret, s3_endpoint]):
            logger.warning("S3 credentials not fully configured")
            return None
        
        try:
            logger.info(f"Fetching GeneratedCanIds.xml from S3 bucket: {s3_bucket}")
            
            # Create S3 client (same pattern as gateway)
            s3 = boto3.client(
                "s3",
                aws_access_key_id=s3_key,
                aws_secret_access_key=s3_secret,
                endpoint_url=s3_endpoint
            )
            
            # Get the file from S3 (same key as gateway uses)
            obj = s3.get_object(Bucket=s3_bucket, Key="GeneratedCanIds.xml")
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(
                mode='wb',
                suffix='.xml',
                delete=False
            )
            
            # Write S3 content to temp file
            temp_file.write(obj["Body"].read())
            temp_file.close()
            
            logger.info(f"Downloaded GeneratedCanIds.xml to: {temp_file.name}")
            return temp_file.name
            
        except ClientError as e:
            logger.error(f"Failed to fetch GeneratedCanIds.xml from S3: {e}")
            return None
    
    def index_definitions(self, xml_path: str):
        """Parse and index all CAN variables."""
        logger.info(f"Indexing CAN definitions from: {xml_path}")
        
        parser = CANDefinesParser(xml_path)
        self.variables = parser.parse_all()
        self.indexed = True
        
        logger.info(f"Indexed {len(self.variables)} CAN variables")
    
    def search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Search for CAN variables using Gemini AI.
        
        Args:
            query: Natural language search query
            limit: Maximum number of results
            min_score: Minimum relevance score (0.0-1.0)
        
        Returns:
            List of matching variables with scores
        """
        if not self.indexed:
            logger.warning("CAN definitions not indexed yet")
            return []
        
        if self.use_gemini and self.gemini_model:
            return self._search_with_gemini(query, limit, min_score)
        else:
            logger.warning("Gemini not available, search disabled")
            return []
    
    def _search_with_gemini(
        self,
        query: str,
        limit: int,
        min_score: float
    ) -> List[Dict[str, Any]]:
        """Use Gemini to intelligently search for CAN variables."""
        try:
            # Create a concise list of all variables for Gemini
            var_list = []
            for i, var in enumerate(self.variables):
                var_info = f"{i}|{var.can_id}|{var.path}|{var.name}|{var.device}|{var.type}|{var.description}|{var.units}"
                var_list.append(var_info)
            
            # Build prompt for Gemini
            prompt = f"""You are a CAN variable search assistant for an electric race car.

Given this search query: "{query}"

Find the most relevant CAN variables from this list. Each line has format:
INDEX|CAN_ID|PATH|NAME|DEVICE|TYPE|DESCRIPTION|UNITS

Variables:
{chr(10).join(var_list[:2000])}  

Return ONLY a JSON array of objects with:
- "index": the variable index number
- "score": relevance score from 0.0 to 1.0
- "reason": brief reason why it matches

Return up to {limit} results with score >= {min_score}.
Only return highly relevant matches.

Example format:
[
  {{"index": 42, "score": 0.95, "reason": "Battery voltage measurement"}},
  {{"index": 17, "score": 0.82, "reason": "Related voltage sensor"}}
]"""

            # Query Gemini
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # Parse Gemini's response
            import json
            gemini_results = json.loads(response_text)
            
            # Build final results
            results = []
            for result in gemini_results:
                idx = result.get("index")
                score = result.get("score", 0.0)
                reason = result.get("reason", "")
                
                if idx is not None and 0 <= idx < len(self.variables):
                    var = self.variables[idx]
                    results.append({
                        "variable": var.to_dict(),
                        "score": int(score * 100),  # Convert to 0-100 scale for consistency
                        "reason": reason,
                        "formatted": var.format_result()
                    })
            
            logger.info(f"Gemini found {len(results)} matches for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Gemini search failed: {e}")
            logger.info("Falling back to simple keyword search")
            return self._fallback_search(query, limit, min_score)
    
    def _fallback_search(
        self,
        query: str,
        limit: int,
        min_score: float
    ) -> List[Dict[str, Any]]:
        """Simple keyword-based fallback search."""
        query_lower = query.lower()
        results = []
        
        for var in self.variables:
            score = 0.0
            searchable = var.to_text().lower()
            
            # Simple keyword matching
            query_words = query_lower.split()
            matches = sum(1 for word in query_words if word in searchable)
            
            if matches > 0:
                score = matches / len(query_words)
                
                if score >= min_score:
                    results.append({
                        "variable": var.to_dict(),
                        "score": int(score * 100),
                        "formatted": var.format_result()
                    })
        
        # Sort by score and limit
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
    
    def get_by_path(self, path: str) -> Optional[CANVariable]:
        """
        Get a variable by its full path.
        
        Args:
            path: Full path like "pdu.sensors.batCurrent"
        
        Returns:
            Variable if found, None otherwise
        """
        for var in self.variables:
            if var.path == path:
                return var
        return None
    
    def get_by_identifier(self, identifier: str) -> List[CANVariable]:
        """
        Get all variables with matching identifier.
        
        Args:
            identifier: Variable identifier
        
        Returns:
            List of matching variables
        """
        return [var for var in self.variables if var.identifier == identifier]
    
    def get_by_device(self, device: str) -> List[CANVariable]:
        """
        Get all variables for a specific device.
        
        Args:
            device: Device name or identifier
        
        Returns:
            List of variables for that device
        """
        return [
            var for var in self.variables
            if var.device.lower() == device.lower() or var.path.startswith(device.lower())
        ]
    
    def list_devices(self) -> List[Dict[str, Any]]:
        """
        List all devices in the CAN definitions.
        
        Returns:
            List of devices with counts
        """
        devices = {}
        for var in self.variables:
            if var.device not in devices:
                devices[var.device] = {
                    "name": var.device,
                    "id": var.device_id,
                    "count": 0
                }
            devices[var.device]["count"] += 1
        
        return list(devices.values())
