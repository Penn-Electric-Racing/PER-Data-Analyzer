import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Global cache for CAN variable definitions from XML
_can_defines_cache = None

def load_can_defines() -> str:
    """
    Parse GeneratedCanIds.xml and cache variable definitions.

    Returns condensed list of all CAN variables with descriptions for LLM context.
    """
    global _can_defines_cache

    # Return cached version if available
    if _can_defines_cache is not None:
        return _can_defines_cache

    # Check if CAN_DEFINES_PATH is configured
    can_defines_path = os.getenv("CAN_DEFINES_PATH")
    if not can_defines_path:
        logger.warning("CAN_DEFINES_PATH not configured - LLM won't have variable descriptions")
        _can_defines_cache = ""
        return ""

    if not os.path.exists(can_defines_path):
        logger.warning(f"CAN defines file not found: {can_defines_path}")
        _can_defines_cache = ""
        return ""

    try:
        from lxml import etree

        logger.info(f"Loading CAN definitions from: {can_defines_path}")
        tree = etree.parse(can_defines_path)
        root = tree.getroot()

        # Parse all variable definitions
        definitions = []
        for can_id_elem in root.findall('CanId'):
            for value_elem in can_id_elem.findall('Value'):
                path = value_elem.get('AccessString', '')
                name = value_elem.get('Name', '')
                desc = value_elem.get('Description', '')
                unit = value_elem.get('Unit', '')

                if path:  # Only include if we have a path
                    # Format: "path: Name - Description (unit)"
                    line = f"{path}: {name}"
                    if desc:
                        line += f" - {desc}"
                    if unit:
                        line += f" ({unit})"
                    definitions.append(line)

        # Cache the formatted list
        _can_defines_cache = "\n".join(definitions)
        logger.info(f"✓ Loaded {len(definitions)} CAN variable definitions")
        return _can_defines_cache

    except ImportError:
        logger.error("lxml not installed - cannot parse CAN defines XML")
        _can_defines_cache = ""
        return ""
    except Exception as e:
        logger.error(f"Failed to parse CAN defines: {e}", exc_info=True)
        _can_defines_cache = ""
        return ""
