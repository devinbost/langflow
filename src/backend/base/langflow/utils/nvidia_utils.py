import json
import logging
from typing import Optional, List, Dict, Any, Union

# Assuming Langchain Core message types are accessible via langflow.schema
# Adjust if direct langchain_core imports are standard for utility modules
from langflow.schema.message import AIMessage, AIMessageChunk
from langflow.field_typing.langchain_types import ToolCall # For AIMessage.tool_calls structure

logger = logging.getLogger(__name__)

def _deduplicate_if_doubled(s: Optional[str]) -> Optional[str]:
    """
    If a string 's' is a simple concatenation of two identical halves,
    returns the first half. Otherwise, returns the original string.
    """
    if not s or len(s) < 2:
        return s
    half_len = len(s) // 2
    if len(s) % 2 == 0 and s[:half_len] == s[half_len:]:
        # logger.debug(f"De-duplicating string '{s}' to '{s[:half_len]}'")
        return s[:half_len]
    return s

def _clean_tool_call_chunk_dict(tc_chunk_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Cleans name, args, and id within a single tool_call_chunk dictionary."""
    cleaned_chunk = tc_chunk_dict.copy()
    if "name" in cleaned_chunk and isinstance(cleaned_chunk["name"], str):
        cleaned_chunk["name"] = _deduplicate_if_doubled(cleaned_chunk["name"])
    if "id" in cleaned_chunk and isinstance(cleaned_chunk["id"], str):
        cleaned_chunk["id"] = _deduplicate_if_doubled(cleaned_chunk["id"])
    
    raw_args_str = cleaned_chunk.get("args")
    if isinstance(raw_args_str, str) and raw_args_str:
        args_half_len = len(raw_args_str) // 2
        is_perfect_double = len(raw_args_str) % 2 == 0 and raw_args_str[:args_half_len] == raw_args_str[args_half_len:]
        if is_perfect_double:
            try:
                json.loads(raw_args_str[:args_half_len]) 
                cleaned_chunk["args"] = raw_args_str[:args_half_len]
                # logger.debug(f"De-duplicating args string '{raw_args_str}' to '{cleaned_chunk['args']}'")
            except json.JSONDecodeError:
                cleaned_chunk["args"] = raw_args_str[:args_half_len]
        # No 'else' needed; if not a perfect double, args remain unchanged.
            
    return cleaned_chunk

def _clean_tool_call_model_dict(tc_model_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Cleans name and id for a ToolCall that is represented as a dictionary."""
    # This is used if AIMessage.tool_calls are dicts instead of Pydantic models directly
    cleaned_tc = tc_model_dict.copy()
    if "name" in cleaned_tc and isinstance(cleaned_tc["name"], str):
        cleaned_tc["name"] = _deduplicate_if_doubled(cleaned_tc["name"])
    if "id" in cleaned_tc and isinstance(cleaned_tc["id"], str):
        cleaned_tc["id"] = _deduplicate_if_doubled(cleaned_tc["id"])
    # Args in ToolCall dict are already parsed, no string de-duplication needed for the args value itself.
    return cleaned_tc

def clean_nvidia_message_content(message: Union[AIMessage, AIMessageChunk]) -> Union[AIMessage, AIMessageChunk]:
    """
    Cleans AIMessage or AIMessageChunk from NVIDIA model outputs
    by de-duplicating tool call name, args, id, and response_metadata fields.
    Modifies the message object in-place if mutable, or returns a modified copy.
    """
    if not isinstance(message, (AIMessage, AIMessageChunk)):
        # logger.warning(f"Attempted to clean non-AIMessage(Chunk) type: {type(message)}")
        return message

    # Langflow AIMessage/AIMessageChunk might be Pydantic models.
    # Direct assignment might be fine if they are mutable.
    # If they are immutable, we'd need to create new instances.
    # Assuming for now that direct assignment is okay for these fields.

    # Clean tool_call_chunks (List[Dict[str, Any]])
    if hasattr(message, "tool_call_chunks") and isinstance(message.tool_call_chunks, list):
        cleaned_chunks_list = []
        for chunk_dict in message.tool_call_chunks:
            if isinstance(chunk_dict, dict):
                 cleaned_chunks_list.append(_clean_tool_call_chunk_dict(chunk_dict))
            else:
                 cleaned_chunks_list.append(chunk_dict) 
        message.tool_call_chunks = cleaned_chunks_list

    # Clean tool_calls (List[ToolCall] for AIMessage, or List[Dict] if not pydantic models)
    if hasattr(message, "tool_calls") and isinstance(message.tool_calls, list):
        cleaned_tool_calls_list = []
        for tc_item in message.tool_calls:
            if isinstance(tc_item, ToolCall): # Pydantic model from langflow.field_typing
                # Assuming ToolCall model fields can be reassigned.
                # If ToolCall is immutable, this would need tc_item.copy(update={...})
                if isinstance(tc_item.name, str):
                    tc_item.name = _deduplicate_if_doubled(tc_item.name)
                if isinstance(tc_item.id, str):
                    tc_item.id = _deduplicate_if_doubled(tc_item.id)
                cleaned_tool_calls_list.append(tc_item)
            elif isinstance(tc_item, dict): # If it's a raw dict
                cleaned_tool_calls_list.append(_clean_tool_call_model_dict(tc_item))
            else:
                cleaned_tool_calls_list.append(tc_item)
        message.tool_calls = cleaned_tool_calls_list
    
    # Clean response_metadata string values
    if hasattr(message, "response_metadata") and isinstance(message.response_metadata, dict):
        cleaned_meta = {}
        for key, value in message.response_metadata.items():
            if isinstance(value, str):
                cleaned_meta[key] = _deduplicate_if_doubled(value)
            else:
                cleaned_meta[key] = value
        message.response_metadata = cleaned_meta
        
    # Clean id and content of the message itself
    if hasattr(message, "id") and isinstance(message.id, str) :
        message.id = _deduplicate_if_doubled(message.id)
    if hasattr(message, "content") and isinstance(message.content, str) :
        # Be careful with content, as it might be legitimate repeated patterns.
        # For now, applying the simple doubler. If issues arise, this could be removed for 'content'.
        message.content = _deduplicate_if_doubled(message.content)

    return message