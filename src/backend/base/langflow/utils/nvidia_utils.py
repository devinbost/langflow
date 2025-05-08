import json
import logging
from typing import Optional, List, Dict, Any, Union

# Assuming Langchain Core message types are accessible via langflow.schema
# Adjust if direct langchain_core imports are standard for utility modules
# Change imports to use langchain_core directly
from langchain_core.messages import AIMessage, AIMessageChunk, ToolCall
# from langflow.schema.message import AIMessage, AIMessageChunk # REMOVED
# from langflow.field_typing.langchain_types import ToolCall # REMOVED

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
        logger.debug(f"NVIDIA_UTILS: De-duplicating string '{s}' to '{s[:half_len]}'")
        return s[:half_len]
    return s

def _clean_tool_call_chunk_dict(tc_chunk_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Cleans name, args, and id within a single tool_call_chunk dictionary."""
    logger.debug(f"NVIDIA_UTILS: _clean_tool_call_chunk_dict attempting to clean chunk: {tc_chunk_dict}")
    cleaned_chunk = tc_chunk_dict.copy()
    original_name = cleaned_chunk.get("name")
    if "name" in cleaned_chunk and isinstance(cleaned_chunk["name"], str):
        cleaned_chunk["name"] = _deduplicate_if_doubled(cleaned_chunk["name"])
        if original_name != cleaned_chunk["name"]:
            logger.debug(f"NVIDIA_UTILS: Cleaned tool_call_chunk name from '{original_name}' to '{cleaned_chunk['name']}'")

    original_id = cleaned_chunk.get("id")
    if "id" in cleaned_chunk and isinstance(cleaned_chunk["id"], str):
        cleaned_chunk["id"] = _deduplicate_if_doubled(cleaned_chunk["id"])
        if original_id != cleaned_chunk["id"]:
            logger.debug(f"NVIDIA_UTILS: Cleaned tool_call_chunk id from '{original_id}' to '{cleaned_chunk['id']}'")

    raw_args_str = cleaned_chunk.get("args")
    logger.debug(f"NVIDIA_UTILS: _clean_tool_call_chunk_dict raw_args_str: {raw_args_str}")
    if isinstance(raw_args_str, str) and raw_args_str:
        args_half_len = len(raw_args_str) // 2
        is_perfect_double = len(raw_args_str) % 2 == 0 and raw_args_str[:args_half_len] == raw_args_str[args_half_len:]
        if is_perfect_double:
            potential_dedup_args = raw_args_str[:args_half_len]
            logger.debug(f"NVIDIA_UTILS: tool_call_chunk args '{raw_args_str}' is a perfect double. Potential dedup: '{potential_dedup_args}'")
            try:
                json.loads(potential_dedup_args)
                original_args = cleaned_chunk["args"]
                cleaned_chunk["args"] = potential_dedup_args
                logger.debug(f"NVIDIA_UTILS: De-duplicating valid JSON args string in tool_call_chunk from '{original_args}' to '{cleaned_chunk['args']}'")
            except json.JSONDecodeError as e:
                logger.debug(f"NVIDIA_UTILS: Potential dedup args '{potential_dedup_args}' is not valid JSON ({e}). Still using for deduplication as per original logic.")
                original_args = cleaned_chunk["args"]
                cleaned_chunk["args"] = potential_dedup_args
                logger.debug(f"NVIDIA_UTILS: De-duplicating non-JSON args string in tool_call_chunk from '{original_args}' to '{cleaned_chunk['args']}'")
        else:
            logger.debug(f"NVIDIA_UTILS: tool_call_chunk args '{raw_args_str}' is not a perfect double, no deduplication applied to args.")
    elif raw_args_str is not None:
        logger.debug(f"NVIDIA_UTILS: tool_call_chunk args is not a string or is empty: {raw_args_str}, type: {type(raw_args_str)}")
    return cleaned_chunk

def _clean_tool_call_model_dict(tc_model_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Cleans name and id for a ToolCall that is represented as a dictionary."""
    logger.debug(f"NVIDIA_UTILS: _clean_tool_call_model_dict attempting to clean model_dict: {tc_model_dict}")
    cleaned_tc = tc_model_dict.copy()
    original_name = cleaned_tc.get("name")
    if "name" in cleaned_tc and isinstance(cleaned_tc["name"], str):
        cleaned_tc["name"] = _deduplicate_if_doubled(cleaned_tc["name"])
        if original_name != cleaned_tc["name"]:
            logger.debug(f"NVIDIA_UTILS: Cleaned tool_call (dict) name from '{original_name}' to '{cleaned_tc['name']}'")

    original_id = cleaned_tc.get("id")
    if "id" in cleaned_tc and isinstance(cleaned_tc["id"], str):
        cleaned_tc["id"] = _deduplicate_if_doubled(cleaned_tc["id"])
        if original_id != cleaned_tc["id"]:
            logger.debug(f"NVIDIA_UTILS: Cleaned tool_call (dict) id from '{original_id}' to '{cleaned_tc['id']}'")

    # Args in ToolCall dict are already parsed, no string de-duplication needed for the args value itself.
    # However, logging them can be useful.
    args_content = cleaned_tc.get("args")
    if args_content is not None:
        logger.debug(f"NVIDIA_UTILS: _clean_tool_call_model_dict args content (type: {type(args_content)}): {str(args_content)[:200]}")
    else:
        logger.debug("NVIDIA_UTILS: _clean_tool_call_model_dict args content is None.")
    return cleaned_tc

def clean_nvidia_message_content(message: Union[AIMessage, AIMessageChunk]) -> Union[AIMessage, AIMessageChunk]:
    """
    Cleans AIMessage or AIMessageChunk from NVIDIA model outputs
    by de-duplicating tool call name, args, id, and response_metadata fields.
    Modifies the message object in-place if mutable, or returns a modified copy.
    """
    if hasattr(message, 'tool_calls') and message.tool_calls:
        logger.debug(f"NVIDIA_UTILS: Entering clean_nvidia_message_content. Initial tool_calls: {message.tool_calls}")
    if hasattr(message, 'tool_call_chunks') and message.tool_call_chunks:
        logger.debug(f"NVIDIA_UTILS: Entering clean_nvidia_message_content. Initial tool_call_chunks: {message.tool_call_chunks}")

    if not isinstance(message, (AIMessage, AIMessageChunk)):
        logger.warning(f"NVIDIA_UTILS: Attempted to clean non-AIMessage(Chunk) type: {type(message)}")
        return message

    # Langflow AIMessage/AIMessageChunk might be Pydantic models.
    # Direct assignment might be fine if they are mutable.
    # If they are immutable, we'd need to create new instances.
    # Assuming for now that direct assignment is okay for these fields.

    # Clean tool_call_chunks (List[Dict[str, Any]])
    if hasattr(message, "tool_call_chunks") and isinstance(message.tool_call_chunks, list):
        logger.debug(f"NVIDIA_UTILS: Cleaning tool_call_chunks: {message.tool_call_chunks}")
        cleaned_chunks_list = []
        for chunk_idx, chunk_dict in enumerate(message.tool_call_chunks):
            if isinstance(chunk_dict, dict):
                 cleaned_chunk = _clean_tool_call_chunk_dict(chunk_dict)
                 if cleaned_chunk != chunk_dict:
                     logger.debug(f"NVIDIA_UTILS: Tool call chunk at index {chunk_idx} was modified from {chunk_dict} to {cleaned_chunk}")
                 cleaned_chunks_list.append(cleaned_chunk)
            else:
                 logger.warning(f"NVIDIA_UTILS: Tool call chunk at index {chunk_idx} is not a dict: {chunk_dict}")
                 cleaned_chunks_list.append(chunk_dict)
        message.tool_call_chunks = cleaned_chunks_list

    # Clean tool_calls (List[ToolCall] for AIMessage, or List[Dict] if not pydantic models)
    if hasattr(message, "tool_calls") and isinstance(message.tool_calls, list):
        logger.debug(f"NVIDIA_UTILS: Cleaning tool_calls: {message.tool_calls}")
        cleaned_tool_calls_list = []
        for tc_idx, tc_item in enumerate(message.tool_calls):
            original_tc_item_repr = str(tc_item)
            modified_tc_item = None
            if isinstance(tc_item, ToolCall): # Pydantic model from langflow.field_typing
                changed = False
                new_name = tc_item.name
                new_id = tc_item.id

                if isinstance(tc_item.name, str):
                    dedup_name = _deduplicate_if_doubled(tc_item.name)
                    if dedup_name != tc_item.name:
                        new_name = dedup_name
                        changed = True
                if isinstance(tc_item.id, str):
                    dedup_id = _deduplicate_if_doubled(tc_item.id)
                    if dedup_id != tc_item.id:
                        new_id = dedup_id
                        changed = True
                
                if changed:
                    try:
                        modified_tc_item = tc_item.copy(update={"name": new_name, "id": new_id})
                    except AttributeError:
                        modified_tc_item = ToolCall(name=new_name, args=tc_item.args, id=new_id, type=tc_item.type) 
                    logger.debug(f"NVIDIA_UTILS: ToolCall object at index {tc_idx} was modified from {original_tc_item_repr} to {modified_tc_item}")
                    cleaned_tool_calls_list.append(modified_tc_item)
                else:
                    cleaned_tool_calls_list.append(tc_item)

            elif isinstance(tc_item, dict): # If it's a raw dict
                cleaned_dict = _clean_tool_call_model_dict(tc_item)
                if cleaned_dict != tc_item:
                    logger.debug(f"NVIDIA_UTILS: ToolCall dict at index {tc_idx} was modified from {tc_item} to {cleaned_dict}")
                cleaned_tool_calls_list.append(cleaned_dict)
            else:
                logger.warning(f"NVIDIA_UTILS: ToolCall item at index {tc_idx} is not a ToolCall object or dict: {tc_item}")
                cleaned_tool_calls_list.append(tc_item)
        message.tool_calls = cleaned_tool_calls_list
    
    # Clean response_metadata string values
    if hasattr(message, "response_metadata") and isinstance(message.response_metadata, dict):
        logger.debug(f"NVIDIA_UTILS: Cleaning response_metadata: {message.response_metadata}")
        cleaned_meta = {}
        metadata_changed = False
        for key, value in message.response_metadata.items():
            original_value = value
            if isinstance(value, str):
                cleaned_value = _deduplicate_if_doubled(value)
                if cleaned_value != original_value:
                    logger.debug(f"NVIDIA_UTILS: Cleaned response_metadata key '{key}' from '{original_value}' to '{cleaned_value}'")
                    metadata_changed = True
                cleaned_meta[key] = cleaned_value
            else:
                cleaned_meta[key] = value
        if metadata_changed:
            logger.debug(f"NVIDIA_UTILS: response_metadata was modified. Original: {message.response_metadata}, New: {cleaned_meta}")
        message.response_metadata = cleaned_meta
        
    # Clean id and content of the message itself
    original_id = getattr(message, "id", None)
    if hasattr(message, "id") and isinstance(message.id, str) :
        message.id = _deduplicate_if_doubled(message.id)
        if original_id != message.id:
            logger.debug(f"NVIDIA_UTILS: Message ID was cleaned from '{original_id}' to '{message.id}'")

    original_content = getattr(message, "content", None)
    if hasattr(message, "content") and isinstance(message.content, str) :
        # Be careful with content, as it might be legitimate repeated patterns.
        # For now, applying the simple doubler. If issues arise, this could be removed for 'content'.
        message.content = _deduplicate_if_doubled(message.content)
        if original_content != message.content:
            logger.debug(f"NVIDIA_UTILS: Message content was cleaned from \'{str(original_content)[:100]}...\' to \'{str(message.content)[:100]}...\'")

    # Option 2: Use implied continuation within parentheses
    has_calls = hasattr(message, 'tool_calls') and message.tool_calls
    has_chunks = hasattr(message, 'tool_call_chunks') and message.tool_call_chunks
    if has_calls or has_chunks:
        logger.debug(f"NVIDIA_UTILS: Exiting clean_nvidia_message_content. Final tool_calls: {getattr(message, 'tool_calls', 'N/A')}, Final tool_call_chunks: {getattr(message, 'tool_call_chunks', 'N/A')}")

    return message

def _validate_and_clean_message_content(msg_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Ensures message content is non-empty for NVIDIA API compatibility."""
    logger.debug(f"NVIDIA_UTILS: _validate_and_clean_message_content input: {msg_dict}")
    cleaned_msg = msg_dict.copy()
    original_content = cleaned_msg.get('content')
    msg_type = cleaned_msg.get('role', 'unknown')
    content = cleaned_msg.get('content')

    # Check for empty or None content
    if content is None or content == "":
        if cleaned_msg.get("tool_calls") or cleaned_msg.get("function_call"):
             cleaned_msg['content'] = None
             if original_content != cleaned_msg['content']:
                logger.debug(f"NVIDIA_UTILS: Content for '{msg_type}' with tool_calls changed from '{original_content}' to None.")
        else:
            placeholder = f"[Internal Note: Empty '{msg_type}' message content replaced]"
            logger.warning(f"NVIDIA_UTILS: Empty content found in '{msg_type}' message. Replacing with placeholder: '{placeholder}'")
            cleaned_msg['content'] = placeholder
            
    elif cleaned_msg.get("tool_calls") and content == "": # Explicitly checking if content was empty string with tool_calls
         cleaned_msg['content'] = None
         if original_content != cleaned_msg['content']:
            logger.debug(f"NVIDIA_UTILS: Content for '{msg_type}' with tool_calls (originally empty string) changed to None.")

    if original_content != cleaned_msg['content']:
        logger.debug(f"NVIDIA_UTILS: _validate_and_clean_message_content output (content modified): {cleaned_msg}")
    else:
        logger.debug(f"NVIDIA_UTILS: _validate_and_clean_message_content output (content not modified from initial): {cleaned_msg}")
    return cleaned_msg