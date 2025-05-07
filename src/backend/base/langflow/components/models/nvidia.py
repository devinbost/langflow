from typing import Any

from loguru import logger
from requests.exceptions import ConnectionError  # noqa: A004
from urllib3.exceptions import MaxRetryError, NameResolutionError

from langflow.base.models.model import LCModelComponent
from langflow.field_typing import LanguageModel
from langflow.field_typing.range_spec import RangeSpec
from langflow.inputs import BoolInput, DropdownInput, IntInput, MessageTextInput, SecretStrInput, SliderInput
from langflow.schema.dotdict import dotdict


class NVIDIAModelComponent(LCModelComponent):
    display_name = "NVIDIA"
    description = "Generates text using NVIDIA LLMs."
    icon = "NVIDIA"

    try:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        all_models = ChatNVIDIA().get_available_models()
    except ImportError as e:
        msg = "Please install langchain-nvidia-ai-endpoints to use the NVIDIA model."
        raise ImportError(msg) from e
    except (ConnectionError, MaxRetryError, NameResolutionError):
        logger.warning(
            "Failed to connect to NVIDIA API. Model list may be unavailable."
            " Please check your internet connection and API credentials."
        )
        all_models = []

    inputs = [
        *LCModelComponent._base_inputs,
        IntInput(
            name="max_tokens",
            display_name="Max Tokens",
            advanced=True,
            info="The maximum number of tokens to generate. Set to 0 for unlimited tokens.",
        ),
        DropdownInput(
            name="model_name",
            display_name="Model Name",
            info="The name of the NVIDIA model to use.",
            advanced=False,
            value=None,
            options=[model.id for model in all_models],
            combobox=True,
            refresh_button=True,
        ),
        BoolInput(
            name="detailed_thinking",
            display_name="Detailed Thinking",
            info="If true, the model will return a detailed thought process. Only supported by reasoning models.",
            value=False,
            show=False,
        ),
        BoolInput(
            name="tool_model_enabled",
            display_name="Enable Tool Models",
            info="If enabled, only show models that support tool-calling.",
            advanced=False,
            value=False,
            real_time_refresh=True,
        ),
        MessageTextInput(
            name="base_url",
            display_name="NVIDIA Base URL",
            value="https://integrate.api.nvidia.com/v1",
            info="The base URL of the NVIDIA API. Defaults to https://integrate.api.nvidia.com/v1.",
        ),
        SecretStrInput(
            name="api_key",
            display_name="NVIDIA API Key",
            info="The NVIDIA API Key.",
            advanced=False,
            value="NVIDIA_API_KEY",
        ),
        SliderInput(
            name="temperature",
            display_name="Temperature",
            value=0.1,
            info="Run inference with this temperature.",
            range_spec=RangeSpec(min=0, max=1, step=0.01),
            advanced=True,
        ),
        IntInput(
            name="seed",
            display_name="Seed",
            info="The seed controls the reproducibility of the job.",
            advanced=True,
            value=1,
        ),
    ]

    def get_models(self, tool_model_enabled: bool | None = None) -> list[str]:
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
        except ImportError as e:
            msg = "Please install langchain-nvidia-ai-endpoints to use the NVIDIA model."
            raise ImportError(msg) from e

        # Note: don't include the previous model, as it may not exist in available models from the new base url
        model = ChatNVIDIA(base_url=self.base_url, api_key=self.api_key)
        if tool_model_enabled:
            tool_models = [m for m in model.get_available_models() if m.supports_tools]
            return [m.id for m in tool_models]
        return [m.id for m in model.available_models]

    def update_build_config(self, build_config: dotdict, _field_value: Any, field_name: str | None = None):
        if field_name in {"model_name", "tool_model_enabled", "base_url", "api_key"}:
            try:
                ids = self.get_models(self.tool_model_enabled)
                build_config["model_name"]["options"] = ids

                if "value" not in build_config["model_name"] or build_config["model_name"]["value"] is None:
                    build_config["model_name"]["value"] = ids[0]
                elif build_config["model_name"]["value"] not in ids:
                    build_config["model_name"]["value"] = None

                # TODO: use api to determine if model supports detailed thinking
                if build_config["model_name"]["value"] == "nemotron":
                    build_config["detailed_thinking"]["show"] = True
                else:
                    build_config["detailed_thinking"]["value"] = False
                    build_config["detailed_thinking"]["show"] = False
            except Exception as e:
                msg = f"Error getting model names: {e}"
                build_config["model_name"]["value"] = None
                build_config["model_name"]["options"] = []
                raise ValueError(msg) from e

        return build_config

    def build_model(self) -> LanguageModel:  # type: ignore[type-var]
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
            from langchain_nvidia_ai_endpoints._common import _NVIDIAClient
        except ImportError as e:
            msg = "Please install langchain-nvidia-ai-endpoints to use the NVIDIA model."
            raise ImportError(msg) from e
        api_key = self.api_key
        temperature = self.temperature
        model_name: str = self.model_name
        max_tokens = self.max_tokens
        seed = self.seed
        # Prepare additional parameters
        kwargs = {
            "max_tokens": max_tokens or None,
            "model": model_name,
            "base_url": self.base_url,
            "api_key": api_key,
            "temperature": temperature or 0.1,
            "seed": seed,
            # Important: Enable tool support for agent use
            "tool_configs": {"type": "json_object"},
        }
        
        # Add detailed_thinking if it's enabled (as a boolean)
        if hasattr(self, "detailed_thinking") and isinstance(self.detailed_thinking, bool) and self.detailed_thinking:
            # The parameter will be carried through to signal this setting is enabled
            kwargs["detailed_thinking"] = True
            logger.info("NVIDIA model: detailed_thinking parameter enabled")
        else:
            logger.info("NVIDIA model: detailed_thinking parameter not enabled")
            
        # Create the model instance
        try:
            # Instead of patching the class method, we'll inject a payload validator directly into the model instance
            original_model = ChatNVIDIA(**kwargs)
            
            # Store original post method
            if hasattr(original_model, '_client') and hasattr(original_model._client, 'get_session_fn'):
                original_session_fn = original_model._client.get_session_fn
                
                # Create patched session function
                def patched_session_fn():
                    session = original_session_fn()
                    original_post = session.post
                    
                    # Create patched post method to fix messages
                    def patched_post(*args, **kwargs):
                        try:
                            # Check if there's a JSON payload with messages
                            if 'json' in kwargs and isinstance(kwargs['json'], dict):
                                payload = kwargs['json']
                                
                                # Process and fix tools if present
                                if 'tools' in payload:
                                    tool_count = len(payload['tools'])
                                    logger.info(f"NVIDIA API payload contains {tool_count} tools")
                                    
                                    # Process each tool to fix potential naming issues
                                    for i, tool in enumerate(payload['tools']):
                                        if 'function' in tool and isinstance(tool['function'], dict):
                                            func_dict = tool['function']
                                            
                                            # Get current name
                                            original_name = func_dict.get('name', 'unknown')
                                            
                                            # Simple function to fix duplicated tool names
                                            def fix_duplicated_name(name):
                                                # Special case for evaluate_expression
                                                if name == "evaluate_expressionevaluate_expression":
                                                    return "calculate"
                                                
                                                # Check for exact duplication
                                                if name and len(name) >= 6:
                                                    half_len = len(name) // 2
                                                    first_half = name[:half_len]
                                                    second_half = name[half_len:]
                                                    
                                                    if first_half == second_half:
                                                        from langflow.logging import logger
                                                        logger.warning(f"Detected duplicated tool name: {name} -> fixing to {first_half}")
                                                        return first_half
                                                
                                                return name
                                            
                                            # Fix the name if needed
                                            fixed_name = fix_duplicated_name(original_name)
                                            if fixed_name != original_name:
                                                func_dict['name'] = fixed_name
                                                logger.warning(f"Fixed duplicated tool name in NVIDIA payload: {original_name} -> {fixed_name}")
                                                logger.warning("This is a workaround for a known issue. The root cause may have been fixed.")
                                            
                                            # Log tool information
                                            logger.info(f"Tool {i}: name={func_dict.get('name', 'unknown')}")
                                
                                # Fix message content in messages array
                                if 'messages' in payload:
                                    message_count = len(payload['messages'])
                                    logger.info(f"NVIDIA API payload has {message_count} messages")
                                    
                                    # Process each message to fix empty content and check for tool calls
                                    for i, msg in enumerate(payload['messages']):
                                        # Get message type
                                        msg_type = msg.get('role', 'unknown')
                                        
                                        # Log tool calls if present
                                        if 'tool_calls' in msg and msg['tool_calls']:
                                            for j, tool_call in enumerate(msg['tool_calls']):
                                                logger.info(f"Message {i} contains tool call {j}: {tool_call}")
                                        
                                        # If this is an assistant message with tool_call_id, log it
                                        if msg_type == 'assistant' and 'tool_call_id' in msg:
                                            logger.info(f"Tool call response in message {i}: tool_call_id={msg.get('tool_call_id')}")
                                            
                                        # Fix messages with empty content
                                        if 'content' not in msg or msg['content'] is None or msg['content'] == '':
                                            logger.warning(f"NVIDIA API: Empty content in '{msg_type}' message at position {i}. Adding default content.")
                                            
                                            # Add default content based on message type
                                            if msg_type == 'system':
                                                msg['content'] = "You are a helpful assistant."
                                            elif msg_type == 'assistant':
                                                msg['content'] = "[AI thinking...]"
                                            elif msg_type == 'user':
                                                msg['content'] = "[User request]"
                                            else:
                                                msg['content'] = f"[Empty {msg_type} message]"
                                        
                                        # Also log if content contains potential tool references
                                        if msg.get('content') and isinstance(msg['content'], str) and 'evaluate_expression' in msg['content']:
                                            tool_reference = msg['content'][:200] + ('...' if len(msg['content']) > 200 else '')
                                            logger.info(f"Message {i} contains potential tool reference: {tool_reference}")
                                    
                                    # Log the updated payload
                                    logger.info(f"NVIDIA API: Fixed {message_count} messages in payload")
                        except Exception as e:
                            logger.error(f"Error in NVIDIA API payload processing: {e}")
                        
                        # Call original post method
                        return original_post(*args, **kwargs)
                    
                    # Replace post method with patched version
                    session.post = patched_post
                    return session
                
                # Replace the session function with our patched version
                original_model._client.get_session_fn = patched_session_fn
                logger.info("Applied NVIDIA API message validation via session patching")
            else:
                logger.warning("Could not patch NVIDIA client - expected attributes not found")
            
            logger.info(f"NVIDIA ChatModel initialized with: model={kwargs.get('model')}, max_tokens={kwargs.get('max_tokens')}")
            return original_model
        except Exception as e:
            logger.error(f"Failed to initialize NVIDIA model: {str(e)}")
            raise
