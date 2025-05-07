from typing import Any, Iterator, AsyncIterator, List, Optional

from loguru import logger
from requests.exceptions import ConnectionError  # noqa: A004
from urllib3.exceptions import MaxRetryError, NameResolutionError

from langflow.base.models.model import LCModelComponent
from langflow.field_typing import LanguageModel
from langflow.field_typing.range_spec import RangeSpec
from langflow.inputs import BoolInput, DropdownInput, IntInput, MessageTextInput, SecretStrInput, SliderInput
from langflow.schema.dotdict import dotdict
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGenerationChunk
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.messages import BaseMessage, AIMessage
from langflow.utils.nvidia_utils import clean_nvidia_message_content


class NvidiaChatModelOutputWrapper(BaseChatModel):
    """
    Wrapper around a ChatNVIDIA model (or any BaseChatModel) 
    to clean its outputs from NVIDIA-specific duplications.
    """
    actual_llm: BaseChatModel 

    def __init__(self, llm: BaseChatModel, **kwargs: Any):
        # Pass kwargs to super to handle Pydantic model initialization if BaseChatModel expects it
        super().__init__(**kwargs) 
        object.__setattr__(self, "actual_llm", llm)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        result: ChatResult = self.actual_llm._generate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )
        # Ensure result.generations is not None and is a list before iterating
        if result.generations and isinstance(result.generations, list):
            for gen_idx, gen_item in enumerate(result.generations):
                # Standard ChatResult has List[ChatGeneration], ChatGeneration has .message
                if isinstance(gen_item, Generation) and hasattr(gen_item, "message") and isinstance(gen_item.message, (AIMessage, AIMessageChunk)):
                    result.generations[gen_idx].message = clean_nvidia_message_content(gen_item.message) # type: ignore
        return result

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        result: ChatResult = await self.actual_llm._agenerate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )
        if result.generations and isinstance(result.generations, list):
            for gen_idx, gen_item in enumerate(result.generations):
                 if isinstance(gen_item, Generation) and hasattr(gen_item, "message") and isinstance(gen_item.message, (AIMessage, AIMessageChunk)):
                    result.generations[gen_idx].message = clean_nvidia_message_content(gen_item.message) # type: ignore
        return result

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        original_iterator = self.actual_llm._stream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )
        for chunk in original_iterator:
            # ChatGenerationChunk has .message which is an AIMessageChunk
            if hasattr(chunk, "message") and isinstance(chunk.message, AIMessageChunk):
                chunk.message = clean_nvidia_message_content(chunk.message) # type: ignore
            yield chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        original_iterator = self.actual_llm._astream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )
        async for chunk in original_iterator:
            if hasattr(chunk, "message") and isinstance(chunk.message, AIMessageChunk):
                chunk.message = clean_nvidia_message_content(chunk.message) # type: ignore
            yield chunk
            
    @property
    def _llm_type(self) -> str:
        # Ensure actual_llm is initialized
        if hasattr(self, 'actual_llm') and hasattr(self.actual_llm, '_llm_type'):
            return self.actual_llm._llm_type + "_cleaned_nvidia_wrapper"
        return "nvidia_chat_model_cleaned_wrapper"


    def __getattr__(self, name: str) -> Any:
        if name == 'actual_llm' or name.startswith('_'): # Avoid recursion and private attributes
             raise AttributeError(f"Attribute {name} not found or is private.")
        try:
            # Ensure actual_llm is initialized before trying to access its attributes
            if hasattr(self, 'actual_llm'):
                return getattr(self.actual_llm, name)
            raise AttributeError("Wrapped LLM 'actual_llm' not initialized.")
        except AttributeError:
            raise AttributeError(
                f"'{self.__class__.__name__}' (wrapping '{getattr(self, 'actual_llm', 'UninitializedLLM').__class__.__name__}') has no attribute '{name}'"
            )
    
    # Delegate common properties required by Langchain
    @property
    def lc_serializable(self) -> bool:
        if hasattr(self, 'actual_llm') and hasattr(self.actual_llm, 'lc_serializable'):
            return self.actual_llm.lc_serializable
        return False 

    @property
    def InputType(self) -> Any: 
        if hasattr(self, 'actual_llm') and hasattr(self.actual_llm, 'InputType'):
            return self.actual_llm.InputType
        return super().InputType # Fallback to BaseChatModel's default

    @property
    def OutputType(self) -> Any: 
        if hasattr(self, 'actual_llm') and hasattr(self.actual_llm, 'OutputType'):
            return self.actual_llm.OutputType
        return super().OutputType # Fallback to BaseChatModel's default

    # Ensure to define all abstract methods from BaseChatModel if any are not covered by delegation
    # For BaseChatModel, _generate, _agenerate are key. _stream, _astream have defaults that call them.
    # If ChatNVIDIA implements _stream/_astream more efficiently, our wrapper should too.


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

    def build_model(self) -> BaseLanguageModel:  # type: ignore[type-var]
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
            # from langchain_nvidia_ai_endpoints._common import _NVIDIAClient # Not needed directly here
        except ImportError as e:
            msg = "Please install langchain-nvidia-ai-endpoints to use the NVIDIA model."
            logger.error(msg) # Added logger
            raise ImportError(msg) from e
        
        api_key_value = self.api_key.get_secret_value() if isinstance(self.api_key, SecretStr) else self.api_key

        kwargs = {
            "max_tokens": self.max_tokens or None,
            "model": self.model_name,
            "base_url": self.base_url,
            "api_key": api_key_value, # Use unwrapped secret
            "temperature": self.temperature if self.temperature is not None else 0.1,
            "seed": self.seed,
            # "tool_configs": {"type": "json_object"}, # This might be specific to older versions or direct API calls
                                                    # langchain_nvidia_ai_endpoints.ChatNVIDIA handles tool binding via .bind_tools()
        }
        
        # Remove None valued keys to prevent issues with ChatNVIDIA constructor
        kwargs = {k: v for k, v in kwargs.items() if v is not None}


        if hasattr(self, "detailed_thinking") and isinstance(self.detailed_thinking, bool) and self.detailed_thinking:
            kwargs["detailed_thinking"] = True # Assuming ChatNVIDIA supports this pass-through
            logger.info("NVIDIA model: detailed_thinking parameter enabled in kwargs")
        else:
            logger.info("NVIDIA model: detailed_thinking parameter not enabled in kwargs")
            
        # The patching logic for session.post can be removed if we are cleaning the response,
        # unless it's vital for request formation (e.g., fixing tool definitions before sending).
        # The user stated workarounds (which included patching) weren't working for the response issue.
        # For now, I'll remove the complex patching to simplify, as we're focused on response cleaning.
        # If request-side tool name fixing is still needed, the `fix_duplicated_name` in `patched_post`
        # should target the `payload['tools'][i]['function']['name']` before the request is made.
        # But let's first ensure response cleaning works.

        try:
            chat_nvidia_model = ChatNVIDIA(**kwargs)
            
            # Wrap the model instance
            # Pass the original kwargs to the wrapper if it needs to initialize BaseChatModel fields
            # Or pass specific fields BaseChatModel expects if its __init__ is not just **kwargs
            wrapper_kwargs = {} # Collect necessary kwargs for BaseChatModel parent if needed
            if hasattr(chat_nvidia_model, 'callbacks'):
                wrapper_kwargs['callbacks'] = chat_nvidia_model.callbacks
            if hasattr(chat_nvidia_model, 'verbose'):
                wrapper_kwargs['verbose'] = chat_nvidia_model.verbose
            # Add other relevant BaseChatModel fields if necessary for initialization
            
            wrapped_model = NvidiaChatModelOutputWrapper(llm=chat_nvidia_model, **wrapper_kwargs) 
            
            logger.info("Successfully built and wrapped ChatNVIDIA model for output cleaning.")
            return wrapped_model

        except Exception as e:
            logger.error(f"Error building or wrapping NVIDIA model: {e}", exc_info=True) # Added exc_info
            raise
