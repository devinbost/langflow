import re
from abc import abstractmethod
from typing import TYPE_CHECKING, cast

from langchain.agents import AgentExecutor, BaseMultiActionAgent, BaseSingleActionAgent
from langchain.agents.agent import RunnableAgent
from langchain_core.runnables import Runnable

from langflow.base.agents.callback import AgentAsyncHandler
from langflow.base.agents.events import ExceptionWithMessageError, process_agent_events
from langflow.base.agents.utils import data_to_messages
from langflow.custom import Component
from langflow.custom.custom_component.component import _get_component_toolkit
from langflow.field_typing import Tool
from langflow.inputs.inputs import InputTypes, MultilineInput
from langflow.io import BoolInput, HandleInput, IntInput, MessageTextInput
from langflow.logging import logger
from langflow.memory import delete_message
from langflow.schema import Data
from langflow.schema.content_block import ContentBlock
from langflow.schema.message import Message
from langflow.template import Output
from langflow.utils.constants import MESSAGE_SENDER_AI

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

    from langflow.schema.log import SendMessageFunctionType


DEFAULT_TOOLS_DESCRIPTION = "A helpful assistant with access to the following tools:"
DEFAULT_AGENT_NAME = "Agent ({tools_names})"


class LCAgentComponent(Component):
    trace_type = "agent"
    _base_inputs: list[InputTypes] = [
        MessageTextInput(
            name="input_value",
            display_name="Input",
            info="The input provided by the user for the agent to process.",
            tool_mode=True,
        ),
        BoolInput(
            name="handle_parsing_errors",
            display_name="Handle Parse Errors",
            value=True,
            advanced=True,
            info="Should the Agent fix errors when reading user input for better processing?",
        ),
        BoolInput(name="verbose", display_name="Verbose", value=True, advanced=True),
        IntInput(
            name="max_iterations",
            display_name="Max Iterations",
            value=15,
            advanced=True,
            info="The maximum number of attempts the agent can make to complete its task before it stops.",
        ),
        MultilineInput(
            name="agent_description",
            display_name="Agent Description [Deprecated]",
            info=(
                "The description of the agent. This is only used when in Tool Mode. "
                f"Defaults to '{DEFAULT_TOOLS_DESCRIPTION}' and tools are added dynamically. "
                "This feature is deprecated and will be removed in future versions."
            ),
            advanced=True,
            value=DEFAULT_TOOLS_DESCRIPTION,
        ),
    ]

    outputs = [
        Output(display_name="Agent", name="agent", method="build_agent", hidden=True, tool_mode=False),
        Output(display_name="Response", name="response", method="message_response"),
    ]

    @abstractmethod
    def build_agent(self) -> AgentExecutor:
        """Create the agent."""

    async def message_response(self) -> Message:
        """Run the agent and return the response."""
        agent = self.build_agent()
        message = await self.run_agent(agent=agent)

        self.status = message
        return message

    def _validate_outputs(self) -> None:
        required_output_methods = ["build_agent"]
        output_names = [output.name for output in self.outputs]
        for method_name in required_output_methods:
            if method_name not in output_names:
                msg = f"Output with name '{method_name}' must be defined."
                raise ValueError(msg)
            if not hasattr(self, method_name):
                msg = f"Method '{method_name}' must be defined."
                raise ValueError(msg)

    def get_agent_kwargs(self, *, flatten: bool = False) -> dict:
        base = {
            "handle_parsing_errors": self.handle_parsing_errors,
            "verbose": self.verbose,
            "allow_dangerous_code": True,
        }
        agent_kwargs = {
            "handle_parsing_errors": self.handle_parsing_errors,
            "max_iterations": self.max_iterations,
        }
        if flatten:
            return {
                **base,
                **agent_kwargs,
            }
        return {**base, "agent_executor_kwargs": agent_kwargs}

    def get_chat_history_data(self) -> list[Data] | None:
        # might be overridden in subclasses
        return None

    async def run_agent(
        self,
        agent: Runnable | BaseSingleActionAgent | BaseMultiActionAgent | AgentExecutor,
    ) -> Message:
        from langflow.logging import logger
        
        if isinstance(agent, AgentExecutor):
            runnable = agent
        else:
            if not hasattr(self, "tools") or not self.tools:
                msg = "Tools are required to run the agent."
                raise ValueError(msg)
            handle_parsing_errors = hasattr(self, "handle_parsing_errors") and self.handle_parsing_errors
            verbose = hasattr(self, "verbose") and self.verbose
            max_iterations = hasattr(self, "max_iterations") and self.max_iterations
            runnable = AgentExecutor.from_agent_and_tools(
                agent=agent,
                tools=self.tools,
                handle_parsing_errors=handle_parsing_errors,
                verbose=verbose,
                max_iterations=max_iterations,
            )
            
        # Initialize input dictionary with sane defaults
        input_dict: dict[str, str | list[BaseMessage]] = {}
        
        # Ensure input_value has content
        if hasattr(self, "input_value") and self.input_value and len(str(self.input_value).strip()) > 0:
            input_dict["input"] = self.input_value
            logger.info(f"Using user-provided input: {str(self.input_value)[:50]}...")
        else:
            default_input = "What can you help me with?"
            input_dict["input"] = default_input
            logger.warning(f"Input value was empty or missing, using default: '{default_input}'")
            # Set it on the component for consistency
            self.input_value = default_input
            
        # Ensure system_prompt has content
        if hasattr(self, "system_prompt") and self.system_prompt and len(str(self.system_prompt).strip()) > 0:
            # Check for NVIDIA model and detailed_thinking
            is_nvidia = False
            has_detailed_thinking = False
            
            # Check if this is a NVIDIA model with detailed_thinking
            if hasattr(self, "agent_llm") and self.agent_llm == "NVIDIA":
                is_nvidia = True
                if hasattr(self, "detailed_thinking") and isinstance(self.detailed_thinking, bool) and self.detailed_thinking:
                    has_detailed_thinking = True
            
            # Special handling for NVIDIA with detailed_thinking
            if is_nvidia and has_detailed_thinking and "detailed thinking on" not in str(self.system_prompt).lower():
                # Add the directive at the beginning of the prompt
                self.system_prompt = f"detailed thinking on\n\n{self.system_prompt}"
                logger.info("Added 'detailed thinking on' directive to system prompt for NVIDIA model")
                
            input_dict["system_prompt"] = self.system_prompt
            logger.info(f"Using system prompt: {str(self.system_prompt)[:50]}...")
        else:
            default_system = "You are a helpful assistant that can use tools to answer questions and perform tasks."
            input_dict["system_prompt"] = default_system
            logger.warning(f"System prompt was empty or missing, using default: '{default_system}'")
            # Set it on the component for consistency
            self.system_prompt = default_system
            
        # Add chat history if available
        if hasattr(self, "chat_history") and self.chat_history:
            chat_messages = data_to_messages(self.chat_history)
            if chat_messages:
                input_dict["chat_history"] = chat_messages
                logger.info(f"Added {len(chat_messages)} chat history messages")
            else:
                logger.warning("Chat history was available but empty after conversion")
        else:
            logger.info("No chat history available")
            
        # Debug log the input dictionary with detailed type information
        logger.info(f"Agent input dictionary prepared with keys: {list(input_dict.keys())}")
        logger.info(f"Input value (type: {type(input_dict.get('input', '')).__name__}): '{str(input_dict.get('input', ''))[:50]}...'")
        logger.info(f"System prompt (type: {type(input_dict.get('system_prompt', '')).__name__}): '{str(input_dict.get('system_prompt', ''))[:50]}...'")
        
        # Double-check for empty values and fix them
        for key, value in list(input_dict.items()):
            if value is None or (isinstance(value, str) and not value.strip()):
                if key == "input":
                    input_dict[key] = "What can you help me with?"
                    logger.warning(f"Empty value detected for {key}, replacing with default")
                elif key == "system_prompt":
                    input_dict[key] = "You are a helpful assistant that can use tools to answer questions and perform tasks."
                    logger.warning(f"Empty value detected for {key}, replacing with default")

        if hasattr(self, "graph"):
            session_id = self.graph.session_id
        elif hasattr(self, "_session_id"):
            session_id = self._session_id
        else:
            session_id = None

        agent_message = Message(
            sender=MESSAGE_SENDER_AI,
            sender_name=self.display_name or "Agent",
            properties={"icon": "Bot", "state": "partial"},
            content_blocks=[ContentBlock(title="Agent Steps", contents=[])],
            session_id=session_id,
        )
        
        # Extra validation and debug to troubleshoot API errors
        for key, value in list(input_dict.items()):
            if isinstance(value, str) and not value.strip():
                # Replace empty strings with defaults to avoid API errors
                if key == "input":
                    input_dict[key] = "What can you help me with?"
                    logger.warning(f"Empty string detected for {key}, using default")
                elif key == "system_prompt":
                    input_dict[key] = "You are a helpful assistant that can use tools to answer questions and perform tasks."
                    logger.warning(f"Empty string detected for {key}, using default")
        
        # Final safety check - ensure absolutely all values are non-empty
        if "input" not in input_dict or not input_dict["input"]:
            input_dict["input"] = "What can you help me with?"
            logger.warning("Missing 'input' in final dictionary check, adding default")
            
        if "system_prompt" not in input_dict or not input_dict["system_prompt"]:
            input_dict["system_prompt"] = "You are a helpful assistant that can use tools to answer questions and perform tasks."
            logger.warning("Missing 'system_prompt' in final dictionary check, adding default")
            
        # Ensure all required prompt variables exist to prevent errors
        # These are typically needed by message templates
        required_keys = ["agent_scratchpad"]
        for key in required_keys:
            if key not in input_dict:
                input_dict[key] = ""
                logger.info(f"Added required key '{key}' to input dictionary with empty value")
            
        # Now check specifically for NVIDIA models with detailed_thinking again
        # This is a safeguard to ensure the detailed_thinking directive is not lost during processing
        if (hasattr(self, "agent_llm") and self.agent_llm == "NVIDIA" and 
            hasattr(self, "detailed_thinking") and isinstance(self.detailed_thinking, bool) and 
            self.detailed_thinking and "system_prompt" in input_dict):
            
            system_prompt = input_dict["system_prompt"]
            if isinstance(system_prompt, str) and "detailed thinking on" not in system_prompt.lower():
                logger.info("Re-adding detailed thinking directive that was lost during processing")
                input_dict["system_prompt"] = f"detailed thinking on\n\n{system_prompt}"
            
        # Final log of sanitized input with type information for debugging
        safe_dict_for_logging = {}
        for k, v in input_dict.items():
            if k == "chat_history":
                safe_dict_for_logging[k] = f"[{len(v)} messages]" if isinstance(v, list) else str(v)
            else:
                safe_dict_for_logging[k] = f"{str(v)[:30]}... (type: {type(v).__name__})"
                
        logger.info(f"Final input dictionary: {safe_dict_for_logging}")
        
        try:
            # Add event stream debug configuration
            config = {
                "callbacks": [AgentAsyncHandler(self.log), *self.get_langchain_callbacks()],
                "version": "v2",
                # Enable more detailed tracing for debugging
                "recursion_limit": 25,  # Allow deeper recursion for complex agent logic
            }
            
            # Stream events from the agent
            logger.info("Starting agent event stream processing")
            result = await process_agent_events(
                runnable.astream_events(
                    input_dict,
                    config=config,
                ),
                agent_message,
                cast("SendMessageFunctionType", self.send_message),
            )
        except ExceptionWithMessageError as e:
            if hasattr(e, "agent_message") and hasattr(e.agent_message, "id"):
                msg_id = e.agent_message.id
                await delete_message(id_=msg_id)
            await self._send_message_event(e.agent_message, category="remove_message")
            logger.error(f"ExceptionWithMessageError: {e}")
            raise
        except Exception as e:
            # Log or handle any other exceptions
            logger.error(f"Error: {e}")
            raise

        self.status = result
        return result

    @abstractmethod
    def create_agent_runnable(self) -> Runnable:
        """Create the agent."""

    def validate_tool_names(self) -> None:
        """Validate tool names to ensure they match the required pattern."""
        pattern = re.compile(r"^[a-zA-Z0-9_-]+$")
        if hasattr(self, "tools") and self.tools:
            for tool in self.tools:
                if not pattern.match(tool.name):
                    msg = (
                        f"Invalid tool name '{tool.name}': must only contain letters, numbers, underscores, dashes,"
                        " and cannot contain spaces."
                    )
                    raise ValueError(msg)


class LCToolsAgentComponent(LCAgentComponent):
    _base_inputs = [
        HandleInput(
            name="tools",
            display_name="Tools",
            input_types=["Tool"],
            is_list=True,
            required=False,
            info="These are the tools that the agent can use to help with tasks.",
        ),
        *LCAgentComponent._base_inputs,
    ]

    def build_agent(self) -> AgentExecutor:
        self.validate_tool_names()
        agent = self.create_agent_runnable()
        return AgentExecutor.from_agent_and_tools(
            agent=RunnableAgent(runnable=agent, input_keys_arg=["input"], return_keys_arg=["output"]),
            tools=self.tools,
            **self.get_agent_kwargs(flatten=True),
        )

    @abstractmethod
    def create_agent_runnable(self) -> Runnable:
        """Create the agent."""

    def get_tool_name(self) -> str:
        return self.display_name or "Agent"

    def get_tool_description(self) -> str:
        return self.agent_description or DEFAULT_TOOLS_DESCRIPTION

    def _build_tools_names(self):
        tools_names = ""
        if self.tools:
            tools_names = ", ".join([tool.name for tool in self.tools])
        return tools_names

    async def _get_tools(self) -> list[Tool]:
        component_toolkit = _get_component_toolkit()
        tools_names = self._build_tools_names()
        agent_description = self.get_tool_description()
        # TODO: Agent Description Depreciated Feature to be removed
        description = f"{agent_description}{tools_names}"
        tools = component_toolkit(component=self).get_tools(
            tool_name=self.get_tool_name(), tool_description=description, callbacks=self.get_langchain_callbacks()
        )
        if hasattr(self, "tools_metadata"):
            tools = component_toolkit(component=self, metadata=self.tools_metadata).update_tools_metadata(tools=tools)
        return tools
