from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

from langflow.base.agents.agent import LCToolsAgentComponent
from langflow.inputs import MessageTextInput
from langflow.inputs.inputs import DataInput, HandleInput
from langflow.logging import logger
from langflow.schema import Data


class ToolCallingAgentComponent(LCToolsAgentComponent):
    display_name: str = "Tool Calling Agent"
    description: str = "An agent designed to utilize various tools seamlessly within workflows."
    icon = "LangChain"
    name = "ToolCallingAgent"

    inputs = [
        *LCToolsAgentComponent._base_inputs,
        HandleInput(
            name="llm",
            display_name="Language Model",
            input_types=["LanguageModel"],
            required=True,
            info="Language model that the agent utilizes to perform tasks effectively.",
        ),
        MessageTextInput(
            name="system_prompt",
            display_name="System Prompt",
            info="System prompt to guide the agent's behavior.",
            value="You are a helpful assistant that can use tools to answer questions and perform tasks.",
        ),
        DataInput(
            name="chat_history",
            display_name="Chat Memory",
            is_list=True,
            advanced=True,
            info="This input stores the chat history, allowing the agent to remember previous conversations.",
        ),
    ]

    def get_chat_history_data(self) -> list[Data] | None:
        return self.chat_history

    def create_agent_runnable(self):
        # Debug log for diagnostic purposes
        logger.info(f"Creating agent runnable with system_prompt: {getattr(self, 'system_prompt', 'NOT SET')}")
        logger.info(f"Creating agent runnable with input_value: {getattr(self, 'input_value', 'NOT SET')}")
        
        # Force system_prompt to have content - be more aggressive in setting defaults
        if not hasattr(self, 'system_prompt') or not self.system_prompt or not str(self.system_prompt).strip():
            system_prompt_default = "You are a helpful assistant that can use tools to answer questions and perform tasks."
            logger.warning(f"System prompt was empty or missing, using default: '{system_prompt_default}'")
            self.system_prompt = system_prompt_default
            
        # Force input_value to have content - be more aggressive in setting defaults
        if not hasattr(self, 'input_value') or not self.input_value or not str(self.input_value).strip():
            input_value_default = "What can you help me with?"
            logger.warning(f"Input value was empty or missing, using default: '{input_value_default}'")
            self.input_value = input_value_default
        
        # Log the actual data being used
        logger.info(f"Using system_prompt: '{self.system_prompt[:50]}...'")
        logger.info(f"Using input_value: '{self.input_value[:50]}...'")
        
        # Define custom message formatter with strict validation
        def validate_and_format_message(message_type, template_content):
            if message_type == "system":
                if hasattr(self, 'system_prompt') and self.system_prompt and str(self.system_prompt).strip():
                    return ("system", self.system_prompt)
                else:
                    default = "You are a helpful assistant that can use tools to answer questions and perform tasks."
                    logger.warning(f"System prompt invalid in formatter, using default: '{default}'")
                    return ("system", default)
            elif message_type == "human":
                if hasattr(self, 'input_value') and self.input_value and str(self.input_value).strip():
                    return ("human", "{input}")
                else:
                    logger.warning("Input value invalid in formatter, using default template")
                    return ("human", "What can you help me with?")
            else:
                if template_content and str(template_content).strip():
                    return (message_type, template_content)
                else:
                    logger.warning(f"Empty {message_type} message content, using placeholder")
                    return (message_type, f"[Empty {message_type} message]")
                
        # Create message template with validated content
        # For NVIDIA models, we need to ensure all placeholder messages have default content
        # to prevent empty string validation errors
        is_nvidia_model = (hasattr(self.llm, 'client') and 'nvidia' in str(self.llm.__class__).lower()) or 'nvidia' in str(self.llm.__class__).lower()
        
        # Always use the same standard template structure to avoid prompt variable issues
        messages = [
            validate_and_format_message("system", "{system_prompt}"),
            ("placeholder", "{chat_history}"),
            validate_and_format_message("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
        
        # Log template choice
        logger.info("Using standard message template structure")
        
        # Log the message structure for debugging
        logger.info(f"Message template structure: {messages}")
        
        prompt = ChatPromptTemplate.from_messages(messages)
        self.validate_tool_names()

        # Add detailed_thinking to system prompt if it's enabled and available
        try:
            # Check if detailed_thinking is set as a valid boolean and is True
            has_detailed_thinking = (
                hasattr(self, 'detailed_thinking') and 
                isinstance(self.detailed_thinking, bool) and 
                self.detailed_thinking
            )
            
            # Check if model is a NVIDIA model (either directly or via client attribute)
            is_nvidia_model = False
            try:
                is_nvidia_model = 'nvidia' in str(self.llm.__class__).lower()
            except:
                pass
            
            logger.info(f"Model class: {self.llm.__class__.__name__}, has_detailed_thinking: {has_detailed_thinking}, is_nvidia_model: {is_nvidia_model}")
            
            # Modify system prompt for NVIDIA models with detailed_thinking enabled
            if has_detailed_thinking and is_nvidia_model:
                # Prepend "detailed thinking on" to the system prompt if not already present
                if "detailed thinking on" not in self.system_prompt.lower():
                    original_prompt = self.system_prompt
                    self.system_prompt = f"detailed thinking on\n\n{original_prompt}"
                    logger.info(f"Added 'detailed thinking on' to system prompt for NVIDIA model. Original: '{original_prompt[:50]}...'")
        except Exception as e:
            # Log error but continue without modifying the prompt
            logger.error(f"Error processing detailed_thinking parameter: {e}")
            # Don't propagate the exception to avoid breaking the agent
            
        try:
            # For NVIDIA models, add special handling for tools
            is_nvidia_model = False
            try:
                is_nvidia_model = 'nvidia' in str(self.llm.__class__).lower()
            except:
                pass
                
            if is_nvidia_model:
                # Make sure we have valid tools with proper structure for NVIDIA
                tools_copy = []
                for tool in (self.tools or []):
                    # Ensure tool has proper structure
                    if hasattr(tool, 'to_openai_tool'):
                        # Already has OpenAI tool format
                        tools_copy.append(tool)
                    else:
                        # Convert to proper format
                        from langchain.tools import StructuredTool
                        try:
                            # Get the tool's original name
                            original_tool_name = getattr(tool, 'name', 'unknown')
                            
                            # Log tool information for debugging
                            logger.info(f"Tool info - class: {tool.__class__.__name__}, name: {original_tool_name}")
                            
                            # Simple function to detect and fix duplicated tool names
                            def fix_duplicated_tool_name(name):
                                # Special case for the evaluate_expression tool (Calculator)
                                if name == "evaluate_expressionevaluate_expression":
                                    return "calculate"
                                
                                # Check for exact duplication (somethingsomething)
                                if name and len(name) >= 6:
                                    half_len = len(name) // 2
                                    first_half = name[:half_len]
                                    second_half = name[half_len:]
                                    
                                    if first_half == second_half:
                                        logger.warning(f"Found exact duplicated tool name: {name} -> {first_half}")
                                        logger.info(f"This may be a symptom of a mismatch between method name and tool_name")
                                        return first_half
                                
                                # No duplication detected
                                return name
                            
                            # Apply name validation
                            tool_name = fix_duplicated_tool_name(original_tool_name)
                            if tool_name != original_tool_name:
                                logger.info(f"Fixed duplicated tool name: {original_tool_name} -> {tool_name}")
                                logger.warning("This is a workaround for a known issue with duplicated tool names. The root cause may have been fixed.")
                            
                            # Create a properly structured tool
                            structured_tool = StructuredTool.from_function(
                                func=tool.run,
                                name=tool_name,
                                description=tool.description,
                                return_direct=False,
                                args_schema=None,
                                coroutine=None,
                                verbose=True,
                            )
                            tools_copy.append(structured_tool)
                            logger.info(f"Converted tool {tool.name} to StructuredTool for NVIDIA compatibility")
                        except Exception as tool_e:
                            logger.warning(f"Could not convert tool {tool.name}: {tool_e}")
                            tools_copy.append(tool)
                
                logger.info(f"Creating NVIDIA tool calling agent with {len(tools_copy)} tools")
                return create_tool_calling_agent(self.llm, tools_copy, prompt)
            else:
                # Standard handling for non-NVIDIA models
                return create_tool_calling_agent(self.llm, self.tools or [], prompt)
                
        except NotImplementedError as e:
            message = f"{self.display_name} does not support tool calling. Please try using a compatible model."
            raise NotImplementedError(message) from e
