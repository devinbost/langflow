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
                # Check based on class name or potential client attribute more robustly
                llm_class_name_lower = str(self.llm.__class__).lower()
                is_nvidia_model = 'nvidia' in llm_class_name_lower
                # Add check for wrapper if it indicates underlying type
                if not is_nvidia_model and hasattr(self.llm, 'actual_llm'):
                    llm_class_name_lower = str(self.llm.actual_llm.__class__).lower()
                    is_nvidia_model = 'nvidia' in llm_class_name_lower

            except Exception as e_check:
                logger.warning(f"Could not reliably determine if model is NVIDIA: {e_check}")
            
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
                # Check based on class name or potential client attribute more robustly
                llm_class_name_lower = str(self.llm.__class__).lower()
                is_nvidia_model = 'nvidia' in llm_class_name_lower
                # Add check for wrapper if it indicates underlying type
                if not is_nvidia_model and hasattr(self.llm, 'actual_llm'):
                    llm_class_name_lower = str(self.llm.actual_llm.__class__).lower()
                    is_nvidia_model = 'nvidia' in llm_class_name_lower

            except Exception as e_check:
                logger.warning(f"Could not reliably determine if model is NVIDIA: {e_check}")

            if is_nvidia_model:
                logger.info("Applying NVIDIA specific tool handling.")
                tools_copy = []
                for tool_obj in (self.tools or []):
                    if hasattr(tool_obj, 'to_openai_tool'):
                        tools_copy.append(tool_obj)
                    else:
                        try:
                            original_tool_name = getattr(tool_obj, 'name', 'unknown')
                            logger.info(f"Tool info - class: {tool_obj.__class__.__name__}, name before NVIDIA fix: {original_tool_name}")

                            # --- MODIFIED FIX LOGIC ---
                            if original_tool_name == "evaluate_expression":
                                tool_name = "calculate"
                                logger.info(f"Forcing tool name 'evaluate_expression' to 'calculate' for NVIDIA agent.")
                            elif original_tool_name == "evaluate_expressionevaluate_expression":
                                tool_name = "calculate"
                                logger.info(f"Fixing duplicated tool name '{original_tool_name}' to 'calculate' for NVIDIA agent.")
                            else:
                                # Apply generic duplication fix if needed
                                if original_tool_name and len(original_tool_name) >= 6:
                                     half_len = len(original_tool_name) // 2
                                     first_half = original_tool_name[:half_len]
                                     second_half = original_tool_name[half_len:]
                                     if first_half == second_half:
                                         logger.warning(f"Fixing generic duplicated name: {original_tool_name} -> {first_half}")
                                         tool_name = first_half
                                     else:
                                         tool_name = original_tool_name
                                else:
                                    tool_name = original_tool_name
                            # --- END MODIFIED FIX LOGIC ---

                            logger.info(f"Final tool name for NVIDIA agent schema: {tool_name}")
                            
                            # Extract description safely
                            tool_description = getattr(tool_obj, 'description', 'No description available')
                            if not isinstance(tool_description, str):
                                tool_description = str(tool_description) # Ensure string
                            
                            # Extract run function safely
                            # Needs the actual execution logic. tool_obj.run might be correct,
                            # but ensure it exists and is callable.
                            run_func = getattr(tool_obj, 'run', None)
                            if not callable(run_func):
                                logger.error(f"Tool object {original_tool_name} does not have a callable 'run' method.")
                                # Decide whether to skip or raise. Skipping for now.
                                continue 

                            # Extract args_schema safely
                            args_schema = getattr(tool_obj, 'args_schema', None)

                            structured_tool = StructuredTool.from_function(
                                func=run_func, # Use the validated run function
                                name=tool_name, # Use the definitively corrected name
                                description=tool_description,
                                return_direct=getattr(tool_obj, 'return_direct', False),
                                args_schema=args_schema, # Pass the schema if it exists
                                # coroutine=... # Add if async support is needed/available
                                # verbose=... # Add if needed
                            )
                            tools_copy.append(structured_tool)
                            logger.info(f"Processed tool {original_tool_name} -> {tool_name} for NVIDIA compatibility")
                        except Exception as tool_e:
                            logger.warning(f"Could not process tool {getattr(tool_obj, 'name', 'unknown')}: {tool_e}", exc_info=True)
                            # Optionally append the original tool if processing fails, or skip
                            # tools_copy.append(tool_obj)
                
                logger.info(f"Creating NVIDIA tool calling agent with {len(tools_copy)} tools")
                return create_tool_calling_agent(self.llm, tools_copy, prompt)
            else:
                # Standard handling for non-NVIDIA models
                logger.info(f"Creating standard tool calling agent with {len(self.tools or [])} tools")
                return create_tool_calling_agent(self.llm, self.tools or [], prompt)
                
        except NotImplementedError as e:
            message = f"{self.display_name} does not support tool calling. Please try using a compatible model."
            logger.error(message, exc_info=True)
            raise NotImplementedError(message) from e
        except Exception as e_main:
             logger.error(f"Failed to create agent runnable: {e_main}", exc_info=True)
             raise
