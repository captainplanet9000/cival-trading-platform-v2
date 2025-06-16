"""
Phase 10: LLM Integration Service
Advanced language model integration for autonomous trading system
Supports multiple LLM providers with unified interface and intelligent routing
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json
import hashlib
from decimal import Decimal

import openai
import anthropic
from transformers import pipeline
import tiktoken
import redis.asyncio as redis

from ..core.service_registry import get_registry
from ..models.llm_models import (
    LLMRequest, LLMResponse, ConversationContext, 
    AgentCommunication, TradingDecision, MarketAnalysis
)

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI_GPT4 = "openai_gpt4"
    OPENAI_GPT35 = "openai_gpt35"
    ANTHROPIC_CLAUDE = "anthropic_claude"
    HUGGINGFACE_LOCAL = "huggingface_local"
    GOOGLE_PALM = "google_palm"
    COHERE = "cohere"

class LLMTaskType(Enum):
    """Types of LLM tasks"""
    MARKET_ANALYSIS = "market_analysis"
    TRADING_DECISION = "trading_decision"
    RISK_ASSESSMENT = "risk_assessment"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"
    AGENT_COMMUNICATION = "agent_communication"
    GOAL_PLANNING = "goal_planning"
    STRATEGY_GENERATION = "strategy_generation"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    NATURAL_LANGUAGE_QUERY = "natural_language_query"

@dataclass
class LLMConfig:
    """LLM provider configuration"""
    provider: LLMProvider
    model_name: str
    api_key: Optional[str]
    endpoint: Optional[str]
    max_tokens: int
    temperature: float
    timeout: int
    cost_per_token: float
    rate_limit_rpm: int
    rate_limit_tpm: int

@dataclass
class AgentPersonality:
    """Agent personality configuration for communication"""
    agent_id: str
    name: str
    role: str
    trading_style: str
    risk_tolerance: float
    communication_style: str
    expertise_areas: List[str]
    personality_traits: Dict[str, float]
    system_prompt: str

@dataclass
class ConversationMessage:
    """Individual conversation message"""
    message_id: str
    sender_id: str
    recipient_id: str
    content: str
    message_type: str
    timestamp: datetime
    context: Dict[str, Any]
    response_to: Optional[str] = None

class LLMIntegrationService:
    """
    Advanced LLM integration service for autonomous trading system
    Phase 10: Multi-provider support with intelligent routing and agent communication
    """
    
    def __init__(self, redis_client=None):
        self.registry = get_registry()
        self.redis = redis_client
        
        # Service dependencies
        self.event_service = None
        self.market_data_service = None
        self.portfolio_service = None
        self.goal_service = None
        
        # LLM providers
        self.providers: Dict[LLMProvider, Any] = {}
        self.provider_configs: Dict[LLMProvider, LLMConfig] = {}
        
        # Agent personalities
        self.agent_personalities: Dict[str, AgentPersonality] = {}
        
        # Conversation management
        self.active_conversations: Dict[str, List[ConversationMessage]] = {}
        self.conversation_contexts: Dict[str, ConversationContext] = {}
        
        # Performance tracking
        self.provider_performance: Dict[LLMProvider, Dict[str, Any]] = {}
        self.token_usage: Dict[str, int] = {}
        self.cost_tracking: Dict[str, float] = {}
        
        # Rate limiting
        self.rate_limiters: Dict[LLMProvider, Dict[str, Any]] = {}
        
        # Caching
        self.response_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 300  # 5 minutes
        
        logger.info("LLMIntegrationService Phase 10 initialized")
    
    async def initialize(self):
        """Initialize the LLM integration service"""
        try:
            # Get required services
            self.event_service = self.registry.get_service("wallet_event_streaming_service")
            self.market_data_service = self.registry.get_service("market_data_service")
            self.portfolio_service = self.registry.get_service("portfolio_management_service")
            self.goal_service = self.registry.get_service("intelligent_goal_service")
            
            # Initialize LLM providers
            await self._initialize_providers()
            
            # Load agent personalities
            await self._load_agent_personalities()
            
            # Start background tasks
            asyncio.create_task(self._conversation_manager())
            asyncio.create_task(self._performance_monitor())
            asyncio.create_task(self._cache_cleanup())
            
            logger.info("LLMIntegrationService initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLMIntegrationService: {e}")
            raise
    
    async def _initialize_providers(self):
        """Initialize all LLM providers"""
        try:
            # OpenAI GPT-4
            if openai.api_key:
                self.providers[LLMProvider.OPENAI_GPT4] = openai
                self.provider_configs[LLMProvider.OPENAI_GPT4] = LLMConfig(
                    provider=LLMProvider.OPENAI_GPT4,
                    model_name="gpt-4-turbo-preview",
                    api_key=openai.api_key,
                    endpoint="https://api.openai.com/v1",
                    max_tokens=4096,
                    temperature=0.7,
                    timeout=60,
                    cost_per_token=0.00003,
                    rate_limit_rpm=500,
                    rate_limit_tpm=30000
                )
            
            # Anthropic Claude
            try:
                claude_client = anthropic.Anthropic()
                self.providers[LLMProvider.ANTHROPIC_CLAUDE] = claude_client
                self.provider_configs[LLMProvider.ANTHROPIC_CLAUDE] = LLMConfig(
                    provider=LLMProvider.ANTHROPIC_CLAUDE,
                    model_name="claude-3-opus-20240229",
                    api_key=None,  # Set via environment
                    endpoint="https://api.anthropic.com",
                    max_tokens=4096,
                    temperature=0.7,
                    timeout=60,
                    cost_per_token=0.000015,
                    rate_limit_rpm=200,
                    rate_limit_tpm=20000
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic Claude: {e}")
            
            # Local HuggingFace models
            try:
                self.providers[LLMProvider.HUGGINGFACE_LOCAL] = pipeline(
                    "text-generation",
                    model="microsoft/DialoGPT-medium",
                    device=-1  # CPU
                )
                self.provider_configs[LLMProvider.HUGGINGFACE_LOCAL] = LLMConfig(
                    provider=LLMProvider.HUGGINGFACE_LOCAL,
                    model_name="microsoft/DialoGPT-medium",
                    api_key=None,
                    endpoint=None,
                    max_tokens=1024,
                    temperature=0.8,
                    timeout=30,
                    cost_per_token=0.0,  # Free local model
                    rate_limit_rpm=1000,
                    rate_limit_tpm=50000
                )
            except Exception as e:
                logger.warning(f"Failed to initialize HuggingFace local model: {e}")
            
            logger.info(f"Initialized {len(self.providers)} LLM providers")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM providers: {e}")
            raise
    
    async def _load_agent_personalities(self):
        """Load agent personality configurations"""
        try:
            # Define agent personalities for different trading strategies
            personalities = [
                AgentPersonality(
                    agent_id="trend_follower_001",
                    name="Marcus Momentum",
                    role="Trend Following Specialist",
                    trading_style="aggressive_momentum",
                    risk_tolerance=0.7,
                    communication_style="confident_analytical",
                    expertise_areas=["trend_analysis", "momentum_indicators", "breakout_patterns"],
                    personality_traits={
                        "confidence": 0.8,
                        "risk_seeking": 0.7,
                        "analytical": 0.9,
                        "collaborative": 0.6
                    },
                    system_prompt="""You are Marcus Momentum, a trend-following trading specialist. 
                    You excel at identifying and riding market trends. You're confident in your analysis 
                    but always back decisions with data. You communicate clearly and decisively, 
                    focusing on momentum indicators and trend strength."""
                ),
                
                AgentPersonality(
                    agent_id="arbitrage_bot_003",
                    name="Alex Arbitrage",
                    role="Arbitrage Opportunity Hunter",
                    trading_style="risk_neutral_arbitrage",
                    risk_tolerance=0.3,
                    communication_style="precise_mathematical",
                    expertise_areas=["price_discrepancies", "cross_exchange_analysis", "statistical_arbitrage"],
                    personality_traits={
                        "confidence": 0.9,
                        "risk_seeking": 0.2,
                        "analytical": 0.95,
                        "collaborative": 0.8
                    },
                    system_prompt="""You are Alex Arbitrage, a precision-focused arbitrage specialist. 
                    You identify risk-free profit opportunities across markets. You communicate with 
                    mathematical precision and always quantify potential returns and risks. 
                    You're highly collaborative and share insights freely."""
                ),
                
                AgentPersonality(
                    agent_id="mean_reversion_002",
                    name="Sophia Reversion",
                    role="Mean Reversion Strategist",
                    trading_style="conservative_mean_reversion",
                    risk_tolerance=0.4,
                    communication_style="thoughtful_cautious",
                    expertise_areas=["oversold_conditions", "support_resistance", "statistical_mean_reversion"],
                    personality_traits={
                        "confidence": 0.6,
                        "risk_seeking": 0.3,
                        "analytical": 0.8,
                        "collaborative": 0.9
                    },
                    system_prompt="""You are Sophia Reversion, a thoughtful mean reversion strategist. 
                    You identify when assets are oversold or overbought and likely to revert to mean. 
                    You're cautious and methodical, always considering multiple scenarios. 
                    You communicate thoughtfully and seek input from other agents."""
                ),
                
                AgentPersonality(
                    agent_id="risk_manager_007",
                    name="Riley Risk",
                    role="Portfolio Risk Manager",
                    trading_style="defensive_risk_management",
                    risk_tolerance=0.2,
                    communication_style="authoritative_protective",
                    expertise_areas=["portfolio_risk", "volatility_analysis", "correlation_monitoring"],
                    personality_traits={
                        "confidence": 0.7,
                        "risk_seeking": 0.1,
                        "analytical": 0.9,
                        "collaborative": 0.7
                    },
                    system_prompt="""You are Riley Risk, the portfolio's guardian and risk manager. 
                    Your primary goal is protecting capital and managing downside risk. 
                    You communicate authoritatively when risks are detected and work to 
                    keep the portfolio within safe parameters. You're analytical and protective."""
                )
            ]
            
            for personality in personalities:
                self.agent_personalities[personality.agent_id] = personality
            
            logger.info(f"Loaded {len(personalities)} agent personalities")
            
        except Exception as e:
            logger.error(f"Failed to load agent personalities: {e}")
    
    async def process_llm_request(
        self,
        request: LLMRequest,
        preferred_provider: Optional[LLMProvider] = None
    ) -> LLMResponse:
        """Process an LLM request with intelligent provider selection"""
        try:
            # Check cache first
            cache_key = self._generate_cache_key(request)
            cached_response = await self._get_cached_response(cache_key)
            if cached_response:
                logger.info(f"Returning cached response for request {request.request_id}")
                return cached_response
            
            # Select optimal provider
            provider = await self._select_optimal_provider(request, preferred_provider)
            
            # Process request
            response = await self._process_with_provider(provider, request)
            
            # Cache response
            await self._cache_response(cache_key, response)
            
            # Track usage and performance
            await self._track_usage(provider, request, response)
            
            # Emit event
            if self.event_service:
                await self.event_service.emit_event({
                    'event_type': 'llm.request_processed',
                    'request_id': request.request_id,
                    'provider': provider.value,
                    'task_type': request.task_type.value,
                    'tokens_used': response.tokens_used,
                    'processing_time': response.processing_time,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to process LLM request {request.request_id}: {e}")
            raise
    
    async def _select_optimal_provider(
        self,
        request: LLMRequest,
        preferred_provider: Optional[LLMProvider] = None
    ) -> LLMProvider:
        """Select the optimal LLM provider for the request"""
        try:
            if preferred_provider and preferred_provider in self.providers:
                return preferred_provider
            
            # Provider selection based on task type and requirements
            task_type = request.task_type
            
            # High-complexity tasks prefer GPT-4 or Claude
            if task_type in [LLMTaskType.PORTFOLIO_OPTIMIZATION, LLMTaskType.STRATEGY_GENERATION]:
                if LLMProvider.OPENAI_GPT4 in self.providers:
                    return LLMProvider.OPENAI_GPT4
                elif LLMProvider.ANTHROPIC_CLAUDE in self.providers:
                    return LLMProvider.ANTHROPIC_CLAUDE
            
            # Communication tasks can use lighter models
            elif task_type == LLMTaskType.AGENT_COMMUNICATION:
                if LLMProvider.HUGGINGFACE_LOCAL in self.providers:
                    return LLMProvider.HUGGINGFACE_LOCAL
                elif LLMProvider.OPENAI_GPT35 in self.providers:
                    return LLMProvider.OPENAI_GPT35
            
            # Default to most capable available provider
            provider_priority = [
                LLMProvider.OPENAI_GPT4,
                LLMProvider.ANTHROPIC_CLAUDE,
                LLMProvider.OPENAI_GPT35,
                LLMProvider.HUGGINGFACE_LOCAL
            ]
            
            for provider in provider_priority:
                if provider in self.providers:
                    return provider
            
            raise ValueError("No suitable LLM provider available")
            
        except Exception as e:
            logger.error(f"Failed to select optimal provider: {e}")
            raise
    
    async def _process_with_provider(
        self,
        provider: LLMProvider,
        request: LLMRequest
    ) -> LLMResponse:
        """Process request with specific provider"""
        try:
            start_time = datetime.now(timezone.utc)
            
            if provider == LLMProvider.OPENAI_GPT4:
                response = await self._process_openai(request, "gpt-4-turbo-preview")
            elif provider == LLMProvider.ANTHROPIC_CLAUDE:
                response = await self._process_anthropic(request)
            elif provider == LLMProvider.HUGGINGFACE_LOCAL:
                response = await self._process_huggingface(request)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            return LLMResponse(
                request_id=request.request_id,
                provider=provider,
                content=response['content'],
                tokens_used=response.get('tokens_used', 0),
                processing_time=processing_time,
                confidence_score=response.get('confidence_score', 0.8),
                metadata=response.get('metadata', {}),
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.error(f"Failed to process with provider {provider}: {e}")
            raise
    
    async def _process_openai(self, request: LLMRequest, model: str) -> Dict[str, Any]:
        """Process request with OpenAI"""
        try:
            messages = [
                {"role": "system", "content": request.system_prompt or "You are a helpful AI assistant for trading analysis."},
                {"role": "user", "content": request.prompt}
            ]
            
            if request.context:
                context_message = f"Context: {json.dumps(request.context, indent=2)}"
                messages.insert(-1, {"role": "user", "content": context_message})
            
            response = await openai.ChatCompletion.acreate(
                model=model,
                messages=messages,
                max_tokens=request.max_tokens or 2048,
                temperature=request.temperature or 0.7,
                timeout=30
            )
            
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            return {
                'content': content,
                'tokens_used': tokens_used,
                'confidence_score': 0.85,
                'metadata': {
                    'model': model,
                    'finish_reason': response.choices[0].finish_reason
                }
            }
            
        except Exception as e:
            logger.error(f"OpenAI processing failed: {e}")
            raise
    
    async def _process_anthropic(self, request: LLMRequest) -> Dict[str, Any]:
        """Process request with Anthropic Claude"""
        try:
            client = self.providers[LLMProvider.ANTHROPIC_CLAUDE]
            
            prompt = request.prompt
            if request.context:
                prompt = f"Context: {json.dumps(request.context, indent=2)}\n\n{prompt}"
            
            if request.system_prompt:
                prompt = f"{request.system_prompt}\n\n{prompt}"
            
            response = await client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=request.max_tokens or 2048,
                temperature=request.temperature or 0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            
            return {
                'content': content,
                'tokens_used': tokens_used,
                'confidence_score': 0.9,
                'metadata': {
                    'model': 'claude-3-opus-20240229',
                    'stop_reason': response.stop_reason
                }
            }
            
        except Exception as e:
            logger.error(f"Anthropic processing failed: {e}")
            raise
    
    async def _process_huggingface(self, request: LLMRequest) -> Dict[str, Any]:
        """Process request with local HuggingFace model"""
        try:
            generator = self.providers[LLMProvider.HUGGINGFACE_LOCAL]
            
            prompt = request.prompt
            if request.context:
                prompt = f"Context: {json.dumps(request.context, indent=2)}\n\n{prompt}"
            
            response = generator(
                prompt,
                max_length=request.max_tokens or 1024,
                temperature=request.temperature or 0.8,
                num_return_sequences=1,
                pad_token_id=generator.tokenizer.eos_token_id
            )
            
            content = response[0]['generated_text']
            # Remove the original prompt from the response
            if content.startswith(prompt):
                content = content[len(prompt):].strip()
            
            return {
                'content': content,
                'tokens_used': len(content.split()),  # Approximate token count
                'confidence_score': 0.7,
                'metadata': {
                    'model': 'microsoft/DialoGPT-medium',
                    'local_processing': True
                }
            }
            
        except Exception as e:
            logger.error(f"HuggingFace processing failed: {e}")
            raise
    
    async def start_agent_conversation(
        self,
        conversation_id: str,
        participants: List[str],
        topic: str,
        context: Dict[str, Any]
    ) -> str:
        """Start a multi-agent conversation"""
        try:
            # Create conversation context
            conversation_context = ConversationContext(
                conversation_id=conversation_id,
                participants=participants,
                topic=topic,
                context=context,
                created_at=datetime.now(timezone.utc)
            )
            
            self.conversation_contexts[conversation_id] = conversation_context
            self.active_conversations[conversation_id] = []
            
            # Generate opening message
            opening_prompt = f"""
            You are facilitating a conversation between trading agents on the topic: {topic}
            
            Participants: {', '.join([self.agent_personalities.get(p, AgentPersonality(agent_id=p, name=p, role='Unknown', trading_style='', risk_tolerance=0.5, communication_style='', expertise_areas=[], personality_traits={}, system_prompt='')).name for p in participants])}
            
            Context: {json.dumps(context, indent=2)}
            
            Generate an opening statement that sets the stage for productive discussion.
            """
            
            request = LLMRequest(
                request_id=f"conversation_start_{conversation_id}",
                task_type=LLMTaskType.AGENT_COMMUNICATION,
                prompt=opening_prompt,
                context=context
            )
            
            response = await self.process_llm_request(request)
            
            # Create opening message
            opening_message = ConversationMessage(
                message_id=f"msg_{conversation_id}_000",
                sender_id="system",
                recipient_id="all",
                content=response.content,
                message_type="conversation_start",
                timestamp=datetime.now(timezone.utc),
                context=context
            )
            
            self.active_conversations[conversation_id].append(opening_message)
            
            # Emit event
            if self.event_service:
                await self.event_service.emit_event({
                    'event_type': 'conversation.started',
                    'conversation_id': conversation_id,
                    'participants': participants,
                    'topic': topic,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
            
            logger.info(f"Started conversation {conversation_id} with {len(participants)} participants")
            return conversation_id
            
        except Exception as e:
            logger.error(f"Failed to start agent conversation: {e}")
            raise
    
    async def send_agent_message(
        self,
        conversation_id: str,
        sender_id: str,
        content: str,
        message_type: str = "discussion",
        context: Optional[Dict[str, Any]] = None
    ) -> ConversationMessage:
        """Send a message in an agent conversation"""
        try:
            if conversation_id not in self.active_conversations:
                raise ValueError(f"Conversation {conversation_id} not found")
            
            # Get agent personality
            personality = self.agent_personalities.get(sender_id)
            if personality:
                # Enhance message with agent personality
                enhanced_prompt = f"""
                {personality.system_prompt}
                
                Current conversation context: {self.conversation_contexts[conversation_id].topic}
                
                Recent messages: {json.dumps([msg.content for msg in self.active_conversations[conversation_id][-3:]], indent=2)}
                
                Your message content: {content}
                
                Respond in character as {personality.name}, keeping your {personality.communication_style} style 
                and expertise in {', '.join(personality.expertise_areas)}.
                """
                
                request = LLMRequest(
                    request_id=f"agent_msg_{sender_id}_{len(self.active_conversations[conversation_id])}",
                    task_type=LLMTaskType.AGENT_COMMUNICATION,
                    prompt=enhanced_prompt,
                    context=context or {},
                    system_prompt=personality.system_prompt
                )
                
                response = await self.process_llm_request(request)
                enhanced_content = response.content
            else:
                enhanced_content = content
            
            # Create message
            message = ConversationMessage(
                message_id=f"msg_{conversation_id}_{len(self.active_conversations[conversation_id]):03d}",
                sender_id=sender_id,
                recipient_id="all",
                content=enhanced_content,
                message_type=message_type,
                timestamp=datetime.now(timezone.utc),
                context=context or {}
            )
            
            # Add to conversation
            self.active_conversations[conversation_id].append(message)
            
            # Emit event
            if self.event_service:
                await self.event_service.emit_event({
                    'event_type': 'conversation.message_sent',
                    'conversation_id': conversation_id,
                    'sender_id': sender_id,
                    'message_id': message.message_id,
                    'message_type': message_type,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
            
            return message
            
        except Exception as e:
            logger.error(f"Failed to send agent message: {e}")
            raise
    
    async def generate_trading_analysis(
        self,
        market_data: Dict[str, Any],
        portfolio_data: Dict[str, Any],
        agent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive trading analysis using LLM"""
        try:
            # Get agent personality if specified
            personality = self.agent_personalities.get(agent_id) if agent_id else None
            
            analysis_prompt = f"""
            Analyze the current market conditions and provide trading recommendations.
            
            Market Data:
            {json.dumps(market_data, indent=2)}
            
            Portfolio Data:
            {json.dumps(portfolio_data, indent=2)}
            
            Provide analysis in the following format:
            1. Market Overview
            2. Key Opportunities
            3. Risk Assessment
            4. Specific Recommendations
            5. Position Sizing Suggestions
            
            Be specific and actionable in your recommendations.
            """
            
            request = LLMRequest(
                request_id=f"trading_analysis_{agent_id}_{int(datetime.now().timestamp())}",
                task_type=LLMTaskType.MARKET_ANALYSIS,
                prompt=analysis_prompt,
                context={
                    'market_data': market_data,
                    'portfolio_data': portfolio_data,
                    'agent_id': agent_id
                },
                system_prompt=personality.system_prompt if personality else None
            )
            
            response = await self.process_llm_request(request)
            
            return {
                'analysis': response.content,
                'confidence': response.confidence_score,
                'agent_id': agent_id,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'tokens_used': response.tokens_used
            }
            
        except Exception as e:
            logger.error(f"Failed to generate trading analysis: {e}")
            raise
    
    async def _generate_cache_key(self, request: LLMRequest) -> str:
        """Generate cache key for request"""
        content = f"{request.task_type.value}:{request.prompt}:{json.dumps(request.context, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def _get_cached_response(self, cache_key: str) -> Optional[LLMResponse]:
        """Get cached response if available"""
        try:
            if self.redis:
                cached_data = await self.redis.get(f"llm_cache:{cache_key}")
                if cached_data:
                    data = json.loads(cached_data)
                    return LLMResponse(**data)
            
            return self.response_cache.get(cache_key)
            
        except Exception as e:
            logger.error(f"Failed to get cached response: {e}")
            return None
    
    async def _cache_response(self, cache_key: str, response: LLMResponse):
        """Cache response"""
        try:
            response_data = asdict(response)
            response_data['timestamp'] = response.timestamp.isoformat()
            
            if self.redis:
                await self.redis.setex(
                    f"llm_cache:{cache_key}",
                    self.cache_ttl,
                    json.dumps(response_data, default=str)
                )
            
            # Also cache in memory
            self.response_cache[cache_key] = response
            
        except Exception as e:
            logger.error(f"Failed to cache response: {e}")
    
    async def _track_usage(self, provider: LLMProvider, request: LLMRequest, response: LLMResponse):
        """Track usage statistics"""
        try:
            # Update token usage
            self.token_usage[provider.value] = self.token_usage.get(provider.value, 0) + response.tokens_used
            
            # Update cost tracking
            config = self.provider_configs.get(provider)
            if config:
                cost = response.tokens_used * config.cost_per_token
                self.cost_tracking[provider.value] = self.cost_tracking.get(provider.value, 0.0) + cost
            
            # Update performance metrics
            if provider not in self.provider_performance:
                self.provider_performance[provider] = {
                    'total_requests': 0,
                    'total_tokens': 0,
                    'avg_response_time': 0.0,
                    'error_rate': 0.0
                }
            
            perf = self.provider_performance[provider]
            perf['total_requests'] += 1
            perf['total_tokens'] += response.tokens_used
            perf['avg_response_time'] = (
                (perf['avg_response_time'] * (perf['total_requests'] - 1) + response.processing_time) /
                perf['total_requests']
            )
            
        except Exception as e:
            logger.error(f"Failed to track usage: {e}")
    
    async def _conversation_manager(self):
        """Background task for managing conversations"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Clean up old conversations
                current_time = datetime.now(timezone.utc)
                conversations_to_remove = []
                
                for conv_id, context in self.conversation_contexts.items():
                    if (current_time - context.created_at).total_seconds() > 3600:  # 1 hour
                        conversations_to_remove.append(conv_id)
                
                for conv_id in conversations_to_remove:
                    del self.conversation_contexts[conv_id]
                    del self.active_conversations[conv_id]
                    logger.info(f"Cleaned up conversation {conv_id}")
                
            except Exception as e:
                logger.error(f"Error in conversation manager: {e}")
    
    async def _performance_monitor(self):
        """Background task for monitoring provider performance"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Log performance metrics
                for provider, metrics in self.provider_performance.items():
                    logger.info(
                        f"Provider {provider.value}: "
                        f"{metrics['total_requests']} requests, "
                        f"{metrics['total_tokens']} tokens, "
                        f"{metrics['avg_response_time']:.2f}s avg response time"
                    )
                
                # Log cost metrics
                total_cost = sum(self.cost_tracking.values())
                logger.info(f"Total LLM costs: ${total_cost:.4f}")
                
            except Exception as e:
                logger.error(f"Error in performance monitor: {e}")
    
    async def _cache_cleanup(self):
        """Background task for cache cleanup"""
        while True:
            try:
                await asyncio.sleep(600)  # Check every 10 minutes
                
                # Clean up in-memory cache
                current_time = datetime.now(timezone.utc)
                keys_to_remove = []
                
                for key, response in self.response_cache.items():
                    if (current_time - response.timestamp).total_seconds() > self.cache_ttl:
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    del self.response_cache[key]
                
                logger.info(f"Cleaned up {len(keys_to_remove)} cache entries")
                
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get service status and health metrics"""
        return {
            "service": "llm_integration_service",
            "status": "running",
            "providers": list(self.providers.keys()),
            "active_conversations": len(self.active_conversations),
            "agent_personalities": len(self.agent_personalities),
            "total_token_usage": sum(self.token_usage.values()),
            "total_costs": sum(self.cost_tracking.values()),
            "cache_size": len(self.response_cache),
            "last_health_check": datetime.now(timezone.utc).isoformat()
        }

# Factory function for service registry
def create_llm_integration_service():
    """Factory function to create LLMIntegrationService instance"""
    registry = get_registry()
    redis_client = registry.get_connection("redis")
    
    service = LLMIntegrationService(redis_client)
    return service