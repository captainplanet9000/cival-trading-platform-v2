# Cival Trading Platform - Technology Stack & APIs Documentation

## 🛠️ Core Technology Stack

### Frontend Technologies

#### Next.js 15 Framework
- **App Router:** Modern routing with layouts and nested routes
- **Server Components:** Server-side rendering for optimal performance
- **TypeScript:** Full type safety across the application
- **Features Used:**
  - Static generation for marketing pages
  - Dynamic rendering for trading interfaces
  - API routes for backend integration
  - Middleware for authentication

#### React 18 Ecosystem
- **React 18:** Latest React with concurrent features
- **Hooks:** useState, useEffect, useCallback, useMemo, useContext
- **Context API:** Global state management for trading data
- **Suspense:** Loading states and code splitting
- **Error Boundaries:** Graceful error handling

#### UI & Styling
- **Tailwind CSS:** Utility-first CSS framework
- **Shadcn/UI:** Premium component library with accessible components
- **Framer Motion:** Advanced animations and transitions
- **Lucide React:** Consistent icon system
- **Responsive Design:** Mobile-first approach

### Backend Technologies

#### FastAPI Python Framework
- **Async/Await:** Non-blocking operations for high performance
- **Pydantic v2:** Data validation and serialization
- **Type Hints:** Full Python type safety
- **Automatic Documentation:** OpenAPI/Swagger integration
- **WebSocket Support:** Real-time bidirectional communication

#### Database & Storage
- **PostgreSQL:** Primary relational database
- **Supabase:** Database hosting with real-time features
- **Row Level Security:** Database-level security policies
- **Redis:** Caching and session management
- **Time-series Optimization:** Efficient storage for trading data

#### Trading Infrastructure
- **Multi-Exchange APIs:** Direct integration with trading platforms
- **WebSocket Feeds:** Real-time market data streaming
- **Order Management:** Unified order routing and tracking
- **Risk Engine:** Real-time risk calculations and limits

## 🔌 Exchange APIs Integration

### Binance Integration
```python
# API Configuration
BINANCE_API_URL = "https://api.binance.com"
BINANCE_WS_URL = "wss://stream.binance.com:9443"

# Endpoints Used
- /api/v3/ticker/price      # Real-time prices
- /api/v3/depth             # Order book data  
- /api/v3/order             # Order placement
- /api/v3/account           # Account information
- /api/v3/myTrades          # Trade history
```

**Features:**
- Spot trading (buy/sell cryptocurrencies)
- Futures trading (leverage positions)
- Real-time price feeds
- Order book streaming
- Account balance tracking

### Coinbase Pro Integration
```python
# API Configuration  
COINBASE_API_URL = "https://api.pro.coinbase.com"
COINBASE_WS_URL = "wss://ws-feed.pro.coinbase.com"

# Endpoints Used
- /products                 # Available trading pairs
- /products/{id}/ticker     # Real-time ticker
- /orders                   # Order management
- /accounts                 # Account balances
- /fills                    # Trade executions
```

**Features:**
- Professional trading interface
- Advanced order types
- Market maker rebates
- USD fiat integration
- Institutional-grade APIs

### Hyperliquid Integration
```python
# API Configuration
HYPERLIQUID_API_URL = "https://api.hyperliquid.xyz"
HYPERLIQUID_WS_URL = "wss://api.hyperliquid.xyz/ws"

# Endpoints Used
- /info/metas              # Market metadata
- /info/user_state         # User positions
- /exchange/order          # Order placement
- /exchange/cancel         # Order cancellation
- /info/user_fills         # Fill history
```

**Features:**
- Perpetual futures trading
- Up to 50x leverage
- Cross-margin trading
- Funding rate optimization
- MEV protection

### Decentralized Exchange (DEX) Integration

#### Uniswap V3
```typescript
// Smart Contract Integration
const UNISWAP_V3_ROUTER = "0xE592427A0AEce92De3Edee1F18E0157C05861564"
const UNISWAP_V3_FACTORY = "0x1F98431c8aD98523631AE4a59f267346ea31F984"

// Key Functions
- exactInputSingle()       # Single-hop swaps
- exactInput()             # Multi-hop swaps  
- mint()                   # Add liquidity
- burn()                   # Remove liquidity
```

#### 1inch Aggregator
```python
# API Configuration
ONEINCH_API_URL = "https://api.1inch.dev"

# Endpoints Used
- /v5.0/{chainId}/quote    # Price quotes
- /v5.0/{chainId}/swap     # Swap execution
- /v5.0/{chainId}/tokens   # Token list
- /v5.0/{chainId}/protocols # DEX protocols
```

**Features:**
- Best price aggregation across 100+ DEXs
- Gas optimization
- MEV protection
- Limit orders
- Cross-chain swaps

## 🤖 AI & Machine Learning APIs

### OpenAI Integration
```python
# API Configuration
OPENAI_API_URL = "https://api.openai.com/v1"

# Models Used
- GPT-4 Turbo             # Market analysis
- GPT-3.5 Turbo           # Fast decision making
- Embeddings              # Sentiment analysis
- Whisper                 # Audio news processing
```

**Use Cases:**
- Market sentiment analysis
- News impact assessment
- Trading strategy generation
- Risk assessment
- Pattern recognition

### Custom ML Models
```python
# Trading Strategies
- LSTM Networks           # Price prediction
- Random Forest           # Feature importance
- SVM                     # Classification
- Reinforcement Learning  # Strategy optimization
```

## 📊 Market Data APIs

### Real-time Data Sources

#### CoinGecko API
```python
# API Configuration
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"

# Endpoints Used
- /simple/price           # Current prices
- /coins/markets          # Market data
- /exchanges/rates        # Exchange rates
- /global                 # Market overview
```

#### Alpha Vantage
```python
# API Configuration
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"

# Data Types
- TIME_SERIES_DAILY       # Stock prices
- CURRENCY_EXCHANGE_RATE  # Forex data
- CRYPTO_INTRADAY         # Crypto prices
- TECHNICAL_INDICATORS    # TA indicators
```

### WebSocket Data Streams
```python
# Real-time Connections
BINANCE_WS = "wss://stream.binance.com:9443/ws"
COINBASE_WS = "wss://ws-feed.pro.coinbase.com"
HYPERLIQUID_WS = "wss://api.hyperliquid.xyz/ws"

# Message Types
- ticker                  # Price updates
- depth                   # Order book changes
- trade                   # Trade executions
- account                 # Account updates
```

## 🔐 Authentication & Security

### JWT Authentication
```typescript
// Token Structure
interface JWTPayload {
  sub: string           // User ID
  email: string         // User email
  role: string          // User role
  exp: number           // Expiration time
  iat: number           // Issued at
}
```

### API Key Management
```python
# Secure Storage
- Environment Variables   # Local development
- Supabase Vault         # Production secrets
- Redis Encryption       # Temporary keys
- Hardware Security      # Key derivation
```

### Rate Limiting
```python
# Implementation
- Redis-based counters   # Request tracking
- Sliding window         # Smooth rate limits
- Tier-based limits      # User-specific rates
- Circuit breakers       # Failure protection
```

## 🌐 External Service APIs

### Supabase Services
```javascript
// Database Operations
const supabase = createClient(url, key)

// Features Used
- Authentication         # User management
- Real-time subscriptions # Live data updates
- Row Level Security     # Data protection
- Edge Functions         # Serverless compute
- Storage               # File management
```

### Railway Deployment
```yaml
# Railway Configuration
services:
  web:
    source: .
    build:
      command: npm run build
    start:
      command: npm start
    env:
      NODE_ENV: production
```

### Redis Caching
```python
# Redis Operations
import redis

# Use Cases
- Session storage        # User sessions
- Rate limiting          # API throttling
- Market data cache      # Fast data access
- Order queue           # Execution queue
```

## 📱 Frontend API Layer

### Backend Client
```typescript
// API Client Configuration
class BackendClient {
  private baseURL = process.env.NEXT_PUBLIC_API_URL
  private token = this.getAuthToken()
  
  // Trading Operations
  async placeOrder(order: OrderRequest)
  async cancelOrder(orderId: string)
  async getPortfolio()
  async getOrders()
  
  // Market Data
  async getMarketData(symbol: string)
  async getOrderBook(symbol: string)
  async getTradeHistory(symbol: string)
}
```

### WebSocket Integration
```typescript
// Real-time Communication
class AGUIProtocol {
  private ws: WebSocket
  
  // Event Types
  subscribe(event: TradingEvent, callback: Function)
  emit(event: TradingEvent, data: any)
  
  // Supported Events
  - 'trade.order_placed'
  - 'trade.order_filled'
  - 'market_data.price_update'
  - 'portfolio.balance_update'
  - 'agent.decision_made'
}
```

## 🔧 Development Tools

### Build & Development
```json
{
  "scripts": {
    "dev": "next dev",
    "build": "next build", 
    "start": "next start",
    "lint": "next lint",
    "type-check": "tsc --noEmit"
  }
}
```

### Testing Framework
```python
# Python Testing
- pytest                # Unit testing
- pytest-asyncio        # Async testing
- httpx                 # HTTP testing
- pytest-mock          # Mocking

# JavaScript Testing  
- Jest                  # Unit testing
- React Testing Library # Component testing
- Playwright           # E2E testing
- MSW                  # API mocking
```

### Code Quality
```yaml
# ESLint Configuration
extends:
  - next/core-web-vitals
  - "@typescript-eslint/recommended"

# Prettier Configuration
semi: false
singleQuote: true
trailingComma: es5
```

## 📈 Performance Optimization

### Frontend Optimization
- **Code Splitting:** Dynamic imports for large components
- **Image Optimization:** Next.js Image component
- **Bundle Analysis:** Webpack bundle analyzer
- **Caching:** Aggressive caching for static assets

### Backend Optimization
- **Connection Pooling:** Database connection optimization
- **Async Operations:** Non-blocking I/O operations
- **Caching Layers:** Redis for frequently accessed data
- **Database Indexing:** Optimized queries for trading data

### API Performance
- **Response Compression:** Gzip compression for large responses
- **Pagination:** Efficient data loading for large datasets
- **WebSocket Batching:** Batch real-time updates
- **CDN Integration:** Global content delivery

---

**Document Version:** 1.0  
**Last Updated:** December 2025  
**Maintained By:** Development Team