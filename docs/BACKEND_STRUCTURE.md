# Cival Trading Platform - Backend Structure & Database Schema

## 🏗️ Backend Architecture Overview

### System Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Next.js)                       │
├─────────────────────────────────────────────────────────────┤
│                    API Gateway Layer                        │
├─────────────────────────────────────────────────────────────┤
│                FastAPI Application Layer                    │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│  │   Trading   │  Portfolio  │    Risk     │   Market    │  │
│  │   Engine    │   Tracker   │  Manager    │    Data     │  │
│  └─────────────┴─────────────┴─────────────┴─────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                   Service Layer                             │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│  │ Exchange    │   Wallet    │   Agent     │  WebSocket  │  │
│  │ Connectors  │  Manager    │Coordinator  │  Manager    │  │
│  └─────────────┴─────────────┴─────────────┴─────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                   Data Layer                                │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│  │ PostgreSQL  │    Redis    │  External   │    File     │  │
│  │ (Supabase)  │   Cache     │   APIs      │  Storage    │  │
│  └─────────────┴─────────────┴─────────────┴─────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Backend File Structure

### Python Backend (`python-ai-services/`)
```
python-ai-services/
├── main_consolidated.py              # Main FastAPI application
├── core/                            # Core infrastructure
│   ├── __init__.py
│   ├── service_registry.py          # Dependency injection
│   ├── config.py                    # Configuration management
│   └── exceptions.py                # Custom exceptions
├── services/                        # Business logic services
│   ├── __init__.py
│   ├── trading_engine.py            # Main trading orchestration
│   ├── market_data_service.py       # Market data aggregation
│   ├── portfolio_tracker.py         # Portfolio management
│   ├── risk_manager.py              # Risk assessment
│   ├── agent_coordinator.py         # AI agent management
│   ├── order_management.py          # Order lifecycle
│   ├── websocket_manager.py         # Real-time communication
│   └── notification_service.py      # Alerts and notifications
├── connectors/                      # Exchange integrations
│   ├── __init__.py
│   ├── binance_connector.py         # Binance API integration
│   ├── coinbase_connector.py        # Coinbase Pro integration
│   ├── hyperliquid_connector.py     # Hyperliquid integration
│   └── dex_connector.py             # DEX aggregation
├── models/                          # Data models (Pydantic)
│   ├── __init__.py
│   ├── trading_models.py            # Trading-related models
│   ├── portfolio_models.py          # Portfolio models
│   ├── market_models.py             # Market data models
│   ├── agent_models.py              # Agent-related models
│   └── user_models.py               # User and authentication
├── database/                        # Database operations
│   ├── __init__.py
│   ├── connection.py                # Database connection
│   ├── models.py                    # SQLAlchemy models
│   ├── migrations/                  # Database migrations
│   └── seeds/                       # Initial data
├── utils/                           # Utility functions
│   ├── __init__.py
│   ├── encryption.py                # Security utilities
│   ├── validators.py                # Input validation
│   └── helpers.py                   # General helpers
└── tests/                           # Test suite
    ├── unit/                        # Unit tests
    ├── integration/                 # Integration tests
    └── fixtures/                    # Test data
```

## 🗄️ Database Schema (PostgreSQL)

### Core Tables

#### 1. Users & Authentication
```sql
-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255),
    full_name VARCHAR(255),
    role VARCHAR(50) DEFAULT 'trader',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- User sessions
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- API keys for exchange integrations
CREATE TABLE user_api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    exchange VARCHAR(50) NOT NULL,
    api_key_encrypted TEXT NOT NULL,
    api_secret_encrypted TEXT NOT NULL,
    passphrase_encrypted TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### 2. Trading Tables
```sql
-- Trading accounts per exchange
CREATE TABLE trading_accounts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    exchange VARCHAR(50) NOT NULL,
    account_id VARCHAR(255) NOT NULL,
    account_type VARCHAR(50) DEFAULT 'spot', -- spot, margin, futures
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, exchange, account_id)
);

-- Trading pairs/symbols
CREATE TABLE trading_pairs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(50) NOT NULL,
    base_asset VARCHAR(20) NOT NULL,
    quote_asset VARCHAR(20) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    min_order_size DECIMAL(20,8),
    max_order_size DECIMAL(20,8),
    tick_size DECIMAL(20,8),
    step_size DECIMAL(20,8),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, exchange)
);

-- Orders
CREATE TABLE orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    account_id UUID REFERENCES trading_accounts(id),
    external_order_id VARCHAR(255), -- Exchange order ID
    symbol VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL, -- buy, sell
    type VARCHAR(20) NOT NULL, -- market, limit, stop, stop_limit
    quantity DECIMAL(20,8) NOT NULL,
    price DECIMAL(20,8),
    stop_price DECIMAL(20,8),
    time_in_force VARCHAR(10) DEFAULT 'GTC', -- GTC, IOC, FOK
    status VARCHAR(20) DEFAULT 'pending', -- pending, filled, cancelled, rejected, partial
    filled_quantity DECIMAL(20,8) DEFAULT 0,
    average_price DECIMAL(20,8),
    fees DECIMAL(20,8) DEFAULT 0,
    exchange VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Trade executions
CREATE TABLE trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id UUID REFERENCES orders(id),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    external_trade_id VARCHAR(255), -- Exchange trade ID
    symbol VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity DECIMAL(20,8) NOT NULL,
    price DECIMAL(20,8) NOT NULL,
    fees DECIMAL(20,8) DEFAULT 0,
    exchange VARCHAR(50) NOT NULL,
    executed_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### 3. Portfolio Tables
```sql
-- Portfolio positions
CREATE TABLE positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    account_id UUID REFERENCES trading_accounts(id),
    symbol VARCHAR(50) NOT NULL,
    size DECIMAL(20,8) NOT NULL,
    average_price DECIMAL(20,8) NOT NULL,
    market_value DECIMAL(20,8),
    unrealized_pnl DECIMAL(20,8),
    exchange VARCHAR(50) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, account_id, symbol)
);

-- Portfolio balances
CREATE TABLE balances (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    account_id UUID REFERENCES trading_accounts(id),
    asset VARCHAR(20) NOT NULL,
    free DECIMAL(20,8) DEFAULT 0,
    locked DECIMAL(20,8) DEFAULT 0,
    total DECIMAL(20,8) DEFAULT 0,
    usd_value DECIMAL(20,8),
    exchange VARCHAR(50) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, account_id, asset)
);

-- Portfolio snapshots for performance tracking
CREATE TABLE portfolio_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    total_value DECIMAL(20,8) NOT NULL,
    total_pnl DECIMAL(20,8),
    daily_pnl DECIMAL(20,8),
    positions_count INTEGER DEFAULT 0,
    snapshot_date DATE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, snapshot_date)
);
```

#### 4. Trading Strategies & Agents
```sql
-- Trading strategies
CREATE TABLE trading_strategies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    strategy_type VARCHAR(50) NOT NULL, -- trend_following, mean_reversion, arbitrage
    parameters JSONB NOT NULL,
    is_active BOOLEAN DEFAULT false,
    performance_metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- AI trading agents
CREATE TABLE trading_agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    agent_type VARCHAR(50) NOT NULL, -- trend_follower, arbitrageur, sentiment
    configuration JSONB NOT NULL,
    status VARCHAR(20) DEFAULT 'inactive', -- active, inactive, error
    performance JSONB,
    last_decision_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Trading signals generated by strategies/agents
CREATE TABLE trading_signals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_id UUID REFERENCES trading_strategies(id),
    agent_id UUID REFERENCES trading_agents(id),
    symbol VARCHAR(50) NOT NULL,
    action VARCHAR(10) NOT NULL, -- buy, sell, hold
    strength DECIMAL(5,2) NOT NULL, -- 0-100
    price DECIMAL(20,8) NOT NULL,
    confidence DECIMAL(5,2) NOT NULL, -- 0-1
    reasoning TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### 5. Market Data Tables
```sql
-- OHLCV candlestick data
CREATE TABLE market_data_ohlcv (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(50) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    timeframe VARCHAR(10) NOT NULL, -- 1m, 5m, 1h, 1d
    open_time TIMESTAMP WITH TIME ZONE NOT NULL,
    close_time TIMESTAMP WITH TIME ZONE NOT NULL,
    open_price DECIMAL(20,8) NOT NULL,
    high_price DECIMAL(20,8) NOT NULL,
    low_price DECIMAL(20,8) NOT NULL,
    close_price DECIMAL(20,8) NOT NULL,
    volume DECIMAL(20,8) NOT NULL,
    quote_volume DECIMAL(20,8),
    trades_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, exchange, timeframe, open_time)
);

-- Real-time price tickers
CREATE TABLE price_tickers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(50) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    price DECIMAL(20,8) NOT NULL,
    bid_price DECIMAL(20,8),
    ask_price DECIMAL(20,8),
    volume_24h DECIMAL(20,8),
    change_24h DECIMAL(10,4),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
    UNIQUE(symbol, exchange)
);

-- Order book snapshots
CREATE TABLE order_book_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(50) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    bids JSONB NOT NULL, -- [[price, quantity], ...]
    asks JSONB NOT NULL, -- [[price, quantity], ...]
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### 6. Risk Management Tables
```sql
-- Risk metrics calculations
CREATE TABLE risk_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    portfolio_risk DECIMAL(10,4),
    value_at_risk DECIMAL(20,8),
    max_drawdown DECIMAL(10,4),
    sharpe_ratio DECIMAL(10,4),
    position_risks JSONB, -- {symbol: risk_score}
    calculated_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Risk alerts and notifications
CREATE TABLE risk_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    alert_type VARCHAR(50) NOT NULL, -- var_breach, drawdown_limit, position_risk
    severity VARCHAR(20) NOT NULL, -- low, medium, high, critical
    message TEXT NOT NULL,
    data JSONB,
    acknowledged BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Trading limits and rules
CREATE TABLE trading_limits (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    limit_type VARCHAR(50) NOT NULL, -- daily_loss, position_size, leverage
    symbol VARCHAR(50), -- NULL for global limits
    limit_value DECIMAL(20,8) NOT NULL,
    current_value DECIMAL(20,8) DEFAULT 0,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### 7. System & Monitoring Tables
```sql
-- System logs
CREATE TABLE system_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    level VARCHAR(20) NOT NULL, -- DEBUG, INFO, WARNING, ERROR, CRITICAL
    module VARCHAR(100) NOT NULL,
    message TEXT NOT NULL,
    data JSONB,
    user_id UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- API rate limiting
CREATE TABLE api_rate_limits (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    endpoint VARCHAR(255) NOT NULL,
    requests_count INTEGER DEFAULT 0,
    window_start TIMESTAMP WITH TIME ZONE NOT NULL,
    window_duration_seconds INTEGER NOT NULL,
    limit_exceeded BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Exchange API status monitoring
CREATE TABLE exchange_status (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    exchange VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL, -- online, offline, maintenance
    last_ping TIMESTAMP WITH TIME ZONE,
    response_time_ms INTEGER,
    error_message TEXT,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(exchange)
);
```

## 🔧 Database Indexes & Performance

### Key Indexes
```sql
-- Performance indexes for frequent queries
CREATE INDEX idx_orders_user_status ON orders(user_id, status);
CREATE INDEX idx_orders_symbol_created ON orders(symbol, created_at DESC);
CREATE INDEX idx_trades_user_executed ON trades(user_id, executed_at DESC);
CREATE INDEX idx_positions_user_symbol ON positions(user_id, symbol);
CREATE INDEX idx_market_data_symbol_time ON market_data_ohlcv(symbol, timeframe, open_time DESC);
CREATE INDEX idx_signals_created ON trading_signals(created_at DESC);
CREATE INDEX idx_logs_level_created ON system_logs(level, created_at DESC);

-- Unique constraints for data integrity
ALTER TABLE positions ADD CONSTRAINT uk_position_user_symbol UNIQUE(user_id, symbol);
ALTER TABLE balances ADD CONSTRAINT uk_balance_user_asset UNIQUE(user_id, asset);
ALTER TABLE price_tickers ADD CONSTRAINT uk_ticker_symbol_exchange UNIQUE(symbol, exchange);
```

### Database Configuration
```sql
-- Enable Row Level Security
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
ALTER TABLE trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE balances ENABLE ROW LEVEL SECURITY;

-- RLS Policies (example for orders)
CREATE POLICY "Users can only see their own orders" ON orders
    FOR ALL USING (auth.uid() = user_id);

-- Performance settings
SET shared_preload_libraries = 'pg_stat_statements';
SET track_activity_query_size = 2048;
SET log_statement = 'all';
```

## 🔄 Data Flow & Relationships

### Entity Relationships
```
Users (1:N) → Trading Accounts (1:N) → Orders (1:N) → Trades
Users (1:N) → Positions (1:1) → Balances
Users (1:N) → Trading Strategies (1:N) → Trading Signals
Users (1:N) → Trading Agents (1:N) → Trading Signals
Users (1:N) → Risk Metrics
Users (1:N) → Risk Alerts
```

### Data Synchronization
```python
# Real-time data sync pattern
async def sync_portfolio_data(user_id: UUID):
    # 1. Fetch latest trades
    recent_trades = await get_recent_trades(user_id)
    
    # 2. Update positions
    for trade in recent_trades:
        await update_position(user_id, trade.symbol, trade)
    
    # 3. Recalculate balances
    await recalculate_balances(user_id)
    
    # 4. Update portfolio snapshot
    await create_portfolio_snapshot(user_id)
    
    # 5. Emit real-time update
    await websocket_manager.emit_portfolio_update(user_id)
```

## 📊 Database Migrations

### Migration Structure
```python
# Example migration file
"""
Add trading signals table

Revision ID: 001_trading_signals
Revises: 000_initial
Create Date: 2025-12-15
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

def upgrade():
    op.create_table(
        'trading_signals',
        sa.Column('id', postgresql.UUID(), nullable=False),
        sa.Column('strategy_id', postgresql.UUID(), nullable=True),
        sa.Column('symbol', sa.VARCHAR(50), nullable=False),
        sa.Column('action', sa.VARCHAR(10), nullable=False),
        sa.Column('strength', sa.DECIMAL(5,2), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

def downgrade():
    op.drop_table('trading_signals')
```

---

**Document Version:** 1.0  
**Last Updated:** December 2025  
**Database Schema Version:** v1.0  
**Total Tables:** 20+  
**Estimated Storage:** 10GB+ for active trading data