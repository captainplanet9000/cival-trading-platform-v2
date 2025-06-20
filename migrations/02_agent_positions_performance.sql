-- Agent Trading Database Schema: Positions and Performance
-- Migration 02: Position tracking and performance metrics

-- Agent positions table (enhanced from existing schema)
CREATE TABLE IF NOT EXISTS agent_positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(50) REFERENCES agent_trading_permissions(agent_id),
    account_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    quantity DECIMAL(18,8) NOT NULL,
    average_price DECIMAL(18,8) NOT NULL,
    current_price DECIMAL(18,8),
    unrealized_pnl DECIMAL(18,8),
    realized_pnl DECIMAL(18,8) DEFAULT 0,
    opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(agent_id, symbol, account_id)
);

-- Agent performance metrics (enhanced from existing schema)
CREATE TABLE IF NOT EXISTS agent_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(50) REFERENCES agent_trading_permissions(agent_id),
    date DATE NOT NULL,
    total_trades INT DEFAULT 0,
    successful_trades INT DEFAULT 0,
    failed_trades INT DEFAULT 0,
    total_profit_loss DECIMAL(18,8) DEFAULT 0,
    win_rate DECIMAL(5,2) DEFAULT 0,
    average_trade_duration INTEGER DEFAULT 0,
    max_drawdown DECIMAL(18,8) DEFAULT 0,
    sharpe_ratio DECIMAL(8,4) DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(agent_id, date)
);

-- Agent status tracking
CREATE TABLE IF NOT EXISTS agent_status (
    agent_id VARCHAR(50) PRIMARY KEY REFERENCES agent_trading_permissions(agent_id),
    status VARCHAR(20) NOT NULL DEFAULT 'idle',
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);