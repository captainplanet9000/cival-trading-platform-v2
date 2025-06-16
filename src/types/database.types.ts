/**
 * Database Types
 * Auto-generated database types
 */

export interface Database {
  public: {
    Tables: {
      agent_trading_permissions: {
        Row: {
          id: string;
          agent_id: string;
          account_id: string;
          max_trade_size: number;
          max_position_size: number;
          max_daily_trades: number;
          allowed_symbols: string[];
          allowed_strategies: string[];
          risk_level: string;
          is_active: boolean;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          agent_id: string;
          account_id: string;
          max_trade_size: number;
          max_position_size: number;
          max_daily_trades: number;
          allowed_symbols: string[];
          allowed_strategies: string[];
          risk_level: string;
          is_active?: boolean;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          agent_id?: string;
          account_id?: string;
          max_trade_size?: number;
          max_position_size?: number;
          max_daily_trades?: number;
          allowed_symbols?: string[];
          allowed_strategies?: string[];
          risk_level?: string;
          is_active?: boolean;
          created_at?: string;
          updated_at?: string;
        };
      };
      agent_trades: {
        Row: {
          id: string;
          agent_id: string;
          symbol: string;
          side: string;
          quantity: number;
          price: number;
          order_type: string;
          status: string;
          strategy: string;
          reasoning: string;
          confidence_score: number;
          exchange: string;
          order_id: string;
          executed_at: string;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          agent_id: string;
          symbol: string;
          side: string;
          quantity: number;
          price: number;
          order_type: string;
          status: string;
          strategy?: string;
          reasoning?: string;
          confidence_score?: number;
          exchange?: string;
          order_id?: string;
          executed_at?: string;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          agent_id?: string;
          symbol?: string;
          side?: string;
          quantity?: number;
          price?: number;
          order_type?: string;
          status?: string;
          strategy?: string;
          reasoning?: string;
          confidence_score?: number;
          exchange?: string;
          order_id?: string;
          executed_at?: string;
          created_at?: string;
          updated_at?: string;
        };
      };
      agent_positions: {
        Row: {
          id: string;
          agent_id: string;
          symbol: string;
          side: string;
          quantity: number;
          average_entry_price: number;
          current_price: number;
          unrealized_pnl: number;
          position_value: number;
          status: string;
          strategy: string;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          agent_id: string;
          symbol: string;
          side: string;
          quantity: number;
          average_entry_price: number;
          current_price: number;
          unrealized_pnl: number;
          position_value: number;
          status: string;
          strategy?: string;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          agent_id?: string;
          symbol?: string;
          side?: string;
          quantity?: number;
          average_entry_price?: number;
          current_price?: number;
          unrealized_pnl?: number;
          position_value?: number;
          status?: string;
          strategy?: string;
          created_at?: string;
          updated_at?: string;
        };
      };
      agent_performance: {
        Row: {
          id: string;
          agent_id: string;
          date: string;
          profit_loss: number;
          profit_loss_percentage: number;
          total_trades: number;
          successful_trades: number;
          win_rate: number;
          max_drawdown: number;
          sharpe_ratio: number;
          average_trade_duration: number;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          agent_id: string;
          date: string;
          profit_loss: number;
          profit_loss_percentage: number;
          total_trades: number;
          successful_trades: number;
          win_rate: number;
          max_drawdown: number;
          sharpe_ratio: number;
          average_trade_duration: number;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          agent_id?: string;
          date?: string;
          profit_loss?: number;
          profit_loss_percentage?: number;
          total_trades?: number;
          successful_trades?: number;
          win_rate?: number;
          max_drawdown?: number;
          sharpe_ratio?: number;
          average_trade_duration?: number;
          created_at?: string;
          updated_at?: string;
        };
      };
      agent_status: {
        Row: {
          id: string;
          agent_id: string;
          status: string;
          health_score: number;
          last_activity: string;
          uptime: number;
          error_message: string;
          cpu_usage: number;
          memory_usage: number;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          agent_id: string;
          status: string;
          health_score: number;
          last_activity: string;
          uptime: number;
          error_message?: string;
          cpu_usage?: number;
          memory_usage?: number;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          agent_id?: string;
          status?: string;
          health_score?: number;
          last_activity?: string;
          uptime?: number;
          error_message?: string;
          cpu_usage?: number;
          memory_usage?: number;
          created_at?: string;
          updated_at?: string;
        };
      };
      agent_market_data_subscriptions: {
        Row: {
          id: string;
          agent_id: string;
          symbol: string;
          data_type: string;
          interval: string;
          source: string;
          is_active: boolean;
          last_updated: string;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          agent_id: string;
          symbol: string;
          data_type: string;
          interval: string;
          source: string;
          is_active?: boolean;
          last_updated?: string;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          agent_id?: string;
          symbol?: string;
          data_type?: string;
          interval?: string;
          source?: string;
          is_active?: boolean;
          last_updated?: string;
          created_at?: string;
          updated_at?: string;
        };
      };
      agent_state: {
        Row: {
          id: string;
          agent_id: string;
          state_data: any;
          last_checkpoint: string;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          agent_id: string;
          state_data: any;
          last_checkpoint?: string;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          agent_id?: string;
          state_data?: any;
          last_checkpoint?: string;
          created_at?: string;
          updated_at?: string;
        };
      };
      agent_checkpoints: {
        Row: {
          id: string;
          agent_id: string;
          checkpoint_data: any;
          created_at: string;
        };
        Insert: {
          id?: string;
          agent_id: string;
          checkpoint_data: any;
          created_at?: string;
        };
        Update: {
          id?: string;
          agent_id?: string;
          checkpoint_data?: any;
          created_at?: string;
        };
      };
      agent_decisions: {
        Row: {
          id: string;
          agent_id: string;
          decision_type: string;
          decision_data: any;
          signals: any;
          confidence: number;
          executed_at: string;
          result: any;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          agent_id: string;
          decision_type: string;
          decision_data: any;
          signals?: any;
          confidence: number;
          executed_at?: string;
          result?: any;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          agent_id?: string;
          decision_type?: string;
          decision_data?: any;
          signals?: any;
          confidence?: number;
          executed_at?: string;
          result?: any;
          created_at?: string;
          updated_at?: string;
        };
      };
    };
    Views: {
      [_ in never]: never;
    };
    Functions: {
      [_ in never]: never;
    };
    Enums: {
      [_ in never]: never;
    };
    CompositeTypes: {
      [_ in never]: never;
    };
  };
}

export type Tables<
  PublicTableNameOrOptions extends
    | keyof (Database['public']['Tables'])
    | { schema: keyof Database },
  TableName extends PublicTableNameOrOptions extends { schema: keyof Database }
    ? keyof (Database[PublicTableNameOrOptions['schema']]['Tables'])
    : never = never
> = PublicTableNameOrOptions extends { schema: keyof Database }
  ? (Database[PublicTableNameOrOptions['schema']]['Tables'])[TableName] extends {
      Row: infer R;
    }
    ? R
    : never
  : PublicTableNameOrOptions extends keyof (Database['public']['Tables'])
  ? (Database['public']['Tables'])[PublicTableNameOrOptions] extends {
      Row: infer R;
    }
    ? R
    : never
  : never;

export type TablesInsert<
  PublicTableNameOrOptions extends
    | keyof (Database['public']['Tables'])
    | { schema: keyof Database },
  TableName extends PublicTableNameOrOptions extends { schema: keyof Database }
    ? keyof (Database[PublicTableNameOrOptions['schema']]['Tables'])
    : never = never
> = PublicTableNameOrOptions extends { schema: keyof Database }
  ? (Database[PublicTableNameOrOptions['schema']]['Tables'])[TableName] extends {
      Insert: infer I;
    }
    ? I
    : never
  : PublicTableNameOrOptions extends keyof (Database['public']['Tables'])
  ? (Database['public']['Tables'])[PublicTableNameOrOptions] extends {
      Insert: infer I;
    }
    ? I
    : never
  : never;

export type TablesUpdate<
  PublicTableNameOrOptions extends
    | keyof (Database['public']['Tables'])
    | { schema: keyof Database },
  TableName extends PublicTableNameOrOptions extends { schema: keyof Database }
    ? keyof (Database[PublicTableNameOrOptions['schema']]['Tables'])
    : never = never
> = PublicTableNameOrOptions extends { schema: keyof Database }
  ? (Database[PublicTableNameOrOptions['schema']]['Tables'])[TableName] extends {
      Update: infer U;
    }
    ? U
    : never
  : PublicTableNameOrOptions extends keyof (Database['public']['Tables'])
  ? (Database['public']['Tables'])[PublicTableNameOrOptions] extends {
      Update: infer U;
    }
    ? U
    : never
  : never;

export type Enums<
  PublicEnumNameOrOptions extends
    | keyof (Database['public']['Enums'])
    | { schema: keyof Database },
  EnumName extends PublicEnumNameOrOptions extends { schema: keyof Database }
    ? keyof (Database[PublicEnumNameOrOptions['schema']]['Enums'])
    : never = never
> = PublicEnumNameOrOptions extends { schema: keyof Database }
  ? (Database[PublicEnumNameOrOptions['schema']]['Enums'])[EnumName]
  : PublicEnumNameOrOptions extends keyof (Database['public']['Enums'])
  ? (Database['public']['Enums'])[PublicEnumNameOrOptions]
  : never;