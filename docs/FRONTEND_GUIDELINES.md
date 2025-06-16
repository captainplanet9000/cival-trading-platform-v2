# Cival Trading Platform - Frontend Development Guidelines

## 🎨 Frontend Architecture & Standards

### 1. Project Structure & Organization

#### 1.1 Next.js 15 App Router Structure
```
src/
├── app/                              # App Router pages and layouts
│   ├── layout.tsx                    # Root layout (auth removed for solo mode)
│   ├── page.tsx                      # Home page
│   ├── dashboard/                    # Main trading dashboard
│   │   ├── layout.tsx               # Dashboard layout
│   │   ├── page.tsx                 # Dashboard overview
│   │   ├── trading/                 # Trading interface
│   │   ├── analytics/               # Analytics dashboard
│   │   ├── agents/                  # Agent management
│   │   └── risk/                    # Risk management
│   ├── trading/                     # Advanced trading features
│   ├── portfolio/                   # Portfolio management
│   ├── analytics/                   # Performance analytics
│   └── api/                         # API routes
│       ├── trading/                 # Trading endpoints
│       ├── portfolio/               # Portfolio endpoints
│       ├── agents/                  # Agent endpoints
│       └── market/                  # Market data endpoints
├── components/                       # Reusable React components
│   ├── ui/                          # Shadcn/UI base components
│   ├── trading/                     # Trading-specific components
│   ├── dashboard/                   # Dashboard components
│   ├── agent-trading/               # Agent management components
│   ├── real-time-dashboard/         # Live monitoring components
│   ├── charts/                      # Chart components
│   ├── analytics/                   # Analytics components
│   └── performance/                 # Performance monitoring
├── lib/                             # Utilities and core logic
│   ├── api/                         # API clients and wrappers
│   ├── trading/                     # Trading engine and connectors
│   ├── websocket/                   # Real-time communication
│   ├── hooks/                       # Custom React hooks
│   ├── utils/                       # Helper functions
│   └── types/                       # TypeScript type definitions
├── styles/                          # Global styles and themes
│   ├── globals.css                  # Global CSS with Tailwind
│   └── components.css               # Component-specific styles
└── types/                           # Global TypeScript definitions
```

#### 1.2 Component Architecture
```typescript
// Component structure pattern
interface ComponentProps {
  // Required props
  data: DataType
  onAction: (action: ActionType) => void
  
  // Optional props with defaults
  variant?: 'default' | 'compact' | 'detailed'
  loading?: boolean
  className?: string
}

export function Component({ 
  data, 
  onAction, 
  variant = 'default',
  loading = false,
  className 
}: ComponentProps) {
  // Component implementation
}
```

### 2. State Management Strategy

#### 2.1 Local State (useState)
```typescript
// Use for component-specific state
const [isLoading, setIsLoading] = useState(false)
const [formData, setFormData] = useState<FormData>({})
const [errors, setErrors] = useState<Record<string, string>>({})
```

#### 2.2 Context API for Global State
```typescript
// Trading context for app-wide trading state
interface TradingContextType {
  orders: Order[]
  positions: Position[]
  portfolio: Portfolio
  activeSignals: TradingSignal[]
  updateOrder: (order: Order) => void
  addPosition: (position: Position) => void
}

const TradingContext = createContext<TradingContextType | null>(null)

export function TradingProvider({ children }: { children: ReactNode }) {
  const [orders, setOrders] = useState<Order[]>([])
  const [positions, setPositions] = useState<Position[]>([])
  
  // Context value and methods
  const value = {
    orders,
    positions,
    updateOrder: (order: Order) => {
      setOrders(prev => prev.map(o => o.id === order.id ? order : o))
    }
  }
  
  return (
    <TradingContext.Provider value={value}>
      {children}
    </TradingContext.Provider>
  )
}
```

#### 2.3 Custom Hooks for Data Management
```typescript
// Custom hook for trading operations
export function useTrading() {
  const context = useContext(TradingContext)
  if (!context) {
    throw new Error('useTrading must be used within TradingProvider')
  }
  
  const placeOrder = async (orderData: OrderRequest) => {
    try {
      const response = await backendClient.placeOrder(orderData)
      context.updateOrder(response.order)
      return response
    } catch (error) {
      throw new TradingError('Order placement failed', error)
    }
  }
  
  return {
    ...context,
    placeOrder
  }
}
```

### 3. Real-time Data Integration

#### 3.1 AG-UI Protocol v2 Integration
```typescript
// WebSocket connection management
export function useRealTimeData() {
  const [isConnected, setIsConnected] = useState(false)
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date())
  
  useEffect(() => {
    // Initialize AG-UI connection
    const eventBus = getAGUIEventBus()
    
    // Subscribe to events
    const subscriptions = [
      subscribe('trade.order_placed', handleOrderPlaced),
      subscribe('portfolio.value_updated', handlePortfolioUpdate),
      subscribe('market_data.price_update', handlePriceUpdate)
    ]
    
    // Connection management
    eventBus.initialize().then(() => {
      setIsConnected(true)
    }).catch(error => {
      console.error('AG-UI connection failed:', error)
      setIsConnected(false)
    })
    
    return () => {
      subscriptions.forEach(sub => sub.unsubscribe())
    }
  }, [])
  
  return { isConnected, lastUpdate }
}
```

#### 3.2 Event-Driven Updates
```typescript
// Event handling pattern
const handleRealTimeEvent = useCallback((event: AGUIEvent) => {
  switch (event.type) {
    case 'trade.order_filled':
      updateOrderStatus(event.data.order_id, 'filled')
      refreshPortfolio()
      break
      
    case 'market_data.price_update':
      updateMarketPrice(event.data.symbol, event.data.price)
      break
      
    case 'risk.alert_triggered':
      showRiskAlert(event.data)
      break
  }
}, [updateOrderStatus, refreshPortfolio, updateMarketPrice])
```

### 4. UI Component Standards

#### 4.1 Shadcn/UI Component Usage
```typescript
// Import and use Shadcn components
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

// Consistent component patterns
export function TradingCard({ title, children, actions }: TradingCardProps) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle>{title}</CardTitle>
        <div className="flex space-x-2">
          {actions}
        </div>
      </CardHeader>
      <CardContent>
        {children}
      </CardContent>
    </Card>
  )
}
```

#### 4.2 Design System Colors
```css
/* CSS Variables for consistent theming */
:root {
  --background: 0 0% 100%;
  --foreground: 222.2 84% 4.9%;
  --primary: 221.2 83.2% 53.3%;
  --primary-foreground: 210 40% 98%;
  --secondary: 210 40% 96%;
  --secondary-foreground: 222.2 84% 4.9%;
  --muted: 210 40% 96%;
  --muted-foreground: 215.4 16.3% 46.9%;
  --accent: 210 40% 96%;
  --accent-foreground: 222.2 84% 4.9%;
  --destructive: 0 84.2% 60.2%;
  --destructive-foreground: 210 40% 98%;
  --border: 214.3 31.8% 91.4%;
  --input: 214.3 31.8% 91.4%;
  --ring: 221.2 83.2% 53.3%;
  --radius: 0.5rem;
  
  /* Trading-specific colors */
  --profit: 142.1 76.2% 36.3%;
  --loss: 0 84.2% 60.2%;
  --neutral: 215.4 16.3% 46.9%;
}
```

#### 4.3 Responsive Design Patterns
```tsx
// Mobile-first responsive components
export function ResponsiveTradingGrid({ children }: { children: ReactNode }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
      {children}
    </div>
  )
}

// Responsive navigation
export function TradingNavigation() {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)
  
  return (
    <nav className="flex items-center justify-between p-4">
      <div className="hidden md:flex space-x-4">
        {/* Desktop navigation */}
      </div>
      <div className="md:hidden">
        <Button 
          variant="ghost" 
          onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
        >
          <Menu className="h-6 w-6" />
        </Button>
      </div>
    </nav>
  )
}
```

### 5. Performance Optimization

#### 5.1 Code Splitting & Lazy Loading
```typescript
// Dynamic imports for heavy components
const TradingCharts = dynamic(() => import('./TradingCharts'), {
  loading: () => <ChartSkeleton />,
  ssr: false
})

const AdvancedAnalytics = dynamic(() => import('./AdvancedAnalytics'), {
  loading: () => <AnalyticsSkeleton />
})

// Route-based code splitting
const TradingPage = dynamic(() => import('./TradingPage'))
const PortfolioPage = dynamic(() => import('./PortfolioPage'))
```

#### 5.2 Memoization & Optimization
```typescript
// Memoize expensive calculations
const memoizedPortfolioValue = useMemo(() => {
  return calculatePortfolioValue(positions, currentPrices)
}, [positions, currentPrices])

// Memoize callbacks to prevent unnecessary re-renders
const handleOrderClick = useCallback((orderId: string) => {
  onOrderSelect(orderId)
}, [onOrderSelect])

// Memoize components that render large lists
const MemoizedOrderRow = memo(({ order, onSelect }: OrderRowProps) => {
  return (
    <tr onClick={() => onSelect(order.id)}>
      <td>{order.symbol}</td>
      <td>{order.side}</td>
      <td>{order.quantity}</td>
    </tr>
  )
})
```

#### 5.3 Virtual Scrolling for Large Datasets
```typescript
// Virtual list for large order books
import { FixedSizeList as List } from 'react-window'

export function VirtualOrderBook({ orders }: { orders: Order[] }) {
  const Row = ({ index, style }: { index: number, style: CSSProperties }) => (
    <div style={style}>
      <OrderRow order={orders[index]} />
    </div>
  )
  
  return (
    <List
      height={400}
      itemCount={orders.length}
      itemSize={35}
      width="100%"
    >
      {Row}
    </List>
  )
}
```

### 6. Error Handling & Loading States

#### 6.1 Error Boundaries
```typescript
// Global error boundary for trading components
interface TradingErrorBoundaryState {
  hasError: boolean
  error?: Error
}

class TradingErrorBoundary extends Component<
  { children: ReactNode },
  TradingErrorBoundaryState
> {
  constructor(props: { children: ReactNode }) {
    super(props)
    this.state = { hasError: false }
  }
  
  static getDerivedStateFromError(error: Error): TradingErrorBoundaryState {
    return { hasError: true, error }
  }
  
  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Trading error caught:', error, errorInfo)
    // Log to monitoring service
  }
  
  render() {
    if (this.state.hasError) {
      return <TradingErrorFallback error={this.state.error} />
    }
    
    return this.props.children
  }
}
```

#### 6.2 Loading States & Skeletons
```typescript
// Skeleton components for loading states
export function OrderBookSkeleton() {
  return (
    <div className="space-y-2">
      {Array.from({ length: 10 }).map((_, i) => (
        <div key={i} className="flex justify-between">
          <Skeleton className="h-4 w-20" />
          <Skeleton className="h-4 w-16" />
        </div>
      ))}
    </div>
  )
}

// Loading state hook
export function useAsyncData<T>(
  fetchFn: () => Promise<T>,
  deps: any[] = []
) {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)
  
  useEffect(() => {
    let cancelled = false
    
    const fetchData = async () => {
      try {
        setLoading(true)
        setError(null)
        const result = await fetchFn()
        if (!cancelled) {
          setData(result)
        }
      } catch (err) {
        if (!cancelled) {
          setError(err as Error)
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }
    
    fetchData()
    
    return () => {
      cancelled = true
    }
  }, deps)
  
  return { data, loading, error }
}
```

### 7. Testing Standards

#### 7.1 Component Testing
```typescript
// Component test example
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { TradingOrderForm } from './TradingOrderForm'

describe('TradingOrderForm', () => {
  it('should submit order with correct data', async () => {
    const mockOnSubmit = jest.fn()
    
    render(
      <TradingOrderForm 
        symbol="BTC-USD" 
        onSubmit={mockOnSubmit} 
      />
    )
    
    // Fill form
    fireEvent.change(screen.getByLabelText('Quantity'), {
      target: { value: '0.1' }
    })
    
    fireEvent.change(screen.getByLabelText('Price'), {
      target: { value: '50000' }
    })
    
    // Submit
    fireEvent.click(screen.getByText('Place Order'))
    
    await waitFor(() => {
      expect(mockOnSubmit).toHaveBeenCalledWith({
        symbol: 'BTC-USD',
        quantity: 0.1,
        price: 50000,
        side: 'buy'
      })
    })
  })
})
```

#### 7.2 Integration Testing
```typescript
// API integration test
import { rest } from 'msw'
import { setupServer } from 'msw/node'

const server = setupServer(
  rest.post('/api/trading/orders', (req, res, ctx) => {
    return res(ctx.json({ 
      success: true, 
      orderId: 'order_123' 
    }))
  })
)

beforeAll(() => server.listen())
afterEach(() => server.resetHandlers())
afterAll(() => server.close())
```

### 8. Accessibility Standards

#### 8.1 ARIA Labels & Semantic HTML
```typescript
// Accessible trading components
export function AccessibleOrderBook({ orders }: { orders: Order[] }) {
  return (
    <div role="region" aria-label="Order Book">
      <table role="table" aria-label="Current orders">
        <thead>
          <tr role="row">
            <th role="columnheader" aria-sort="none">Price</th>
            <th role="columnheader" aria-sort="none">Quantity</th>
          </tr>
        </thead>
        <tbody>
          {orders.map((order, index) => (
            <tr 
              key={order.id} 
              role="row"
              aria-rowindex={index + 1}
              aria-selected={order.isSelected}
            >
              <td role="cell">{order.price}</td>
              <td role="cell">{order.quantity}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
```

#### 8.2 Keyboard Navigation
```typescript
// Keyboard navigation support
export function KeyboardNavigableChart() {
  const handleKeyDown = (event: KeyboardEvent) => {
    switch (event.key) {
      case 'ArrowLeft':
        moveToPreviousDataPoint()
        break
      case 'ArrowRight':
        moveToNextDataPoint()
        break
      case 'Enter':
        selectCurrentDataPoint()
        break
    }
  }
  
  return (
    <div 
      tabIndex={0}
      onKeyDown={handleKeyDown}
      role="application"
      aria-label="Trading Chart"
    >
      {/* Chart content */}
    </div>
  )
}
```

### 9. Security Best Practices

#### 9.1 Input Validation & Sanitization
```typescript
// Input validation for trading forms
import { z } from 'zod'

const OrderSchema = z.object({
  symbol: z.string().min(1, 'Symbol is required'),
  quantity: z.number().positive('Quantity must be positive'),
  price: z.number().positive('Price must be positive'),
  side: z.enum(['buy', 'sell'])
})

export function validateOrderInput(data: unknown) {
  try {
    return OrderSchema.parse(data)
  } catch (error) {
    throw new ValidationError('Invalid order data', error)
  }
}
```

#### 9.2 Secure API Communication
```typescript
// Secure API client with error handling
class SecureAPIClient {
  private async makeRequest<T>(
    endpoint: string, 
    options: RequestInit = {}
  ): Promise<T> {
    const headers = {
      'Content-Type': 'application/json',
      ...options.headers
    }
    
    const response = await fetch(`${this.baseURL}${endpoint}`, {
      ...options,
      headers,
      credentials: 'include' // Include cookies for session
    })
    
    if (!response.ok) {
      throw new APIError(`Request failed: ${response.status}`)
    }
    
    return response.json()
  }
}
```

### 10. Deployment & Build Configuration

#### 10.1 Next.js Configuration
```javascript
// next.config.js
/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    appDir: true
  },
  images: {
    domains: ['example.com']
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${process.env.BACKEND_URL}/api/:path*`
      }
    ]
  },
  webpack: (config, { isServer }) => {
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        net: false,
        tls: false
      }
    }
    return config
  }
}

module.exports = nextConfig
```

#### 10.2 Environment Configuration
```typescript
// Environment validation
const envSchema = z.object({
  NEXT_PUBLIC_API_URL: z.string().url(),
  NEXT_PUBLIC_WS_URL: z.string().url(),
  DATABASE_URL: z.string().url(),
  REDIS_URL: z.string().url().optional()
})

export const env = envSchema.parse(process.env)
```

---

**Document Version:** 1.0  
**Last Updated:** December 2025  
**Standards Compliance:** React 18, Next.js 15, TypeScript 5.0  
**Accessibility:** WCAG 2.1 AA Compliant