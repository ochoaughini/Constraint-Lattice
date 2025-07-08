import React, { createContext, useState, useEffect, ReactNode, useRef } from 'react';

interface AffectiveState {
  name: string;
  value: number;
}

interface AffectiveContextType {
  data: AffectiveState[];
  loading: boolean;
  error: string | null;
}

const AffectiveContext = createContext<AffectiveContextType>({
  data: [],
  loading: true,
  error: null,
});

export const AffectiveProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [data, setData] = useState<AffectiveState[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttempt = useRef(0);
  const maxReconnectAttempts = 5;

  const connectWebSocket = () => {
    // REPLACE WITH YOUR ACTUAL WEBSOCKET ENDPOINT
    const ws = new WebSocket('wss://your-actual-production-websocket-endpoint');

    ws.onopen = () => {
      console.log('WebSocket connected');
      setError(null);
      setLoading(false);
      reconnectAttempt.current = 0; // Reset reconnect attempts
    };

    ws.onmessage = (event) => {
      try {
        const newData: AffectiveState[] = JSON.parse(event.data);
        setData(newData);
      } catch (err) {
        setError('Error parsing WebSocket data');
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setError('WebSocket connection error');
    };

    ws.onclose = (event) => {
      console.log('WebSocket closed:', event.reason);
      setLoading(false);
      
      if (reconnectAttempt.current < maxReconnectAttempts) {
        const delay = Math.min(1000 * 2 ** reconnectAttempt.current, 30000);
        reconnectAttempt.current++;
        
        setTimeout(() => {
          console.log(`Reconnecting... attempt ${reconnectAttempt.current}`);
          connectWebSocket();
        }, delay);
      } else {
        setError('Failed to establish WebSocket connection after multiple attempts');
      }
    };

    wsRef.current = ws;
  };

  useEffect(() => {
    connectWebSocket();
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  return (
    <AffectiveContext.Provider value={{ data, loading, error }}>
      {children}
    </AffectiveContext.Provider>
  );
};

export default AffectiveContext;
