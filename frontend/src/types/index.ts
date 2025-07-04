export interface InferenceLog {
  id: number;
  model_name: string;
  timestamp: string;
  input_shape: number[];
  latency_ms: number;
  confidence: number;
  output_class: string;
}

export interface AlertThreshold {
  id: number;
  model_name: string | null;
  metric_name: 'latency_ms' | 'confidence';
  threshold_value: number;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface Alert {
  model_name: string;
  metric_name: string;
  threshold_value: number;
  actual_value: number;
  timestamp: string;
}

export interface AnalyticsSummary {
  avg_latency: number;
  avg_confidence: number;
  total_requests: number;
  requests_per_minute: number;
  model_distribution: Record<string, number>;
  class_distribution: Record<string, number>;
  latency_over_time: Array<{
    timestamp: string;
    avg_latency: number;
  }>;
  confidence_distribution: Array<{
    confidence_range: string;
    count: number;
  }>;
}