import axios from 'axios';
import { InferenceLog, AlertThreshold, AnalyticsSummary } from '../types';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Logs API
export const fetchLogs = async (params?: { 
  model_name?: string; 
  output_class?: string;
  skip?: number;
  limit?: number;
}) => {
  const response = await api.get<InferenceLog[]>('/logs', { params });
  return response.data;
};

// Analytics API
export const fetchAnalytics = async (params?: {
  model_name?: string;
  time_range?: number;
}) => {
  const response = await api.get<AnalyticsSummary>('/analytics', { params });
  return response.data;
};

// Alert Thresholds API
export const fetchAlertThresholds = async (params?: {
  model_name?: string;
  metric_name?: string;
  is_active?: boolean;
}) => {
  const response = await api.get<AlertThreshold[]>('/alert-thresholds', { params });
  return response.data;
};

export const createAlertThreshold = async (threshold: Omit<AlertThreshold, 'id' | 'created_at' | 'updated_at'>) => {
  const response = await api.post<AlertThreshold>('/alert-thresholds', threshold);
  return response.data;
};

export const updateAlertThreshold = async (id: number, threshold: Partial<Omit<AlertThreshold, 'id' | 'created_at' | 'updated_at'>>) => {
  const response = await api.put<AlertThreshold>(`/alert-thresholds/${id}`, threshold);
  return response.data;
};

export const deleteAlertThreshold = async (id: number) => {
  const response = await api.delete(`/alert-thresholds/${id}`);
  return response.data;
};

export default api;