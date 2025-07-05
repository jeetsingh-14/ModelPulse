import { useEffect, useState } from 'react';
import { Row, Col, Card, Statistic, Table, Alert, Spin } from 'antd';
import { ArrowUpOutlined, ArrowDownOutlined, ClockCircleOutlined } from '@ant-design/icons';
import { fetchAnalytics, fetchLogs, fetchAlertThresholds } from '../services/api';
import { InferenceLog, AnalyticsSummary, Alert as AlertType, AlertThreshold } from '../types';
import AlertBanner from '../components/AlertBanner';
import moment from 'moment';

const Dashboard = () => {
  const [analytics, setAnalytics] = useState<AnalyticsSummary | null>(null);
  const [recentLogs, setRecentLogs] = useState<InferenceLog[]>([]);
  const [alerts, setAlerts] = useState<AlertType[]>([]);
  const [thresholds, setThresholds] = useState<AlertThreshold[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const [analyticsData, logsResponse, alertThresholds] = await Promise.all([
          fetchAnalytics(),
          fetchLogs({ limit: 5 }),
          fetchAlertThresholds({ is_active: true })
        ]);

        setAnalytics(analyticsData);
        setThresholds(alertThresholds);

        // Handle logs and alerts from the response
        if (Array.isArray(logsResponse)) {
          // If the response is just an array of logs (old API format)
          setRecentLogs(logsResponse);
          setAlerts([]);
        } else if (logsResponse && typeof logsResponse === 'object') {
          // If the response is an object with logs and alerts (new API format)
          const { log, alerts } = logsResponse;
          if (log) {
            // Single log with alerts
            setRecentLogs([log]);
            setAlerts(alerts || []);
          } else if (Array.isArray(logsResponse)) {
            // Just logs
            setRecentLogs(logsResponse);
            setAlerts([]);
          }
        }

        setError(null);
      } catch (err) {
        console.error('Error fetching dashboard data:', err);
        setError('Failed to load dashboard data. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchData();

    // Refresh data every 30 seconds
    const intervalId = setInterval(fetchData, 30000);

    return () => clearInterval(intervalId);
  }, []);

  // Helper functions to check if values exceed thresholds
    const checkLatencyThreshold = (latency: number) => {
      const latencyThreshold = thresholds.find(t => 
        t.metric_name === 'latency_ms' && 
        (t.model_name === null || t.model_name === '')
      );

      if (latencyThreshold && latency > latencyThreshold.threshold_value) {
        return latency > latencyThreshold.threshold_value * 1.5 ? 'error' : 'warning';
      }
      return 'normal';
    };

    const checkConfidenceThreshold = (confidence: number) => {
      const confidenceThreshold = thresholds.find(t => 
        t.metric_name === 'confidence' && 
        (t.model_name === null || t.model_name === '')
      );

      if (confidenceThreshold && confidence < confidenceThreshold.threshold_value) {
        return confidence < confidenceThreshold.threshold_value * 0.8 ? 'error' : 'warning';
      }
      return 'normal';
    };

    const getStatusColor = (status: 'normal' | 'warning' | 'error') => {
      switch (status) {
        case 'warning':
          return '#faad14';
        case 'error':
          return '#f5222d';
        default:
          return '#3f8600';
      }
    };

  const columns = [
    {
      title: 'Model',
      dataIndex: 'model_name',
      key: 'model_name',
    },
    {
      title: 'Timestamp',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (text: string) => moment(text).format('YYYY-MM-DD HH:mm:ss'),
    },
    {
      title: 'Latency (ms)',
      dataIndex: 'latency_ms',
      key: 'latency_ms',
      render: (latency: number) => latency.toFixed(2),
    },
    {
      title: 'Confidence',
      dataIndex: 'confidence',
      key: 'confidence',
      render: (confidence: number) => (confidence * 100).toFixed(2) + '%',
    },
    {
      title: 'Output Class',
      dataIndex: 'output_class',
      key: 'output_class',
    },
  ];

  if (loading && !analytics) {
    return <Spin size="large" tip="Loading dashboard..." />;
  }

  return (
    <div>
      <h1>Dashboard</h1>

      {error && (
        <Alert
          message="Error"
          description={error}
          type="error"
          showIcon
          className="alert-banner"
        />
      )}

      <AlertBanner alerts={alerts} />

      <Row gutter={16} className="card-container">
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Average Latency"
              value={analytics?.avg_latency || 0}
              precision={2}
              valueStyle={{ color: getStatusColor(checkLatencyThreshold(analytics?.avg_latency || 0)) }}
              suffix="ms"
              prefix={<ClockCircleOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Average Confidence"
              value={(analytics?.avg_confidence || 0) * 100}
              precision={2}
              valueStyle={{ color: getStatusColor(checkConfidenceThreshold(analytics?.avg_confidence || 0)) }}
              suffix="%"
              prefix={<ArrowUpOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Total Requests"
              value={analytics?.total_requests || 0}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Requests Per Minute"
              value={analytics?.requests_per_minute || 0}
              precision={2}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
      </Row>

      <Card title="Recent Inference Logs" className="card-container">
        <Table
          dataSource={recentLogs}
          columns={columns}
          rowKey="id"
          pagination={false}
          loading={loading}
        />
      </Card>
    </div>
  );
};

export default Dashboard;
