import { useEffect, useState } from 'react';
import { Card, Row, Col, Select, Button, Alert, Spin, Typography } from 'antd';
import { ReloadOutlined } from '@ant-design/icons';
import { fetchAnalytics } from '../services/api';
import { AnalyticsSummary } from '../types';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, PieChart, Pie, Cell, Tooltip as RechartsTooltip
} from 'recharts';
import moment from 'moment';

const { Option } = Select;
const { Title } = Typography;

// Colors for charts
const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d', '#ffc658', '#8dd1e1'];

const Analytics = () => {
  const [analytics, setAnalytics] = useState<AnalyticsSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [modelFilter, setModelFilter] = useState<string | undefined>(undefined);
  const [timeRange, setTimeRange] = useState<number>(24); // Default to 24 hours

  const fetchData = async () => {
    try {
      setLoading(true);
      const data = await fetchAnalytics({
        model_name: modelFilter,
        time_range: timeRange,
      });
      setAnalytics(data);
      setError(null);
    } catch (err) {
      console.error('Error fetching analytics:', err);
      setError('Failed to load analytics data. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    
    // Refresh data every minute
    const intervalId = setInterval(fetchData, 60000);
    
    return () => clearInterval(intervalId);
  }, [modelFilter, timeRange]);

  // Prepare data for model distribution pie chart
  const modelDistributionData = analytics?.model_distribution
    ? Object.entries(analytics.model_distribution).map(([name, value]) => ({
        name,
        value,
      }))
    : [];

  // Prepare data for class distribution bar chart
  const classDistributionData = analytics?.class_distribution
    ? Object.entries(analytics.class_distribution).map(([name, value]) => ({
        name,
        value,
      }))
    : [];

  // Format timestamp for latency over time chart
  const latencyOverTimeData = analytics?.latency_over_time.map(item => ({
    ...item,
    timestamp: moment(item.timestamp).format('HH:mm'),
  })) || [];

  if (loading && !analytics) {
    return <Spin size="large" tip="Loading analytics..." />;
  }

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <Title level={2}>Analytics</Title>
        <div>
          <Select
            placeholder="Filter by model"
            allowClear
            style={{ width: 200, marginRight: 16 }}
            value={modelFilter}
            onChange={value => setModelFilter(value)}
          >
            {modelDistributionData.map(model => (
              <Option key={model.name} value={model.name}>{model.name}</Option>
            ))}
          </Select>
          <Select
            placeholder="Time range"
            style={{ width: 150, marginRight: 16 }}
            value={timeRange}
            onChange={value => setTimeRange(value)}
          >
            <Option value={1}>Last hour</Option>
            <Option value={6}>Last 6 hours</Option>
            <Option value={24}>Last 24 hours</Option>
            <Option value={72}>Last 3 days</Option>
            <Option value={168}>Last week</Option>
          </Select>
          <Button 
            icon={<ReloadOutlined />} 
            onClick={fetchData}
          >
            Refresh
          </Button>
        </div>
      </div>
      
      {error && (
        <Alert
          message="Error"
          description={error}
          type="error"
          showIcon
          className="alert-banner"
        />
      )}
      
      <Row gutter={16}>
        <Col xs={24} lg={12}>
          <Card title="Latency Over Time" className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={latencyOverTimeData}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" />
                <YAxis name="Latency (ms)" />
                <Tooltip />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="avg_latency" 
                  name="Avg Latency (ms)" 
                  stroke="#8884d8" 
                  activeDot={{ r: 8 }} 
                />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card title="Confidence Distribution" className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={analytics?.confidence_distribution || []}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="confidence_range" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="count" name="Count" fill="#82ca9d" />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </Col>
      </Row>
      
      <Row gutter={16} style={{ marginTop: 16 }}>
        <Col xs={24} lg={12}>
          <Card title="Model Distribution" className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={modelDistributionData}
                  cx="50%"
                  cy="50%"
                  labelLine={true}
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {modelDistributionData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <RechartsTooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card title="Class Distribution" className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={classDistributionData}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                layout="vertical"
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="name" />
                <Tooltip />
                <Legend />
                <Bar dataKey="value" name="Count" fill="#8884d8" />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Analytics;