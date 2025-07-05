import { useEffect, useState } from 'react';
import { Table, Card, Input, Select, Button, Space, Alert, Tag } from 'antd';
import { SearchOutlined, ReloadOutlined } from '@ant-design/icons';
import { fetchLogs, fetchAlertThresholds } from '../services/api';
import { InferenceLog, AlertThreshold } from '../types';
import moment from 'moment';

const { Option } = Select;

const LogsTable = () => {
  const [logs, setLogs] = useState<InferenceLog[]>([]);
  const [thresholds, setThresholds] = useState<AlertThreshold[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pagination, setPagination] = useState({
    current: 1,
    pageSize: 10,
    total: 0,
  });
  const [filters, setFilters] = useState({
    model_name: undefined as string | undefined,
    output_class: undefined as string | undefined,
  });

  const fetchData = async (page = 1, pageSize = 10) => {
    try {
      setLoading(true);
      const skip = (page - 1) * pageSize;
      const data = await fetchLogs({
        ...filters,
        skip,
        limit: pageSize,
      });
      setLogs(data);
      setPagination({
        ...pagination,
        current: page,
        total: data.length + skip, // This is an approximation since we don't have a count endpoint
      });
      setError(null);
    } catch (err) {
      console.error('Error fetching logs:', err);
      setError('Failed to load logs. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  // Fetch alert thresholds
  const fetchThresholds = async () => {
    try {
      const data = await fetchAlertThresholds({ is_active: true });
      setThresholds(data);
    } catch (err) {
      console.error('Error fetching alert thresholds:', err);
    }
  };

  useEffect(() => {
    fetchData(pagination.current, pagination.pageSize);
    fetchThresholds();

    // Refresh data every 30 seconds
    const intervalId = setInterval(() => {
      fetchData(pagination.current, pagination.pageSize);
      fetchThresholds();
    }, 30000);

    return () => clearInterval(intervalId);
  }, [filters]);

  const handleTableChange = (pagination: any) => {
    fetchData(pagination.current, pagination.pageSize);
  };

  const handleFilterChange = (key: string, value: string | undefined) => {
    setFilters({
      ...filters,
      [key]: value,
    });
    setPagination({
      ...pagination,
      current: 1, // Reset to first page when filters change
    });
  };

  const handleReset = () => {
    setFilters({
      model_name: undefined,
      output_class: undefined,
    });
    setPagination({
      ...pagination,
      current: 1,
    });
  };

  // Helper functions to check if values exceed thresholds
  const checkLatencyThreshold = (latency: number, model: string) => {
    // First check for model-specific threshold
    const modelSpecificThreshold = thresholds.find(t => 
      t.metric_name === 'latency_ms' && 
      t.model_name === model
    );

    if (modelSpecificThreshold && latency > modelSpecificThreshold.threshold_value) {
      return latency > modelSpecificThreshold.threshold_value * 1.5 ? 'error' : 'warning';
    }

    // Then check for global threshold
    const globalThreshold = thresholds.find(t => 
      t.metric_name === 'latency_ms' && 
      (t.model_name === null || t.model_name === '')
    );

    if (globalThreshold && latency > globalThreshold.threshold_value) {
      return latency > globalThreshold.threshold_value * 1.5 ? 'error' : 'warning';
    }

    return 'normal';
  };

  const checkConfidenceThreshold = (confidence: number, model: string) => {
    // First check for model-specific threshold
    const modelSpecificThreshold = thresholds.find(t => 
      t.metric_name === 'confidence' && 
      t.model_name === model
    );

    if (modelSpecificThreshold && confidence < modelSpecificThreshold.threshold_value) {
      return confidence < modelSpecificThreshold.threshold_value * 0.8 ? 'error' : 'warning';
    }

    // Then check for global threshold
    const globalThreshold = thresholds.find(t => 
      t.metric_name === 'confidence' && 
      (t.model_name === null || t.model_name === '')
    );

    if (globalThreshold && confidence < globalThreshold.threshold_value) {
      return confidence < globalThreshold.threshold_value * 0.8 ? 'error' : 'warning';
    }

    return 'normal';
  };

  // Determine row-level status based on all metrics
  const getRowStatus = (record: InferenceLog) => {
    const latencyStatus = checkLatencyThreshold(record.latency_ms, record.model_name);
    const confidenceStatus = checkConfidenceThreshold(record.confidence, record.model_name);

    // If any metric has error status, the row has error status
    if (latencyStatus === 'error' || confidenceStatus === 'error') {
      return 'error';
    }

    // If any metric has warning status, the row has warning status
    if (latencyStatus === 'warning' || confidenceStatus === 'warning') {
      return 'warning';
    }

    return 'normal';
  };

  const columns = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
    },
    {
      title: 'Model',
      dataIndex: 'model_name',
      key: 'model_name',
      render: (text: string) => <Tag color="blue">{text}</Tag>,
    },
    {
      title: 'Timestamp',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (text: string) => moment(text).format('YYYY-MM-DD HH:mm:ss'),
      sorter: (a: InferenceLog, b: InferenceLog) => 
        moment(a.timestamp).valueOf() - moment(b.timestamp).valueOf(),
    },
    {
      title: 'Input Shape',
      dataIndex: 'input_shape',
      key: 'input_shape',
      render: (shapes: number[]) => shapes.join(' × '),
    },
    {
      title: 'Latency (ms)',
      dataIndex: 'latency_ms',
      key: 'latency_ms',
      render: (latency: number, record: InferenceLog) => {
        const status = checkLatencyThreshold(latency, record.model_name);
        let color = '';

        switch (status) {
          case 'warning':
            color = '#faad14';
            break;
          case 'error':
            color = '#f5222d';
            break;
          default:
            color = 'inherit';
        }

        return <span style={{ color }}>{latency.toFixed(2)}</span>;
      },
      sorter: (a: InferenceLog, b: InferenceLog) => a.latency_ms - b.latency_ms,
    },
    {
      title: 'Confidence',
      dataIndex: 'confidence',
      key: 'confidence',
      render: (confidence: number, record: InferenceLog) => {
        const status = checkConfidenceThreshold(confidence, record.model_name);
        let color = '';

        switch (status) {
          case 'warning':
            color = '#faad14';
            break;
          case 'error':
            color = '#f5222d';
            break;
          default:
            color = 'inherit';
        }

        return <span style={{ color }}>{(confidence * 100).toFixed(2) + '%'}</span>;
      },
      sorter: (a: InferenceLog, b: InferenceLog) => a.confidence - b.confidence,
    },
    {
      title: 'Output Class',
      dataIndex: 'output_class',
      key: 'output_class',
      render: (text: string) => <Tag color="green">{text}</Tag>,
    },
  ];

  return (
    <div>
      <h1>Inference Logs</h1>

      {error && (
        <Alert
          message="Error"
          description={error}
          type="error"
          showIcon
          className="alert-banner"
        />
      )}

      <Card className="card-container">
        <Space style={{ marginBottom: 16 }}>
          <Input
            placeholder="Filter by model"
            value={filters.model_name}
            onChange={(e) => handleFilterChange('model_name', e.target.value || undefined)}
            style={{ width: 200 }}
            prefix={<SearchOutlined />}
            allowClear
          />
          <Input
            placeholder="Filter by class"
            value={filters.output_class}
            onChange={(e) => handleFilterChange('output_class', e.target.value || undefined)}
            style={{ width: 200 }}
            prefix={<SearchOutlined />}
            allowClear
          />
          <Button 
            icon={<ReloadOutlined />} 
            onClick={handleReset}
          >
            Reset
          </Button>
        </Space>

        <Table
          dataSource={logs}
          columns={columns}
          rowKey="id"
          pagination={pagination}
          loading={loading}
          onChange={handleTableChange}
          rowClassName={(record) => {
            const status = getRowStatus(record);
            return status === 'normal' ? '' : `row-${status}`;
          }}
          onRow={(record) => {
            const status = getRowStatus(record);
            return {
              style: {
                backgroundColor: status === 'warning' ? 'rgba(250, 173, 20, 0.1)' : 
                                status === 'error' ? 'rgba(245, 34, 45, 0.1)' : 
                                'inherit'
              }
            };
          }}
        />
      </Card>
    </div>
  );
};

export default LogsTable;
