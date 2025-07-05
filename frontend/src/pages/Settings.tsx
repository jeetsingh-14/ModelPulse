import { useEffect, useState } from 'react';
import { 
  Card, Table, Button, Form, Input, Select, InputNumber, Switch, 
  Space, Popconfirm, Alert, message, Typography, Tag 
} from 'antd';
import { 
  PlusOutlined, DeleteOutlined, EditOutlined, SaveOutlined, CloseOutlined 
} from '@ant-design/icons';
import { 
  fetchAlertThresholds, createAlertThreshold, updateAlertThreshold, deleteAlertThreshold 
} from '../services/api';
import { AlertThreshold } from '../types';

const { Option } = Select;
const { Title } = Typography;

interface EditingThreshold {
  id: number;
  model_name: string | null;
  metric_name: 'latency_ms' | 'confidence';
  threshold_value: number;
  is_active: boolean;
}

const Settings = () => {
  const [thresholds, setThresholds] = useState<AlertThreshold[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [editingKey, setEditingKey] = useState<number | 'new' | ''>('');
  const [form] = Form.useForm();

  const fetchData = async () => {
    try {
      setLoading(true);
      const data = await fetchAlertThresholds();
      setThresholds(data);
      setError(null);
    } catch (err) {
      console.error('Error fetching alert thresholds:', err);
      setError('Failed to load alert thresholds. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const isEditing = (record: AlertThreshold) => record.id === editingKey;

  const edit = (record: AlertThreshold) => {
    form.setFieldsValue({
      model_name: record.model_name,
      metric_name: record.metric_name,
      threshold_value: record.metric_name === 'confidence' ? record.threshold_value * 100 : record.threshold_value,
      is_active: record.is_active,
    });
    setEditingKey(record.id);
  };

  const cancel = () => {
    setEditingKey('');
  };

  const addNew = () => {
    form.resetFields();
    form.setFieldsValue({
      model_name: null,
      metric_name: 'latency_ms',
      threshold_value: 0,
      is_active: true,
    });
    setEditingKey('new');
  };

  const save = async (id: number | 'new') => {
    try {
      const row = await form.validateFields();
      
      // Convert confidence percentage back to decimal
      if (row.metric_name === 'confidence') {
        row.threshold_value = row.threshold_value / 100;
      }
      
      if (id === 'new') {
        // Create new threshold
        const newThreshold = await createAlertThreshold(row);
        setThresholds([...thresholds, newThreshold]);
        message.success('Alert threshold created successfully');
      } else {
        // Update existing threshold
        const updatedThreshold = await updateAlertThreshold(id as number, row);
        setThresholds(thresholds.map(item => item.id === id ? updatedThreshold : item));
        message.success('Alert threshold updated successfully');
      }
      setEditingKey('');
    } catch (err) {
      console.error('Error saving alert threshold:', err);
      message.error('Failed to save alert threshold');
    }
  };

  const handleDelete = async (id: number) => {
    try {
      await deleteAlertThreshold(id);
      setThresholds(thresholds.filter(item => item.id !== id));
      message.success('Alert threshold deleted successfully');
    } catch (err) {
      console.error('Error deleting alert threshold:', err);
      message.error('Failed to delete alert threshold');
    }
  };

  const columns = [
    {
      title: 'Model',
      dataIndex: 'model_name',
      key: 'model_name',
      render: (_: any, record: AlertThreshold) => {
        const editable = isEditing(record);
        return editable ? (
          <Form.Item
            name="model_name"
            style={{ margin: 0 }}
            rules={[{ required: false, message: 'Leave empty for all models' }]}
          >
            <Input placeholder="Leave empty for all models" />
          </Form.Item>
        ) : (
          <span>{record.model_name || <Tag color="blue">All Models</Tag>}</span>
        );
      },
    },
    {
      title: 'Metric',
      dataIndex: 'metric_name',
      key: 'metric_name',
      render: (_: any, record: AlertThreshold) => {
        const editable = isEditing(record);
        return editable ? (
          <Form.Item
            name="metric_name"
            style={{ margin: 0 }}
            rules={[{ required: true, message: 'Please select a metric' }]}
          >
            <Select>
              <Option value="latency_ms">Latency (ms)</Option>
              <Option value="confidence">Confidence (%)</Option>
            </Select>
          </Form.Item>
        ) : (
          <Tag color={record.metric_name === 'latency_ms' ? 'orange' : 'green'}>
            {record.metric_name === 'latency_ms' ? 'Latency (ms)' : 'Confidence (%)'}
          </Tag>
        );
      },
    },
    {
      title: 'Threshold Value',
      dataIndex: 'threshold_value',
      key: 'threshold_value',
      render: (_: any, record: AlertThreshold) => {
        const editable = isEditing(record);
        const isConfidence = form.getFieldValue('metric_name') === 'confidence';
        
        return editable ? (
          <Form.Item
            name="threshold_value"
            style={{ margin: 0 }}
            rules={[{ required: true, message: 'Please enter a threshold value' }]}
          >
            <InputNumber 
              min={0} 
              step={isConfidence ? 1 : 10}
              formatter={isConfidence ? value => `${value}%` : undefined}
              parser={isConfidence ? value => value!.replace('%', '') : undefined}
            />
          </Form.Item>
        ) : (
          <span>
            {record.metric_name === 'confidence' 
              ? `${(record.threshold_value * 100).toFixed(2)}%` 
              : `${record.threshold_value.toFixed(2)} ms`}
          </span>
        );
      },
    },
    {
      title: 'Active',
      dataIndex: 'is_active',
      key: 'is_active',
      render: (_: any, record: AlertThreshold) => {
        const editable = isEditing(record);
        return editable ? (
          <Form.Item
            name="is_active"
            style={{ margin: 0 }}
            valuePropName="checked"
          >
            <Switch />
          </Form.Item>
        ) : (
          <Tag color={record.is_active ? 'green' : 'red'}>
            {record.is_active ? 'Active' : 'Inactive'}
          </Tag>
        );
      },
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: any, record: AlertThreshold) => {
        const editable = isEditing(record);
        return editable ? (
          <Space>
            <Button 
              type="primary" 
              icon={<SaveOutlined />} 
              onClick={() => save(record.id)}
            >
              Save
            </Button>
            <Button 
              icon={<CloseOutlined />} 
              onClick={cancel}
            >
              Cancel
            </Button>
          </Space>
        ) : (
          <Space>
            <Button 
              icon={<EditOutlined />} 
              onClick={() => edit(record)}
              disabled={editingKey !== ''}
            >
              Edit
            </Button>
            <Popconfirm
              title="Are you sure you want to delete this alert threshold?"
              onConfirm={() => handleDelete(record.id)}
              okText="Yes"
              cancelText="No"
            >
              <Button 
                danger 
                icon={<DeleteOutlined />}
                disabled={editingKey !== ''}
              >
                Delete
              </Button>
            </Popconfirm>
          </Space>
        );
      },
    },
  ];

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <Title level={2}>Alert Settings</Title>
        <Button 
          type="primary" 
          icon={<PlusOutlined />} 
          onClick={addNew}
          disabled={editingKey !== ''}
        >
          Add New Threshold
        </Button>
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
      
      <Card className="card-container">
        <Form form={form} component={false}>
          <Table
            dataSource={editingKey === 'new' ? [...thresholds, { id: 'new', created_at: '', updated_at: '' } as any] : thresholds}
            columns={columns}
            rowKey="id"
            loading={loading}
            pagination={false}
          />
        </Form>
      </Card>
      
      <Card title="About Alert Thresholds" className="card-container">
        <p>Configure alerts to be notified when model metrics exceed thresholds:</p>
        <ul>
          <li><strong>Latency Threshold</strong>: Alert when inference latency exceeds the specified value in milliseconds.</li>
          <li><strong>Confidence Threshold</strong>: Alert when model confidence falls below the specified percentage.</li>
        </ul>
        <p>Leave the model field empty to apply the threshold to all models.</p>
      </Card>
    </div>
  );
};

export default Settings;