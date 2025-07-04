import React from 'react';
import { Alert, Space } from 'antd';
import { Alert as AlertType } from '../types';
import moment from 'moment';

interface AlertBannerProps {
  alerts: AlertType[];
}

const AlertBanner: React.FC<AlertBannerProps> = ({ alerts }) => {
  if (!alerts || alerts.length === 0) {
    return null;
  }

  return (
    <Space direction="vertical" style={{ width: '100%' }} className="alert-banner">
      {alerts.map((alert, index) => {
        const isLatency = alert.metric_name === 'latency_ms';
        const message = isLatency
          ? `High Latency Alert: ${alert.model_name}`
          : `Low Confidence Alert: ${alert.model_name}`;
        
        const description = isLatency
          ? `Latency of ${alert.actual_value.toFixed(2)}ms exceeds threshold of ${alert.threshold_value.toFixed(2)}ms`
          : `Confidence of ${(alert.actual_value * 100).toFixed(2)}% is below threshold of ${(alert.threshold_value * 100).toFixed(2)}%`;
        
        return (
          <Alert
            key={index}
            message={message}
            description={
              <div>
                {description}
                <div style={{ marginTop: 8, fontSize: '0.9em', opacity: 0.8 }}>
                  {moment(alert.timestamp).format('YYYY-MM-DD HH:mm:ss')}
                </div>
              </div>
            }
            type={isLatency ? 'warning' : 'error'}
            showIcon
            closable
          />
        );
      })}
    </Space>
  );
};

export default AlertBanner;