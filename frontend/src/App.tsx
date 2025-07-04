import { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { Layout, Menu, Typography, theme } from 'antd';
import {
  DashboardOutlined,
  TableOutlined,
  BarChartOutlined,
  SettingOutlined,
} from '@ant-design/icons';

// Pages
import Dashboard from './pages/Dashboard';
import LogsTable from './pages/LogsTable';
import Analytics from './pages/Analytics';
import Settings from './pages/Settings';

const { Header, Sider, Content } = Layout;
const { Title } = Typography;

function App() {
  const [collapsed, setCollapsed] = useState(false);
  const { token } = theme.useToken();

  return (
    <Router>
      <Layout className="app-container" style={{ minHeight: '100vh' }}>
        <Header style={{ 
          background: token.colorBgContainer, 
          padding: '0 24px',
          display: 'flex',
          alignItems: 'center',
          boxShadow: '0 1px 4px rgba(0,21,41,.08)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <Title level={3} style={{ margin: 0, marginRight: '16px' }}>
              ModelPulse
            </Title>
            <span style={{ color: token.colorTextSecondary }}>Real-time ML Monitoring</span>
          </div>
        </Header>
        <Layout className="content-container">
          <Sider 
            collapsible 
            collapsed={collapsed} 
            onCollapse={(value) => setCollapsed(value)}
            width={200}
            style={{ background: token.colorBgContainer }}
          >
            <Menu
              mode="inline"
              defaultSelectedKeys={['dashboard']}
              style={{ height: '100%', borderRight: 0 }}
              items={[
                {
                  key: 'dashboard',
                  icon: <DashboardOutlined />,
                  label: <Link to="/">Dashboard</Link>,
                },
                {
                  key: 'logs',
                  icon: <TableOutlined />,
                  label: <Link to="/logs">Logs</Link>,
                },
                {
                  key: 'analytics',
                  icon: <BarChartOutlined />,
                  label: <Link to="/analytics">Analytics</Link>,
                },
                {
                  key: 'settings',
                  icon: <SettingOutlined />,
                  label: <Link to="/settings">Settings</Link>,
                },
              ]}
            />
          </Sider>
          <Content className="main-content">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/logs" element={<LogsTable />} />
              <Route path="/analytics" element={<Analytics />} />
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </Content>
        </Layout>
      </Layout>
    </Router>
  );
}

export default App;