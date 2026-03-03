import { useState } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import ChatTab from './components/ChatTab';
import ErrorBoundary from './components/ErrorBoundary';
import { ToastProvider } from './components/Toast';

function App() {
  const [activeTab, setActiveTab] = useState('chat');

  return (
    <ErrorBoundary>
      <ToastProvider>
        <div className="flex h-screen overflow-hidden bg-gradient-to-br from-darker to-dark">
          <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />
          
          <main className="flex-1 flex flex-col overflow-hidden">
            <Header />
            
            <div className="flex-1 overflow-hidden">
              <ChatTab />
            </div>
          </main>
        </div>
      </ToastProvider>
    </ErrorBoundary>
  );
}

export default App;
