import { useState } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import ChatTab from './components/ChatTab';
import TopTenListsTab from './components/TopTenListsTab';
import ArticlesTab from './components/ArticlesTab';
import GraphBackground from './components/GraphBackground';
import ErrorBoundary from './components/ErrorBoundary';
import { ToastProvider } from './components/Toast';

function App() {
  const [activeTab, setActiveTab] = useState('chat');

  return (
    <ErrorBoundary>
      <ToastProvider>
        <div className="flex h-screen overflow-hidden bg-slate-900 relative">
          {/* Animated Neo4j Graph Background */}
          {/* <GraphBackground /> */}
          
          <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />
          
          <main className="flex-1 flex flex-col overflow-hidden relative z-20">
            <Header />
            
            <div className="flex-1 overflow-hidden">
              {activeTab === 'chat' && <ChatTab />}
              {activeTab === 'toplists' && <TopTenListsTab />}
              {activeTab === 'articles' && <ArticlesTab />}
            </div>
          </main>
        </div>
      </ToastProvider>
    </ErrorBoundary>
  );
}

export default App;
