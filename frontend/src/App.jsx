import { useState } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import ChatTab from './components/ChatTab';
import TopTenListsTab from './components/TopTenListsTab';
import ArticlesTab from './components/ArticlesTab';
import GraphBackground from './components/GraphBackground';
import ErrorBoundary from './components/ErrorBoundary';
import { ToastProvider } from './components/Toast';
import { MessageCircle, ListOrdered, Newspaper } from 'lucide-react';

function App() {
  const [activeTab, setActiveTab] = useState('chat');
  const tabs = [
    { key: 'chat', label: 'Finder', icon: MessageCircle },
    { key: 'toplists', label: '10 Best', icon: ListOrdered },
    { key: 'articles', label: 'Articles', icon: Newspaper },
  ];

  return (
    <ErrorBoundary>
      <ToastProvider>
        <div className="flex h-dvh overflow-hidden bg-slate-900 relative">
          {/* Animated Neo4j Graph Background */}
          {/* <GraphBackground /> */}
          
          <div className="hidden lg:flex">
            <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />
          </div>
          
          <main className="flex-1 flex flex-col overflow-hidden relative z-20">
            <Header />
            
            <div className="flex-1 overflow-hidden">
              {activeTab === 'chat' && <ChatTab />}
              {activeTab === 'toplists' && <TopTenListsTab />}
              {activeTab === 'articles' && <ArticlesTab />}
            </div>

            <nav className="lg:hidden border-t border-slate-700/50 bg-slate-900/90 backdrop-blur-xl px-2 pt-2 pb-[calc(env(safe-area-inset-bottom)+0.5rem)]">
              <div className="grid grid-cols-3 gap-2">
                {tabs.map((tab) => {
                  const Icon = tab.icon;
                  const isActive = activeTab === tab.key;

                  return (
                    <button
                      key={tab.key}
                      onClick={() => setActiveTab(tab.key)}
                      className={`min-h-11 flex flex-col items-center justify-center gap-1 rounded-lg py-2 text-xs font-medium transition-all ${
                        isActive
                          ? 'bg-primary/20 text-primary border border-primary/30'
                          : 'text-slate-300 hover:bg-slate-800/60'
                      }`}
                    >
                      <Icon className="w-4 h-4" />
                      <span>{tab.label}</span>
                    </button>
                  );
                })}
              </div>
            </nav>
          </main>
        </div>
      </ToastProvider>
    </ErrorBoundary>
  );
}

export default App;
