import { useState } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import ChatTab from './components/ChatTab';
import PosterTab from './components/PosterTab';

function App() {
  const [activeTab, setActiveTab] = useState('chat');

  return (
    <div className="flex h-screen overflow-hidden bg-gradient-to-br from-darker to-dark">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />
      
      <main className="flex-1 flex flex-col overflow-hidden">
        <Header />
        
        <div className="flex-1 overflow-hidden">
          {activeTab === 'chat' ? <ChatTab /> : <PosterTab />}
        </div>
      </main>
    </div>
  );
}

export default App;
