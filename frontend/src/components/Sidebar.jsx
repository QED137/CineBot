export default function Sidebar({ activeTab, setActiveTab }) {
  return (
    <aside className="w-80 bg-slate-800/50 backdrop-blur-sm border-r border-slate-700 flex flex-col">
      <div className="p-6 border-b border-slate-700">
        <h2 className="text-2xl font-bold bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
          CineBot
        </h2>
      </div>

      <nav className="p-4 space-y-2">
        <button
          onClick={() => setActiveTab('chat')}
          className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
            activeTab === 'chat'
              ? 'bg-primary text-white shadow-lg shadow-primary/50'
              : 'text-slate-300 hover:bg-slate-700/50'
          }`}
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
          </svg>
          <span className="font-medium">Movie Finder</span>
        </button>
        
        <div className="mt-4 p-4 bg-slate-700/30 rounded-lg">
          <h3 className="text-sm font-semibold mb-2 text-slate-300">Features</h3>
          <ul className="space-y-2 text-xs text-slate-400">
            <li className="flex items-center gap-2">
              <svg className="w-4 h-4 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
              Text-based queries
            </li>
            <li className="flex items-center gap-2">
              <svg className="w-4 h-4 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
              Poster image search
            </li>
            <li className="flex items-center gap-2">
              <svg className="w-4 h-4 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
              Conversation history
            </li>
            <li className="flex items-center gap-2">
              <svg className="w-4 h-4 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
              Follow-up questions
            </li>
          </ul>
        </div>
      </nav>

      <div className="mt-auto p-6 border-t border-slate-700">
        <div className="bg-gradient-to-br from-primary/20 to-secondary/20 rounded-lg p-4 border border-primary/30">
          <h3 className="font-semibold text-sm mb-2">Open to Work</h3>
          <p className="text-xs text-slate-300 mb-2">
            Looking for roles in AI & Backend Engineering
          </p>
          <a
            href="https://www.janmajay.de"
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-primary hover:text-secondary transition-colors underline"
          >
            janmajay.de
          </a>
        </div>
        <p className="text-center text-sm text-slate-400 mt-4">✍️ JANMAJAY KUMAR</p>
      </div>
    </aside>
  );
}
