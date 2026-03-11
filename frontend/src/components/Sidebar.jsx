import { motion } from 'framer-motion';
import { MessageCircle, CheckCircle, Image, Sparkles, ExternalLink, Film, Heart } from 'lucide-react';

export default function Sidebar({ activeTab, setActiveTab }) {
  const features = [
    { icon: MessageCircle, text: 'Text-based queries' },
    { icon: Image, text: 'Poster image search' },
    { icon: Film, text: 'Conversation history' },
    { icon: Sparkles, text: 'Follow-up questions' },
  ];

  return (
    <aside className="w-80 bg-gradient-to-b from-slate-800/60 to-slate-900/60 backdrop-blur-xl border-r border-slate-700/50 flex flex-col shadow-2xl relative z-20">
      {/* Logo Area */}
      <div className="p-6 border-b border-slate-700/50">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center gap-3"
        >
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/50">
            <Film className="w-6 h-6 text-white" />
          </div>
          <h2 className="text-2xl font-bold bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">
            CineBot
          </h2>
        </motion.div>
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.1 }}
          className="text-xs text-slate-400 mt-2"
        >
          Your AI Movie Companion
        </motion.p>
      </div>

      {/* Navigation */}
      <nav className="p-4 space-y-2">
        <motion.button
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          whileHover={{ x: 4 }}
          whileTap={{ scale: 0.98 }}
          onClick={() => setActiveTab('chat')}
          className={`w-full flex items-center justify-between gap-3 px-4 py-3.5 rounded-xl transition-all group ${
            activeTab === 'chat'
              ? 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white shadow-lg shadow-indigo-500/30'
              : 'text-slate-300 hover:bg-slate-700/50 hover:text-white'
          }`}
        >
          <div className="flex items-center gap-3">
            <MessageCircle className="w-5 h-5" />
            <span className="font-medium">Movie Finder</span>
          </div>
          <span className="text-xs opacity-60">Ctrl+K</span>
        </motion.button>

        <motion.button
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.25 }}
          whileHover={{ x: 4 }}
          whileTap={{ scale: 0.98 }}
          onClick={() => setActiveTab('watchlist')}
          className={`w-full flex items-center justify-between gap-3 px-4 py-3.5 rounded-xl transition-all group ${
            activeTab === 'watchlist'
              ? 'bg-gradient-to-r from-pink-600 to-red-600 text-white shadow-lg shadow-pink-500/30'
              : 'text-slate-300 hover:bg-slate-700/50 hover:text-white'
          }`}
        >
          <div className="flex items-center gap-3">
            <Heart className={`w-5 h-5 ${activeTab === 'watchlist' ? 'fill-current' : ''}`} />
            <span className="font-medium">Watchlist</span>
          </div>
        </motion.button>
        
        {/* Features Box */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="mt-6 p-4 bg-gradient-to-br from-slate-800/50 to-slate-900/50 rounded-xl border border-slate-700/50 backdrop-blur-sm"
        >
          <h3 className="text-sm font-semibold mb-3 text-slate-200 flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-indigo-400" />
            Features
          </h3>
          <ul className="space-y-2.5">
            {features.map((feature, idx) => (
              <motion.li
                key={feature.text}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.4 + idx * 0.1 }}
                className="flex items-center gap-2.5 text-xs text-slate-400"
              >
                <div className="flex-shrink-0 w-5 h-5 rounded-full bg-indigo-500/20 flex items-center justify-center">
                  <feature.icon className="w-3 h-3 text-indigo-400" />
                </div>
                <span>{feature.text}</span>
              </motion.li>
            ))}
          </ul>
        </motion.div>
      </nav>

      {/* Footer */}
      <div className="mt-auto p-6 border-t border-slate-700/50">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="bg-gradient-to-br from-indigo-500/20 via-purple-500/20 to-pink-500/20 rounded-xl p-4 border border-indigo-500/30 backdrop-blur-sm"
        >
          <h3 className="font-semibold text-sm mb-2 text-white flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-indigo-400" />
            Open to Work
          </h3>
          <p className="text-xs text-slate-300 mb-3 leading-relaxed">
            Looking for roles in AI & Backend Engineering
          </p>
          <motion.a
            whileHover={{ x: 4 }}
            href="https://www.janmajay.de"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 text-xs text-indigo-400 hover:text-indigo-300 transition-colors font-medium group"
          >
            <span>janmajay.de</span>
            <ExternalLink className="w-3 h-3 group-hover:translate-x-0.5 transition-transform" />
          </motion.a>
        </motion.div>
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
          className="text-center text-sm text-slate-500 mt-4 font-medium"
        >
          JANMAJAY KUMAR
        </motion.p>
      </div>
    </aside>
  );
}
