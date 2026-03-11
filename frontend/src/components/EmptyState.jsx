import { motion } from 'framer-motion';
import { Film, Sparkles, Image, MessageCircle } from 'lucide-react';

export default function EmptyState() {
  const suggestions = [
    { icon: Film, text: '"Show me sci-fi movies like Inception"' },
    { icon: Sparkles, text: '"I want a feel-good comedy"' },
    { icon: MessageCircle, text: '"Recommend thriller movies from the 90s"' },
    { icon: Image, text: 'Upload a movie poster for visual similarity search' },
  ];

  return (
    <div className="flex-1 flex items-center justify-center p-8 relative overflow-hidden">
      {/* Ambient Background Effects */}
      <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-purple-600/5 to-pink-600/5 animate-pulse" style={{ animationDuration: '8s' }} />
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/10 rounded-full blur-3xl animate-pulse" style={{ animationDuration: '6s', animationDelay: '1s' }} />
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-600/10 rounded-full blur-3xl animate-pulse" style={{ animationDuration: '7s', animationDelay: '2s' }} />
      
      <div className="relative text-center max-w-3xl">
        {/* Branded Hero Icon */}
        <motion.div
          initial={{ scale: 0.5, opacity: 0, rotate: -10 }}
          animate={{ scale: 1, opacity: 1, rotate: 0 }}
          transition={{ 
            duration: 0.8, 
            type: "spring",
            stiffness: 100,
            damping: 10
          }}
          className="mb-10 inline-flex items-center justify-center w-32 h-32 bg-gradient-to-br from-primary/40 via-purple-500/40 to-pink-500/40 rounded-3xl shadow-2xl shadow-primary/30 border border-primary/30 backdrop-blur-xl relative group"
        >
          <div className="absolute inset-0 bg-gradient-to-br from-primary/20 to-purple-600/20 rounded-3xl blur-xl group-hover:blur-2xl transition-all duration-500" />
          <Film className="w-16 h-16 text-white relative z-10" strokeWidth={1.5} />
        </motion.div>
        
        {/* Branded Title */}
        <motion.div
          initial={{ y: 30, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.2, duration: 0.6 }}
          className="mb-6"
        >
          <h1 className="text-5xl md:text-6xl font-black mb-3 tracking-tight">
            <span className="bg-gradient-to-r from-primary via-purple-500 to-pink-500 bg-clip-text text-transparent drop-shadow-2xl">
              CineBot
            </span>
          </h1>
          <p className="text-lg md:text-xl text-slate-300 font-medium tracking-wide">
            Your AI-Powered Movie Companion
          </p>
        </motion.div>
        
        {/* Tagline */}
        <motion.p
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.35, duration: 0.6 }}
          className="text-slate-400 mb-10 text-lg md:text-xl leading-relaxed max-w-2xl mx-auto"
        >
          Discover your next favorite film with intelligent recommendations powered by advanced AI and a database of <span className="text-primary font-semibold">4,799 movies</span>.
        </motion.p>
        
        {/* Feature Highlights */}
        <motion.div
          initial={{ y: 30, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.45, duration: 0.6 }}
          className="mb-12 grid grid-cols-1 sm:grid-cols-3 gap-6 max-w-4xl mx-auto"
        >
          <div className="flex flex-col items-center p-5 bg-slate-800/20 backdrop-blur-sm rounded-2xl border border-slate-700/30 hover:border-primary/30 transition-all hover:scale-105">
            <div className="w-12 h-12 bg-gradient-to-br from-primary/30 to-purple-600/30 rounded-xl flex items-center justify-center mb-3 border border-primary/20">
              <svg className="w-6 h-6 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <h3 className="text-white font-semibold mb-1">Lightning Fast</h3>
            <p className="text-slate-400 text-sm text-center">Query 4,799 movies with cached responses</p>
          </div>
          
          <div className="flex flex-col items-center p-5 bg-slate-800/20 backdrop-blur-sm rounded-2xl border border-slate-700/30 hover:border-purple-500/30 transition-all hover:scale-105">
            <div className="w-12 h-12 bg-gradient-to-br from-purple-500/30 to-pink-500/30 rounded-xl flex items-center justify-center mb-3 border border-purple-500/20">
              <svg className="w-6 h-6 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
            </div>
            <h3 className="text-white font-semibold mb-1">Multimodal Search</h3>
            <p className="text-slate-400 text-sm text-center">Text queries or upload movie posters</p>
          </div>
          
          <div className="flex flex-col items-center p-5 bg-slate-800/20 backdrop-blur-sm rounded-2xl border border-slate-700/30 hover:border-pink-500/30 transition-all hover:scale-105">
            <div className="w-12 h-12 bg-gradient-to-br from-pink-500/30 to-red-500/30 rounded-xl flex items-center justify-center mb-3 border border-pink-500/20">
              <svg className="w-6 h-6 text-pink-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </div>
            <h3 className="text-white font-semibold mb-1">AI-Powered</h3>
            <p className="text-slate-400 text-sm text-center">Context-aware recommendations</p>
          </div>
        </motion.div>
        
        {/* Example Queries */}
        <motion.div
          initial={{ y: 30, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.55, duration: 0.6 }}
          className="mb-10"
        >
          <p className="text-slate-500 text-sm uppercase tracking-wider font-semibold mb-5">Try asking...</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-3xl mx-auto">
            {suggestions.map((suggestion, index) => (
              <motion.button
                key={index}
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: 0.6 + index * 0.1, duration: 0.4 }}
                className="group relative p-5 bg-gradient-to-br from-slate-800/40 to-slate-900/40 hover:from-slate-700/50 hover:to-slate-800/50 border border-slate-700/50 hover:border-primary/50 rounded-2xl transition-all hover:scale-105 hover:shadow-xl hover:shadow-primary/10 text-left overflow-hidden"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-primary/0 via-primary/5 to-primary/0 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                <div className="relative flex items-start gap-4">
                  <div className="flex-shrink-0 w-12 h-12 bg-gradient-to-br from-primary/30 to-purple-600/30 rounded-xl flex items-center justify-center border border-primary/20 group-hover:border-primary/40 transition-all group-hover:scale-110 group-hover:rotate-6">
                    <suggestion.icon className="w-6 h-6 text-primary" strokeWidth={2} />
                  </div>
                  <p className="text-slate-300 group-hover:text-white transition-colors flex-1 leading-relaxed text-base pt-2">
                    {suggestion.text}
                  </p>
                </div>
              </motion.button>
            ))}
          </div>
        </motion.div>
        
        {/* Keyboard Shortcut Hint */}
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.9, duration: 0.5 }}
          className="flex items-center justify-center gap-2 text-sm text-slate-500"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span>Pro tip: Press</span>
          <kbd className="px-2.5 py-1 bg-slate-800/50 rounded-lg border border-slate-700/50 text-slate-400 font-mono text-xs shadow-inner">
            Ctrl
          </kbd>
          <span>+</span>
          <kbd className="px-2.5 py-1 bg-slate-800/50 rounded-lg border border-slate-700/50 text-slate-400 font-mono text-xs shadow-inner">
            K
          </kbd>
          <span>to start searching instantly</span>
        </motion.div>
      </div>
    </div>
  );
}
