import { motion } from 'framer-motion';
import { Sparkles, Database, Brain, Zap } from 'lucide-react';

export default function Header() {
  const techStack = [
    { icon: Brain, label: 'OpenAI GPT', color: 'text-emerald-400' },
    { icon: Database, label: 'Neo4j Graph', color: 'text-blue-400' },
    { icon: Zap, label: 'FastAPI', color: 'text-purple-400' },
    { icon: Sparkles, label: 'CLIP Vision', color: 'text-pink-400' },
  ];

  return (
    <header className="bg-gradient-to-r from-slate-800/50 via-slate-800/30 to-slate-800/50 backdrop-blur-xl border-b border-slate-700/50 px-6 py-5 shadow-lg">
      {/* Warning Banner */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="overflow-hidden mb-4"
      >
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-yellow-500/10 border border-yellow-500/20">
          <Sparkles className="w-4 h-4 text-yellow-400 animate-pulse" />
          <p className="text-xs text-yellow-300 font-medium">
            Running on Neo4j Free Tier & Public APIs — Performance may vary
          </p>
        </div>
      </motion.div>
      
      {/* Main Title */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <motion.h1
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="text-3xl md:text-4xl font-bold tracking-tight mb-2"
          >
            <span className="bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
              CineBot
            </span>
            <span className="text-slate-200 ml-3 text-2xl md:text-3xl font-medium">
              AI Movie Recommender
            </span>
          </motion.h1>
          
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="text-sm text-slate-400 max-w-2xl"
          >
            Powered by Multimodal RAG with Graph Database, LLMs, and Vision Models
          </motion.p>
        </div>

        {/* Tech Stack Badges */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
          className="flex gap-2 flex-wrap"
        >
          {techStack.map((tech, idx) => (
            <motion.div
              key={tech.label}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.4 + idx * 0.1 }}
              whileHover={{ scale: 1.05 }}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-slate-900/50 border border-slate-700/50 backdrop-blur-sm"
            >
              <tech.icon className={`w-3.5 h-3.5 ${tech.color}`} />
              <span className="text-xs font-medium text-slate-300">{tech.label}</span>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </header>
  );
}
