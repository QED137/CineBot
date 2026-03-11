import { useEffect, useRef, useState } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { forceCollide, forceCenter } from 'd3-force';

export default function GraphBackground() {
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [dimensions, setDimensions] = useState({
    width: window.innerWidth,
    height: window.innerHeight
  });

  const graphRef = useRef(null);

  useEffect(() => {
    const fetchGraphData = async () => {
      try {
        const apiUrl = import.meta.env.VITE_API_URL
          ? `${import.meta.env.VITE_API_URL}/graph-data`
          : '/api/graph-data';

        const response = await fetch(apiUrl);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();

        // Initialize nodes around world center (0,0), not screen center
        const nodes = (data.nodes || []).map(node => ({
          ...node,
          x: (Math.random() - 0.5) * 300,
          y: (Math.random() - 0.5) * 300,
          vx: (Math.random() - 0.5) * 0.5,
          vy: (Math.random() - 0.5) * 0.5
        }));

        setGraphData({
          nodes,
          links: data.links || []
        });
      } catch (error) {
        console.error('Failed to load graph data:', error);
      }
    };

    fetchGraphData();

    const handleResize = () => {
      setDimensions({
        width: window.innerWidth,
        height: window.innerHeight
      });
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  useEffect(() => {
    if (!graphRef.current || graphData.nodes.length === 0) return;

    const graph = graphRef.current;

    // Forces
    graph.d3Force('charge').strength(-60);

    graph.d3Force('link')
      .distance(140)
      .strength(0.08);

    graph.d3Force(
      'collision',
      forceCollide()
        .radius(node => ((node.val || 4) * 3) + 6)
        .strength(0.9)
    );

    // Correct center force: world center is (0, 0)
    graph.d3Force('center', forceCenter(0, 0));

    // Keep camera centered on graph world center
    graph.centerAt(0, 0);

    // Reheating + drift loop
    const interval = setInterval(() => {
      graphData.nodes.forEach(node => {
        const drift = 0.25;

        node.vx = (node.vx || 0) + (Math.random() - 0.5) * drift;
        node.vy = (node.vy || 0) + (Math.random() - 0.5) * drift;

        // gentle floating tendency
        node.vy += Math.sin(Date.now() * 0.001 + node.x * 0.01) * 0.03;
        node.vx += Math.cos(Date.now() * 0.001 + node.y * 0.01) * 0.03;

        // clamp max speed
        const maxVel = 1.5;
        node.vx = Math.max(-maxVel, Math.min(maxVel, node.vx));
        node.vy = Math.max(-maxVel, Math.min(maxVel, node.vy));
      });

      // Important: keep the engine alive
      graph.d3ReheatSimulation();
    }, 120);

    return () => clearInterval(interval);
  }, [graphData]);

  const getNodeColor = (node) => {
    switch (node.type) {
      case 'movie':
        return '#FF1493';
      case 'genre':
        return '#9D4EDD';
      case 'person':
        return '#00D4FF';
      default:
        return '#64748b';
    }
  };

  const getLinkColor = () => 'rgba(157, 78, 221, 0.35)';

  const nodeCanvasObject = (node, ctx) => {
    if (
      !node ||
      typeof node.x !== 'number' ||
      typeof node.y !== 'number' ||
      !isFinite(node.x) ||
      !isFinite(node.y)
    ) {
      return;
    }

    const nodeSize = (node.val || 4) * 3;

    ctx.fillStyle = getNodeColor(node);
    ctx.beginPath();
    ctx.arc(node.x, node.y, nodeSize, 0, 2 * Math.PI);
    ctx.fill();
  };

  if (graphData.nodes.length === 0) return null;

  return (
    <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none">
      <div className="absolute inset-0 opacity-95">
        <ForceGraph2D
          ref={graphRef}
          graphData={graphData}
          width={dimensions.width}
          height={dimensions.height}
          backgroundColor="transparent"
          nodeCanvasObject={nodeCanvasObject}
          linkColor={getLinkColor}
          linkWidth={1.5}
          linkDirectionalParticles={2}
          linkDirectionalParticleWidth={1.5}
          linkDirectionalParticleSpeed={0.0025}
          enableNodeDrag={false}
          enableZoomInteraction={false}
          enablePanInteraction={false}
          cooldownTime={Infinity}
          d3AlphaDecay={0.002}
          d3VelocityDecay={0.82}
          warmupTicks={80}
          onEngineStop={() => graphRef.current?.d3ReheatSimulation()}
        />
      </div>

      <div className="absolute inset-0 bg-gradient-radial from-transparent via-transparent to-slate-900/40 pointer-events-none" />
    </div>
  );
}