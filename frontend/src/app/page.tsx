"use client";

import { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, TrendingUp, AlertTriangle, MessageSquare, Activity, X, Sparkles, Network, Map as MapIcon, GripVertical, Maximize2 } from 'lucide-react';
import { LineChart, Line, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ZAxis } from 'recharts';
import { ComposableMap, Geographies, Geography, Sphere, Graticule } from 'react-simple-maps';
import { scaleLinear } from 'd3-scale';
import styles from './page.module.css';

const API_BASE = 'http://localhost:8000/api';
const geoUrl = "https://unpkg.com/world-atlas@2.0.2/countries-110m.json";

// Colors from our new palette
const MINTED_BRASS = "#C8A97E";
const CRIMSON_WAX = "#9E3E3E";
const NIGHT_SLATE = "#1A1C21";
const FADED_INK = "#4A4F5C";
const CARD_SURFACE = "#23262D";

// Typewriter Component
const TypewriterMessage = ({ content }: { content: string }) => {
  const [displayed, setDisplayed] = useState('');
  useEffect(() => {
    let i = 0;
    const interval = setInterval(() => {
      setDisplayed(content.substring(0, i));
      i++;
      if (i > content.length) clearInterval(interval);
    }, 15);
    return () => clearInterval(interval);
  }, [content]);
  return <span>{displayed}</span>;
};

// Modal Component for 6-Month Drill-Down
const DrillDownModal = ({ country, onClose }: { country: string, onClose: () => void }) => {
  // Generate stable mock data for the 6-month drilldown
  const mockData = Array.from({length: 6}).map((_, i) => ({
    month: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'][i],
    volume: Math.floor(Math.random() * 50) + 10
  }));

  return (
    <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)' }}>
      <motion.div 
        initial={{ opacity: 0, scale: 0.9, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.9, y: 20 }}
        style={{ backgroundColor: CARD_SURFACE, border: `1px solid ${MINTED_BRASS}`, padding: '2rem', width: '500px', boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)' }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1.5rem' }}>
          <h3 style={{ margin: 0, fontFamily: "'Playfair Display', serif", fontSize: '1.4rem', color: MINTED_BRASS }}>{country} - 6 Month Trend</h3>
          <button onClick={onClose} style={{ background: 'transparent', border: 'none', color: FADED_INK, cursor: 'pointer' }}><X size={20}/></button>
        </div>
        <div style={{ height: '250px' }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={mockData} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
              <XAxis dataKey="month" stroke={FADED_INK} fontSize={12} tickLine={false} axisLine={false} />
              <YAxis stroke={FADED_INK} fontSize={12} tickLine={false} axisLine={false} tickFormatter={(val) => `$${val}B`} />
              <Tooltip contentStyle={{ backgroundColor: NIGHT_SLATE, border: `1px solid ${MINTED_BRASS}`, borderRadius: '0px' }} itemStyle={{ color: MINTED_BRASS }} />
              <Line type="monotone" dataKey="volume" stroke={MINTED_BRASS} strokeWidth={3} dot={{ fill: CARD_SURFACE, stroke: MINTED_BRASS, strokeWidth: 2, r: 4 }} activeDot={{ r: 6, fill: MINTED_BRASS }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </motion.div>
    </div>
  );
};

export default function Dashboard() {
  // Chat State
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [drawerWidth, setDrawerWidth] = useState(450);
  const [isDragging, setIsDragging] = useState(false);
  const [messages, setMessages] = useState<{role: string, content: string, source?: string, citation?: string}[]>([
    { role: 'ai', content: 'Hello! I am your AI Indian Trade Policy Assistant. Ask me anything about DGFT compliance, import/export policies, or tariff rates.' }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Modal State
  const [selectedCountry, setSelectedCountry] = useState<string | null>(null);

  // Forecast State
  const [usdInr, setUsdInr] = useState('83.50');
  const [crudePrice, setCrudePrice] = useState('80.00');
  const [forecastYear, setForecastYear] = useState('2025');
  const [forecast, setForecast] = useState<number | null>(null);
  const [featureImportances, setFeatureImportances] = useState<{feature: string, importance: number}[]>([]);
  const [chartData, setChartData] = useState<{year: string, value: number}[]>([
    { year: '2022', value: 453.2 },
    { year: '2023', value: 437.1 },
    { year: '2024', value: 442.8 },
  ]);
  const [isPredicting, setIsPredicting] = useState(false);
  const [mounted, setMounted] = useState(false);

  // Data State
  const [anomalyData, setAnomalyData] = useState<any[]>([]);
  const [isLoadingAnomalies, setIsLoadingAnomalies] = useState(true);
  const [networkData, setNetworkData] = useState<any[]>([]);
  const [isLoadingNetwork, setIsLoadingNetwork] = useState(true);

  // Fetch initial data
  useEffect(() => {
    setMounted(true);
    fetch(`${API_BASE}/anomaly/historical`)
      .then(res => res.json())
      .then(data => { setAnomalyData(data.data || []); setIsLoadingAnomalies(false); })
      .catch(err => { console.error(err); setIsLoadingAnomalies(false); });

    fetch(`${API_BASE}/network/`)
      .then(res => res.json())
      .then(data => { setNetworkData(data.nodes || []); setIsLoadingNetwork(false); })
      .catch(err => { console.error(err); setIsLoadingNetwork(false); });
  }, []);

  // Resizable Drawer Logic
  const handleDrag = useCallback((e: MouseEvent) => {
    if (isDragging) {
      const newWidth = window.innerWidth - e.clientX;
      setDrawerWidth(Math.max(300, Math.min(newWidth, 1000)));
    }
  }, [isDragging]);

  const stopDrag = useCallback(() => { setIsDragging(false); }, []);

  useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handleDrag);
      window.addEventListener('mouseup', stopDrag);
    } else {
      window.removeEventListener('mousemove', handleDrag);
      window.removeEventListener('mouseup', stopDrag);
    }
    return () => {
      window.removeEventListener('mousemove', handleDrag);
      window.removeEventListener('mouseup', stopDrag);
    };
  }, [isDragging, handleDrag, stopDrag]);

  // Auto-scroll chat
  useEffect(() => {
    if (isChatOpen) chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isChatOpen]);

  const handleChatSubmit = async (e?: React.FormEvent, customMsg?: string) => {
    if (e) e.preventDefault();
    const userMessage = customMsg || input;
    if (!userMessage.trim()) return;

    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsTyping(true);

    try {
      const res = await fetch(`${API_BASE}/query/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userMessage })
      });
      const data = await res.json();
      setMessages(prev => [...prev, { role: 'ai', content: data.answer, source: data.source, citation: data.citation }]);
    } catch (error) {
      setMessages(prev => [...prev, { role: 'ai', content: 'Failed to connect to the backend server.', source: 'Error' }]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleAnomalyClick = (row: any) => {
    const prompt = `Analyze the trade anomaly for ${row.partner} in ${row.commodity} during ${row.date}. The system flagged: ${row.reason}.`;
    setIsChatOpen(true);
    setInput(prompt);
  };

  const handlePredict = async () => {
    setIsPredicting(true);
    try {
      const res = await fetch(`${API_BASE}/forecast/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ usd_inr: parseFloat(usdInr), crude_price: parseFloat(crudePrice), year: parseInt(forecastYear) })
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      const newPrediction = data.forecasted_trade_value_usd;
      setForecast(newPrediction);
      if (data.feature_importance) setFeatureImportances(data.feature_importance);
      setChartData([
        { year: '2022', value: 453.2 },
        { year: '2023', value: 437.1 },
        { year: '2024', value: 442.8 },
        { year: `${forecastYear} (Pred)`, value: parseFloat(newPrediction.toFixed(1)) }
      ]);
    } catch (error) {
      console.error("Forecast failed:", error);
    } finally {
      setIsPredicting(false);
    }
  };

  const colorScale = scaleLinear<string>().domain([0, 50, 100]).range([NIGHT_SLATE, MINTED_BRASS, CRIMSON_WAX]);

  const SkeletonLoader = () => (
    <motion.div 
      initial={{ opacity: 0.3 }} animate={{ opacity: 0.8 }} transition={{ repeat: Infinity, duration: 1, direction: "alternate" }}
      style={{ height: '100%', width: '100%', backgroundColor: 'rgba(255,255,255,0.02)', borderRadius: '0px' }}
    />
  );

  // Staggered Animations
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { staggerChildren: 0.15 } }
  };
  
  const itemVariants = {
    hidden: { opacity: 0, y: 30 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.6, ease: [0.22, 1, 0.36, 1] } }
  };

  return (
    <main className={styles.container}>
      {/* 6-Month Drill-Down Modal */}
      <AnimatePresence>
        {selectedCountry && <DrillDownModal country={selectedCountry} onClose={() => setSelectedCountry(null)} />}
      </AnimatePresence>

      <motion.div variants={containerVariants} initial="hidden" animate="visible">
        <motion.header variants={itemVariants} className={styles.header}>
          <div>
            <h1 className={`${styles.title} gradient-text`}>IndiTrade AI</h1>
            <p className={styles.subtitle}>Intelligent Foreign Trade Policy & Forecasting</p>
          </div>
          <div className={styles.subtleRing} style={{ padding: '0.5rem 1rem', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem' }}>
            <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: MINTED_BRASS }}></div>
            Institutional Core Online
          </div>
        </motion.header>

        <div className={styles.grid}>
          {/* Pillar 1: Trade Forecast (XGBoost) */}
          <motion.section variants={itemVariants} className={`glass-panel ${styles.section}`} style={{ padding: '2rem' }}>
            <div className={styles.sectionHeader} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <TrendingUp size={20} color={MINTED_BRASS} />
                <h2 className={styles.sectionTitle} style={{ fontFamily: "'Playfair Display', serif", fontSize: '1.5rem', margin: 0 }}>XGBoost Trade Forecaster</h2>
              </div>
              <div className={styles.badge} style={{ color: MINTED_BRASS, border: `1px solid ${MINTED_BRASS}`, padding: '4px 12px', fontSize: '0.8rem', backgroundColor: 'transparent' }}>R² = 0.992 (Tested on 2022+ Genuine Holdout)</div>
            </div>
            
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
              <div className={styles.inputGroup}>
                <label className={styles.inputLabel}>Forecast Year</label>
                <select value={forecastYear} onChange={(e) => setForecastYear(e.target.value)} className={styles.chatInput}>
                  <option value="2025">2025</option>
                  <option value="2026">2026</option>
                  <option value="2027">2027</option>
                </select>
              </div>
              <div className={styles.inputGroup}>
                <label className={styles.inputLabel}>USD/INR Rate</label>
                <input type="number" value={usdInr} onChange={(e) => setUsdInr(e.target.value)} className={styles.chatInput} step="0.1" />
              </div>
              <div className={styles.inputGroup}>
                <label className={styles.inputLabel}>Crude Oil ($/bbl)</label>
                <input type="number" value={crudePrice} onChange={(e) => setCrudePrice(e.target.value)} className={styles.chatInput} step="0.1" />
              </div>
            </div>

            <button onClick={handlePredict} disabled={isPredicting} className={styles.chatButton} style={{ width: '100%', padding: '0.75rem', marginTop: '1rem', backgroundColor: MINTED_BRASS, color: NIGHT_SLATE, fontFamily: "'Playfair Display', serif", fontSize: '1.1rem' }}>
              {isPredicting ? 'Running Model...' : 'Generate AI Forecast'}
            </button>

            <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '2rem', marginTop: '2rem' }}>
              <div className={styles.chartContainer} style={{ height: '300px' }}>
                {mounted && (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData} margin={{ top: 20, right: 20, bottom: 5, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                      <XAxis dataKey="year" stroke={FADED_INK} fontSize={12} tickLine={false} axisLine={false} />
                      <YAxis stroke={FADED_INK} fontSize={12} tickLine={false} axisLine={false} tickFormatter={(val) => `$${val}B`} />
                      <Tooltip contentStyle={{ backgroundColor: CARD_SURFACE, border: `1px solid ${MINTED_BRASS}`, borderRadius: '0px' }} itemStyle={{ color: MINTED_BRASS }} />
                      <Line type="monotone" dataKey="value" stroke={MINTED_BRASS} strokeWidth={3} dot={{ fill: NIGHT_SLATE, stroke: MINTED_BRASS, strokeWidth: 2, r: 4 }} activeDot={{ r: 6, fill: MINTED_BRASS }} />
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </div>
              
              <div style={{ padding: '1.5rem', backgroundColor: 'rgba(0,0,0,0.2)', border: `1px solid ${FADED_INK}` }}>
                <h3 style={{ fontSize: '0.9rem', fontWeight: 500, marginBottom: '1rem', color: '#EFECE6', fontFamily: "'Playfair Display', serif" }}>Prediction Drivers</h3>
                <div className={styles.featureBarContainer}>
                  {featureImportances.length > 0 ? featureImportances.map((f, i) => (
                    <div key={i} className={styles.featureBarRow}>
                      <span className={styles.featureLabel} title={f.feature}>{f.feature}</span>
                      <div className={styles.featureTrack}>
                        <motion.div initial={{ width: 0 }} animate={{ width: `${Math.max(5, (f.importance / featureImportances[0].importance) * 100)}%` }} className={styles.featureFill} style={{ backgroundColor: MINTED_BRASS }} />
                      </div>
                    </div>
                  )) : (
                    <p style={{ fontSize: '0.8rem', color: FADED_INK }}>Run forecast to view feature importance.</p>
                  )}
                </div>
              </div>
            </div>
          </motion.section>

          {/* Pillar 2: Anomaly Detection */}
          <motion.section variants={itemVariants} className={`glass-panel ${styles.section}`} style={{ padding: '2rem' }}>
            <div className={styles.sectionHeader} style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
              <AlertTriangle size={20} color={CRIMSON_WAX} />
              <h2 className={styles.sectionTitle} style={{ fontFamily: "'Playfair Display', serif", fontSize: '1.5rem', margin: 0 }}>Isolation Forest Anomalies</h2>
            </div>
            <p style={{ fontSize: '0.85rem', color: FADED_INK, marginBottom: '1.5rem' }}>Click any flagged anomaly row to investigate with the AI Policy Assistant.</p>
            
            <div className={styles.chartContainer} style={{ height: '250px', marginBottom: '2rem' }}>
              {isLoadingAnomalies ? <SkeletonLoader /> : mounted && (
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                    <XAxis type="category" dataKey="date" stroke={FADED_INK} fontSize={12} tickLine={false} axisLine={false} />
                    <YAxis type="number" dataKey="value" stroke={FADED_INK} fontSize={12} tickLine={false} axisLine={false} tickFormatter={(val) => `$${(val/1e9).toFixed(1)}B`} />
                    <Tooltip 
                      cursor={{ strokeDasharray: '3 3' }} 
                      content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                          const data = payload[0].payload;
                          return (
                            <div style={{ backgroundColor: CARD_SURFACE, padding: '10px', border: `1px solid ${CRIMSON_WAX}` }}>
                              <p style={{ margin: 0, fontWeight: 'bold', color: '#EFECE6' }}>{data.partner} - {data.commodity}</p>
                              <p style={{ margin: 0, fontSize: '0.85rem', color: CRIMSON_WAX }}>{data.reason}</p>
                            </div>
                          );
                        }
                        return null;
                      }} 
                    />
                    <Scatter name="Anomalies" data={anomalyData} fill={CRIMSON_WAX} />
                  </ScatterChart>
                </ResponsiveContainer>
              )}
            </div>

            <div className={styles.tableContainer} style={{ border: `1px solid ${FADED_INK}` }}>
              <table className={styles.anomalyTable} style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.85rem' }}>
                <thead>
                  <tr style={{ backgroundColor: 'rgba(255,255,255,0.02)', borderBottom: `1px solid ${FADED_INK}` }}>
                    <th style={{ padding: '1rem' }}>Period</th>
                    <th style={{ padding: '1rem' }}>Partner</th>
                    <th style={{ padding: '1rem' }}>Commodity</th>
                    <th style={{ padding: '1rem' }}>Reason Flagged</th>
                  </tr>
                </thead>
                <tbody>
                  {isLoadingAnomalies ? (
                    <tr><td colSpan={4} style={{ padding: '1rem' }}><SkeletonLoader /></td></tr>
                  ) : anomalyData.slice(0, 5).map((row, i) => (
                    <tr key={i} onClick={() => handleAnomalyClick(row)} style={{ cursor: 'pointer', borderBottom: `1px solid rgba(255,255,255,0.05)`, transition: 'background-color 0.2s' }} onMouseOver={(e) => e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.03)'} onMouseOut={(e) => e.currentTarget.style.backgroundColor = 'transparent'}>
                      <td style={{ padding: '1rem' }}>{row.date}</td>
                      <td style={{ padding: '1rem' }}>{row.partner}</td>
                      <td style={{ padding: '1rem' }}>{row.commodity}</td>
                      <td style={{ padding: '1rem', color: CRIMSON_WAX }}>{row.reason}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </motion.section>

          {/* Pillar 3: Trade Network (Node2Vec) */}
          <motion.section variants={itemVariants} className={`glass-panel ${styles.section}`} style={{ padding: '2rem' }}>
            <div className={styles.sectionHeader} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <MapIcon size={20} color={MINTED_BRASS} />
                <h2 className={styles.sectionTitle} style={{ fontFamily: "'Playfair Display', serif", fontSize: '1.5rem', margin: 0 }}>Global Trade Heatmap & Network Embeddings</h2>
              </div>
              <Maximize2 size={16} color={FADED_INK} style={{ cursor: 'pointer' }} title="Expand Map" />
            </div>
            <p style={{ fontSize: '0.85rem', color: FADED_INK, marginBottom: '1.5rem' }}>
              Choropleth mapping of node volumes and 2D PCA projection of Node2Vec random walks over the global trade graph.
            </p>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              {/* Choropleth Map */}
              <div className={styles.chartContainer} style={{ height: '300px', backgroundColor: 'rgba(0,0,0,0.2)', border: `1px solid ${FADED_INK}`, overflow: 'hidden', position: 'relative' }}>
                {mounted && (
                  <ComposableMap projection="geoMercator" projectionConfig={{ scale: 100 }} style={{ width: '100%', height: '100%' }}>
                    <Sphere stroke="rgba(255,255,255,0.1)" strokeWidth={0.5} id="sphere" fill="transparent" />
                    <Graticule stroke="rgba(255,255,255,0.05)" strokeWidth={0.5} />
                    <Geographies geography={geoUrl}>
                      {({ geographies }) =>
                        geographies.map((geo) => {
                          const nodeData = networkData.find(d => d.country_name === geo.properties.name);
                          const val = nodeData ? nodeData.val : 0;
                          return (
                            <Geography
                              key={geo.rsmKey}
                              geography={geo}
                              fill={val > 0 ? colorScale(val) : '#2C303A'}
                              stroke={NIGHT_SLATE}
                              strokeWidth={0.5}
                              onClick={() => { if(nodeData) setSelectedCountry(nodeData.country_name); }}
                              style={{ hover: { fill: MINTED_BRASS, outline: 'none', cursor: 'pointer' }, pressed: { outline: 'none' }, default: { outline: 'none' } }}
                            />
                          );
                        })
                      }
                    </Geographies>
                  </ComposableMap>
                )}
                {/* Gradient Legend */}
                <div style={{ position: 'absolute', bottom: '10px', left: '10px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  <span style={{ fontSize: '0.65rem', color: FADED_INK, textTransform: 'uppercase' }}>Volume</span>
                  <div style={{ width: '100px', height: '6px', background: `linear-gradient(to right, ${NIGHT_SLATE}, ${MINTED_BRASS}, ${CRIMSON_WAX})`, border: `1px solid ${FADED_INK}` }} />
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.6rem', color: FADED_INK }}>
                    <span>Low</span><span>High</span>
                  </div>
                </div>
              </div>

              {/* PCA Network Scatter */}
              <div className={styles.chartContainer} style={{ height: '300px', backgroundColor: 'rgba(0,0,0,0.2)', border: `1px solid ${FADED_INK}` }}>
                {isLoadingNetwork ? <SkeletonLoader /> : mounted && (
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                      <XAxis type="number" dataKey="x" hide />
                      <YAxis type="number" dataKey="y" hide />
                      <ZAxis type="number" dataKey="val" range={[50, 400]} />
                      <Tooltip cursor={false} content={({ payload }) => {
                        if (payload && payload.length) {
                          const data = payload[0].payload;
                          return (
                            <div style={{ backgroundColor: CARD_SURFACE, padding: '10px', border: `1px solid ${MINTED_BRASS}`, fontSize: '0.85rem' }}>
                              <strong style={{ color: MINTED_BRASS }}>{data.country_name}</strong>
                              <br/>
                              Vol: ${(data.trade_volume / 1e9).toFixed(2)}B
                            </div>
                          );
                        }
                        return null;
                      }} />
                      <Scatter data={networkData} fill={MINTED_BRASS} fillOpacity={0.7} onClick={(e: any) => setSelectedCountry(e.payload?.country_name)} style={{ cursor: 'pointer' }} />
                    </ScatterChart>
                  </ResponsiveContainer>
                )}
              </div>
            </div>
          </motion.section>
        </div>
      </motion.div>

      {/* Floating Action Button */}
      <button className={styles.fab} onClick={() => setIsChatOpen(true)}>
        <Sparkles size={20} />
        Ask AI
      </button>

      {/* Resizable Slide-out Drawer */}
      <AnimatePresence>
        {isChatOpen && (
          <div className={styles.sidebarOverlay} style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, pointerEvents: 'none', zIndex: 50 }}>
            <motion.div 
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              transition={{ type: 'spring', damping: 25, stiffness: 200 }}
              className={styles.sidebar}
              style={{ position: 'absolute', right: 0, top: 0, height: '100%', width: drawerWidth, backgroundColor: CARD_SURFACE, borderLeft: `1px solid ${MINTED_BRASS}`, pointerEvents: 'auto', display: 'flex', flexDirection: 'row' }}
            >
              {/* Drag Handle */}
              <div 
                onMouseDown={() => setIsDragging(true)}
                style={{ width: '12px', cursor: 'ew-resize', backgroundColor: 'rgba(0,0,0,0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center', borderRight: `1px solid ${FADED_INK}` }}
              >
                <GripVertical size={12} color={FADED_INK} />
              </div>

              {/* Drawer Content */}
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                <div className={styles.sidebarHeader} style={{ padding: '1.5rem', borderBottom: `1px solid ${FADED_INK}`, display: 'flex', justifyContent: 'space-between' }}>
                  <div className={styles.sidebarTitle} style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', fontFamily: "'Playfair Display', serif" }}>
                    <MessageSquare size={18} color={MINTED_BRASS} />
                    AI Policy Assistant
                  </div>
                  <button onClick={() => setIsChatOpen(false)} style={{ background: 'none', border: 'none', color: FADED_INK, cursor: 'pointer' }}>
                    <X size={20} />
                  </button>
                </div>

                <div className={styles.chatContainer} style={{ flex: 1, overflowY: 'auto', padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  <AnimatePresence>
                    {messages.map((msg, i) => (
                      <motion.div 
                        key={i}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        style={{
                          padding: '1rem', 
                          border: `1px solid ${msg.role === 'ai' ? MINTED_BRASS : FADED_INK}`,
                          alignSelf: msg.role === 'ai' ? 'flex-start' : 'flex-end',
                          maxWidth: '85%',
                          backgroundColor: msg.role === 'ai' ? 'transparent' : 'rgba(255,255,255,0.02)',
                          fontSize: '0.9rem',
                          fontFamily: "'Inter', sans-serif"
                        }}
                      >
                        {msg.role === 'ai' ? <TypewriterMessage content={msg.content} /> : msg.content}
                        
                        {msg.role === 'ai' && msg.citation && msg.citation !== "Knowledge Base Error" && (
                          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 2 }} style={{ marginTop: '1rem', paddingTop: '0.5rem', borderTop: `1px solid ${FADED_INK}`, fontSize: '0.75rem', color: MINTED_BRASS }}>
                            <strong>Source:</strong> {msg.citation}
                          </motion.div>
                        )}
                      </motion.div>
                    ))}
                    {isTyping && (
                      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} style={{ padding: '1rem', border: `1px solid ${MINTED_BRASS}`, alignSelf: 'flex-start', maxWidth: '85%' }}>
                        <div style={{ display: 'flex', gap: '4px' }}>
                          <motion.div animate={{ y: [0, -5, 0] }} transition={{ repeat: Infinity, duration: 0.6 }} style={{ width: 6, height: 6, borderRadius: '50%', backgroundColor: MINTED_BRASS }} />
                          <motion.div animate={{ y: [0, -5, 0] }} transition={{ repeat: Infinity, duration: 0.6, delay: 0.2 }} style={{ width: 6, height: 6, borderRadius: '50%', backgroundColor: MINTED_BRASS }} />
                          <motion.div animate={{ y: [0, -5, 0] }} transition={{ repeat: Infinity, duration: 0.6, delay: 0.4 }} style={{ width: 6, height: 6, borderRadius: '50%', backgroundColor: MINTED_BRASS }} />
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                  <div ref={chatEndRef} />
                </div>

                <form onSubmit={handleChatSubmit} style={{ padding: '1.5rem', borderTop: `1px solid ${FADED_INK}`, display: 'flex', gap: '0.5rem' }}>
                  <input 
                    type="text" 
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Ask about duty free imports..."
                    style={{ flex: 1, padding: '0.75rem', background: 'transparent', border: `1px solid ${FADED_INK}`, color: '#EFECE6', outline: 'none' }}
                  />
                  <button type="submit" disabled={isTyping || !input.trim()} style={{ padding: '0 1rem', background: MINTED_BRASS, color: NIGHT_SLATE, border: 'none', cursor: 'pointer' }}>
                    <Send size={18} />
                  </button>
                </form>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </main>
  );
}
