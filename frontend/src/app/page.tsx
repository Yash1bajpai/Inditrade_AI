"use client";
import { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, TrendingUp, AlertTriangle, MessageSquare, X, Sparkles, Map as MapIcon, GripVertical, Maximize2 } from 'lucide-react';
import { LineChart, Line, ScatterChart, Scatter, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ZAxis } from 'recharts';
import { ComposableMap, Geographies, Geography, Sphere, Graticule } from 'react-simple-maps';
import { scaleLinear } from 'd3-scale';
import styles from './page.module.css';
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';
const geoUrl = "https://unpkg.com/world-atlas@2.0.2/countries-110m.json";
const MINTED_BRASS = "#C8A97E";
const CRIMSON_WAX = "#9E3E3E";
const NIGHT_SLATE = "#1A1C21";
const FADED_INK = "#4A4F5C";
const CARD_SURFACE = "#23262D";
export interface AnomalyRow {
  date: string;
  partner: string;
  commodity: string;
  reason: string;
  anomaly_score: number;
}
export interface NetworkNode {
  id: string;
  country_name: string;
  original_country: string;
  trade_volume: number;
  x: number;
  y: number;
  val: number;
}
const SkeletonLoader = () => (
  <motion.div
    initial={{ opacity: 0.3 }} animate={{ opacity: 0.8 }} transition={{ repeat: Infinity, duration: 1, repeatType: "mirror" }}
    style={{ height: '100%', width: '100%', backgroundColor: 'rgba(255,255,255,0.02)', borderRadius: '0px' }}
  />
);
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
const DrillDownModal = ({ country, originalCountry, onClose }: { country: string, originalCountry: string, onClose: () => void }) => {
  const [historyData, setHistoryData] = useState<{period: string, volume: number}[]>([]);
  const [domains, setDomains] = useState<{name: string, value: number}[]>([]);
  const [latestYear, setLatestYear] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    fetch(`${API_BASE}/network/history/${encodeURIComponent(originalCountry)}`)
      .then(res => res.json())
      .then(data => { 
        setHistoryData(data.history || []); 
        setDomains(data.domains || []);
        setLatestYear(data.latest_year || null);
        setLoading(false); 
      })
      .catch(err => { console.error(err); setLoading(false); });
  }, [originalCountry]);
  return (
    <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)' }}>
      <motion.div
        initial={{ opacity: 0, scale: 0.9, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.9, y: 20 }}
        style={{ backgroundColor: CARD_SURFACE, border: `1px solid ${MINTED_BRASS}`, padding: '2rem', width: '500px', boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)' }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1.5rem' }}>
          <h3 style={{ margin: 0, fontFamily: "'Playfair Display', serif", fontSize: '1.4rem', color: MINTED_BRASS }}>{country} - Historical Trade</h3>
          <button onClick={onClose} style={{ background: 'transparent', border: 'none', color: FADED_INK, cursor: 'pointer' }} aria-label="Close modal"><X size={20}/></button>
        </div>
        <div style={{ height: '200px', marginBottom: '1.5rem' }}>
          {loading ? (
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', color: FADED_INK }}>Loading history...</div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={historyData} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="period" stroke={FADED_INK} fontSize={12} tickLine={false} axisLine={false} />
                <YAxis stroke={FADED_INK} fontSize={12} tickLine={false} axisLine={false} tickFormatter={(val) => `$${val}B`} />
                <Tooltip contentStyle={{ backgroundColor: NIGHT_SLATE, border: `1px solid ${MINTED_BRASS}`, borderRadius: '0px' }} itemStyle={{ color: MINTED_BRASS }} />
                <Line type="monotone" dataKey="volume" stroke={MINTED_BRASS} strokeWidth={3} dot={{ fill: CARD_SURFACE, stroke: MINTED_BRASS, strokeWidth: 2, r: 4 }} activeDot={{ r: 6, fill: MINTED_BRASS }} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
        
        <div>
           <h4 style={{ color: MINTED_BRASS, marginBottom: '0.5rem', fontFamily: "'Playfair Display', serif", fontSize: '1.1rem' }}>Top Traded Domains ({latestYear || 'Recent'})</h4>
           <div style={{ display: 'grid', gap: '0.5rem', maxHeight: '150px', overflowY: 'auto', paddingRight: '0.5rem' }}>
             {loading ? <span style={{ color: FADED_INK, fontSize: '0.85rem' }}>Loading domains...</span> : domains.map((d, i) => (
                <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem', backgroundColor: NIGHT_SLATE, border: `1px solid ${FADED_INK}`, borderRadius: '4px' }}>
                  <span style={{ fontSize: '0.85rem', color: '#e2e8f0' }}>{d.name}</span>
                  <span style={{ color: MINTED_BRASS, fontWeight: 'bold', fontSize: '0.85rem' }}>${d.value.toFixed(2)}B</span>
                </div>
             ))}
             {domains.length === 0 && !loading && <span style={{ color: FADED_INK, fontSize: '0.85rem' }}>No domain data available.</span>}
           </div>
        </div>
      </motion.div>
    </div>
  );
};
export default function Dashboard() {
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [drawerWidth, setDrawerWidth] = useState(450);
  const [isDragging, setIsDragging] = useState(false);
  const [messages, setMessages] = useState<{role: string, content: string, source?: string, citation?: string}[]>([
    { role: 'ai', content: 'Hello! I am your AI Indian Trade Policy Assistant. Ask me anything about DGFT compliance, import/export policies, or tariff rates.' }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const [selectedCountry, setSelectedCountry] = useState<string | null>(null);
  const [isMapEnlarged, setIsMapEnlarged] = useState(false);
  const [usdInr, setUsdInr] = useState('83.50');
  const [crudePrice, setCrudePrice] = useState('80.00');
  const [forecastYear, setForecastYear] = useState('2025');
  const [partnerCode, setPartnerCode] = useState('156');
  const [commodityCode, setCommodityCode] = useState('27');

  const [featureImportances, setFeatureImportances] = useState<{feature: string, importance: number}[]>([]);
  const [chartData, setChartData] = useState<{year: string, value: number}[]>([
    { year: '2022', value: 453.2 },
    { year: '2023', value: 437.1 },
    { year: '2024', value: 442.8 },
  ]);
  const [isPredicting, setIsPredicting] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [anomalyChartData, setAnomalyChartData] = useState<AnomalyRow[]>([]);
  const [anomalyTableData, setAnomalyTableData] = useState<AnomalyRow[]>([]);
  const [isLoadingAnomalies, setIsLoadingAnomalies] = useState(true);
  const [networkData, setNetworkData] = useState<NetworkNode[]>([]);
  const [isLoadingNetwork, setIsLoadingNetwork] = useState(true);
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setMounted(true);
  }, []);

  useEffect(() => {
    const abortController = new AbortController();
    fetch(`${API_BASE}/anomaly/historical`, { signal: abortController.signal })
      .then(res => {
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        return res.json();
      })
      .then(data => { setAnomalyChartData(data.chart_data || []); setAnomalyTableData(data.table_data || []); setIsLoadingAnomalies(false); })
      .catch(err => { if (err.name !== 'AbortError') { console.error(err); setIsLoadingAnomalies(false); }});
    fetch(`${API_BASE}/network/`, { signal: abortController.signal })
      .then(res => {
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        return res.json();
      })
      .then(data => { setNetworkData(data.nodes || []); setIsLoadingNetwork(false); })
      .catch(err => { if (err.name !== 'AbortError') { console.error(err); setIsLoadingNetwork(false); }});
    fetch(`${API_BASE}/forecast/history`, { signal: abortController.signal })
      .then(res => {
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        return res.json();
      })
      .then(data => {
        if (data.history) {
          setChartData(data.history.map((item: {year: number, value: number}) => ({
            year: item.year.toString(),
            value: parseFloat(item.value.toFixed(1))
          })));
        }
      })
      .catch(err => { if (err.name !== 'AbortError') console.error(err); });
    return () => abortController.abort();
  }, []);
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
    } catch {
      setMessages(prev => [...prev, { role: 'ai', content: 'Failed to connect to the backend server.', source: 'Error' }]);
    } finally {
      setIsTyping(false);
    }
  };
  const handleAnomalyClick = (row: AnomalyRow) => {
    const prompt = `Analyze the trade anomaly for ${row.partner} in ${row.commodity} during ${row.date}. The system flagged: ${row.reason}. Severity score: ${row.anomaly_score}`;
    setIsChatOpen(true);
    setInput(prompt);
    handleChatSubmit(undefined, prompt);
  };
  const handlePredict = async () => {
    setIsPredicting(true);
    try {
      const parsedUsdInr = parseFloat(usdInr);
      const parsedCrude = parseFloat(crudePrice);
      const parsedYear = parseInt(forecastYear);
      if (isNaN(parsedUsdInr) || isNaN(parsedCrude) || isNaN(parsedYear)) {
        throw new Error("Invalid input values");
      }
      const res = await fetch(`${API_BASE}/forecast/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ usd_inr: parsedUsdInr, crude_price: parsedCrude, year: parsedYear, partner_code: partnerCode, commodity_code: commodityCode })
      });
      const data = await res.json();
      
      if (data.error) {
        alert("Forecast Error: " + data.error);
        setIsPredicting(false);
        return;
      }
      if (data.error) throw new Error(data.error);
      const newPrediction = data.forecasted_trade_value_usd / 1e9;

      if (data.feature_importance) setFeatureImportances(data.feature_importance);
      setChartData(prev => [
        ...prev.filter(d => !d.year.includes('Pred')),
        { year: `${parsedYear} (Pred)`, value: parseFloat(newPrediction.toFixed(1)) }
      ]);
    } catch (error) {
      console.error("Forecast failed:", error);
    } finally {
      setIsPredicting(false);
    }
  };
  const colorScale = scaleLinear<string>().domain([0, 50, 100]).range([NIGHT_SLATE, MINTED_BRASS, CRIMSON_WAX]);
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { staggerChildren: 0.15 } }
  };
  const itemVariants = {
    hidden: { opacity: 0, y: 30 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.6 } }
  };
  return (
    <main className={styles.container}>
      {}
      <AnimatePresence>
        {selectedCountry && <DrillDownModal country={selectedCountry} originalCountry={selectedCountry} onClose={() => setSelectedCountry(null)} />}
        
        {isMapEnlarged && (
          <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, zIndex: 90, display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: 'rgba(0,0,0,0.8)', backdropFilter: 'blur(4px)' }}>
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              style={{ backgroundColor: CARD_SURFACE, border: `1px solid ${MINTED_BRASS}`, padding: '2rem', width: '90%', height: '90%', boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)', display: 'flex', flexDirection: 'column' }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1.5rem' }}>
                <h3 style={{ margin: 0, fontFamily: "'Playfair Display', serif", fontSize: '1.4rem', color: MINTED_BRASS }}>Enlarged Global Trade Heatmap</h3>
                <button onClick={() => setIsMapEnlarged(false)} style={{ background: 'transparent', border: 'none', color: FADED_INK, cursor: 'pointer' }} aria-label="Close modal"><X size={24}/></button>
              </div>
              <div style={{ flex: 1, position: 'relative' }}>
                {mounted && (
                  <ComposableMap projection="geoMercator" projectionConfig={{ scale: 150 }} style={{ width: '100%', height: '100%' }}>
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
                              onClick={() => { if(nodeData) { setSelectedCountry(nodeData.country_name); setIsMapEnlarged(false); } }}
                              style={{ hover: { fill: MINTED_BRASS, outline: 'none', cursor: 'pointer' }, pressed: { outline: 'none' }, default: { outline: 'none' } }}
                            />
                          );
                        })
                      }
                    </Geographies>
                  </ComposableMap>
                )}
              </div>
            </motion.div>
          </div>
        )}
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
          {}
          <motion.section variants={itemVariants} className={`glass-panel ${styles.section}`}>
            <div className={styles.sectionHeader}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <TrendingUp size={20} color={MINTED_BRASS} />
                <h2 className={styles.sectionTitle}>XGBoost Bilateral Trade Forecaster</h2>
              </div>
              <div className={styles.badge} style={{ color: MINTED_BRASS, border: `1px solid ${MINTED_BRASS}`, padding: '4px 12px', fontSize: '0.8rem', backgroundColor: 'transparent' }}>R&sup2; = 0.992 (Log-Scale)</div>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
              <div className={styles.inputGroup}>
                <label className={styles.inputLabel}>Partner</label>
                <select value={partnerCode} onChange={(e) => setPartnerCode(e.target.value)} className={styles.chatInput}>
                  <option value="156">China</option>
                  <option value="842">USA</option>
                  <option value="784">United Arab Emirates</option>
                  <option value="682">Saudi Arabia</option>
                  <option value="643">Russian Federation</option>
                </select>
              </div>
              <div className={styles.inputGroup}>
                <label className={styles.inputLabel}>Commodity</label>
                <select value={commodityCode} onChange={(e) => setCommodityCode(e.target.value)} className={styles.chatInput}>
                  <option value="27">Mineral Fuels & Oils</option>
                  <option value="71">Precious Metals & Stones</option>
                  <option value="85">Electrical Machinery & Electronics</option>
                  <option value="84">Machinery & Mechanical Appliances</option>
                  <option value="29">Organic Chemicals</option>
                </select>
              </div>
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
                <h3 style={{ fontSize: '0.9rem', fontWeight: 500, marginBottom: '1rem', color: '#EFECE6', fontFamily: "'Playfair Display', serif" }}>Global Model Feature Importance</h3>
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
          {}
          <motion.section variants={itemVariants} className={`glass-panel ${styles.section}`}>
            <div className={styles.sectionHeader}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <AlertTriangle size={20} color={CRIMSON_WAX} />
                <h2 className={styles.sectionTitle}>Isolation Forest Anomalies</h2>
              </div>
            </div>
            <p style={{ fontSize: '0.85rem', color: FADED_INK, marginBottom: '2rem' }}>
              Click any flagged anomaly row to investigate with the AI Policy Assistant.
            </p>
            <div className={styles.chartContainer} style={{ height: '200px', marginBottom: '2rem' }}>
              {isLoadingAnomalies ? <SkeletonLoader /> : mounted && (
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 10, right: 10, bottom: 0, left: -20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                    <XAxis dataKey="date" stroke={FADED_INK} fontSize={10} tickLine={false} axisLine={false} />
                    <YAxis dataKey="value" stroke={FADED_INK} fontSize={10} tickLine={false} axisLine={false} tickFormatter={(val) => `$${(val / 1e9).toFixed(1)}B`} />
                    <Tooltip 
                      cursor={{ strokeDasharray: '3 3' }} 
                      contentStyle={{ backgroundColor: NIGHT_SLATE, border: `1px solid ${MINTED_BRASS}` }}
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
                    <Scatter name="Anomalies" data={anomalyChartData} fill={CRIMSON_WAX} />
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
                    <th style={{ padding: '1rem' }}>Severity</th>
                  </tr>
                </thead>
                <tbody>
                  {isLoadingAnomalies ? (
                    <tr><td colSpan={5} style={{ padding: '1rem' }}><SkeletonLoader /></td></tr>
                  ) : anomalyTableData.slice(0, 5).map((row, i) => (
                    <tr
                      key={i}
                      onClick={() => handleAnomalyClick(row)}
                      tabIndex={0}
                      onKeyDown={(e) => { if(e.key === 'Enter') handleAnomalyClick(row); }}
                      style={{ cursor: 'pointer', borderBottom: `1px solid rgba(255,255,255,0.05)`, transition: 'background-color 0.2s' }}
                      onFocus={(e) => e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.03)'}
                      onBlur={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
                      onMouseOver={(e) => e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.03)'}
                      onMouseOut={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
                    >
                      <td style={{ padding: '1rem' }}>{row.date}</td>
                      <td style={{ padding: '1rem' }}>{row.partner}</td>
                      <td style={{ padding: '1rem' }}>{row.commodity}</td>
                      <td style={{ padding: '1rem', color: CRIMSON_WAX }}>{row.reason}</td>
                      <td style={{ padding: '1rem', color: row.anomaly_score < -0.1 ? CRIMSON_WAX : MINTED_BRASS }}>
                        {row.anomaly_score ? row.anomaly_score.toFixed(3) : "N/A"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </motion.section>
          {}
          <motion.section variants={itemVariants} className={`glass-panel ${styles.section}`}>
            <div className={styles.sectionHeader}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <MapIcon size={20} color={MINTED_BRASS} />
                <h2 className={styles.sectionTitle}>Global Trade Heatmap & Network Embeddings</h2>
              </div>
              <button 
                onClick={() => setIsMapEnlarged(true)} 
                style={{ background: 'transparent', border: 'none', color: FADED_INK, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem' }}
              >
                <Maximize2 size={16} /> Enlarge Map
              </button>
            </div>
            <p style={{ fontSize: '0.85rem', color: FADED_INK, marginBottom: '1.5rem' }}>
              Choropleth mapping of node volumes and 2D PCA projection of Node2Vec random walks over the global trade graph.
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              {}
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
                {}
                <div style={{ position: 'absolute', bottom: '10px', left: '10px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  <span style={{ fontSize: '0.65rem', color: FADED_INK, textTransform: 'uppercase' }}>Flagged Anomaly Value</span>
                  <div style={{ width: '100px', height: '6px', background: `linear-gradient(to right, ${NIGHT_SLATE}, ${MINTED_BRASS}, ${CRIMSON_WAX})`, border: `1px solid ${FADED_INK}` }} />
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.6rem', color: FADED_INK }}>
                    <span>Low</span><span>High</span>
                  </div>
                </div>
              </div>
              {}
              <div className={styles.chartContainer} style={{ height: '300px', backgroundColor: 'rgba(0,0,0,0.2)', border: `1px solid ${FADED_INK}` }}>
                {isLoadingNetwork ? <SkeletonLoader /> : mounted && (
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                      <XAxis type="number" dataKey="x" hide />
                      <YAxis type="number" dataKey="y" hide />
                      <ZAxis type="number" dataKey="val" range={[50, 400]} />
                      <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: CARD_SURFACE, border: 'none', borderRadius: '4px' }} content={({ payload }) => {
                        if (payload && payload.length) {
                          const data = payload[0].payload;
                          return (
                            <div style={{ backgroundColor: CARD_SURFACE, padding: '10px', border: `1px solid ${MINTED_BRASS}`, fontSize: '0.85rem' }}>
                              <strong style={{ color: MINTED_BRASS }}>{data.country_name}</strong>
                              <br/>
                              Anomaly Value: ${(data.trade_volume / 1e9).toFixed(2)}B
                            </div>
                          );
                        }
                        return null;
                      }} />
                      <Scatter data={networkData} fill={MINTED_BRASS} fillOpacity={0.7} onClick={(e: { payload?: { country_name?: string } }) => setSelectedCountry(e.payload?.country_name || null)} style={{ cursor: 'pointer' }} />
                    </ScatterChart>
                  </ResponsiveContainer>
                )}
              </div>
            </div>
          </motion.section>
        </div>
      </motion.div>
      {}
      <button className={styles.fab} onClick={() => setIsChatOpen(true)}>
        <Sparkles size={20} />
        Ask AI
      </button>
      {}
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
              {}
              <div
                onMouseDown={() => setIsDragging(true)}
                style={{ width: '12px', cursor: 'ew-resize', backgroundColor: 'rgba(0,0,0,0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center', borderRight: `1px solid ${FADED_INK}` }}
              >
                <GripVertical size={12} color={FADED_INK} />
              </div>
              {}
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
                          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 2 }} style={{ marginTop: '1rem', paddingTop: '0.5rem', borderTop: `1px solid ${FADED_INK}`, fontSize: '0.75rem', color: MINTED_BRASS, display: 'flex', justifyContent: 'space-between' }}>
                            <span><strong>Source:</strong> {msg.citation}</span>
                            {msg.source && <span style={{ color: FADED_INK }}>{msg.source}</span>}
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
