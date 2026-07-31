"use client";
import { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, TrendingUp, AlertTriangle, MessageSquare, X, Sparkles, Map as MapIcon, GripVertical, Maximize2, Menu, Code, User, Mail } from 'lucide-react';
import { LineChart, Line, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ZAxis, ComposedChart, Bar } from 'recharts';
import { ComposableMap, Geographies, Geography, Sphere, Graticule } from 'react-simple-maps';
import { Tooltip as ReactTooltip } from "react-tooltip";
import { geoMercator } from 'd3-geo';
import CountUp from 'react-countup';
import styles from './page.module.css';

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.05, delayChildren: 0 }
  }
};

const itemVariants = {
  hidden: { opacity: 0, y: 15 },
  visible: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 120, damping: 14 } }
};
let API_BASE = (process.env.NEXT_PUBLIC_API_URL || 'https://inditrade-backend.onrender.com/api').replace(/\/$/, "");
if (!API_BASE.endsWith('/api')) {
  API_BASE += '/api';
}
const geoUrl = "/countries-110m.json";
const MINTED_BRASS = "#C8A97E";
const CRIMSON_WAX = "#9E3E3E";
const NIGHT_SLATE = "#1A1C21";
const FADED_INK = "#4A4F5C";
const CARD_SURFACE = "#23262D";

const proj = geoMercator().scale(150).translate([400, 225]);

declare global {
  interface Window {
    VanijyaChat?: {
      open: () => void;
      ask: (prompt: string) => void;
      mount: (el: HTMLElement | null, config: Record<string, unknown>) => void;
      destroy: () => void;
    };
  }
}

const CMD_MAP: Record<string, string> = {'01': 'Live Animals', '02': 'Meat', '03': 'Fish', '04': 'Dairy', '05': 'Animal Products', '06': 'Trees/Plants', '07': 'Vegetables', '08': 'Fruits/Nuts', '09': 'Coffee/Tea/Spices', '10': 'Cereals', '11': 'Milling Products', '12': 'Oil Seeds', '13': 'Gums/Resins', '14': 'Vegetable Plaiting', '15': 'Fats/Oils', '16': 'Prepared Meat/Fish', '17': 'Sugars', '18': 'Cocoa', '19': 'Cereal Preps', '20': 'Vegetable/Fruit Preps', '21': 'Misc Edibles', '22': 'Beverages', '23': 'Food Waste/Fodder', '24': 'Tobacco', '25': 'Salt/Earths/Stone', '26': 'Ores/Slag/Ash', '27': 'Mineral Fuels', '28': 'Inorganic Chemicals', '29': 'Organic Chemicals', '30': 'Pharmaceuticals', '31': 'Fertilizers', '32': 'Tanning/Dyes', '33': 'Essential Oils/Cosmetics', '34': 'Soap/Waxes', '35': 'Albuminoids', '36': 'Explosives', '37': 'Photographic Goods', '38': 'Misc Chemicals', '39': 'Plastics', '40': 'Rubber', '41': 'Raw Hides/Skins', '42': 'Leather Articles', '43': 'Furskins', '44': 'Wood', '45': 'Cork', '46': 'Straw/Esparto', '47': 'Wood Pulp', '48': 'Paper/Paperboard', '49': 'Printed Books', '50': 'Silk', '51': 'Wool', '52': 'Cotton', '53': 'Vegetable Textile Fibers', '54': 'Man-made Filaments', '55': 'Man-made Staple Fibers', '56': 'Wadding/Felt/Yarn', '57': 'Carpets', '58': 'Special Woven Fabrics', '59': 'Impregnated Fabrics', '60': 'Knitted Fabrics', '61': 'Knitted Apparel', '62': 'Non-knitted Apparel', '63': 'Other Textiles', '64': 'Footwear', '65': 'Headgear', '66': 'Umbrellas', '67': 'Prepared Feathers', '68': 'Stone/Plaster Articles', '69': 'Ceramics', '70': 'Glass', '71': 'Precious Stones/Metals', '72': 'Iron/Steel', '73': 'Articles of Iron/Steel', '74': 'Copper', '75': 'Nickel', '76': 'Aluminum', '78': 'Lead', '79': 'Zinc', '80': 'Tin', '81': 'Other Base Metals', '82': 'Tools/Cutlery', '83': 'Misc Base Metal Articles', '84': 'Nuclear Reactors/Boilers/Machinery', '85': 'Electrical Machinery', '86': 'Railway/Tramway', '87': 'Vehicles', '88': 'Aircraft/Spacecraft', '89': 'Ships/Boats', '90': 'Optical/Medical Instruments', '91': 'Clocks/Watches', '92': 'Musical Instruments', '93': 'Arms/Ammunition', '94': 'Furniture', '95': 'Toys/Sports', '96': 'Misc Manufactured', '97': 'Works of Art', '98': 'Special Classification', '99': 'Special Classification'};

const COUNTRY_COORDS: Record<string, [number, number]> = {
  '36': [133.7751, -25.2744], // Australia
  '56': [4.4699, 50.5039], // Belgium
  '156': [104.1954, 35.8617], // China
  '276': [10.4515, 51.1657], // Germany
  '344': [114.1694, 22.3193], // Hong Kong
  '360': [113.9213, -0.7893], // Indonesia
  '368': [43.6793, 33.2232], // Iraq
  '392': [138.2529, 36.2048], // Japan
  '410': [127.7669, 35.9078], // South Korea
  '458': [101.9758, 4.2105], // Malaysia
  '528': [5.2913, 52.1326], // Netherlands
  '643': [105.3188, 61.5240], // Russia
  '682': [45.0792, 23.8859], // Saudi Arabia
  '702': [103.8198, 1.3521], // Singapore
  '704': [108.2772, 14.0583], // Vietnam
  '784': [53.8478, 23.4241], // UAE
  '826': [-3.4359, 55.3781], // UK
  '842': [-95.7129, 37.0902], // USA
};
const INDIA_COORDS: [number, number] = [78.9629, 20.5937];

const drawCurve = (startCoords: [number, number], endCoords: [number, number], curvature: number = 0.2) => {
  const start = proj(startCoords);
  const end = proj(endCoords);
  if (!start || !end) return "";
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  const midX = (start[0] + end[0]) / 2;
  const midY = (start[1] + end[1]) / 2;
  const cx = midX - dy * curvature;
  const cy = midY + dx * curvature;
  return `M ${start[0]},${start[1]} Q ${cx},${cy} ${end[0]},${end[1]}`;
};

const formatMoney = (value: number | undefined | null) => {
  if (value === null || value === undefined || isNaN(value)) return "N/A";
  if (value >= 1) return `$${value.toFixed(2)}B`;
  if (value > 0 && value < 1) return `$${Math.round(value * 1000)}M`;
  if (value === 0) return "$0";
  return formatSigned(value); // fallback for any accidental negatives
};

const formatSigned = (value: number | undefined | null) => {
  if (value === null || value === undefined || isNaN(value)) return "N/A";
  const sign = value >= 0 ? '+' : '-';
  const absVal = Math.abs(value);
  if (absVal >= 1) return `${sign}$${absVal.toFixed(2)}B`;
  if (absVal > 0 && absVal < 1) return `${sign}$${Math.round(absVal * 1000)}M`;
  return "$0";
};

export interface AnomalyRow {
  date: string;
  partner: string;
  commodity: string;
  reason: string;
  reason_code: string;
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
const AnimatedMoney = ({ value }: { value: number | undefined | null }) => {
  if (value === null || value === undefined || isNaN(value)) return <span>N/A</span>;
  if (value === 0) return <span>$0</span>;
  
  const absVal = Math.abs(value);
  const isBillion = absVal >= 1;
  const displayVal = isBillion ? absVal : absVal * 1000;
  const suffix = isBillion ? 'B' : 'M';
  const prefix = value < 0 ? '-$' : '$';
  
  return (
    <CountUp 
      start={0} 
      end={displayVal} 
      duration={1} 
      decimals={isBillion ? 2 : 0} 
      prefix={prefix} 
      suffix={suffix} 
      useEasing={true}
    />
  );
};
const SkeletonLoader = () => (
  <motion.div
    initial={{ opacity: 0.3 }} animate={{ opacity: 0.8 }} transition={{ repeat: Infinity, duration: 1, repeatType: "mirror" }}
    style={{ height: '100%', width: '100%', backgroundColor: 'rgba(255,255,255,0.02)', borderRadius: '0px' }}
  />
);
// eslint-disable-next-line @typescript-eslint/no-unused-vars
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
  const [historyData, setHistoryData] = useState<{year: string, value_billions: number, import_billions?: number, export_billions?: number}[]>([]);
  const [domains, setDomains] = useState<{code: string, name: string, value_billions: number}[]>([]);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    if (!originalCountry) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setHistoryData([]);
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setDomains([]);
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setLoading(false);
      return;
    }
    setLoading(true);
    fetch(`${API_BASE}/forecast/country_series?partner_code=${encodeURIComponent(originalCountry)}`)
      .then(res => { if (!res.ok) throw new Error("API failed"); return res.json(); })
      .then(data => { 
        setHistoryData(data.yearly || []); 
        setDomains(data.top_commodities || []);
        setLoading(false); 
      })
      .catch(err => { console.error(err); setLoading(false); });
  }, [originalCountry]);
  return (
    <div onClick={onClose} style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)' }}>
      <motion.div
        initial={{ opacity: 0, scale: 0.9, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.9, y: 20 }}
        onClick={(e) => e.stopPropagation()} 
        style={{ backgroundColor: CARD_SURFACE, border: `1px solid ${MINTED_BRASS}`, padding: '2rem', width: 'clamp(300px, 90vw, 500px)', boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)' }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1.5rem', alignItems: 'center' }}>
          <div>
            <h3 style={{ margin: 0, fontFamily: "'Playfair Display', serif", fontSize: '1.4rem', color: MINTED_BRASS }}>{country} - Historical Trade</h3>
            <span style={{ fontSize: '0.75rem', color: FADED_INK, border: `1px solid rgba(255,255,255,0.1)`, padding: '2px 6px', borderRadius: '4px', marginTop: '4px', display: 'inline-block' }}>Yearly 2015-2024 · monthly view coming next update</span>
          </div>
          <button onClick={onClose} style={{ background: 'transparent', border: 'none', color: FADED_INK, cursor: 'pointer' }} aria-label="Close modal"><X size={20}/></button>
        </div>
        <div style={{ height: '200px', marginBottom: '1.5rem' }}>
          {loading ? (
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', color: FADED_INK }}>Loading history...</div>
          ) : historyData.length === 0 ? (
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', color: FADED_INK }}>
              {!originalCountry ? "No bilateral trade data for this country." : "No historical trade data available."}
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              {historyData[0].import_billions !== undefined ? (
                <ComposedChart data={historyData} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                  <XAxis dataKey="year" stroke={FADED_INK} fontSize={12} tickLine={false} axisLine={false} />
                  <YAxis stroke={FADED_INK} fontSize={12} tickLine={false} axisLine={false} tickFormatter={(val) => formatMoney(val)} />
                  <Tooltip contentStyle={{ backgroundColor: NIGHT_SLATE, border: `1px solid ${MINTED_BRASS}`, borderRadius: '0px' }} itemStyle={{ color: MINTED_BRASS }} formatter={(v: any, name: any) => [formatMoney(Number(v)), name]} />
                  <Bar dataKey="import_billions" name="Imports" fill={CRIMSON_WAX} barSize={20} />
                  <Bar dataKey="export_billions" name="Exports" fill="#4a90e2" barSize={20} />
                  <Line type="monotone" dataKey="value_billions" name="Total Trade" stroke={MINTED_BRASS} strokeWidth={3} dot={{ fill: CARD_SURFACE, stroke: MINTED_BRASS, strokeWidth: 2, r: 4 }} activeDot={{ r: 6, fill: MINTED_BRASS }} />
                </ComposedChart>
              ) : (
                <LineChart data={historyData} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                  <XAxis dataKey="year" stroke={FADED_INK} fontSize={12} tickLine={false} axisLine={false} />
                  <YAxis stroke={FADED_INK} fontSize={12} tickLine={false} axisLine={false} tickFormatter={(val) => formatMoney(val)} />
                  <Tooltip contentStyle={{ backgroundColor: NIGHT_SLATE, border: `1px solid ${MINTED_BRASS}`, borderRadius: '0px' }} itemStyle={{ color: MINTED_BRASS }} formatter={(v: any, name: any) => [formatMoney(Number(v)), name]} />
                  <Line type="monotone" dataKey="value_billions" stroke={MINTED_BRASS} strokeWidth={3} dot={{ fill: CARD_SURFACE, stroke: MINTED_BRASS, strokeWidth: 2, r: 4 }} activeDot={{ r: 6, fill: MINTED_BRASS }} />
                </LineChart>
              )}
            </ResponsiveContainer>
          )}
        </div>
        
        <div>
           <h4 style={{ color: MINTED_BRASS, marginBottom: '0.5rem', fontFamily: "'Playfair Display', serif", fontSize: '1.1rem' }}>Top Commodities</h4>
           <div style={{ display: 'grid', gap: '0.5rem', maxHeight: '150px', overflowY: 'auto', paddingRight: '0.5rem' }}>
             {loading ? <span style={{ color: FADED_INK, fontSize: '0.85rem' }}>Loading domains...</span> : domains.map((d, i) => (
                <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem', backgroundColor: NIGHT_SLATE, border: `1px solid ${FADED_INK}`, borderRadius: '4px' }}>
                  <span style={{ fontSize: '0.85rem', color: '#e2e8f0' }}>{d.name}</span>
                  <span style={{ color: MINTED_BRASS, fontWeight: 'bold', fontSize: '0.85rem' }}>{formatMoney(d.value_billions)}</span>
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
  const [flowMode, setFlowMode] = useState<'off' | 'exports' | 'imports'>('off');
  const [activeTab, setActiveTab] = useState<'dashboard' | 'anomalies' | 'network'>('dashboard');
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [topExports, setTopExports] = useState<any[]>([]);
  const [topImports, setTopImports] = useState<any[]>([]);

  const [selectedCountry, setSelectedCountry] = useState<{name: string, code: string} | null>(null);
    const [usdInr, setUsdInr] = useState('83.50');
  const [crudePrice, setCrudePrice] = useState('80.00');
  const [forecastYear, setForecastYear] = useState('2025');
  const [partnerCode, setPartnerCode] = useState('156');
  const [commodityCode, setCommodityCode] = useState('27');
  const [forecastError, setForecastError] = useState<string | null>(null);
  const [partnerList, setPartnerList] = useState<{code:string, name:string}[]>([]);
  const [validMap, setValidMap] = useState<Record<string, string[]>>({});
  const [suggestedCommodities, setSuggestedCommodities] = useState<any[]>([]);
  const [r2Metric, setR2Metric] = useState<number>(0.992);

  const [featureImportances, setFeatureImportances] = useState<{feature: string, importance: number}[]>([]);
  const [chartData, setChartData] = useState<{year: string, value: number}[]>([]);
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

  const getForecastHistory = async (partner: string, commodity: string, signal?: AbortSignal) => {
    try {
      const res = await fetch(`${API_BASE}/forecast/history?partner_code=${encodeURIComponent(partner)}&commodity_code=${encodeURIComponent(commodity)}`, { signal });
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      const data = await res.json();
      if (data.history) {
        return data.history.map((item: {year: string | number, value: number}) => ({
          year: String(item.year),
          value: Number(Number(item.value || 0).toFixed(3))
        }));
      }
      return [];
    } catch (err: unknown) {
      if (err instanceof Error && err.name !== 'AbortError') console.error(err);
      return [];
    }
  };

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
      .then(data => { 
        setNetworkData(data.nodes || []); 
        setTopExports(data.top_exports || []);
        setTopImports(data.top_imports || []);
        setIsLoadingNetwork(false); 
      })
      .catch(err => { if (err.name !== 'AbortError') { console.error(err); setIsLoadingNetwork(false); }});
    return () => abortController.abort();
  }, []);

  useEffect(() => {
    const abortController = new AbortController();
    getForecastHistory(partnerCode, commodityCode, abortController.signal).then(history => {
      setChartData(history);
    });
    return () => abortController.abort();
  }, [partnerCode, commodityCode]);
  const handleAnomalyClick = (row: AnomalyRow) => {
    const prompt = `Analyze the trade anomaly for ${row.partner} in ${row.commodity} during ${row.date}. The system flagged: ${row.reason_code === 'no_baseline' ? 'Insufficient baseline data (not a true anomaly)' : row.reason}. Severity score: ${row.anomaly_score}`;
    if (window.VanijyaChat) {
      window.VanijyaChat.open();
      window.VanijyaChat.ask(prompt);
    }
  };

  // Vanijya Chat Injection
  useEffect(() => {
    const cssId = 'vanijya-chat-css';
    const jsId = 'vanijya-chat-js';
    
    if (!document.getElementById(cssId)) {
      const link = document.createElement('link');
      link.id = cssId;
      link.rel = 'stylesheet';
      link.href = '/chat.css';
      document.head.appendChild(link);
    }
    if (!document.getElementById(jsId)) {
      const script = document.createElement('script');
      script.id = jsId;
      script.src = '/chat.js';
      script.onload = () => {
        if (window.VanijyaChat) {
          window.VanijyaChat.mount(document.getElementById('vanijya-chat-root'), {
            apiBase: API_BASE,
            chatEndpoint: '/query/'
          });
        }
      };
      document.body.appendChild(script);
    }
    
    return () => {
      if (window.VanijyaChat) {
        window.VanijyaChat.destroy();
      }
      const existingScript = document.getElementById(jsId);
      if (existingScript) existingScript.remove();
      const existingLink = document.getElementById(cssId);
      if (existingLink) existingLink.remove();
    };
  }, []);
  


  const predictAbortController = useRef<AbortController | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/forecast/valid_combinations`)
      .then(res => { if (!res.ok) throw new Error("API failed"); return res.json(); })
      .then(data => {
        if (data && data.partners) {
          setPartnerList(data.partners);
          setValidMap(data.map || {});
        }
      }).catch(console.error);
  }, []);

  const handlePartnerChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const p = e.target.value;
    setPartnerCode(p);
    setForecastError(null);
    setSuggestedCommodities([]);
    try {
      const res = await fetch(`${API_BASE}/forecast/partner_signature?partner_code=${p}`);
      if (!res.ok) throw new Error("Failed to load partner signature");
      const data = await res.json();
      if (data && data.length > 0) {
        setCommodityCode(data[0].code);
      } else if (validMap[p] && validMap[p].length > 0) {
        setCommodityCode(validMap[p][0]);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handlePredict = async () => {
    setIsPredicting(true);
    setForecastError(null);
    try {
      const parsedUsdInr = parseFloat(usdInr);
      const parsedCrude = parseFloat(crudePrice);
      const parsedYear = parseInt(forecastYear);
      if (isNaN(parsedUsdInr) || isNaN(parsedCrude) || isNaN(parsedYear)) {
        throw new Error("Invalid input values");
      }
      if (predictAbortController.current) {
        predictAbortController.current.abort();
      }
      const abortController = new AbortController();
      predictAbortController.current = abortController;

      const res = await fetch(`${API_BASE}/forecast/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ usd_inr: parsedUsdInr, crude_price: parsedCrude, year: parsedYear, partner_code: partnerCode, commodity_code: commodityCode }),
        signal: abortController.signal
      });
      const data = await res.json();
      
      if (!res.ok) {
        if (data?.suggested_commodities) {
          setSuggestedCommodities(data.suggested_commodities);
        }
        const errMsg = data?.error || data?.detail?.[0]?.msg || data?.detail || `HTTP error ${res.status}`;
        throw new Error(errMsg);
      }
      
      if (data.error) throw new Error(data.error);

      const usdVal = Number(data.forecasted_trade_value_usd);
      if (!Number.isFinite(usdVal)) {
        throw new Error("Model returned an invalid prediction");
      }

      const billions = usdVal / 1e9;
      let formattedBillions;
      if (billions >= 100) {
        formattedBillions = Number(billions.toFixed(1));
      } else if (billions >= 1) {
        formattedBillions = Number(billions.toFixed(2));
      } else {
        formattedBillions = Number(billions.toFixed(3));
      }

      if (Array.isArray(data.feature_importance)) {
        setFeatureImportances(data.feature_importance);
      }
      
      if (data.metrics && data.metrics.test_log_scale_r2) {
        setR2Metric(data.metrics.test_log_scale_r2);
      }
      
      const history = await getForecastHistory(partnerCode, commodityCode);
      setChartData([
        ...history,
        { year: `${parsedYear} (Pred)`, value: formattedBillions }
      ]);
    } catch (error: unknown) {
      if (error instanceof Error && error.name === 'AbortError') {
        console.log("Forecast request aborted");
        return;
      }
      console.error("Forecast failed:", error);
      setForecastError(error instanceof Error ? error.message : "Forecast failed");
    } finally {
      setIsPredicting(false);
    }
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { staggerChildren: 0.15 } }
  };
  const itemVariants = {
    hidden: { opacity: 0, y: 30 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.6 } }
  };
  return (
    <div className={styles.container}>
      <aside className={`${styles.sidebar} ${isSidebarOpen ? styles.open : ''}`}>
        <div className={styles.sidebarLogo}>
          IndiTrade AI
          <button className={styles.mobileCloseBtn} onClick={() => setIsSidebarOpen(false)} aria-label="Close menu">
            <X size={20} />
          </button>
        </div>
        <nav className={styles.sidebarNav}>
          <a onClick={(e) => { e.preventDefault(); setActiveTab('dashboard'); setIsSidebarOpen(false); }} className={`${styles.sidebarNavItem} ${activeTab === 'dashboard' ? styles.active : ''}`} aria-label="Navigate to Dashboard"><TrendingUp size={18} /> Dashboard</a>
          <a onClick={(e) => { e.preventDefault(); setActiveTab('anomalies'); setIsSidebarOpen(false); }} className={`${styles.sidebarNavItem} ${activeTab === 'anomalies' ? styles.active : ''}`} aria-label="Navigate to Anomalies"><AlertTriangle size={18} /> Anomalies</a>
          <a onClick={(e) => { e.preventDefault(); setActiveTab('network'); setIsSidebarOpen(false); }} className={`${styles.sidebarNavItem} ${activeTab === 'network' ? styles.active : ''}`} aria-label="Navigate to Trade Network"><MapIcon size={18} /> Trade Network</a>
          <button className={styles.sidebarNavItem} onClick={() => { setIsSidebarOpen(false); window.VanijyaChat?.open(); }} aria-label="Open AI Chatbot" style={{ background: 'transparent', border: 'none', color: 'inherit', font: 'inherit', width: '100%', textAlign: 'left', cursor: 'pointer' }}><MessageSquare size={18} /> Vanijya AI</button>
        </nav>
      </aside>

      {isSidebarOpen && <div className={styles.overlay} onClick={() => setIsSidebarOpen(false)} />}

      <div className={styles.mainWrapper}>
        <header className={styles.topHeader}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <button className={styles.menuToggle} onClick={() => setIsSidebarOpen(true)} aria-label="Open menu">
              <Menu size={24} color={MINTED_BRASS} />
            </button>
            <div className={styles.headerTitle}>Global Trade Intelligence</div>
          </div>
          <div className={styles.headerControls}>
             <div className={styles.subtleRing} style={{ padding: '0.4rem 0.8rem', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem' }}>
              <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: MINTED_BRASS }}></div>
              System Online
            </div>
          </div>
        </header>

        <main className={styles.mainContent}>
          <AnimatePresence>
            {selectedCountry && <DrillDownModal key="drilldown" country={selectedCountry.name} originalCountry={selectedCountry.code} onClose={() => setSelectedCountry(null)} />}
            
            </AnimatePresence>
          <motion.div variants={containerVariants} initial="hidden" animate="visible">
            
            {activeTab === 'dashboard' && (
              <div className={styles.kpiRow}>
                <div className={styles.kpiCard}>
                  <span className={styles.kpiLabel}>Partner</span>
                  <select value={partnerCode} onChange={handlePartnerChange} className={styles.chatInput}>
                    {partnerList.map(p => <option key={p.code} value={p.code}>{p.name}</option>)}
                  </select>
                </div>
                <div className={styles.kpiCard}>
                  <span className={styles.kpiLabel}>Commodity</span>
                  <select value={commodityCode} onChange={(e) => { setCommodityCode(e.target.value); setForecastError(null); setSuggestedCommodities([]); }} className={styles.chatInput} disabled={!validMap[partnerCode] || validMap[partnerCode].length === 0}>
                    {validMap[partnerCode]?.length > 0 ? validMap[partnerCode].map(c => <option key={c} value={c}>{c} — {CMD_MAP[c] || c}</option>) : <option value="">No trade data</option>}
                  </select>
                </div>
                <div className={styles.kpiCard}>
                  <span className={styles.kpiLabel}>Forecast Year</span>
                  <select value={forecastYear} onChange={(e) => setForecastYear(e.target.value)} className={styles.chatInput}>
                    <option value="2025">2025</option><option value="2026">2026</option>
                  </select>
                </div>
                <div className={styles.kpiCard}>
                  <span className={styles.kpiLabel}>USD/INR Rate</span>
                  <input type="number" value={usdInr} onChange={(e) => setUsdInr(e.target.value)} className={styles.chatInput} step="0.1" min="0" required />
                </div>
                <div className={styles.kpiCard}>
                  <span className={styles.kpiLabel}>Crude Oil ($/bbl)</span>
                  <input type="number" value={crudePrice} onChange={(e) => setCrudePrice(e.target.value)} className={styles.chatInput} step="0.1" min="0" required />
                </div>
                <div className={styles.kpiCard} style={{ justifyContent: 'flex-end', background: 'transparent', border: 'none' }}>
                  <button onClick={handlePredict} disabled={isPredicting} className={styles.chatButton} style={{ width: '100%', padding: '1rem', backgroundColor: MINTED_BRASS, color: NIGHT_SLATE, fontFamily: "'Playfair Display', serif", fontSize: '1.1rem', borderRadius: '4px', border: 'none', cursor: 'pointer', fontWeight: 'bold' }}>
                    {isPredicting ? 'Running AI Model...' : 'Generate AI Forecast'}
                  </button>
                </div>
              </div>
            )}

            <div className={styles.grid}>
              {activeTab === 'dashboard' && (
                <motion.section id="dashboard" variants={itemVariants} whileHover={{ scale: 1.01, y: -4, boxShadow: "0 20px 40px -10px rgba(0,0,0,0.5)" }} transition={{ duration: 0.2 }} className={`glass-panel ${styles.section}`}>
                <div className={styles.sectionHeader}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <TrendingUp size={20} color={MINTED_BRASS} />
                    <h2 className={styles.sectionTitle}>XGBoost Bilateral Trade Forecaster</h2>
                  </div>
                  <div style={{ display: 'flex', gap: '1rem' }}>
                    {(() => {
                      const latestChartPoint = chartData.length > 0 ? chartData[chartData.length - 1] : null;
                      return (
                        <div className={styles.badge} style={{ color: MINTED_BRASS, border: `1px solid ${MINTED_BRASS}`, padding: '4px 12px', fontSize: '0.8rem', backgroundColor: 'transparent' }}>
                          Latest: <AnimatedMoney value={latestChartPoint ? latestChartPoint.value : null} />
                        </div>
                      );
                    })()}
                    <div className={styles.badge} style={{ color: MINTED_BRASS, border: `1px solid ${MINTED_BRASS}`, padding: '4px 12px', fontSize: '0.8rem', backgroundColor: 'transparent' }}>R&sup2; = {r2Metric.toFixed(3)} (Log-Scale)</div>
                  </div>
                </div>
            {forecastError && (
              <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: 'rgba(220, 38, 38, 0.1)', border: `1px solid ${CRIMSON_WAX}`, color: CRIMSON_WAX, borderRadius: '4px' }}>
                <div style={{ marginBottom: '0.5rem' }}>Forecast Error: {forecastError}</div>
                {suggestedCommodities && suggestedCommodities.length > 0 && (
                  <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginTop: '0.5rem' }}>
                    {suggestedCommodities.map(c => (
                      <button 
                        key={c.code}
                        onClick={() => {
                          setCommodityCode(c.code);
                          setForecastError(null);
                          setSuggestedCommodities([]);
                        }}
                        style={{ background: 'rgba(255,255,255,0.1)', border: `1px solid ${CRIMSON_WAX}`, color: CRIMSON_WAX, padding: '4px 8px', borderRadius: '4px', cursor: 'pointer', fontSize: '0.8rem' }}
                      >
                        {c.code} — {c.name}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}
            <div className={styles.forecastGrid}>
              <div className={styles.chartContainer} style={{ height: '300px', position: 'relative' }}>
                {chartData.length === 0 ? (
                  <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: FADED_INK }}>
                    No historical data available for this partner/commodity combination.
                  </div>
                ) : mounted && (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData} margin={{ top: 20, right: 20, bottom: 5, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                      <XAxis dataKey="year" stroke={FADED_INK} fontSize={12} tickLine={false} axisLine={false} />
                      <YAxis stroke={FADED_INK} fontSize={12} tickLine={false} axisLine={false} tickFormatter={(val) => formatMoney(val)} />
                      <Tooltip contentStyle={{ backgroundColor: CARD_SURFACE, border: `1px solid ${MINTED_BRASS}`, borderRadius: '0px' }} itemStyle={{ color: MINTED_BRASS }} formatter={(v: any, name: any) => [formatMoney(Number(v)), name]} />
                      <Line type="monotone" dataKey="value" stroke={MINTED_BRASS} strokeWidth={3} dot={(props: any) => {
                        const { cx, cy, payload, key } = props;
                        const isPred = payload?.year?.includes('Pred');
                        return (
                          <circle key={key} cx={cx} cy={cy} r={isPred ? 6 : 4} fill={isPred ? CRIMSON_WAX : NIGHT_SLATE} stroke={MINTED_BRASS} strokeWidth={2} />
                        );
                      }} activeDot={{ r: 6, fill: MINTED_BRASS }} />
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </div>
              <div style={{ padding: '1.5rem', backgroundColor: 'rgba(0,0,0,0.2)', border: `1px solid ${FADED_INK}` }}>
                <h3 style={{ fontSize: '0.9rem', fontWeight: 500, marginBottom: '1rem', color: '#EFECE6', fontFamily: "'Playfair Display', serif" }}>Global Model Feature Importance</h3>
                <div className={styles.featureBarContainer}>
                  {(() => {
                    if (featureImportances.length === 0) {
                      return <p style={{ fontSize: '0.8rem', color: FADED_INK }}>Run forecast to view feature importance.</p>;
                    }
                    const maxImportance = Math.max(...featureImportances.map(f => f.importance), 1);
                    return featureImportances.map((f, i) => (
                      <div key={i} className={styles.featureBarRow}>
                        <span className={styles.featureLabel} title={f.feature}>{f.feature}</span>
                        <div className={styles.featureTrack}>
                          <motion.div initial={{ width: 0 }} animate={{ width: `${(f.importance / maxImportance) * 100}%` }} className={styles.featureFill} style={{ backgroundColor: MINTED_BRASS }} />
                        </div>
                      </div>
                    ));
                  })()}
                </div>
              </div>
            </div>
          </motion.section>
              )}
              {activeTab === 'anomalies' && (
          <motion.section id="anomalies" variants={itemVariants} whileHover={{ scale: 1.01, y: -4, boxShadow: "0 20px 40px -10px rgba(0,0,0,0.5)" }} transition={{ duration: 0.2 }} className={`glass-panel ${styles.section}`}>
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
                      <td style={{ padding: '1rem', color: row.reason_code === 'no_baseline' ? FADED_INK : CRIMSON_WAX }}>{row.reason_code === 'no_baseline' ? 'Insufficient baseline (not a true anomaly)' : row.reason}</td>
                      <td style={{ padding: '1rem', color: row.anomaly_score > 0.05 ? CRIMSON_WAX : MINTED_BRASS }}>
                        {row.anomaly_score ? row.anomaly_score.toFixed(3) : "N/A"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </motion.section>
              )}
              {activeTab === 'network' && (
          <motion.section id="network" variants={itemVariants} whileHover={{ scale: 1.01, y: -4, boxShadow: "0 20px 40px -10px rgba(0,0,0,0.5)" }} transition={{ duration: 0.2 }} className={`glass-panel ${styles.section}`}>
            <div className={styles.sectionHeader}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <MapIcon size={20} color={MINTED_BRASS} />
                <h2 className={styles.sectionTitle}>Global Trade Heatmap & Network Embeddings</h2>
              </div>
              <div style={{ display: 'flex', gap: '0.5rem', background: 'rgba(0,0,0,0.3)', padding: '0.25rem', borderRadius: '20px', border: `1px solid ${FADED_INK}` }}>
                      <button onClick={() => setFlowMode('off')} style={{ padding: '0.25rem 1rem', borderRadius: '15px', border: 'none', cursor: 'pointer', fontSize: '0.85rem', background: flowMode === 'off' ? FADED_INK : 'transparent', color: flowMode === 'off' ? '#fff' : FADED_INK, transition: 'all 0.3s' }}>Off</button>
                      <button onClick={() => setFlowMode('exports')} style={{ padding: '0.25rem 1rem', borderRadius: '15px', border: 'none', cursor: 'pointer', fontSize: '0.85rem', background: flowMode === 'exports' ? '#FF9F43' : 'transparent', color: flowMode === 'exports' ? NIGHT_SLATE : '#FF9F43', transition: 'all 0.3s' }}>Exports</button>
                      <button onClick={() => setFlowMode('imports')} style={{ padding: '0.25rem 1rem', borderRadius: '15px', border: 'none', cursor: 'pointer', fontSize: '0.85rem', background: flowMode === 'imports' ? '#00E5FF' : 'transparent', color: flowMode === 'imports' ? NIGHT_SLATE : '#00E5FF', transition: 'all 0.3s' }}>Imports</button>
                    </div>
            </div>
            <p style={{ fontSize: '0.85rem', color: FADED_INK, marginBottom: '1.5rem' }}>
              Choropleth mapping of node volumes and 2D PCA projection of Node2Vec random walks over the global trade graph.
            </p>
            <div style={{ marginTop: "1.5rem" }}>
<div style={{ height: '600px', position: 'relative', overflow: 'hidden', backgroundColor: 'rgba(0,0,0,0.2)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}>
                    {mounted && (
                      <>
                        <ReactTooltip id="map-tooltip" />
                        <ComposableMap projection="geoMercator" projectionConfig={{ scale: 150 }} width={800} height={450} style={{ width: '100%', height: 'auto', display: 'block' }}>
                            <defs>
                              <marker id="arrow-export" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                                <path d="M 0 0 L 10 5 L 0 10 z" fill="#FF9F43" />
                              </marker>
                              <marker id="arrow-import" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                                <path d="M 0 0 L 10 5 L 0 10 z" fill="#00E5FF" />
                              </marker>
                            </defs>
                            <Sphere stroke="rgba(255,255,255,0.1)" strokeWidth={0.5} id="sphere" fill="transparent" />
                            <Graticule stroke="rgba(255,255,255,0.05)" strokeWidth={0.5} />
                            <Geographies geography={geoUrl}>
                              {({ geographies }) =>
                                geographies.map((geo) => {
                                  const nodeData = networkData.find(d => d.country_name === geo.properties.name);
                                  const val = nodeData?.val;
                                  let color = '#2C303A';
                                  if (val) {
                                    color = `rgba(200, 169, 126, ${val / 100})`;
                                  }
                                  const isSelected = selectedCountry?.name === geo.properties.name;
                                  return (
                                    <Geography
                                      key={geo.rsmKey}
                                      geography={geo}
                                      data-tooltip-id="map-tooltip"
                                      data-tooltip-content={`${geo.properties.name}: ${val ? formatMoney(val) : 'N/A'}`}
                                      fill={isSelected ? MINTED_BRASS : color}
                                      stroke={NIGHT_SLATE}
                                      strokeWidth={0.5}
                                      onClick={(e) => { 
                                        e.stopPropagation();
                                        let geoCode = parseInt(geo.id).toString();
                                        if (geoCode === "840") geoCode = "842"; // USA Comtrade override
                                        const partner = partnerList.find(p => p.code === geoCode);
                                        if (partner) {
                                          setSelectedCountry({ name: geo.properties.name, code: geoCode });
                                        } else {
                                          setSelectedCountry({ name: geo.properties.name, code: "" });
                                        }

                                      }}
                                      style={{ hover: { fill: MINTED_BRASS, outline: 'none', cursor: 'pointer' }, pressed: { outline: 'none' }, default: { outline: 'none' } }}
                                    />
                                  );
                                })
                              }
                            </Geographies>
                            {flowMode === 'exports' && topExports.map((item) => {
                              const coords = COUNTRY_COORDS[item.code];
                              if (!coords) return null;
                              return (
                                <path
                                  key={`export-${item.code}`}
                                  d={drawCurve(INDIA_COORDS, coords, 0.3)}
                                  fill="none"
                                  stroke="#FF9F43"
                                  strokeWidth={2}
                                  strokeLinecap="round"
                                  markerEnd="url(#arrow-export)"
                                  className={styles.animatedPath}
                                />
                              );
                            })}
                            {flowMode === 'imports' && topImports.map((item) => {
                              const coords = COUNTRY_COORDS[item.code];
                              if (!coords) return null;
                              return (
                                <path
                                  key={`import-${item.code}`}
                                  d={drawCurve(coords, INDIA_COORDS, 0.3)}
                                  fill="none"
                                  stroke="#00E5FF"
                                  strokeWidth={2}
                                  strokeLinecap="round"
                                  markerEnd="url(#arrow-import)"
                                  className={styles.animatedPath}
                                />
                              );
                            })}
                          </ComposableMap>
                          {flowMode !== 'off' && (
                            <div style={{ position: 'absolute', bottom: '20px', right: '20px', background: 'rgba(11, 14, 20, 0.85)', padding: '1rem', borderRadius: '8px', border: `1px solid ${flowMode === 'exports' ? '#FF9F43' : '#00E5FF'}`, width: '250px', backdropFilter: 'blur(4px)' }}>
                              <h4 style={{ margin: '0 0 0.75rem 0', color: flowMode === 'exports' ? '#FF9F43' : '#00E5FF', fontSize: '0.9rem', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '0.5rem' }}>Top 5 {flowMode === 'exports' ? 'Destinations' : 'Sources'}</h4>
                              <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                                {(flowMode === 'exports' ? topExports : topImports).map((item: any, idx: number) => (
                                  <li key={idx} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#fff' }}>
                                    <span style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: '140px' }} title={item.name}>{item.name}</span>
                                    <span style={{ fontFamily: 'monospace', color: flowMode === 'exports' ? '#FF9F43' : '#00E5FF' }}>${(item.amount / 1e9).toFixed(1)}B</span>
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                      </>
                    )}
                  </div>
</div>
          </motion.section>
              )}
            </div>
      </motion.div>
      {}
      <button className={styles.fab} onClick={() => window.VanijyaChat?.open()}>
        <Sparkles size={20} />
        Ask AI
      </button>
      {}
      <AnimatePresence>
        <div id="vanijya-chat-root"></div>

      </AnimatePresence>
        {/* Data Sources & Footer */}
        <div className={styles.footerSection}>
          <div className={styles.dataSourcesContainer}>
            <p className={styles.sourceTag}>TRANSPARENT SOURCING</p>
            <h2 className={styles.sourceTitle}>Data Sources</h2>
            <p className={styles.sourceSubtitle}>Every prediction and response is grounded in empirical global trade volumes and official Indian trade policies.</p>
            
            <div className={styles.sourceGrid}>
              <div className={styles.sourceCard}>
                <div className={styles.sourceIcon} style={{ background: '#1e3a8a' }}>UN</div>
                <div>
                  <h4>UN Comtrade</h4>
                  <p>Global trade physical volumes, import/export network flows, and historical anomaly data.</p>
                </div>
              </div>
              <div className={styles.sourceCard}>
                <div className={styles.sourceIcon} style={{ background: '#047857' }}>IN</div>
                <div>
                  <h4>DGFT (India)</h4>
                  <p>Directorate General of Foreign Trade official policies, tariff notices, and regulations.</p>
                </div>
              </div>
            </div>
          </div>

          <div className={styles.footer}>
            <div className={styles.footerLeft}>
              <h3>Vanijya AI</h3>
              <p>Global Trade Ensemble Architecture</p>
              <span>© 2026 Yash Bajpai. All rights reserved.</span>
            </div>
            <div className={styles.footerRight}>
              <p>Built by <strong>Yash Bajpai</strong></p>
              <div className={styles.socialLinks}>
                <a href="https://github.com/Yash1bajpai" target="_blank" rel="noopener noreferrer"><Code size={16}/> GitHub</a>
                <a href="https://linkedin.com/in/yash-bajpai" target="_blank" rel="noopener noreferrer"><User size={16}/> LinkedIn</a>
                <a href="mailto:your-email@example.com"><Mail size={16}/> Email</a>
              </div>
            </div>
          </div>
        </div>
        </main>
      </div>
    </div>
  );
}
