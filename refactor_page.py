import re

def refactor():
    with open('frontend/src/app/page.tsx', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Imports
    content = content.replace("LineChart, Line, ScatterChart, Scatter, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ZAxis",
                              "LineChart, Line, ScatterChart, Scatter, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ZAxis, ComposedChart, Bar")

    # 2. State additions
    state_additions = """  const [partnerList, setPartnerList] = useState<{code:string, name:string}[]>([]);
  const [validMap, setValidMap] = useState<Record<string, string[]>>({});
  const [suggestedCommodities, setSuggestedCommodities] = useState<{code:string, name:string}[]>([]);"""
    content = content.replace("const [forecastError, setForecastError] = useState<string | null>(null);", 
                              "const [forecastError, setForecastError] = useState<string | null>(null);\n" + state_additions)
                              
    # 3. useEffect for valid_combinations and handlePartnerChange
    use_effect_combos = """
  useEffect(() => {
    fetch(`${API_BASE}/forecast/valid_combinations`)
      .then(res => res.json())
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
"""
    content = content.replace("const handlePredict = async () => {", use_effect_combos + "\n  const handlePredict = async () => {")

    # 4. handlePredict error modification
    old_error = """      if (data.error) {
        setForecastError(data.error);
        return;
      }"""
    new_error = """      if (data.error) {
        setForecastError(data.error);
        if (data.suggested_commodities) {
          setSuggestedCommodities(data.suggested_commodities);
        } else {
          setSuggestedCommodities([]);
        }
        return;
      }"""
    content = content.replace(old_error, new_error)
    
    # 5. Error Banner modification
    old_banner = """              {forecastError && (
                <div style={{ padding: '12px', marginBottom: '16px', border: '1px solid #ff4d4f', backgroundColor: 'rgba(255, 77, 79, 0.1)', color: '#ff4d4f', borderRadius: '4px' }}>
                  {forecastError}
                </div>
              )}"""
    new_banner = """              {forecastError && (
                <div style={{ padding: '12px', marginBottom: '16px', border: '1px solid #ff4d4f', backgroundColor: 'rgba(255, 77, 79, 0.1)', color: '#ff4d4f', borderRadius: '4px' }}>
                  {forecastError}
                  {suggestedCommodities.length > 0 && (
                    <div style={{ marginTop: '8px', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                      {suggestedCommodities.map(c => (
                         <button key={c.code} onClick={() => setCommodityCode(c.code)} style={{ padding: '4px 8px', border: '1px solid #ff4d4f', background: 'transparent', color: '#ff4d4f', borderRadius: '12px', cursor: 'pointer', fontSize: '12px' }}>{c.name}</button>
                      ))}
                    </div>
                  )}
                </div>
              )}"""
    content = content.replace(old_banner, new_banner)

    # 6. Dropdowns
    old_dropdown_partner = """                <select value={partnerCode} onChange={(e) => setPartnerCode(e.target.value)} className={styles.chatInput}>
                  <option value="156">China</option>
                  <option value="842">USA</option>
                  <option value="784">United Arab Emirates</option>
                  <option value="682">Saudi Arabia</option>
                  <option value="643">Russian Federation</option>
                </select>"""
    new_dropdown_partner = """                <select value={partnerCode} onChange={handlePartnerChange} className={styles.chatInput}>
                  {partnerList.map(p => <option key={p.code} value={p.code}>{p.name}</option>)}
                </select>"""
    content = content.replace(old_dropdown_partner, new_dropdown_partner)

    old_dropdown_cmd = """                <select value={commodityCode} onChange={(e) => setCommodityCode(e.target.value)} className={styles.chatInput}>
                  <option value="27">Mineral Fuels & Oils</option>
                  <option value="71">Precious Metals & Stones</option>
                  <option value="85">Electrical Machinery & Electronics</option>
                  <option value="84">Machinery & Mechanical Appliances</option>
                  <option value="29">Organic Chemicals</option>
                </select>"""
    new_dropdown_cmd = """                <select value={commodityCode} onChange={(e) => { setCommodityCode(e.target.value); setForecastError(null); setSuggestedCommodities([]); }} className={styles.chatInput}>
                  {validMap[partnerCode]?.map(c => <option key={c} value={c}>HS {c}</option>) || <option value="">No valid commodities</option>}
                </select>"""
    content = content.replace(old_dropdown_cmd, new_dropdown_cmd)

    # 7. Motion circle to float-node
    old_motion = "const { cx, cy, fill } = props;"
    new_motion = "const { cx, cy, fill } = props;\n                        const randomDur = 3 + Math.random() * 2;\n                        const randomDelay = Math.random() * -5;"
    content = content.replace(old_motion, new_motion)
    
    content = re.sub(r'<motion\.circle\s+cx=\{cx\}\s+cy=\{cy\}\s+r=\{12\}.*?/>', r'<circle className="float-node" cx={cx} cy={cy} r={12} fill={fill} style={{ animationDuration: randomDur + "s", animationDelay: randomDelay + "s" }} />', content, flags=re.DOTALL)
    content = re.sub(r'<motion\.circle\s+cx=\{cx\}\s+cy=\{cy\}\s+r=\{14\}.*?/>', r'<circle className="float-node" cx={cx} cy={cy} r={14} fill={fill} style={{ animationDuration: randomDur + "s", animationDelay: randomDelay + "s" }} opacity={0.3} />', content, flags=re.DOTALL)
    
    # Also replace import { motion, MotionConfig } from 'framer-motion';
    # actually, leaving it doesn't break anything, but we don't need it.

    # 8. Heatmap Enlarge fit
    old_modal = """      <div style={{ position: 'fixed', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: 'rgba(0,0,0,0.8)', zIndex: 1000 }} onClick={() => setIsMapEnlarged(false)}>
        <div style={{ width: '90vw', height: '90vh', backgroundColor: CARD_SURFACE, padding: '2rem', border: `1px solid ${MINTED_BRASS}`, borderRadius: '8px' }} onClick={e => e.stopPropagation()}>
          <ComposableMap projection="geoMercator" projectionConfig={{ scale: 150 }} width={800} height={450} style={{ width: '100%', height: 'auto', display: 'block' }}>"""
    new_modal = """      <div style={{ position: 'fixed', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: 'rgba(0,0,0,0.8)', zIndex: 1000 }} onClick={() => setIsMapEnlarged(false)}>
        <div style={{ width: 'min(95vw, 1200px)', maxHeight: '90vh', overflow: 'hidden', backgroundColor: CARD_SURFACE, padding: '1rem', border: `1px solid ${MINTED_BRASS}`, borderRadius: '8px' }} onClick={e => e.stopPropagation()}>
          <div style={{ position: 'relative', width: '100%', aspectRatio: '800 / 450', maxHeight: '80vh' }}>
          <ComposableMap preserveAspectRatio="xMidYMid meet" projection="geoMercator" projectionConfig={{ scale: 150 }} width={800} height={450} style={{ width: '100%', height: '100%' }}>"""
    content = content.replace(old_modal, new_modal)
    content = content.replace('            </Geographies>\n          </ComposableMap>\n        </div>\n      </div>', '            </Geographies>\n          </ComposableMap>\n          </div>\n        </div>\n      </div>')

    # 9. Country Drawer split
    old_chart = """                  <LineChart data={historyData} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                    <XAxis dataKey="year" stroke={FADED_INK} fontSize={12} tickLine={false} axisLine={false} />
                    <YAxis stroke={FADED_INK} fontSize={12} tickLine={false} axisLine={false} tickFormatter={(val) => formatMoney(val)} />
                    <Tooltip contentStyle={{ backgroundColor: CARD_SURFACE, border: `1px solid ${MINTED_BRASS}`, borderRadius: '0px' }} itemStyle={{ color: MINTED_BRASS }} />
                    <Line type="monotone" dataKey="value_billions" stroke={MINTED_BRASS} strokeWidth={3} dot={false} />
                  </LineChart>"""
    
    new_chart = """                  {historyData.length > 0 && historyData[0].import_billions !== undefined ? (
                    <ComposedChart data={historyData} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                      <XAxis dataKey="year" stroke={FADED_INK} fontSize={12} tickLine={false} axisLine={false} />
                      <YAxis stroke={FADED_INK} fontSize={12} tickLine={false} axisLine={false} tickFormatter={(val) => formatMoney(val)} />
                      <Tooltip contentStyle={{ backgroundColor: CARD_SURFACE, border: `1px solid ${MINTED_BRASS}`, borderRadius: '0px' }} itemStyle={{ color: MINTED_BRASS }} />
                      <Bar dataKey="import_billions" name="Imports" fill={CRIMSON_WAX} barSize={20} />
                      <Bar dataKey="export_billions" name="Exports" fill="#4a90e2" barSize={20} />
                      <Line type="monotone" dataKey="value_billions" name="Total Trade" stroke={MINTED_BRASS} strokeWidth={3} dot={false} />
                    </ComposedChart>
                  ) : (
                    <LineChart data={historyData} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                      <XAxis dataKey="year" stroke={FADED_INK} fontSize={12} tickLine={false} axisLine={false} />
                      <YAxis stroke={FADED_INK} fontSize={12} tickLine={false} axisLine={false} tickFormatter={(val) => formatMoney(val)} />
                      <Tooltip contentStyle={{ backgroundColor: CARD_SURFACE, border: `1px solid ${MINTED_BRASS}`, borderRadius: '0px' }} itemStyle={{ color: MINTED_BRASS }} />
                      <Line type="monotone" dataKey="value_billions" stroke={MINTED_BRASS} strokeWidth={3} dot={false} />
                    </LineChart>
                  )}"""
    content = content.replace(old_chart, new_chart)

    with open('frontend/src/app/page.tsx', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    refactor()
