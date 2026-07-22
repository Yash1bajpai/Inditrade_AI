import sys
import re

with open('frontend/src/app/page.tsx', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Update formatMoney and add formatSigned
orig_format = '''const formatMoney = (value: number | undefined | null) => {
  if (value === null || value === undefined || isNaN(value)) return "N/A";
  if (value >= 1) return `$${value.toFixed(2)}B`;
  if (value > 0 && value < 1) return `$${Math.round(value * 1000)}M`;
  return "$0";
};'''

new_format = '''const formatMoney = (value: number | undefined | null) => {
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
};'''
code = code.replace(orig_format, new_format)

# 2. Add Tooltip import and render it
if 'from "react-tooltip"' not in code:
    code = code.replace('import { ComposableMap, Geographies, Geography } from "react-simple-maps";',
                        'import { ComposableMap, Geographies, Geography } from "react-simple-maps";\nimport { Tooltip as ReactTooltip } from "react-tooltip";')

    # Insert <ReactTooltip id="map-tooltip" /> right before <ComposableMap>
    code = code.replace('<ComposableMap ', '<ReactTooltip id="map-tooltip" />\n                <ComposableMap ')

# 3. Fix Heatmap Fit (H2)
orig_map = '''<ComposableMap projection="geoMercator" projectionConfig={{ scale: 120, center: [0, 20] }} style={{ width: '100%', height: '100%' }} viewBox="0 0 800 450">'''
new_map = '''<ComposableMap projection="geoMercator" projectionConfig={{ scale: 120, center: [0, 20] }} width={800} height={450} style={{ width: '100%', height: 'auto', display: 'block' }}>'''
code = code.replace(orig_map, new_map)

# Update Modal container
orig_modal = '''<div style={{ position: 'fixed', inset: 0, backgroundColor: 'rgba(0,0,0,0.8)', zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center' }} onClick={() => setIsMapEnlarged(false)}>
              <div style={{ width: '90vw', height: '90vh', backgroundColor: CARD_SURFACE, border: `1px solid ${MINTED_BRASS}`, borderRadius: '8px', padding: '1rem', position: 'relative' }} onClick={(e) => e.stopPropagation()}>'''

new_modal = '''<div style={{ position: 'fixed', inset: 0, backgroundColor: 'rgba(0,0,0,0.8)', zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center' }} onClick={() => setIsMapEnlarged(false)}>
              <div style={{ width: '90vw', maxHeight: '90vh', overflow: 'hidden', backgroundColor: CARD_SURFACE, border: `1px solid ${MINTED_BRASS}`, borderRadius: '8px', padding: '1rem', position: 'relative' }} onClick={(e) => e.stopPropagation()}>'''
code = code.replace(orig_modal, new_modal)


# 4. H1 & L4: Replace framer-motion animation with CSS animation
orig_node = '''<motion.circle
                          style={{ filter: "drop-shadow(0px 0px 4px rgba(200,169,126,0.3))" }}
                          animate={{ y: ["-3px", "3px", "-3px"] }}
                          transition={{ repeat: Infinity, duration: 3 + Math.random()*2, ease: "easeInOut" }}
                          cx={node.x}
                          cy={node.y}'''
new_node = '''<motion.circle
                          className="floating-node"
                          style={{ 
                            filter: "drop-shadow(0px 0px 4px rgba(200,169,126,0.3))",
                            animationDelay: `${Math.random() * 2}s`,
                            animationDuration: `${3 + Math.random() * 2}s`
                          }}
                          cx={node.x}
                          cy={node.y}'''
code = code.replace(orig_node, new_node)


# 5. L2: cleanup useEffect VanijyaChat
orig_chat_effect = '''// Vanijya Chat Injection
  useEffect(() => {
    if (!document.getElementById('vanijya-chat-css')) {
      const link = document.createElement('link');
      link.id = 'vanijya-chat-css';
      link.rel = 'stylesheet';
      link.href = '/chat.css';
      document.head.appendChild(link);
    }
    if (!document.getElementById('vanijya-chat-js')) {
      const script = document.createElement('script');
      script.id = 'vanijya-chat-js';
      script.src = '/chat.js';
      script.onload = () => {
        if ((window as any).VanijyaChat) {
          (window as any).VanijyaChat.mount(document.getElementById('vanijya-chat-root'), {
            apiBase: API_BASE,
            chatEndpoint: '/api/query/'
          });
        }
      };
      document.body.appendChild(script);
    }
  }, []);'''
new_chat_effect = '''// Vanijya Chat Injection
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
        if ((window as any).VanijyaChat) {
          (window as any).VanijyaChat.mount(document.getElementById('vanijya-chat-root'), {
            apiBase: API_BASE,
            chatEndpoint: '/api/query/'
          });
        }
      };
      document.body.appendChild(script);
    }
    
    return () => {
      if ((window as any).VanijyaChat) {
        (window as any).VanijyaChat.destroy();
      }
      const existingScript = document.getElementById(jsId);
      if (existingScript) existingScript.remove();
      const existingLink = document.getElementById(cssId);
      if (existingLink) existingLink.remove();
    };
  }, []);'''
code = code.replace(orig_chat_effect, new_chat_effect)

with open('frontend/src/app/page.tsx', 'w', encoding='utf-8') as f:
    f.write(code)

# Add CSS keyframes for floating-node to globals.css
with open('frontend/src/app/globals.css', 'a', encoding='utf-8') as f:
    f.write('''
@keyframes float {
  0% { transform: translateY(-3px); }
  50% { transform: translateY(3px); }
  100% { transform: translateY(-3px); }
}

.floating-node {
  animation-name: float;
  animation-timing-function: ease-in-out;
  animation-iteration-count: infinite;
}

@media (prefers-reduced-motion: reduce) {
  .floating-node {
    animation: none;
  }
}
''')

print('Success')
