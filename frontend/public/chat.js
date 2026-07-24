(function() {
  if (window.VanijyaChat) return;

  const STORAGE_KEY = 'vanijya_chat_history';
  let rootElement = null;
  let apiBaseUrl = '';
  let chatEndpoint = '';
  
  let isOpen = false;
  let isDragging = false;
  let messages = [];
  let isTyping = false;
  
  let lastAskPrompt = '';
  let lastAskTime = 0;
  
  let containerEl, messagesEl, inputEl, sendBtnEl, typingEl;
  
  function saveHistory() {
    // History persistence is disabled
  }
  
  function loadHistory() {
    // Load fresh start message every time, disregarding localStorage
    messages = [{ role: 'ai', content: 'Hello! I am your AI Indian Trade Policy Assistant. Ask me anything about DGFT compliance, import/export policies, or tariff rates.', sent: true, answered: true }];
    isTyping = false; // C4: clear typing flag on mount
  }

  function renderMessages() {
    if (!messagesEl) return;
    messagesEl.innerHTML = '';
    messages.forEach(msg => {
      const msgDiv = document.createElement('div');
      msgDiv.className = `vanijya-message ${msg.role}`;
      
      const avatar = document.createElement('div');
      avatar.className = `vanijya-avatar ${msg.role}`;
      avatar.innerHTML = msg.role === 'ai' 
        ? `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21 16-4 4-4-4"/><path d="M17 20V4"/><path d="m3 8 4-4 4 4"/><path d="M7 4v16"/></svg>`
        : `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`;
      
      const content = document.createElement('div');
      content.className = 'vanijya-content';
      
      const p = document.createElement('p');
      if (msg.typewriter && msg.role === 'ai') {
        p.textContent = '';
        content.appendChild(p);
        let i = 0;
        const speed = 15;
        msg.typewriter = false; // Prevent re-triggering on subsequent renders
        function typeWriter() {
          if (i < msg.content.length) {
            p.textContent += msg.content.charAt(i);
            i++;
            if (messagesEl) messagesEl.scrollTop = messagesEl.scrollHeight;
            setTimeout(typeWriter, speed);
          }
        }
        typeWriter();
      } else {
        p.textContent = msg.content;
        content.appendChild(p);
      }
      
      if (msg.source && msg.role === 'ai') {
        const src = document.createElement('div');
        src.className = 'vanijya-source';
        src.textContent = `Source: ${msg.source} ${msg.citation ? ' | ' + msg.citation : ''}`;
        content.appendChild(src);
      }
      
      msgDiv.appendChild(avatar);
      msgDiv.appendChild(content);
      messagesEl.appendChild(msgDiv);
    });
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }
  
  function updateTypingState() {
    if (typingEl) {
      if (isTyping) typingEl.classList.add('visible');
      else typingEl.classList.remove('visible');
    }
    if (sendBtnEl) {
      sendBtnEl.disabled = isTyping;
    }
    if (inputEl) {
      inputEl.disabled = isTyping;
    }
  }

  async function submitMessage(text) {
    if (!text.trim() || isTyping) return;
    
    // C4: create new message flag sent: true, answered: false
    const userMsg = { role: 'user', content: text, sent: true, answered: false };
    messages.push(userMsg);
    
    inputEl.value = '';
    isTyping = true;
    renderMessages();
    updateTypingState();
    
    try {
      const res = await fetch(`${apiBaseUrl}${chatEndpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: text })
      });
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      const data = await res.json();
      
      const aiMsg = { role: 'ai', content: data.answer, source: data.source, citation: data.citation, sent: true, answered: true, typewriter: true };
      userMsg.answered = true; // both are valid now
      messages.push(aiMsg);
      saveHistory(); // C4: persist
      
    } catch (e) {
      const errorMsg = { role: 'ai', content: 'Failed to connect to the backend server.', source: 'Error', sent: true, answered: true };
      userMsg.answered = true;
      messages.push(errorMsg);
      // We don't save errors to history typically, but per instruction, if answered=true we can.
    } finally {
      isTyping = false;
      renderMessages();
      updateTypingState();
      inputEl.focus();
    }
  }

  window.VanijyaChat = {
    mount(root, config) {
      rootElement = root;
      apiBaseUrl = (config.apiBase || '').replace(/\/$/, "");
      chatEndpoint = config.chatEndpoint || '/api/query/';
      
      loadHistory();
      
      rootElement.innerHTML = `
        <div id="vanijya-chat-container">
          <div class="vanijya-drag-handle" id="vanijya-drag"></div>
          <div class="vanijya-header">
            <div class="vanijya-header-title">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
              <h2>Vanijya AI</h2>
            </div>
            <button class="vanijya-close-btn" id="vanijya-close" aria-label="Close Chat">
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
            </button>
          </div>
          <div class="vanijya-messages" id="vanijya-messages"></div>
          <div class="vanijya-typing" id="vanijya-typing">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
          </div>
          <form class="vanijya-input-area" id="vanijya-form">
            <input type="text" class="vanijya-input" id="vanijya-input" placeholder="Ask about Trade Policy..." autocomplete="off"/>
            <button type="submit" class="vanijya-send-btn" id="vanijya-send">
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
            </button>
          </form>
        </div>
      `;
      
      containerEl = document.getElementById('vanijya-chat-container');
      messagesEl = document.getElementById('vanijya-messages');
      inputEl = document.getElementById('vanijya-input');
      sendBtnEl = document.getElementById('vanijya-send');
      typingEl = document.getElementById('vanijya-typing');
      const formEl = document.getElementById('vanijya-form');
      const closeBtn = document.getElementById('vanijya-close');
      const dragHandle = document.getElementById('vanijya-drag');
      
      closeBtn.addEventListener('click', () => this.close());
      
      formEl.addEventListener('submit', (e) => {
        e.preventDefault();
        submitMessage(inputEl.value);
      });
      
      // Resizing logic (C1, C3)
      let rafId = null;
      
      const onMouseMove = (e) => {
        if (!isDragging) return;
        const newWidth = Math.max(300, Math.min(window.innerWidth - e.clientX, 1000));
        if (rafId) cancelAnimationFrame(rafId);
        rafId = requestAnimationFrame(() => {
          document.documentElement.style.setProperty('--chat-width', newWidth + 'px');
        });
      };
      
      const onMouseUp = () => {
        isDragging = false;
        dragHandle.classList.remove('dragging');
        document.removeEventListener('mousemove', onMouseMove);
        document.removeEventListener('mouseup', onMouseUp);
      };
      
      // Store these handlers to remove them in destroy()
      this._onMouseMove = onMouseMove;
      this._onMouseUp = onMouseUp;
      
      dragHandle.addEventListener('mousedown', () => {
        isDragging = true;
        dragHandle.classList.add('dragging');
        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);
      });
      
      renderMessages();
      updateTypingState();
    },
    
    open() {
      isOpen = true;
      if (containerEl) {
        containerEl.classList.add('open');
        setTimeout(() => { if (inputEl) inputEl.focus(); }, 300);
      }
    },
    
    close() {
      isOpen = false;
      if (containerEl) {
        containerEl.classList.remove('open');
      }
    },
    
    ask(prompt) {
      // C2: idempotent check
      const now = Date.now();
      if (prompt === lastAskPrompt && (now - lastAskTime) < 500) {
        return;
      }
      lastAskPrompt = prompt;
      lastAskTime = now;
      
      if (!isOpen) this.open();
      submitMessage(prompt);
    },
    
    destroy() {
      if (this._onMouseMove) document.removeEventListener('mousemove', this._onMouseMove);
      if (this._onMouseUp) document.removeEventListener('mouseup', this._onMouseUp);
      
      if (rootElement) {
        rootElement.innerHTML = '';
      }
      isOpen = false;
      messages = [];
    }
  };
})();
