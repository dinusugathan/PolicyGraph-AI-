document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatMessages = document.getElementById('chat-messages');
    const channelSelect = document.getElementById('channel-select');
    const activeChannelBadge = document.getElementById('active-channel-badge');
    const obsStatusText = document.getElementById('obs-status-text');
    const scenarioButtons = document.querySelectorAll('.scenario-btn');

    // Steps
    const stepRouter = document.getElementById('step-router');
    const stepRetriever = document.getElementById('step-retriever');
    const stepAgent = document.getElementById('step-agent');
    const stepGuardrail = document.getElementById('step-guardrail');

    // Update channel badge
    channelSelect.addEventListener('change', () => {
        const text = channelSelect.options[channelSelect.selectedIndex].text;
        activeChannelBadge.textContent = text;
    });

    // Fetch LangSmith status
    fetchObservabilityStatus();

    async function fetchObservabilityStatus() {
        try {
            const res = await fetch('/api/observability');
            const data = await res.json();
            if (data.langsmith_active) {
                obsStatusText.innerHTML = `🟢 <strong>Connected</strong><br>Project: <code>${data.project}</code>`;
            } else {
                obsStatusText.innerHTML = `⚪ Local Mock Tracing<br>Project: <code>${data.project}</code><br><span style="color:#64748b;font-size:0.7rem;">Set LANGCHAIN_API_KEY for cloud</span>`;
            }
        } catch (e) {
            obsStatusText.textContent = 'Observability: Local Mode';
        }
    }

    // Scenario button clicks
    scenarioButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const query = btn.getAttribute('data-query');
            const channel = btn.getAttribute('data-channel');
            channelSelect.value = channel;
            activeChannelBadge.textContent = channelSelect.options[channelSelect.selectedIndex].text;
            userInput.value = query;
            chatForm.dispatchEvent(new Event('submit'));
        });
    });

    // Animate pipeline
    function animatePipeline(intent, callback) {
        // Reset
        [stepRouter, stepRetriever, stepAgent, stepGuardrail].forEach(s => s.classList.remove('active'));

        stepRouter.classList.add('active');
        setTimeout(() => {
            stepRetriever.classList.add('active');
            setTimeout(() => {
                stepAgent.classList.add('active');
                setTimeout(() => {
                    stepGuardrail.classList.add('active');
                    if (callback) callback();
                }, 150);
            }, 150);
        }, 150);
    }

    function formatMarkdown(text) {
        if (!text) return '';

        // Tables
        let formatted = text.replace(/\|(.+)\|/g, (match) => {
            return match; // Keep markdown table lines intact for custom handling or render basic
        });

        // Basic conversions
        formatted = formatted
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            .replace(/\n\n/g, '<br><br>')
            .replace(/\n/g, '<br>');

        return formatted;
    }

    // Handle submit
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = userInput.value.trim();
        if (!query) return;

        const channel = channelSelect.value;

        // Append user message
        appendMessage('user', query);
        userInput.value = '';
        userInput.disabled = true;

        // Create temporary typing placeholder
        const placeholder = appendMessage('assistant', '<div class="typing-indicator"><span>Thinking with LangGraph...</span></div>');

        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, channel })
            });

            if (!res.ok) throw new Error('Failed to fetch graph response');
            const data = await res.json();

            animatePipeline(data.intent, () => {
                renderAssistantMessage(placeholder, data);
            });

        } catch (err) {
            placeholder.querySelector('.msg-content').innerHTML = `<p style="color:#ef4444;">Error executing graph: ${err.message}</p>`;
        } finally {
            userInput.disabled = false;
            userInput.focus();
        }
    });

    function appendMessage(sender, htmlContent) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender}-message`;

        const headerDiv = document.createElement('div');
        headerDiv.className = 'msg-header';
        headerDiv.innerHTML = sender === 'user' 
            ? '<span>You</span>' 
            : '<span class="avatar">🦜</span> <strong>PolicyGraph Agent</strong>';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'msg-content';
        contentDiv.innerHTML = htmlContent;

        msgDiv.appendChild(headerDiv);
        msgDiv.appendChild(contentDiv);
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        return msgDiv;
    }

    function renderAssistantMessage(msgDiv, data) {
        const contentDiv = msgDiv.querySelector('.msg-content');
        
        let html = `<div>${formatMarkdown(data.response)}</div>`;

        // Render Citations
        if (data.citations && data.citations.length > 0) {
            html += `<div class="citation-tags">`;
            data.citations.forEach(c => {
                html += `<span class="citation-tag">📜 ${c}</span>`;
            });
            html += `</div>`;
        }

        // Render Trace Inspector
        if (data.trace_steps && data.trace_steps.length > 0) {
            const totalDuration = data.trace_steps.reduce((acc, s) => acc + (s.duration_ms || 0), 0);
            html += `
                <div class="trace-inspector">
                    <button class="trace-summary-btn" onclick="this.nextElementSibling.classList.toggle('hidden');">
                        <span>🕸️ LangGraph Trace (${data.trace_steps.length} nodes, ${totalDuration.toFixed(1)}ms)</span>
                        <span>▼</span>
                    </button>
                    <div class="trace-details">
            `;
            data.trace_steps.forEach(s => {
                html += `
                    <div class="trace-step-item">
                        <span class="trace-node">${s.node}</span>
                        <span>${s.output_summary}</span>
                        <span class="trace-dur">${s.duration_ms}ms</span>
                    </div>
                `;
            });
            html += `
                    </div>
                </div>
            `;
        }

        contentDiv.innerHTML = html;
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
});
