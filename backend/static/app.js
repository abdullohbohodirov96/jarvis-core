/* =============================================================================
   JARVIS Web Dashboard Frontend Logic
   ============================================================================= */

// App State
const state = {
    activeTab: 'chat',
    clientId: Math.random().toString(36).substring(2, 15),
    conversationId: Math.random().toString(36).substring(2, 15),
    ws: null,
    latencyInterval: null,
    
    // Voice recording
    mediaRecorder: null,
    audioChunks: [],
    isRecording: false,
    recordingTimer: null,
    recordingSeconds: 0,

    // Telegram
    telegramConnected: false,
    chats: [],
    selectedChatId: null,
    selectedChatName: '',
    messages: [],
    chatsInterval: null,
    messagesInterval: null,
};

// Config & Selectors
const DOM = {
    // Navigation
    navItems: document.querySelectorAll('.nav-item'),
    panels: document.querySelectorAll('.tab-panel'),
    tabTitle: document.getElementById('current-tab-title'),
    tabSubtitle: document.getElementById('current-tab-subtitle'),

    // Statuses
    apiStatusDot: document.getElementById('api-status-dot'),
    dbStatusDot: document.getElementById('db-status-dot'),
    tgStatusDot: document.getElementById('tg-status-dot'),
    systemStatusPill: document.getElementById('system-status-pill'),
    latencyVal: document.querySelector('#network-latency span'),

    // Chat
    chatHistory: document.getElementById('chat-history'),
    chatInput: document.getElementById('chat-input-text'),
    chatSendBtn: document.getElementById('chat-send-btn'),
    voiceRecordBtn: document.getElementById('voice-record-btn'),
    voiceStatusBar: document.getElementById('voice-status-bar'),
    voiceStatusText: document.getElementById('voice-status-text'),
    voiceTimer: document.getElementById('voice-timer'),

    // Telegram UI
    tgChatsList: document.getElementById('tg-chats-list'),
    tgChatSearch: document.getElementById('tg-chat-search'),
    tgMessagesBox: document.getElementById('tg-messages-box'),
    tgActiveChatName: document.getElementById('tg-active-chat-name'),
    tgActiveChatStatus: document.getElementById('tg-active-chat-status'),
    tgMessageInput: document.getElementById('tg-message-input'),
    tgSendBtn: document.getElementById('tg-send-btn'),
    tgInputBox: document.getElementById('tg-input-box'),
    tgConnStatus: document.getElementById('tg-conn-status'),

    // Tasks
    taskForm: document.getElementById('task-create-form'),
    taskTitle: document.getElementById('task-title'),
    taskDesc: document.getElementById('task-desc'),
    taskPriority: document.getElementById('task-priority'),
    taskDue: document.getElementById('task-due'),
    tasksContainer: document.getElementById('tasks-container'),
    taskFilterBtns: document.querySelectorAll('.filter-btn'),

    // Settings
    settingsForm: document.getElementById('telegram-settings-form'),
    settingsApiId: document.getElementById('settings-api-id'),
    settingsApiHash: document.getElementById('settings-api-hash'),
    settingsPhone: document.getElementById('settings-phone'),
    authStatusLabel: document.getElementById('auth-status-label'),
    authSendCodeBtn: document.getElementById('auth-send-code-btn'),
    authStep2: document.getElementById('auth-step-2'),
    authCodeInput: document.getElementById('auth-code-input'),
    authPasswordInput: document.getElementById('auth-password-input'),
    authVerifyCodeBtn: document.getElementById('auth-verify-code-btn'),

    // Toasts
    toastContainer: document.getElementById('toast-container'),
};

const TAB_META = {
    chat: { title: 'Suhbat Assistant', subtitle: "JARVIS bilan matn va ovoz orqali muloqot qiling" },
    telegram: { title: 'Telegram Boshqaruvi', subtitle: 'Telegram dialoglarini o\'qing va xabarlar yozing' },
    tasks: { title: 'Vazifalar Rejasi', subtitle: 'Kunlik amallar va rejalarni shakllantiring' },
    settings: { title: 'Tizim Sozlamalari', subtitle: 'Telegram API integratsiyasi va ulanish boshqaruvi' },
};

// =============================================================================
// Helper Functions
// =============================================================================

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <i class="fa-solid ${type === 'success' ? 'fa-circle-check' : type === 'error' ? 'fa-triangle-exclamation' : 'fa-circle-info'}"></i>
        <div class="toast-content">${message}</div>
    `;
    DOM.toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'toastIn 0.3s reverse forwards';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// Auto-grow Textarea
if (DOM.chatInput) {
    DOM.chatInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });
}

// =============================================================================
// API Requests Wrapper
// =============================================================================

async function apiRequest(endpoint, options = {}) {
    const url = `/api/v1${endpoint}`;
    const start = performance.now();
    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        const duration = Math.round(performance.now() - start);
        DOM.latencyVal.innerText = `${duration} ms`;

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || `HTTP Error: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error(`API Error (${endpoint}):`, error);
        throw error;
    }
}

// =============================================================================
// WebSocket Integration (Chat Tab)
// =============================================================================

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/chat/${state.clientId}`;
    
    state.ws = new WebSocket(wsUrl);

    state.ws.onopen = () => {
        console.log('Chat WebSocket connected');
        DOM.apiStatusDot.className = 'stat-val online';
    };

    state.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'token') {
            appendOrUpdateBotMessage(data.content);
        } else if (data.type === 'done') {
            finalizeBotMessage();
        } else if (data.type === 'error') {
            showToast(data.message, 'error');
            finalizeBotMessage(true);
        }
    };

    state.ws.onclose = () => {
        console.warn('Chat WebSocket closed. Reconnecting in 3 seconds...');
        DOM.apiStatusDot.className = 'stat-val';
        setTimeout(connectWebSocket, 3000);
    };
}

let activeBotMessageBubble = null;

function appendUserMessage(text) {
    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message user';
    msgDiv.innerHTML = `
        <div class="message-content">
            <p>${escapeHtml(text)}</p>
            <span class="message-time">${timeStr}</span>
        </div>
    `;
    DOM.chatHistory.appendChild(msgDiv);
    DOM.chatHistory.scrollTop = DOM.chatHistory.scrollHeight;
}

function appendOrUpdateBotMessage(token) {
    if (!activeBotMessageBubble) {
        const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message system';
        msgDiv.innerHTML = `
            <div class="message-content">
                <p class="bot-text-bubble"></p>
                <span class="message-time">${timeStr}</span>
            </div>
        `;
        DOM.chatHistory.appendChild(msgDiv);
        activeBotMessageBubble = msgDiv.querySelector('.bot-text-bubble');
    }
    
    // Accumulate tokens
    activeBotMessageBubble.innerHTML += token.replace(/\n/g, '<br>');
    DOM.chatHistory.scrollTop = DOM.chatHistory.scrollHeight;
}

function finalizeBotMessage(isError = false) {
    if (isError && activeBotMessageBubble) {
        activeBotMessageBubble.innerHTML += ` <span style="color:#ff3333;">[Xatolik yuz berdi]</span>`;
    }
    activeBotMessageBubble = null;
}

function escapeHtml(text) {
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function sendChatMessage() {
    const text = DOM.chatInput.value.trim();
    if (!text) return;

    appendUserMessage(text);
    DOM.chatInput.value = '';
    DOM.chatInput.style.height = 'auto';

    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        state.ws.send(JSON.stringify({
            message: text,
            conversation_id: state.conversationId
        }));
    } else {
        showToast('Server ulanishi mavjud emas. WebSocket qayta yuklanmoqda.', 'error');
    }
}

// =============================================================================
// Voice / Microphone Recording (STT Input)
// =============================================================================

async function toggleVoiceRecording() {
    if (state.isRecording) {
        stopVoiceRecording();
    } else {
        await startVoiceRecording();
    }
}

async function startVoiceRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        state.mediaRecorder = new MediaRecorder(stream);
        state.audioChunks = [];

        state.mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                state.audioChunks.push(event.data);
            }
        };

        state.mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(state.audioChunks, { type: 'audio/wav' });
            await sendAudioForTranscription(audioBlob);
            
            // Turn off microphone tracks
            stream.getTracks().forEach(track => track.stop());
        };

        state.mediaRecorder.start();
        state.isRecording = true;
        DOM.voiceRecordBtn.classList.add('recording');
        DOM.voiceStatusBar.classList.remove('hidden');
        
        state.recordingSeconds = 0;
        DOM.voiceTimer.innerText = '00:00';
        DOM.voiceStatusText.innerText = 'JARVIS eshitmoqda...';

        state.recordingTimer = setInterval(() => {
            state.recordingSeconds++;
            const mins = Math.floor(state.recordingSeconds / 60).toString().padStart(2, '0');
            const secs = (state.recordingSeconds % 60).toString().padStart(2, '0');
            DOM.voiceTimer.innerText = `${mins}:${secs}`;
        }, 1000);

    } catch (err) {
        console.error('Mikrofonga ruxsat berilmadi:', err);
        showToast('Mikrofonga ulanib bo\'lmadi. Ruxsatni tekshiring.', 'error');
    }
}

function stopVoiceRecording() {
    if (state.mediaRecorder && state.isRecording) {
        state.mediaRecorder.stop();
        state.isRecording = false;
        DOM.voiceRecordBtn.classList.remove('recording');
        DOM.voiceStatusBar.classList.add('hidden');
        clearInterval(state.recordingTimer);
    }
}

async function sendAudioForTranscription(blob) {
    showToast('Ovoz matnga o\'tkazilmoqda...', 'info');
    
    const formData = new FormData();
    formData.append('audio', blob, 'recording.wav');

    try {
        const response = await fetch('/api/v1/voice/transcribe', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`STT Error: ${response.status}`);
        }

        const data = await response.json();
        if (data.transcript && data.transcript.trim()) {
            const text = data.transcript.trim();
            showToast('Ovoz aniqlandi!', 'success');
            
            // Process it just like text message
            appendUserMessage(text);
            if (state.ws && state.ws.readyState === WebSocket.OPEN) {
                state.ws.send(JSON.stringify({
                    message: text,
                    conversation_id: state.conversationId
                }));
            }
        } else {
            showToast('Ovoz eshitilmadi yoki aniqlab bo\'lmadi.', 'warning');
        }
    } catch (err) {
        console.error('STT Upload failed:', err);
        showToast('Ovozli xabarni yuborishda xatolik yuz berdi.', 'error');
    }
}

// =============================================================================
// Telegram Boshqaruvi (Telegram Tab)
// =============================================================================

async function checkTelegramStatus() {
    try {
        const data = await apiRequest('/telegram/status');
        state.telegramConnected = data.connected;
        
        const dot = DOM.tgStatusDot;
        const pill = DOM.authStatusLabel;
        
        if (data.connected) {
            dot.className = 'stat-val online';
            dot.innerText = '●';
            pill.className = 'badge success';
            pill.innerText = 'Ulangan';
            DOM.tgConnStatus.className = 'badge success';
            DOM.tgConnStatus.innerText = 'ON';
            
            // Enable Telegram sidebars
            if (state.activeTab === 'telegram' && state.chats.length === 0) {
                loadTelegramChats();
            }
        } else {
            dot.className = 'stat-val';
            dot.innerText = '○';
            pill.className = 'badge offline';
            pill.innerText = 'Ulanmagan';
            DOM.tgConnStatus.className = 'badge offline';
            DOM.tgConnStatus.innerText = 'OFF';
        }
    } catch (err) {
        DOM.tgStatusDot.className = 'stat-val';
        DOM.tgStatusDot.innerText = '○';
    }
}

async function loadTelegramChats() {
    if (!state.telegramConnected) return;

    try {
        const chats = await apiRequest('/telegram/chats?limit=25');
        state.chats = chats;
        renderTelegramChats(chats);
    } catch (err) {
        console.error('Failed to load chats:', err);
    }
}

function renderTelegramChats(chats) {
    if (chats.length === 0) {
        DOM.tgChatsList.innerHTML = `<div class="list-placeholder">Hech qanday chat topilmadi.</div>`;
        return;
    }

    DOM.tgChatsList.innerHTML = chats.map(chat => {
        const unreadBadge = chat.unread_count > 0 ? `<span class="badge unread">${chat.unread_count}</span>` : '';
        const lastMsgTime = chat.last_message_at ? new Date(chat.last_message_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';
        const activeClass = state.selectedChatId === chat.chat_id ? 'active' : '';

        return `
            <div class="tg-chat-item ${activeClass}" onclick="selectTelegramChat(${chat.chat_id}, '${escapeHtml(chat.name)}')">
                <div class="chat-header-row">
                    <span class="chat-title">${escapeHtml(chat.name)}</span>
                    <span class="chat-time">${lastMsgTime}</span>
                </div>
                <div class="chat-preview-row">
                    <span class="chat-preview">${escapeHtml(chat.last_message || '[Media / Xabar yo\'q]')}</span>
                    ${unreadBadge}
                </div>
            </div>
        `;
    }).join('');
}

async function selectTelegramChat(chatId, name) {
    state.selectedChatId = chatId;
    state.selectedChatName = name;
    
    // Update Active UI
    DOM.tgActiveChatName.innerText = name;
    DOM.tgActiveChatStatus.innerText = `Chat ID: ${chatId} | Yuklanmoqda...`;
    DOM.tgInputBox.classList.remove('hidden');

    // Remove active class from previous, add to new
    document.querySelectorAll('.tg-chat-item').forEach(item => {
        item.classList.remove('active');
    });
    
    // Refresh chats rendering to match active selection
    renderTelegramChats(state.chats);

    await loadTelegramMessages(chatId);
}

async function loadTelegramMessages(chatId) {
    if (!chatId) return;

    try {
        const msgs = await apiRequest(`/telegram/messages/${chatId}?limit=50`);
        // Reverse array to render oldest messages first (scrolling down to newest)
        state.messages = msgs.reverse();
        renderTelegramMessages(state.messages);
        DOM.tgActiveChatStatus.innerText = `Chat ID: ${chatId} | Oxirgi 50 ta xabar`;
    } catch (err) {
        console.error('Failed to load messages:', err);
        DOM.tgActiveChatStatus.innerText = `Chat ID: ${chatId} | Yuklashda xatolik`;
    }
}

function renderTelegramMessages(messages) {
    if (messages.length === 0) {
        DOM.tgMessagesBox.innerHTML = `<div class="pane-placeholder">Xabarlar mavjud emas.</div>`;
        return;
    }

    DOM.tgMessagesBox.innerHTML = messages.map(msg => {
        const direction = msg.is_outgoing ? 'outgoing' : 'incoming';
        const msgTime = new Date(msg.date).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const sender = msg.sender && !msg.is_outgoing ? `<strong style="font-size:0.75rem;margin-bottom:2px;display:block;color:var(--primary);">${escapeHtml(msg.sender)}</strong>` : '';

        return `
            <div class="tg-message ${direction}">
                ${sender}
                <div class="tg-msg-text">${escapeHtml(msg.text || '[Fayl / Rasm xabari]')}</div>
                <div class="tg-msg-meta">${msgTime}</div>
            </div>
        `;
    }).join('');

    DOM.tgMessagesBox.scrollTop = DOM.tgMessagesBox.scrollHeight;
}

async function sendTelegramMessage() {
    const text = DOM.tgMessageInput.value.trim();
    if (!text || !state.selectedChatId) return;

    DOM.tgMessageInput.value = '';
    
    try {
        const response = await apiRequest('/telegram/send', {
            method: 'POST',
            body: JSON.stringify({
                chat_id: state.selectedChatId,
                text: text
            })
        });

        if (response.message_id) {
            // Append message locally and scroll
            const localMsg = {
                message_id: response.message_id,
                chat_id: state.selectedChatId,
                sender: 'Siz',
                text: text,
                date: response.sent_at,
                is_outgoing: true
            };
            state.messages.push(localMsg);
            renderTelegramMessages(state.messages);
            showToast('Xabar Telegram orqali yuborildi.', 'success');
        }
    } catch (err) {
        showToast(`Telegram xabarini yuborib bo'lmadi: ${err.message}`, 'error');
    }
}

// Telegram Search Filter
if (DOM.tgChatSearch) {
    DOM.tgChatSearch.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase();
        const filtered = state.chats.filter(c => c.name.toLowerCase().includes(query));
        renderTelegramChats(filtered);
    });
}

// =============================================================================
// Task Manager (Vazifalar Tab)
// =============================================================================

let currentTaskFilter = 'all';

async function loadTasks() {
    try {
        const data = await apiRequest('/tasks?page_size=100');
        renderTasks(data.tasks);
    } catch (err) {
        console.error('Failed to load tasks:', err);
    }
}

function renderTasks(tasks) {
    let filteredTasks = tasks;
    if (currentTaskFilter === 'pending') {
        filteredTasks = tasks.filter(t => t.status === 'pending');
    } else if (currentTaskFilter === 'completed') {
        filteredTasks = tasks.filter(t => t.status === 'completed');
    }

    if (filteredTasks.length === 0) {
        DOM.tasksContainer.innerHTML = `<div class="list-placeholder">Vazifalar ro'yxati bo'sh.</div>`;
        return;
    }

    DOM.tasksContainer.innerHTML = filteredTasks.map(task => {
        const completedClass = task.status === 'completed' ? 'completed' : '';
        const dueDate = task.due_date ? `<span class="task-due-date"><i class="fa-solid fa-clock"></i> ${new Date(task.due_date).toLocaleDateString()}</span>` : '';
        
        return `
            <div class="task-card ${completedClass}" id="task-${task.task_id}">
                <div class="task-card-left">
                    <div class="task-checkbox" onclick="toggleTaskStatus('${task.task_id}', '${task.status}')">
                        <i class="fa-solid fa-check"></i>
                    </div>
                    <div class="task-text">
                        <h4>${escapeHtml(task.title)}</h4>
                        <p>${escapeHtml(task.description || '')}</p>
                        <div class="task-meta">
                            <span class="task-badge p-${task.priority}">${task.priority.toUpperCase()}</span>
                            ${dueDate}
                        </div>
                    </div>
                </div>
                <div class="task-card-right">
                    <button class="delete-task-btn" onclick="deleteTask('${task.task_id}')" title="O'chirish">
                        <i class="fa-solid fa-trash-can"></i>
                    </button>
                </div>
            </div>
        `;
    }).join('');
}

async function createTask(e) {
    e.preventDefault();
    const title = DOM.taskTitle.value.trim();
    const desc = DOM.taskDesc.value.trim();
    const priority = DOM.taskPriority.value;
    const due = DOM.taskDue.value ? new Date(DOM.taskDue.value).toISOString() : null;

    try {
        await apiRequest('/tasks', {
            method: 'POST',
            body: JSON.stringify({
                title: title,
                description: desc || null,
                priority: priority,
                due_date: due
            })
        });
        showToast('Vazifa muvaffaqiyatli qo\'shildi.', 'success');
        DOM.taskForm.reset();
        loadTasks();
    } catch (err) {
        showToast(`Vazifa yaratib bo'lmadi: ${err.message}`, 'error');
    }
}

async function toggleTaskStatus(taskId, currentStatus) {
    try {
        if (currentStatus === 'completed') {
            // Re-open task (update status back to pending)
            await apiRequest(`/tasks/${taskId}`, {
                method: 'PUT',
                body: JSON.stringify({ status: 'pending' })
            });
            showToast('Vazifa qaytadan ochildi.', 'info');
        } else {
            // Complete task
            await apiRequest(`/tasks/${taskId}/complete`, { method: 'POST' });
            showToast('Vazifa bajarildi deb belgilandi.', 'success');
        }
        loadTasks();
    } catch (err) {
        showToast(`Statusni o'zgartirib bo'lmadi: ${err.message}`, 'error');
    }
}

async function deleteTask(taskId) {
    if (!confirm('Ushbu vazifani o\'chirishni xohlaysizmi?')) return;
    try {
        await apiRequest(`/tasks/${taskId}`, { method: 'DELETE' });
        showToast('Vazifa o\'chirildi.', 'success');
        loadTasks();
    } catch (err) {
        showToast(`O'chirib bo'lmadi: ${err.message}`, 'error');
    }
}

// =============================================================================
// Settings & Authentication Flow (Telegram Auth)
// =============================================================================

async function loadTelegramSettings() {
    try {
        const settings = await apiRequest('/telegram/settings');
        DOM.settingsApiId.value = settings.api_id || '';
        DOM.settingsApiHash.value = settings.api_hash || '';
        DOM.settingsPhone.value = settings.phone || '';
    } catch (err) {
        console.error('Failed to load settings:', err);
    }
}

async function saveTelegramSettings(e) {
    e.preventDefault();
    const id = parseInt(DOM.settingsApiId.value);
    const hash = DOM.settingsApiHash.value.trim();
    const phone = DOM.settingsPhone.value.trim();

    try {
        await apiRequest('/telegram/settings', {
            method: 'POST',
            body: JSON.stringify({
                api_id: id,
                api_hash: hash,
                phone: phone
            })
        });
        showToast('Sozlamalar saqlandi.', 'success');
        checkTelegramStatus();
    } catch (err) {
        showToast(`Sozlamalarni saqlashda xatolik: ${err.message}`, 'error');
    }
}

async function sendTelegramAuthCode() {
    showToast('Telegram ga kod so\'rovi yuborilmoqda...', 'info');
    try {
        const response = await apiRequest('/telegram/connect', {
            method: 'POST',
            body: JSON.stringify({})
        });

        if (response.status === 'awaiting_code') {
            showToast('SMS kod yuborildi. Kodni kiriting.', 'success');
            DOM.authStep2.classList.remove('hidden');
        } else if (response.status === 'connected') {
            showToast('Siz allaqachon ulangansiz!', 'success');
            checkTelegramStatus();
        } else {
            showToast(response.detail || 'Xatolik yuz berdi.', 'error');
        }
    } catch (err) {
        showToast(`Kod yuborib bo'lmadi: ${err.message}`, 'error');
    }
}

async function verifyTelegramAuthCode() {
    const code = DOM.authCodeInput.value.trim();
    const password = DOM.authPasswordInput.value.trim();

    if (!code) {
        showToast('Iltimos, verification kodni kiriting.', 'warning');
        return;
    }

    showToast('Kodni tekshirish boshlandi...', 'info');
    try {
        const response = await apiRequest('/telegram/connect', {
            method: 'POST',
            body: JSON.stringify({
                phone_code: code,
                password: password || null
            })
        });

        if (response.status === 'connected') {
            showToast('Muvaffaqiyatli ulandi!', 'success');
            DOM.authStep2.classList.add('hidden');
            DOM.authCodeInput.value = '';
            DOM.authPasswordInput.value = '';
            checkTelegramStatus();
        } else if (response.status === 'awaiting_password') {
            showToast('2FA parol zarur. Parolni kiritib qayta bosing.', 'warning');
        } else {
            showToast(response.detail || 'Ulanishda xatolik.', 'error');
        }
    } catch (err) {
        showToast(`Ulanib bo'lmadi: ${err.message}`, 'error');
    }
}

// =============================================================================
// Init & Lifecycle
// =============================================================================

function switchTab(tabId) {
    state.activeTab = tabId;

    // Toggle menu buttons active state
    DOM.navItems.forEach(btn => {
        if (btn.getAttribute('data-tab') === tabId) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });

    // Toggle panels visibility
    DOM.panels.forEach(panel => {
        if (panel.id === `tab-${tabId}`) {
            panel.classList.add('active');
        } else {
            panel.classList.remove('active');
        }
    });

    // Update Header Meta
    const meta = TAB_META[tabId];
    if (meta) {
        DOM.tabTitle.innerText = meta.title;
        DOM.tabSubtitle.innerText = meta.subtitle;
    }

    // Tab-specific initial loads and polling settings
    clearInterval(state.chatsInterval);
    clearInterval(state.messagesInterval);

    if (tabId === 'telegram') {
        loadTelegramChats();
        // Poll for updates
        state.chatsInterval = setInterval(loadTelegramChats, 12000);
        state.messagesInterval = setInterval(() => {
            if (state.selectedChatId) {
                loadTelegramMessages(state.selectedChatId);
            }
        }, 6000);
    } else if (tabId === 'tasks') {
        loadTasks();
    } else if (tabId === 'settings') {
        loadTelegramSettings();
    }
}

// Setup Event Listeners
function setupEvents() {
    // Navigation
    DOM.navItems.forEach(btn => {
        btn.addEventListener('click', () => {
            switchTab(btn.getAttribute('data-tab'));
        });
    });

    // Chat Actions
    DOM.chatSendBtn.addEventListener('click', sendChatMessage);
    DOM.chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendChatMessage();
        }
    });
    DOM.voiceRecordBtn.addEventListener('click', toggleVoiceRecording);

    // Telegram UI
    DOM.tgSendBtn.addEventListener('click', sendTelegramMessage);
    DOM.tgMessageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            sendTelegramMessage();
        }
    });

    // Tasks Actions
    DOM.taskForm.addEventListener('submit', createTask);
    DOM.taskFilterBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            DOM.taskFilterBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentTaskFilter = btn.getAttribute('data-filter');
            loadTasks();
        });
    });

    // Settings & Auth Actions
    DOM.settingsForm.addEventListener('submit', saveTelegramSettings);
    DOM.authSendCodeBtn.addEventListener('click', sendTelegramAuthCode);
    DOM.authVerifyCodeBtn.addEventListener('click', verifyTelegramAuthCode);
}

// System Status Checks (Database & General API availability)
async function performSystemDiagnostics() {
    try {
        const start = performance.now();
        // Check health ready endpoint under API v1
        const response = await fetch('/api/v1/health/ready');
        const duration = Math.round(performance.now() - start);
        DOM.latencyVal.innerText = `${duration} ms`;

        if (response.ok) {
            const data = await response.json();
            DOM.apiStatusDot.className = 'stat-val online';
            DOM.apiStatusDot.innerText = '●';
            
            // Check database status in checks
            const dbCheck = data.checks && data.checks.database;
            if (dbCheck && dbCheck.status === 'ok') {
                DOM.dbStatusDot.className = 'stat-val online';
                DOM.dbStatusDot.innerText = '●';
            } else {
                DOM.dbStatusDot.className = 'stat-val';
                DOM.dbStatusDot.innerText = '○';
            }
        } else {
            DOM.apiStatusDot.className = 'stat-val';
            DOM.apiStatusDot.innerText = '○';
            DOM.dbStatusDot.className = 'stat-val';
            DOM.dbStatusDot.innerText = '○';
        }
    } catch (e) {
        DOM.apiStatusDot.className = 'stat-val';
        DOM.apiStatusDot.innerText = '○';
        DOM.dbStatusDot.className = 'stat-val';
        DOM.dbStatusDot.innerText = '○';
    }

    // Check Telegram
    await checkTelegramStatus();
}

// Initialise Application
function init() {
    setupEvents();
    connectWebSocket();
    
    // First Diagnostic Run
    performSystemDiagnostics();
    // Diagnostic loop
    setInterval(performSystemDiagnostics, 10000);
}

// Run on Load
window.addEventListener('DOMContentLoaded', init);
