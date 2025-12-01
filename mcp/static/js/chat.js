const uploadBtn = document.getElementById('uploadBtn');
const clearBtn = document.getElementById('clearBtn');
const fileInput = document.getElementById('fileInput');
const uploadForm = document.getElementById('uploadForm');
const clearForm = document.getElementById('clearForm');
const chatForm = document.getElementById('chatForm');
const messageInput = document.getElementById('messageInput');
const messagesContainer = document.getElementById('messages');

if (uploadBtn) {
    uploadBtn.addEventListener('click', () => fileInput.click());
}

if (clearBtn) {
    clearBtn.addEventListener('click', () => {
        if (confirm('Clear all chat history?')) {
            clearForm.submit();
        }
    });
}

if (fileInput) {
    fileInput.addEventListener('change', async () => {
        const formData = new FormData(uploadForm);
        await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        location.reload();
    });
}

if (chatForm) {
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const message = messageInput.value.trim();
        if (!message) return;

        // Immediately show user message
        addMessage('user', message);
        messageInput.value = '';

        // Show loading indicator
        const loadingId = addLoading();

        try {
            // Send message to backend
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message })
            });

            const result = await response.json();

            // Remove loading
            removeLoading(loadingId);

            // Show assistant response
            if (result.type === 'error') {
                addMessage('error', result.text);
            } else {
                addMessage('assistant', result.text, result.images || result.image);
            }
        } catch (error) {
            removeLoading(loadingId);
            addMessage('error', `Error: ${error.message}`);
        }
    });
}

function addMessage(type, text, imageData = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    // Parse markdown for assistant and error messages, keep user messages as plain text
    if (type === 'assistant' || type === 'error') {
        contentDiv.innerHTML = marked.parse(text);
    } else {
        contentDiv.textContent = text;
    }

    messageDiv.appendChild(contentDiv);

    const images = Array.isArray(imageData)
        ? imageData.filter(Boolean)
        : imageData ? [imageData] : [];

    images.forEach((imgData) => {
        const img = document.createElement('img');
        img.className = 'message-image';
        img.src = `data:image/png;base64,${imgData}`;
        img.alt = 'Graph';
        messageDiv.appendChild(img);
    });

    messagesContainer.appendChild(messageDiv);
    scrollToBottom();
}

function addLoading() {
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message assistant';
    loadingDiv.id = `loading-${Date.now()}`;

    const loadingContent = document.createElement('div');
    loadingContent.className = 'loading';
    loadingContent.innerHTML = '<span></span><span></span><span></span>';

    loadingDiv.appendChild(loadingContent);
    messagesContainer.appendChild(loadingDiv);
    scrollToBottom();

    return loadingDiv.id;
}

function removeLoading(loadingId) {
    const loadingDiv = document.getElementById(loadingId);
    if (loadingDiv) {
        loadingDiv.remove();
    }
}

function scrollToBottom() {
    if (messagesContainer) {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
}

// Auto-scroll to bottom on page load
scrollToBottom();

// Log selector functionality
const logSelector = document.getElementById('logSelector');
const loadLogBtn = document.getElementById('loadLogBtn');
const localLogSelector = document.getElementById('localLogSelector');
const loadLocalLogBtn = document.getElementById('loadLocalLogBtn');

// Fetch available logs from S3 on page load
async function fetchLogs() {
    try {
        const response = await fetch('/api/logs');
        if (!response.ok) {
            console.error('Failed to fetch logs');
            return;
        }

        const data = await response.json();
        const logs = data.logs || [];

        // Clear existing options (except the first placeholder)
        logSelector.innerHTML = '<option value="">S3 logs...</option>';

        // Add log options
        logs.forEach(log => {
            const option = document.createElement('option');
            option.value = log.name;

            // Format the display text with name and date
            const date = log.upload_time ? new Date(log.upload_time).toLocaleString() : 'Unknown date';
            const size = log.size ? formatBytes(log.size) : '';
            option.textContent = `${log.name} (${date}${size ? ', ' + size : ''})`;

            logSelector.appendChild(option);
        });

        console.log(`Loaded ${logs.length} logs from S3`);
    } catch (error) {
        console.error('Error fetching logs:', error);
    }
}

// Fetch available local logs on page load
async function fetchLocalLogs() {
    try {
        const response = await fetch('/api/local-logs');
        if (!response.ok) {
            console.error('Failed to fetch local logs');
            return;
        }

        const data = await response.json();
        const logGroups = data.logs || [];

        // Clear existing options (except the first placeholder)
        localLogSelector.innerHTML = '<option value="">Local logs...</option>';

        // Add log options grouped by folder
        logGroups.forEach(group => {
            const optgroup = document.createElement('optgroup');
            optgroup.label = group.folder;

            group.files.forEach(file => {
                const option = document.createElement('option');
                option.value = file.path;

                // Format the display text with name, date, and size
                const date = file.modified ? new Date(file.modified).toLocaleDateString() : '';
                const size = file.size ? formatBytes(file.size) : '';
                option.textContent = `${file.name}${date ? ' (' + date : ''}${size ? ', ' + size : ''}${date ? ')' : ''}`;

                optgroup.appendChild(option);
            });

            localLogSelector.appendChild(optgroup);
        });

        console.log(`Loaded ${logGroups.length} log folders from local storage`);
    } catch (error) {
        console.error('Error fetching local logs:', error);
    }
}

// Format bytes to human-readable size
function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

// Handle load log button click (S3)
if (loadLogBtn) {
    loadLogBtn.addEventListener('click', async () => {
        const selectedLog = logSelector.value;

        if (!selectedLog) {
            alert('Please select a log file first');
            return;
        }

        // Disable button during loading
        loadLogBtn.disabled = true;
        loadLogBtn.textContent = 'Loading...';

        try {
            const response = await fetch('/api/logs/select', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ filename: selectedLog })
            });

            if (response.ok) {
                // Reload page to show the loaded file
                location.reload();
            } else {
                const error = await response.json();
                alert(`Failed to load log: ${error.error || 'Unknown error'}`);
                loadLogBtn.disabled = false;
                loadLogBtn.textContent = 'Load from S3';
            }
        } catch (error) {
            console.error('Error loading log:', error);
            alert('Failed to load log file');
            loadLogBtn.disabled = false;
            loadLogBtn.textContent = 'Load from S3';
        }
    });
}

// Handle load local log button click
if (loadLocalLogBtn) {
    loadLocalLogBtn.addEventListener('click', async () => {
        const selectedLog = localLogSelector.value;

        if (!selectedLog) {
            alert('Please select a local log file first');
            return;
        }

        // Disable button during loading
        loadLocalLogBtn.disabled = true;
        loadLocalLogBtn.textContent = 'Loading...';

        try {
            const response = await fetch('/api/local-logs/select', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ filepath: selectedLog })
            });

            if (response.ok) {
                // Reload page to show the loaded file
                location.reload();
            } else {
                const error = await response.json();
                alert(`Failed to load local log: ${error.error || 'Unknown error'}`);
                loadLocalLogBtn.disabled = false;
                loadLocalLogBtn.textContent = 'Load Local';
            }
        } catch (error) {
            console.error('Error loading local log:', error);
            alert('Failed to load local log file');
            loadLocalLogBtn.disabled = false;
            loadLocalLogBtn.textContent = 'Load Local';
        }
    });
}

// Fetch logs on page load
fetchLogs();
fetchLocalLogs();

