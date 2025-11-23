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
    contentDiv.textContent = text;

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
