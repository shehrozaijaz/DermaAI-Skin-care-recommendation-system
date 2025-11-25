document.addEventListener('DOMContentLoaded', () => {
    const chatContainer = document.getElementById('chat-container');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const modal = document.getElementById('explanation-modal');
    const closeModal = document.querySelector('.close-modal');

    let lastQuery = '';

    function appendMessage(text, isUser, isUrgent = false, showExplain = false) {
        const div = document.createElement('div');
        div.className = `message ${isUser ? 'user-message' : 'bot-message'} ${isUrgent ? 'urgent' : ''}`;

        // Convert newlines to breaks for bot messages
        if (!isUser) {
            text = text.replace(/\\n/g, '<br>').replace(/\n/g, '<br>');
        }

        div.innerHTML = text;

        // Add explain button for bot messages
        if (!isUser && showExplain) {
            const explainBtn = document.createElement('button');
            explainBtn.className = 'explain-btn';
            explainBtn.textContent = '🔍 Explain';
            explainBtn.onclick = () => showExplanation(lastQuery);
            div.appendChild(explainBtn);
        }

        chatContainer.appendChild(div);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function showTyping() {
        const div = document.createElement('div');
        div.className = 'typing-indicator';
        div.id = 'typing';
        div.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';
        chatContainer.appendChild(div);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function removeTyping() {
        const typing = document.getElementById('typing');
        if (typing) typing.remove();
    }

    async function sendMessage() {
        const text = userInput.value.trim();
        if (!text) return;

        // User Message
        appendMessage(text, true);
        lastQuery = text;
        userInput.value = '';

        // Loading
        showTyping();

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query: text })
            });

            const data = await response.json();
            removeTyping();

            if (data.error) {
                appendMessage("Sorry, I encountered an error. Please try again.", false);
            } else {
                const isUrgent = data.severity === 'urgent' || data.severity === 'requires_medical';
                appendMessage(data.advice, false, isUrgent, true);
            }

        } catch (error) {
            removeTyping();
            appendMessage("Error connecting to the server. Please check if the backend is running.", false);
            console.error(error);
        }
    }

    async function showExplanation(query) {
        if (!query) return;

        modal.style.display = 'block';
        document.getElementById('token-importance').innerHTML = '<p>Loading...</p>';
        document.getElementById('retrieval-explanation').innerHTML = '<p>Loading...</p>';
        document.getElementById('confidence-breakdown').innerHTML = '<p>Loading...</p>';

        try {
            const response = await fetch('/explain', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query: query })
            });

            const data = await response.json();

            // Render NER explanation
            renderTokenImportance(data.ner_explanation);

            // Render retrieval explanation
            renderRetrievalExplanation(data.retrieval_explanation);

            // Render confidence breakdown
            renderConfidenceBreakdown(data.confidence_breakdown);

        } catch (error) {
            console.error('Explanation error:', error);
            document.getElementById('token-importance').innerHTML = '<p>Error loading explanation</p>';
        }
    }

    function renderTokenImportance(nerExplanation) {
        const container = document.getElementById('token-importance');

        if (!nerExplanation || !nerExplanation.tokens) {
            container.innerHTML = '<p>No token data available</p>';
            return;
        }

        let html = '<div>';
        nerExplanation.tokens.forEach(token => {
            let importanceClass = 'low-importance';
            if (token.importance > 0.7) importanceClass = 'high-importance';
            else if (token.importance > 0.4) importanceClass = 'medium-importance';

            html += `<span class="token ${importanceClass}" title="Importance: ${token.importance.toFixed(2)}, Tag: ${token.predicted_tag}">${token.word}</span>`;
        });
        html += '</div>';
        html += `<p style="margin-top: 10px; font-size: 0.9rem; color: #666;">Overall confidence: ${(nerExplanation.overall_confidence * 100).toFixed(1)}%</p>`;

        container.innerHTML = html;
    }

    function renderRetrievalExplanation(retrievalExplanation) {
        const container = document.getElementById('retrieval-explanation');

        if (!retrievalExplanation || retrievalExplanation.length === 0) {
            container.innerHTML = '<p>No retrieval data available</p>';
            return;
        }

        let html = '';
        retrievalExplanation.forEach((item, idx) => {
            html += `
                <div class="retrieval-item">
                    <strong>Document ${idx + 1}</strong> (Score: ${(item.score * 100).toFixed(1)}%)
                    <p>${item.explanation}</p>
                    ${item.overlapping_keywords.length > 0 ?
                    `<p style="font-size: 0.85rem; color: #666;">Keywords: ${item.overlapping_keywords.join(', ')}</p>`
                    : ''}
                </div>
            `;
        });

        container.innerHTML = html;
    }

    function renderConfidenceBreakdown(breakdown) {
        const container = document.getElementById('confidence-breakdown');

        if (!breakdown) {
            container.innerHTML = '<p>No confidence data available</p>';
            return;
        }

        const html = `
            <div>
                <p><strong>Overall Confidence:</strong> ${(breakdown.overall * 100).toFixed(1)}%</p>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${breakdown.overall * 100}%"></div>
                </div>
                
                <p style="margin-top: 10px;"><strong>Top Match:</strong> ${(breakdown.top_match * 100).toFixed(1)}%</p>
                <p><strong>Average Match:</strong> ${(breakdown.average_match * 100).toFixed(1)}%</p>
                <p><strong>Consistency:</strong> ${(breakdown.consistency * 100).toFixed(1)}%</p>
                
                <p style="margin-top: 15px; padding: 10px; background: #f0f2f5; border-radius: 8px; font-style: italic;">
                    ${breakdown.interpretation}
                </p>
            </div>
        `;

        container.innerHTML = html;
    }

    // Modal controls
    closeModal.onclick = () => {
        modal.style.display = 'none';
    };

    window.onclick = (event) => {
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    };

    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
});
