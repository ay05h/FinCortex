document.addEventListener('DOMContentLoaded', function() {
    const inputText = document.getElementById('input-text');
    const analyzeBtn = document.getElementById('analyze-btn');
    const loading = document.getElementById('loading');
    const result = document.getElementById('result');
    const error = document.getElementById('error');
    const errorMessage = document.querySelector('.error-message');
    const causesList = document.getElementById('causes-list');
    const effectsList = document.getElementById('effects-list');

    // Add input animations
    inputText.addEventListener('focus', function() {
        this.parentElement.classList.add('focused');
    });

    inputText.addEventListener('blur', function() {
        this.parentElement.classList.remove('focused');
    });

    // Add hover effect to analyze button
    analyzeBtn.addEventListener('mouseenter', function() {
        this.classList.add('hover');
    });

    analyzeBtn.addEventListener('mouseleave', function() {
        this.classList.remove('hover');
    });

    // Function to show loading state
    function showLoading() {
        loading.style.display = 'block';
        result.style.display = 'none';
        error.style.display = 'none';
        
        // Add animation to spinner
        const spinner = document.querySelector('.spinner');
        spinner.style.animation = 'spin 1s linear infinite';
    }

    // Function to show error
    function showError(message) {
        loading.style.display = 'none';
        result.style.display = 'none';
        error.style.display = 'flex';
        errorMessage.textContent = message;
        
        // Add shake animation
        error.style.animation = 'shake 0.5s ease-out';
    }

    // Function to update results
    function updateResults(data) {
        loading.style.display = 'none';
        result.style.display = 'block';
        error.style.display = 'none';
        
        // Clear previous lists
        causesList.innerHTML = '';
        effectsList.innerHTML = '';
        
        // Add causes with animation delay
        if (data.causes && data.causes.length > 0) {
            data.causes.forEach((cause, index) => {
                setTimeout(() => {
                    const li = document.createElement('li');
                    li.innerHTML = `
                        <span class="cause-text">${cause.text}</span>
                        <span class="confidence-badge">${(cause.confidence * 100).toFixed(1)}%</span>
                    `;
                    causesList.appendChild(li);
                }, 100 * index);
            });
        } else {
            const li = document.createElement('li');
            li.innerHTML = '<span class="no-results">No causes found</span>';
            causesList.appendChild(li);
        }
        
        // Add effects with animation delay
        if (data.effects && data.effects.length > 0) {
            data.effects.forEach((effect, index) => {
                setTimeout(() => {
                    const li = document.createElement('li');
                    li.innerHTML = `
                        <span class="effect-text">${effect.text}</span>
                        <span class="confidence-badge">${(effect.confidence * 100).toFixed(1)}%</span>
                    `;
                    effectsList.appendChild(li);
                }, 100 * index);
            });
        } else {
            const li = document.createElement('li');
            li.innerHTML = '<span class="no-results">No effects found</span>';
            effectsList.appendChild(li);
        }
    }

    // Handle form submission
    function handleSubmit() {
        const text = inputText.value.trim();
        
        if (!text) {
            showError('Please enter some text to analyze');
            return;
        }
        
        showLoading();
        
        fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text }),
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                showError(data.error);
            } else {
                updateResults(data);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showError('An error occurred while analyzing the text. Please try again.');
        });
    }

    // Add event listeners
    analyzeBtn.addEventListener('click', handleSubmit);
    
    // Add keyboard shortcut (Ctrl+Enter)
    inputText.addEventListener('keydown', function(e) {
        if (e.ctrlKey && e.key === 'Enter') {
            handleSubmit();
        }
    });
    
    // Add copy to clipboard functionality
    const copyBtn = document.createElement('button');
    copyBtn.className = 'copy-btn';
    copyBtn.innerHTML = '<i class="fas fa-copy"></i>';
    copyBtn.title = 'Copy results to clipboard';
    copyBtn.addEventListener('click', function() {
        const resultText = `
Causes:
${Array.from(causesList.children).map(li => li.textContent).join('\n')}

Effects:
${Array.from(effectsList.children).map(li => li.textContent).join('\n')}
        `;
        
        navigator.clipboard.writeText(resultText).then(() => {
            const originalText = this.innerHTML;
            this.innerHTML = '<i class="fas fa-check"></i>';
            this.title = 'Copied!';
            
            setTimeout(() => {
                this.innerHTML = originalText;
                this.title = 'Copy results to clipboard';
            }, 2000);
        });
    });
    
    result.appendChild(copyBtn);
});