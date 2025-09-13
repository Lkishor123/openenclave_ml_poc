document.getElementById('send-button').addEventListener('click', async () => {
    const inputArea = document.getElementById('prompt-input');
    const outputArea = document.getElementById('output-area');
    const inputText = inputArea.value;

    if (!inputText) {
        outputArea.textContent = 'Error: Input cannot be empty.';
        return;
    }

    outputArea.textContent = 'Analyzing sentiment...';

    try {
        const response = await fetch('/infer', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ input: inputText }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Backend error: ${response.status} ${errorText}`);
        }

        const result = await response.json();

        // MODIFIED: Display the final sentiment from the new 'sentiment' field.
        if (result.sentiment) {
            outputArea.textContent = `Predicted Sentiment: ${result.sentiment}`;
        } else if (result.error) {
            outputArea.textContent = `Error: ${result.error}`;
        } else {
            // Fallback for unexpected response format
            outputArea.textContent = JSON.stringify(result, null, 2);
        }

    } catch (error) {
        console.error('Inference error:', error);
        outputArea.textContent = `Error: ${error.message}`;
    }
});
