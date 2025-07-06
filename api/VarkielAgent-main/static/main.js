document.addEventListener('DOMContentLoaded', function() {
    const processBtn = document.getElementById('processBtn');
    const inputText = document.getElementById('inputText');
    const minVal = document.getElementById('minVal');
    const maxVal = document.getElementById('maxVal');
    const normVal = document.getElementById('normVal');
    
    // Initialize chart
    const ctx = document.getElementById('outputChart').getContext('2d');
    const chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Output Vector',
                data: [],
                borderColor: '#4e54c8',
                tension: 0.4,
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { min: -1, max: 1 }
            }
        }
    });
    
    processBtn.addEventListener('click', async function() {
        const text = inputText.value.trim();
        if (!text) return;
        
        try {
            const response = await fetch('/process', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ input: text })
            });
            
            const result = await response.json();
            minVal.textContent = result.min_val.toFixed(4);
            maxVal.textContent = result.max_val.toFixed(4);
            normVal.textContent = result.norm.toFixed(4);
            
            // Update chart
            chart.data.labels = Array.from({length: result.output_vector.length}, (_, i) => i);
            chart.data.datasets[0].data = result.output_vector;
            chart.update();
            
        } catch (error) {
            console.error('Error:', error);
        }
    });
});
