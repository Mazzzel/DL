const ctx = document.getElementById('costChart').getContext('2d');
const costChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Log Loss',
            data: [],
            backgroundColor: 'rgba(255,99,132,0.2)',
            borderColor: 'rgba(255,99,132,1)',
            borderWidth: 2,
            fill: true,
            tension: 0.3
        }]
    },
    options: {
        responsive: true,
        plugins: { legend: { display: true } },
        scales: {
            x: { title: { display: true, text: 'Epochs' } },
            y: { title: { display: true, text: 'Loss' } }
        }
    }
});

document.getElementById('trainBtn').addEventListener('click', () => {
    fetch('/train', {method: 'POST'})
        .then(res => res.json())
        .then(data => {
            costChart.data.labels = data.map((_, i) => i + 1);
            costChart.data.datasets[0].data = data;
            costChart.update();
        });
});

document.getElementById('irisForm').addEventListener('submit', e=>{
    e.preventDefault();
    const form = e.target;
    const data = [
        [
            parseFloat(form.sepal_length.value),
            parseFloat(form.sepal_width.value),
            parseFloat(form.petal_length.value),
            parseFloat(form.petal_width.value)
        ]
    ]; // 4x1 pour 1 échantillon
    fetch('/predict',{
        method:'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify(data)
    })
        .then(res=>res.json())
        .then(pred=>{
            document.getElementById('predResult').innerText =
                `Setosa: ${pred[0][0].toFixed(2)}, Versicolor: ${pred[0][1].toFixed(2)}, Virginica: ${pred[0][2].toFixed(2)}`;
        });
});

const hiddenCountInput = document.getElementById("hiddenCount");
const hiddenContainer = document.getElementById("hiddenLayersContainer");

hiddenCountInput.addEventListener("input", () => {
    const count = parseInt(hiddenCountInput.value) || 0;

    hiddenContainer.innerHTML = "";

    for (let i = 0; i < count; i++) {
        const div = document.createElement("div");
        div.style.display = "flex";
        div.style.alignItems = "center";
        div.style.gap = "10px";
        const label = document.createElement("label");
        label.textContent = `Neurones couche cachée ${i+1} :`;
        label.htmlFor = `hidden_layer_${i+1}`;

        const input = document.createElement("input");
        input.type = "number";
        input.step = "1";
        input.name = `hidden_layer_${i+1}`;
        input.placeholder = "Ex: 10";
        input.id = `hidden_layer_${i+1}`;

        div.appendChild(label);
        div.appendChild(input);
        hiddenContainer.appendChild(div);
    }
});


document.getElementById("setLayer").addEventListener("submit", async (e) => {
    e.preventDefault();

    const formData = new FormData(e.target);
    const data = {
        input_layer: parseInt(formData.get("input_layer")),
        output_layer: parseInt(formData.get("output_layer")),
        hidden_layers: []
    };

    // récupère dynamiquement toutes les couches cachées
    formData.forEach((value, key) => {
        if (key.startsWith("hidden_layer_")) {
            data.hidden_layers.push(parseInt(value));
        }
    });

    const response = await fetch("/createNN", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    });

    const text = await response.text();
    console.log(text);

    try {
        const result = JSON.parse(text);
        console.log(result);
    } catch (err) {
        console.error("Erreur JSON :", err);
    }
});