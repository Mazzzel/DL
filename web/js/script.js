// === Gestion dynamique des couches cachées ===
const hiddenCountInput = document.getElementById("hiddenCount");
const hiddenContainer = document.getElementById("hiddenLayersContainer");

hiddenCountInput.addEventListener("input", () => {
    const count = parseInt(hiddenCountInput.value) || 0;
    hiddenContainer.innerHTML = "";

    for (let i = 0; i < count; i++) {
        const div = document.createElement("div");
        div.className = "input-group";

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

// === Variables principales ===
const setLayerForm = document.getElementById('setLayer');
const trainBtn = document.getElementById('trainBtn');
const predictBtn = document.getElementById('predictBtn');
const saveBtn = document.getElementById("saveNetworkBtn");
const networkNameInput = document.getElementById("networkName");

// --- Créer réseau ---

// === Génération dynamique des inputs de prédiction ===
let currentInputCount = 0;

function generatePredictionInputs(count) {
    const container = document.getElementById('predictionInputs');
    container.innerHTML = '';
    currentInputCount = count;

    if (count === 0) {
        container.innerHTML = '<p style="text-align: center; color: #9ca3af; grid-column: 1/-1;">"Créez d\'abord un réseau ou chargez des données"</p>';
        return;
    }

    for (let i = 0; i < count; i++) {
        const input = document.createElement('input');
        input.type = 'number';
        input.step = '0.01';
        input.placeholder = `Feature X${i + 1}`;
        input.className = 'prediction-input';
        input.required = true;
        container.appendChild(input);
    }
}

setLayerForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const inputLayer = parseInt(document.getElementById('inputLayer').value);
    const outputLayer = parseInt(document.getElementById('outputLayer').value);
    const hiddenLayers = Array.from(document.querySelectorAll('#hiddenLayersContainer input')).map(i => parseInt(i.value));

    await fetch('/createNN', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            input_layer: inputLayer,
            output_layer: outputLayer,
            hidden_layers: hiddenLayers
        })
    });

    // Générer les inputs de prédiction
    generatePredictionInputs(inputLayer);

    trainBtn.disabled = false;
    predictBtn.disabled = true;
});

// --- Entraînement ---
let costChart = null;
let decisionChart = null;

trainBtn.addEventListener('click', async () => {
    trainBtn.disabled = true;

    // Détruire les anciens graphiques s'ils existent
    if (costChart) {
        costChart.destroy();
        costChart = null;
    }
    if (decisionChart) {
        decisionChart.destroy();
        decisionChart = null;
    }

    // Appel unique au serveur pour l'entraînement
    try {
        const res = await fetch('/train', { method: 'POST' });
        const data = await res.json();

        console.log('Données reçues du serveur:', data);

        // Vérifier que les données sont présentes
        if (!data.costs || !data.training_data || !data.grid_predictions) {
            console.error('Données manquantes:', data);
            alert('Erreur: données incomplètes du serveur');
            trainBtn.disabled = false;
            return;
        }

        // Initialiser le graphique de coût
        const ctx1 = document.getElementById('costChart').getContext('2d');
        costChart = new Chart(ctx1, {
            type: 'line',
            data: {
                labels: data.costs.map((_, i) => (i * 200)),
                datasets: [{
                    label: 'Log Loss',
                    data: data.costs,
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

        // Créer le graphique de frontière de décision
        createDecisionBoundary(data);

        predictBtn.disabled = false;
    } catch (error) {
        console.error('Erreur lors de l\'entraînement:', error);
        alert('Erreur lors de l\'entraînement: ' + error.message);
        trainBtn.disabled = false;
    }
});

// --- Création de la frontière de décision ---
function createDecisionBoundary(data) {
    console.log('Création du graphique de décision...');
    console.log('Training data length:', data.training_data.length);
    console.log('Grid predictions length:', data.grid_predictions.length);
    console.log('Limites X:', data.x_min, '-', data.x_max);
    console.log('Limites Y:', data.y_min, '-', data.y_max);

    const ctx = document.getElementById('decisionChart');
    if (!ctx) {
        console.error('Canvas decisionChart non trouvé!');
        return;
    }

    const ctx2d = ctx.getContext('2d');
    const trainingData = data.training_data;
    const gridPredictions = data.grid_predictions;

    // Préparer les datasets pour les 3 classes
    const setosaPoints = [];
    const versicolorPoints = [];
    const virginicaPoints = [];

    // Séparer les points d'entraînement par classe
    trainingData.forEach(point => {
        const dataPoint = { x: point.x, y: point.y };
        if (point.class === 0) setosaPoints.push(dataPoint);
        else if (point.class === 1) versicolorPoints.push(dataPoint);
        else if (point.class === 2) virginicaPoints.push(dataPoint);
    });

    console.log('Setosa:', setosaPoints.length, 'Versicolor:', versicolorPoints.length, 'Virginica:', virginicaPoints.length);

    // Préparer les points de la grille de prédiction
    const gridSetosa = [];
    const gridVersicolor = [];
    const gridVirginica = [];

    gridPredictions.forEach(point => {
        const dataPoint = { x: point.x, y: point.y };
        if (point.predicted === 0) gridSetosa.push(dataPoint);
        else if (point.predicted === 1) gridVersicolor.push(dataPoint);
        else if (point.predicted === 2) gridVirginica.push(dataPoint);
    });

    console.log('Grille Setosa:', gridSetosa.length, 'Versicolor:', gridVersicolor.length, 'Virginica:', gridVirginica.length);

    decisionChart = new Chart(ctx2d, {
        type: 'scatter',
        data: {
            datasets: [
                // Grille de fond (frontière de décision)
                {
                    label: 'Région Setosa',
                    data: gridSetosa,
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    pointRadius: 5,
                    pointHoverRadius: 5,
                    showLine: false,
                    order: 3
                },
                {
                    label: 'Région Versicolor',
                    data: gridVersicolor,
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    pointRadius: 5,
                    pointHoverRadius: 5,
                    showLine: false,
                    order: 3
                },
                {
                    label: 'Région Virginica',
                    data: gridVirginica,
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    pointRadius: 5,
                    pointHoverRadius: 5,
                    showLine: false,
                    order: 3
                },
                // Points d'entraînement
                {
                    label: 'Setosa (données)',
                    data: setosaPoints,
                    backgroundColor: 'rgba(255, 99, 132, 1)',
                    borderColor: 'rgba(255, 255, 255, 1)',
                    borderWidth: 3,
                    pointRadius: 10,
                    pointHoverRadius: 12,
                    showLine: false,
                    order: 1
                },
                {
                    label: 'Versicolor (données)',
                    data: versicolorPoints,
                    backgroundColor: 'rgba(54, 162, 235, 1)',
                    borderColor: 'rgba(255, 255, 255, 1)',
                    borderWidth: 3,
                    pointRadius: 10,
                    pointHoverRadius: 12,
                    showLine: false,
                    order: 1
                },
                {
                    label: 'Virginica (données)',
                    data: virginicaPoints,
                    backgroundColor: 'rgba(75, 192, 192, 1)',
                    borderColor: 'rgba(255, 255, 255, 1)',
                    borderWidth: 3,
                    pointRadius: 10,
                    pointHoverRadius: 12,
                    showLine: false,
                    order: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                title: {
                    display: true,
                    text: 'Frontière de décision (Longueur vs Largeur pétale)',
                    font: { size: 16 }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: 'Longueur pétale (normalisée)',
                        font: { size: 14 }
                    },
                    min: data.x_min,
                    max: data.x_max
                },
                y: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: 'Largeur pétale (normalisée)',
                        font: { size: 14 }
                    },
                    min: data.y_min,
                    max: data.y_max
                }
            }
        }
    });

    console.log('Graphique de décision créé avec succès');
}

// --- Décodage Iris ---
const predictionForm = document.getElementById('predictionForm');

predictionForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const inputs = predictionForm.querySelectorAll('.prediction-input');

    if (inputs.length === 0) {
        alert('Aucune donnée à prédire');
        return;
    }

    const X_input = [[...inputs].map(input => parseFloat(input.value))];

    const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(X_input)
    });

    const result = await res.json();

    // Afficher les résultats
    const predResultDiv = document.getElementById('predResult');
    let resultHTML = '<strong>Prédiction:</strong><br>';

    result[0].forEach((val, idx) => {
        const percentage = (val * 100).toFixed(1);
        resultHTML += `Classe ${idx + 1}: ${percentage}%<br>`;
    });

    predResultDiv.innerHTML = resultHTML;
});

// --- Sauvegarder réseau ---
saveBtn.addEventListener("click", async () => {
    const name = networkNameInput.value.trim();
    if (!name) return alert("Donne un nom à ton réseau !");
    await fetch("/saveNN", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name })
    });
    alert("Réseau sauvegardé !");
    loadNetworks();
});

// === Charger la liste des réseaux ===
async function loadNetworks() {
    const res = await fetch("/listNN");
    const networks = await res.json();

    const networkListContainer = document.getElementById('networkList');
    networkListContainer.innerHTML = '';

    if (networks.length === 0) {
        networkListContainer.innerHTML = '<div class="empty-state">Aucun réseau sauvegardé</div>';
        return;
    }

    networks.forEach(networkName => {
        const div = document.createElement('div');
        div.className = 'network-item';
        div.dataset.network = networkName;

        const nameSpan = document.createElement('span');
        nameSpan.className = 'network-name';
        nameSpan.textContent = networkName;

        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'network-actions';

        const loadBtn = document.createElement('button');
        loadBtn.className = 'btn-load-single';
        loadBtn.textContent = '📂 Charger';
        loadBtn.title = 'Charger ce réseau';
        loadBtn.onclick = async (e) => {
            e.stopPropagation();
            await loadNetwork(networkName);
        };

        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'btn-delete';
        deleteBtn.textContent = '🗑️';
        deleteBtn.title = 'Supprimer ce réseau';
        deleteBtn.onclick = async (e) => {
            e.stopPropagation();
            if (confirm(`Supprimer le réseau "${networkName}" ?`)) {
                await deleteNetwork(networkName);
            }
        };

        actionsDiv.appendChild(loadBtn);
        actionsDiv.appendChild(deleteBtn);

        div.appendChild(nameSpan);
        div.appendChild(actionsDiv);

        // Clic sur l'élément pour le sélectionner
        div.addEventListener('click', () => {
            document.querySelectorAll('.network-item').forEach(item => {
                item.classList.remove('selected');
            });
            div.classList.add('selected');
            selectedNetwork = networkName;
        });

        networkListContainer.appendChild(div);
    });
}

// --- Supprimer un réseau ---
async function deleteNetwork(name) {
    try {
        const res = await fetch('/deleteNetwork', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name })
        });

        if (res.ok) {
            alert(`Réseau "${name}" supprimé !`);
            loadNetworks();
        } else {
            alert('Erreur lors de la suppression');
        }
    } catch (error) {
        console.error('Erreur:', error);
        alert('Erreur lors de la suppression');
    }
}

// Gestion du menu
const menuToggle = document.getElementById('menuToggle');
const sidebar = document.getElementById('networkSidebar');
const overlay = document.getElementById('overlay');
const closeSidebar = document.getElementById('closeSidebar');

function openMenu() {
    sidebar.classList.add('active');
    overlay.classList.add('active');
    document.body.style.overflow = 'hidden';
}

function closeMenu() {
    sidebar.classList.remove('active');
    overlay.classList.remove('active');
    document.body.style.overflow = '';
}

menuToggle.addEventListener('click', openMenu);
closeSidebar.addEventListener('click', closeMenu);
overlay.addEventListener('click', closeMenu);

// === Upload CSV ===
const csvFileInput = document.getElementById('csvFile');
const uploadBtn = document.getElementById('uploadBtn');
const fileNameSpan = document.getElementById('fileName');
const dataInfo = document.getElementById('dataInfo');

csvFileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        fileNameSpan.textContent = file.name;
        uploadBtn.disabled = false;
    } else {
        fileNameSpan.textContent = 'Aucun fichier sélectionné';
        uploadBtn.disabled = true;
    }
});

uploadBtn.addEventListener('click', async () => {
    const file = csvFileInput.files[0];
    if (!file) return;

    uploadBtn.disabled = true;
    uploadBtn.textContent = '⏳ Chargement...';
    dataInfo.classList.remove('visible', 'error');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/uploadCSV', {
            method: 'POST',
            body: formData
        });

        const text = await res.text();
        console.log("Raw response:", text);

        let result;
        try {
            result = JSON.parse(text);
        } catch (e) {
            throw new Error("Réponse serveur invalide JSON");
        }

        if (res.ok) {
            dataInfo.classList.add('visible');
            dataInfo.classList.remove('error');
            dataInfo.innerHTML = `
                ✅ <strong>Données chargées avec succès !</strong><br>
                📊 Features (X) : ${result.x_rows} × ${result.x_cols}<br>
                🎯 Labels (Y) : ${result.y_rows} × ${result.y_cols}
            `;

            // Générer les inputs de prédiction
            generatePredictionInputs(result.x_rows);

            trainBtn.disabled = false;
        } else {
            throw new Error(result.error || 'Erreur inconnue');
        }

    } catch (error) {
        console.error('Erreur upload CSV:', error);
        dataInfo.classList.add('visible', 'error');
        dataInfo.textContent = `❌ Erreur : ${error.message}`;
    } finally {
        uploadBtn.disabled = false;
        uploadBtn.textContent = '⬆️ Charger les données';
    }
});

// --- Initialisation ---
loadNetworks();