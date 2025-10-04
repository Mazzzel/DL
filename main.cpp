#include <iostream>
#include <httplib.h>
#include <fstream>
#include <json.hpp>
#include "NN.h"
#include "saveData.h"
using json = nlohmann::json;

std::vector<Layer> nn;

Matrice X(4, 30);
Matrice Y(3, 30);

float random_float() {
    return static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
}

void init_weights(std::vector<Layer>& reseau) {
    srand(static_cast<unsigned int>(time(0)));

    for (size_t l = 0; l < reseau.size(); ++l) {
        Layer& layer = reseau[l];

        for (int i = 0; i < layer.W.getLignes(); ++i) {
            for (int j = 0; j < layer.W.getColonnes(); ++j) {
                float scale = sqrt(1.0f / layer.W.getColonnes()) * 0.5f;
                layer.W(i, j) = random_float() * scale;
            }
        }

        for (int i = 0; i < layer.b.getLignes(); ++i) {
            for (int j = 0; j < layer.b.getColonnes(); ++j) {
                layer.b(i, j) = 0.0f;
            }
        }
    }
}

void initializeDataset() {
    // Classe Setosa (10 échantillons)
    X(0,0)=0.22; X(1,0)=0.63; X(2,0)=0.07; X(3,0)=0.04;
    X(0,1)=0.17; X(1,1)=0.42; X(2,1)=0.07; X(3,1)=0.04;
    X(0,2)=0.11; X(1,2)=0.50; X(2,2)=0.05; X(3,2)=0.04;
    X(0,3)=0.08; X(1,3)=0.46; X(2,3)=0.08; X(3,3)=0.04;
    X(0,4)=0.19; X(1,4)=0.67; X(2,4)=0.07; X(3,4)=0.04;
    X(0,5)=0.31; X(1,5)=0.79; X(2,5)=0.12; X(3,5)=0.08;
    X(0,6)=0.08; X(1,6)=0.58; X(2,6)=0.07; X(3,6)=0.08;
    X(0,7)=0.19; X(1,7)=0.58; X(2,7)=0.08; X(3,7)=0.04;
    X(0,8)=0.03; X(1,8)=0.38; X(2,8)=0.07; X(3,8)=0.04;
    X(0,9)=0.17; X(1,9)=0.54; X(2,9)=0.08; X(3,9)=0.04;

    // Classe Versicolor (10 échantillons)
    X(0,10)=0.69; X(1,10)=0.42; X(2,10)=0.42; X(3,10)=0.17;
    X(0,11)=0.56; X(1,11)=0.29; X(2,11)=0.41; X(3,11)=0.13;
    X(0,12)=0.65; X(1,12)=0.42; X(2,12)=0.44; X(3,12)=0.17;
    X(0,13)=0.44; X(1,13)=0.29; X(2,13)=0.39; X(3,13)=0.17;
    X(0,14)=0.61; X(1,14)=0.42; X(2,14)=0.47; X(3,14)=0.21;
    X(0,15)=0.47; X(1,15)=0.25; X(2,15)=0.37; X(3,15)=0.13;
    X(0,16)=0.58; X(1,16)=0.42; X(2,16)=0.41; X(3,16)=0.17;
    X(0,17)=0.50; X(1,17)=0.33; X(2,17)=0.42; X(3,17)=0.21;
    X(0,18)=0.44; X(1,18)=0.21; X(2,18)=0.37; X(3,18)=0.13;
    X(0,19)=0.50; X(1,19)=0.33; X(2,19)=0.37; X(3,19)=0.17;

    // Classe Virginica (10 échantillons)
    X(0,20)=0.64; X(1,20)=0.33; X(2,20)=0.61; X(3,20)=0.25;
    X(0,21)=0.69; X(1,21)=0.33; X(2,21)=0.58; X(3,21)=0.21;
    X(0,22)=0.72; X(1,22)=0.46; X(2,22)=0.64; X(3,22)=0.25;
    X(0,23)=0.64; X(1,23)=0.42; X(2,23)=0.59; X(3,23)=0.25;
    X(0,24)=0.72; X(1,24)=0.33; X(2,24)=0.61; X(3,24)=0.25;
    X(0,25)=1.00; X(1,25)=0.42; X(2,25)=0.68; X(3,25)=0.25;
    X(0,26)=0.58; X(1,26)=0.33; X(2,26)=0.61; X(3,26)=0.25;
    X(0,27)=0.78; X(1,27)=0.38; X(2,27)=0.67; X(3,27)=0.29;
    X(0,28)=0.61; X(1,28)=0.29; X(2,28)=0.54; X(3,28)=0.21;
    X(0,29)=0.64; X(1,29)=0.42; X(2,29)=0.58; X(3,29)=0.25;

    // Matrice de sortie Y (one-hot encoding)
    for (int i = 0; i < 10; ++i) {
        Y(0,i)=1; Y(1,i)=0; Y(2,i)=0;  // Setosa
    }
    for (int i = 10; i < 20; ++i) {
        Y(0,i)=0; Y(1,i)=1; Y(2,i)=0;  // Versicolor
    }
    for (int i = 20; i < 30; ++i) {
        Y(0,i)=0; Y(1,i)=0; Y(2,i)=1;  // Virginica
    }
}

json generateDecisionBoundary(std::vector<Layer>& reseau, const Matrice& X_data, const Matrice& Y_data) {
    json result;
    result["training_data"] = json::array();
    result["grid_predictions"] = json::array();

    std::cout << "Génération de la frontière de décision..." << std::endl;
    std::cout << "X dimensions: " << X_data.getLignes() << "x" << X_data.getColonnes() << std::endl;
    std::cout << "Y dimensions: " << Y_data.getLignes() << "x" << Y_data.getColonnes() << std::endl;

    // Trouver les min/max pour les features 2 et 3 (longueur et largeur pétale)
    float min_x = X_data(2, 0), max_x = X_data(2, 0);
    float min_y = X_data(3, 0), max_y = X_data(3, 0);

    for (int i = 0; i < X_data.getColonnes(); ++i) {
        if (X_data(2, i) < min_x) min_x = X_data(2, i);
        if (X_data(2, i) > max_x) max_x = X_data(2, i);
        if (X_data(3, i) < min_y) min_y = X_data(3, i);
        if (X_data(3, i) > max_y) max_y = X_data(3, i);
    }

    std::cout << "Min X: " << min_x << ", Max X: " << max_x << std::endl;
    std::cout << "Min Y: " << min_y << ", Max Y: " << max_y << std::endl;

    // Ajouter une marge de 10%
    float margin_x = (max_x - min_x) * 0.1f;
    float margin_y = (max_y - min_y) * 0.1f;
    min_x -= margin_x;
    max_x += margin_x;
    min_y -= margin_y;
    max_y += margin_y;

    result["x_min"] = min_x;
    result["x_max"] = max_x;
    result["y_min"] = min_y;
    result["y_max"] = max_y;

    // Ajouter les données d'entraînement
    for (int i = 0; i < X_data.getColonnes(); ++i) {
        json point;
        point["x"] = X_data(2, i);
        point["y"] = X_data(3, i);

        int class_idx = 0;
        if (Y_data(1, i) == 1) class_idx = 1;
        else if (Y_data(2, i) == 1) class_idx = 2;
        point["class"] = class_idx;

        result["training_data"].push_back(point);
    }

    std::cout << "Points d'entraînement ajoutés: " << result["training_data"].size() << std::endl;

    // Créer une grille de points pour visualiser la frontière de décision
    int grid_size = 40;
    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            float x = min_x + (max_x - min_x) * i / (float)(grid_size - 1);
            float y = min_y + (max_y - min_y) * j / (float)(grid_size - 1);

            Matrice test_point(4, 1);
            test_point(0, 0) = 0.5f;
            test_point(1, 0) = 0.5f;
            test_point(2, 0) = x;
            test_point(3, 0) = y;

            Matrice prediction = predict(test_point, reseau);

            int predicted_class = 0;
            float max_val = prediction(0, 0);
            for (int k = 1; k < prediction.getLignes(); ++k) {
                if (prediction(k, 0) > max_val) {
                    max_val = prediction(k, 0);
                    predicted_class = k;
                }
            }

            json grid_point;
            grid_point["x"] = x;
            grid_point["y"] = y;
            grid_point["predicted"] = predicted_class;
            result["grid_predictions"].push_back(grid_point);
        }
    }

    std::cout << "Points de grille générés: " << result["grid_predictions"].size() << std::endl;

    return result;
}

Matrice parse_json_to_matrice(const std::string& body) {
    json j = json::parse(body);
    int rows = j[0].size();
    int cols = j.size();
    Matrice X_input(rows, cols);

    for(int c = 0; c < cols; ++c){
        for(int r = 0; r < rows; ++r){
            X_input(r, c) = j[c][r].get<float>();
        }
    }
    return X_input;
}

std::string predict_from_input(const Matrice& X_input, std::vector<Layer>& reseau){
    if(reseau.empty()){
        return "[]";
    }
    Matrice Y_pred = predict(X_input, reseau);
    std::string json_str = "[";
    for(int j=0;j<Y_pred.getColonnes();++j){
        json_str += "[";
        for(int i=0;i<Y_pred.getLignes();++i){
            json_str += std::to_string(Y_pred(i,j));
            if(i!=Y_pred.getLignes()-1) json_str+=",";
        }
        json_str += "]";
        if(j!=Y_pred.getColonnes()-1) json_str+=",";
    }
    json_str += "]";
    return json_str;
}

void createNNFromJson(std::vector<Layer>& reseau, const json& j) {
    int inputLayer = j["input_layer"];
    int outputLayer = j["output_layer"];
    std::vector<int> hiddenLayers;

    if (j.contains("hidden_layers") && !j["hidden_layers"].empty()) {
        hiddenLayers = j["hidden_layers"].get<std::vector<int>>();
    }

    reseau.clear();

    if (!hiddenLayers.empty()) {
        reseau.emplace_back(inputLayer, hiddenLayers[0]);
        for (size_t i = 0; i < hiddenLayers.size() - 1; ++i)
            reseau.emplace_back(hiddenLayers[i], hiddenLayers[i + 1]);
        reseau.emplace_back(hiddenLayers.back(), outputLayer);
    } else {
        reseau.emplace_back(inputLayer, outputLayer);
    }
}

int main() {
    httplib::Server svr;
    int port = 8080;

    // Initialiser le dataset au démarrage
    initializeDataset();
    std::cout << "Dataset initialisé" << std::endl;

    svr.set_mount_point("/css", "./web/css");
    svr.set_mount_point("/js", "./web/js");

    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        std::ifstream file("./web/index.html");
        if (!file) {
            res.status=404;
            res.set_content("index.html not found","text/plain");
            return;
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        res.set_content(buffer.str(), "text/html");
    });

    svr.Post("/createNN", [](const httplib::Request& req, httplib::Response& res) {
        auto j = json::parse(req.body);
        createNNFromJson(nn, j);
        init_weights(nn);
        std::cout << "Réseau créé avec " << nn.size() << " couches" << std::endl;
        res.set_content(R"({"status":"ok"})","application/json");
    });

    svr.Post("/train", [](const httplib::Request&, httplib::Response& res) {
        std::cout << "Entraînement lancé!" << std::endl;

        if (nn.empty()) {
            res.status = 400;
            res.set_content(R"({"error":"Réseau non créé"})", "application/json");
            return;
        }

        // Historique des coûts
        std::vector<float> cost_history;
        for (int epoch = 0; epoch < 5000; ++epoch) {
            Matrice y_pred = forward_pass(X, nn);
            backprop(X, Y, 0.5f, nn);

            if (epoch % 200 == 0) {
                float cost = y_pred.logLoss(Y);
                cost_history.push_back(cost);
                std::cout << "Epoch " << epoch << ", Loss: " << cost << std::endl;
            }
        }

        std::cout << "Entraînement terminé" << std::endl;

        // Générer le JSON avec les coûts et la frontière de décision
        json result = generateDecisionBoundary(nn, X, Y);
        result["costs"] = cost_history;

        std::cout << "JSON généré, envoi au client..." << std::endl;
        res.set_content(result.dump(), "application/json");
    });

    svr.Post("/predict", [](const httplib::Request& req, httplib::Response& res){
        auto body = req.body;
        Matrice X_input = parse_json_to_matrice(body);
        std::string result = predict_from_input(X_input, nn);
        res.set_content(result,"application/json");
    });

    svr.Post("/saveNN", [](const httplib::Request& req, httplib::Response& res) {
        auto j = json::parse(req.body);
        std::string name = j["name"];
        auto netJson = loadNetwork(name);
        saveNetwork(nn, netJson);
        saveXY(X, Y, name);
        res.set_content(R"({"status":"ok"})","application/json");
    });

    svr.Post("/loadNN", [](const httplib::Request& req, httplib::Response& res) {
        try {
            auto j = json::parse(req.body);
            std::string name = j["name"];

            json netJson = loadNetwork(name);
            createNNFromJson(nn, netJson);

            loadXY(X, Y, name);
            std::cout << "X colonnes: " << X.getColonnes() << ", X lignes: " << X.getLignes() << std::endl;

            res.set_content(R"({"status":"ok"})", "application/json");
        }
        catch (const std::exception& e) {
            std::cerr << "Erreur /loadNN: " << e.what() << std::endl;
            res.status = 500;
            res.set_content(R"({"status":"error","message":")" + std::string(e.what()) + "\"}", "application/json");
        }
    });

    svr.Get("/listNN", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(listNetworks().dump(), "application/json");
    });

    svr.Post("/deleteNetwork", [](const httplib::Request& req, httplib::Response& res) {
        try {
            auto j = json::parse(req.body);

            if (!j.contains("name")) {
                res.status = 400;
                res.set_content(R"({"status":"error","message":"Nom manquant"})", "application/json");
                return;
            }

            std::string name = j["name"];

            if (deleteNetwork(name)) {
                std::cout << "Réseau '" << name << "' supprimé avec succès" << std::endl;
                res.set_content(R"({"status":"ok"})", "application/json");
            } else {
                res.status = 404;
                res.set_content(R"({"status":"error","message":"Réseau non trouvé"})", "application/json");
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Erreur /deleteNetwork: " << e.what() << std::endl;
            res.status = 500;
            res.set_content(R"({"status":"error","message":"Erreur serveur"})", "application/json");
        }
    });

    std::cout << "Serveur démarré sur http://localhost:" << port << std::endl;
    svr.listen("0.0.0.0", port);

    return 0;
}