//
// Created by Trouba Maël on 26/09/2025.
//

#include "saveData.h"
#include <httplib.h>
#include <fstream>
#include <json.hpp>

using json = nlohmann::json;
namespace fs = std::filesystem;
const std::string SAVE_DIR = "saves";

// === Sauvegarde réseau ===
void saveNetwork(std::vector<Layer> nn, const std::string& name) {
    fs::create_directories(SAVE_DIR);

    json j;
    j["input_layer"] = nn.front().W.getColonnes();          // nb neurones entrée
    j["output_layer"] = nn.back().W.getLignes();          // nb neurones sortie

    std::vector<int> hidden;
    for (size_t i = 0; i < nn.size() - 1; ++i) {
        hidden.push_back(nn[i].W.getLignes());
    }
    j["hidden_layers"] = hidden;

    std::ofstream out(SAVE_DIR + "/" + name + ".json");
    out << j.dump(4);
}

// === Liste réseaux sauvegardés ===
json listNetworks() {
    json arr = json::array();
    if (!fs::exists(SAVE_DIR)) return arr;

    for (auto& p : fs::directory_iterator(SAVE_DIR)) {
        if (p.path().extension() == ".json") {
            std::string filename = p.path().filename().string();
            if (filename.find("XY") == std::string::npos) {
                arr.push_back(p.path().stem().string());
            }
        }
    }
    return arr;
}

// === Chargement réseau ===
json loadNetwork(const std::string& name) {
    std::ifstream in(SAVE_DIR + "/" + name + ".json");
    if (!in.is_open()) throw std::runtime_error("Fichier introuvable");

    json j;
    in >> j;

    return j;
}

// === Matrice en json pour sauvegarde des données ===
json matrice_to_json(const Matrice& M) {
    json j = json::array();
    for (int i = 0; i < M.getLignes(); ++i) {
        json row = json::array();
        for (int jcol = 0; jcol < M.getColonnes(); ++jcol) {
            row.push_back(M(i,jcol));
        }
        j.push_back(row);
    }
    return j;
}

// === Sauvegarde XY ===
void saveXY(const Matrice& X, const Matrice& Y, const std::string& filename) {
    json j;
    j["X"] = matrice_to_json(X);
    j["Y"] = matrice_to_json(Y);

    std::string name = filename + "XY";

    std::ofstream file(SAVE_DIR + "/" + name + ".json");
    if (!file.is_open()) {
        std::cerr << "Erreur ouverture fichier " << name << "\n";
        return;
    }
    file << j.dump(4); // dump avec indentation
    file.close();
    std::cout << "Matrices sauvegardées dans " << name << "\n";
}

// === Déserialiser le json ===
Matrice json_to_matrice(const json& j) {
    int rows = j.size();          // nombre de lignes
    int cols = j[0].size();       // nombre de colonnes
    Matrice X(rows, cols);

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            X(r, c) = j[r][c].get<float>();
        }
    }
    return X;
}

// === Charger XY ===
void loadXY(Matrice& X, Matrice& Y, const std::string& filename) {
    std::ifstream file(SAVE_DIR + "/" + filename + "XY.json");
    if (!file.is_open()) throw std::runtime_error("Fichier " + filename + "XY.json introuvable");

    json j;
    file >> j;
    file.close();

    if (!j.contains("X") || !j.contains("Y"))
        throw std::runtime_error("JSON XY invalide");

    X = json_to_matrice(j["X"]);
    Y = json_to_matrice(j["Y"]);
}

// === Suppression réseau ===
bool deleteNetwork(const std::string& name) {
    try {
        std::string json_path = SAVE_DIR + "/" + name + ".json";
        std::string xy_path = SAVE_DIR + "/" + name + "XY.json";

        bool deleted = false;

        // Supprimer le fichier de configuration du réseau
        if (fs::exists(json_path)) {
            fs::remove(json_path);
            deleted = true;
            std::cout << "Fichier " << json_path << " supprimé" << std::endl;
        }

        // Supprimer le fichier de données XY associé
        if (fs::exists(xy_path)) {
            fs::remove(xy_path);
            std::cout << "Fichier " << xy_path << " supprimé" << std::endl;
        }

        return deleted;
    }
    catch (const std::exception& e) {
        std::cerr << "Erreur lors de la suppression: " << e.what() << std::endl;
        return false;
    }
}