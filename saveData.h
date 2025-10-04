//
// Created by Trouba Maël on 26/09/2025.
//

#include <json.hpp>
#include "NN.h"

#ifndef DL_SAVEDATA_H
#define DL_SAVEDATA_H

using json = nlohmann::json;

void saveNetwork(std::vector<Layer> nn, const std::string& name);
json listNetworks();
json loadNetwork(const std::string& name);

void saveXY(const Matrice& X, const Matrice& Y, const std::string& filename);
void loadXY(Matrice& X, Matrice& Y, const std::string& filename);
bool deleteNetwork(const std::string& name);

#endif //DL_SAVEDATA_H