//
// Created by Trouba Maël on 19/09/2025.
//

#include <stdexcept>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include "NN.h"
Matrice::Matrice() {}

Matrice::Matrice(int l, int c) : lignes(l), colonnes(c) {
    elements.resize(lignes, std::vector<float>(colonnes, 0.0f));
}

int Matrice::getLignes() const { return lignes; }
int Matrice::getColonnes() const { return colonnes; }

float& Matrice::operator()(int i, int j) {
    if (i < 0 || i >= lignes || j < 0 || j >= colonnes)
        throw std::out_of_range("Index hors limites dans Matrice::operator()");
    return elements[i][j];
}

float Matrice::operator()(int i, int j) const {
    if (i < 0 || i >= lignes || j < 0 || j >= colonnes)
        throw std::out_of_range("Index hors limites dans Matrice::operator() const");
    return elements[i][j];
}


void Matrice::afficher() const {
    for (int i = 0; i < lignes; ++i) {
        for (int j = 0; j < colonnes; ++j) {
            std::cout << elements[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

Matrice Matrice::operator+(const Matrice& m) const {
    // Cas 1 : dimensions identiques
    if (m.lignes == lignes && m.colonnes == colonnes) {
        Matrice result(lignes, colonnes);
        for (int i = 0; i < lignes; ++i)
            for (int j = 0; j < colonnes; ++j)
                result(i, j) = elements[i][j] + m(i, j);
        return result;
    }

    // Cas 2 : broadcast des biais (m est un vecteur colonne)
    if (m.lignes == lignes && m.colonnes == 1) {
        Matrice result(lignes, colonnes);
        for (int i = 0; i < lignes; ++i)
            for (int j = 0; j < colonnes; ++j)
                result(i, j) = elements[i][j] + m(i, 0);
        return result;
    }

    throw std::invalid_argument("Dimensions incompatibles pour addition de matrices (operator+)");
}

Matrice Matrice::operator-(const Matrice& m) const {
    if (m.lignes != lignes || m.colonnes != colonnes)
        throw std::invalid_argument("Dimensions incompatibles pour soustraction de matrices");

    Matrice result(lignes, colonnes);
    for (int i = 0; i < lignes; ++i) {
        for (int j = 0; j < colonnes; ++j) {
            result(i, j) = elements[i][j] - m(i, j);
        }
    }
    return result;
}

Matrice Matrice::operator-(float val) const {
    Matrice result(lignes, colonnes);
    for (int i = 0; i < lignes; ++i) {
        for (int j = 0; j < colonnes; ++j) {
            result(i, j) = elements[i][j] - val;
        }
    }
    return result;
}

Matrice operator-(float val, const Matrice& m) {
    Matrice result(m.getLignes(), m.getColonnes());
    for (int i = 0; i < m.getLignes(); ++i) {
        for (int j = 0; j < m.getColonnes(); ++j) {
            result(i, j) = val - m(i, j);
        }
    }
    return result;
}

Matrice Matrice::operator*(float val) const {
    Matrice result(lignes, colonnes);
    for (int i = 0; i < lignes; ++i) {
        for (int j = 0; j < colonnes; ++j) {
            result(i, j) = elements[i][j] * val;
        }
    }
    return result;
}

Matrice operator*(float val, const Matrice& m) {
    Matrice result(m.getLignes(), m.getColonnes());
    for (int i = 0; i < m.getLignes(); ++i) {
        for (int j = 0; j < m.getColonnes(); ++j) {
            result(i, j) = val * m(i, j);
        }
    }
    return result;
}

Matrice Matrice::dot(const Matrice& b) const {
    if (colonnes != b.getLignes())
        throw std::invalid_argument("Dimensions incompatibles pour produit matriciel (colonnes de A != lignes de B)");

    Matrice result(lignes, b.getColonnes());

    for (int i = 0; i < lignes; ++i) {
        for (int j = 0; j < b.getColonnes(); ++j) {
            float sum = 0.0f;
            for (int k = 0; k < colonnes; ++k) {
                sum += (*this)(i, k) * b(k, j);
            }
            result(i, j) = sum;
        }
    }

    return result;
}

Matrice Matrice::multiply_elementwise(const Matrice& m) const {
    if (m.lignes != lignes || m.colonnes != colonnes)
        throw std::invalid_argument("Dimensions incompatibles pour multiplication élément par élément");

    Matrice result(lignes, colonnes);

    for (int i = 0; i < lignes; ++i) {
        for (int j = 0; j < colonnes; ++j) {
            result(i, j) = (*this)(i, j) * m(i, j);
        }
    }

    return result;
}

Matrice Matrice::T() const {
    Matrice result(colonnes, lignes);

    for (int i = 0; i < lignes; ++i) {
        for (int j = 0; j < colonnes; ++j) {
            result(j, i) = (*this)(i, j);
        }
    }

    return result;
}

Matrice Matrice::sum_columns() const {
    if (lignes == 0 || colonnes == 0)
        throw std::runtime_error("Impossible de sommer les colonnes d'une matrice vide");

    Matrice result(lignes, 1);
    for (int j = 0; j < colonnes; ++j) {
        for (int i = 0; i < lignes; ++i) {
            result(i, 0) += (*this)(i, j);
        }
    }

    return result;
}

Matrice Matrice::sigmoid() const {
    if (lignes == 0 || colonnes == 0)
        throw std::runtime_error("Sigmoid sur une matrice vide");

    Matrice result(lignes, colonnes);

    for (int i = 0; i < lignes; ++i) {
        for (int j = 0; j < colonnes; ++j) {
            float x = (*this)(i, j);
            result(i, j) = 1.0f / (1.0f + std::exp(-x));
        }
    }
    return result;
}

Matrice Matrice::sigmoid_prime() const {
    if (lignes == 0 || colonnes == 0)
        throw std::runtime_error("Sigmoid_prime sur une matrice vide");

    return (*this).multiply_elementwise(1.0f - (*this));
}

Matrice Matrice::relu() const {
    if (lignes == 0 || colonnes == 0)
        throw std::runtime_error("ReLU sur une matrice vide");

    Matrice result(lignes, colonnes);

    for (int i = 0; i < lignes; ++i) {
        for (int j = 0; j < colonnes; ++j) {
            float x = (*this)(i, j);
            result(i, j) = std::max(0.0f, x);
        }
    }
    return result;
}

Matrice Matrice::relu_prime() const {
    if (lignes == 0 || colonnes == 0)
        throw std::runtime_error("ReLU_prime sur une matrice vide");

    Matrice result(lignes, colonnes);

    for (int i = 0; i < lignes; ++i) {
        for (int j = 0; j < colonnes; ++j) {
            float x = (*this)(i, j);
            result(i, j) = (x > 0.0f) ? 1.0f : 0.0f;
        }
    }
    return result;
}

Matrice Matrice::softmax() const {
    Matrice result(lignes, colonnes);

    for (int j = 0; j < colonnes; ++j) { // chaque échantillon = colonne
        // 1️⃣ Trouver la valeur max pour stabilité numérique
        float max_val = (*this)(0, j);
        for (int i = 1; i < lignes; ++i)
            if ((*this)(i, j) > max_val)
                max_val = (*this)(i, j);

        // 2️⃣ Calcul exponentiel
        float sum_exp = 0.0f;
        for (int i = 0; i < lignes; ++i) {
            result(i, j) = std::exp((*this)(i, j) - max_val);
            sum_exp += result(i, j);
        }

        // 3️⃣ Normalisation
        for (int i = 0; i < lignes; ++i) {
            result(i, j) /= sum_exp;
        }
    }

    return result;
}

float Matrice::logLoss(const Matrice& y_true) const {
    if (y_true.lignes != lignes || y_true.colonnes != colonnes)
        throw std::invalid_argument("Dimensions incompatibles pour logLoss");

    float sum = 0.0f;
    int n = lignes * colonnes;

    for (int i = 0; i < lignes; ++i) {
        for (int j = 0; j < colonnes; ++j) {
            float y = y_true(i, j);
            float p = (*this)(i, j);

            float eps = 1e-9f;
            p = std::min(std::max(p, eps), 1.0f - eps);

            sum += y * std::log(p) + (1 - y) * std::log(1 - p);
        }
    }

    return -sum / n;
}

Layer::Layer(int input_size, int output_size)
    : W(output_size, input_size), b(output_size, 1) {}

Matrice forward_pass(const Matrice& X, std::vector<Layer>& reseau) {
    Matrice A_prev = X;

    for (size_t l = 0; l < reseau.size(); ++l) {
        reseau[l].Z = reseau[l].W.dot(A_prev) + reseau[l].b;

        bool is_last_layer = (l == reseau.size() - 1);

        if (is_last_layer) {
            if (reseau[l].Z.getLignes() == 1) {
                reseau[l].A = reseau[l].Z.sigmoid(); // binaire
            } else {
                reseau[l].A = reseau[l].Z.softmax(); // multi-classes
            }
        } else {
            reseau[l].A = reseau[l].Z.relu();
        }

        A_prev = reseau[l].A;
    }

    return A_prev;
}

void backprop(const Matrice& X, const Matrice& Y, float learning_rate, std::vector<Layer>& reseau) {
    int L = reseau.size();
    int m = X.getColonnes(); // nombre d'exemples = colonnes
    std::vector<Matrice> dZ(L), dW(L), db(L);

    // Dernière couche
    dZ[L - 1] = reseau[L - 1].A - Y;

    // Pour la dernière couche, l'entrée est la sortie de l'avant-dernière couche
    if (L > 1) {
        dW[L - 1] = (1.0f / m) * dZ[L - 1].dot(reseau[L - 2].A.T());
    }
    else {
        // Si une seule couche, l'entrée est X
        dW[L - 1] = (1.0f / m) * dZ[L - 1].dot(X.T());
    }
    db[L - 1] = (1.0f / m) * dZ[L - 1].sum_columns();

    // Couches intermédiaires (de L-2 à 0)
    for (int l = L - 2; l >= 0; --l) {
        dZ[l] = (reseau[l + 1].W.T().dot(dZ[l + 1])).multiply_elementwise(reseau[l].A.relu_prime());

        if (l == 0) {
            dW[l] = (1.0f / m) * dZ[l].dot(X.T());
        }
        else {
            dW[l] = (1.0f / m) * dZ[l].dot(reseau[l - 1].A.T());
        }
        db[l] = (1.0f / m) * dZ[l].sum_columns();
    }

    // Mise à jour
    for (int l = 0; l < L; ++l) {
        reseau[l].W = reseau[l].W - dW[l] * learning_rate;
        reseau[l].b = reseau[l].b - db[l] * learning_rate;
    }
}

Matrice predict(const Matrice& X, std::vector<Layer>& reseau) {
    Matrice A_final = forward_pass(X, reseau);

    if (A_final.getLignes() == 1) {
        // binaire
        for (int i = 0; i < A_final.getLignes(); ++i)
            for (int j = 0; j < A_final.getColonnes(); ++j)
                A_final(i, j) = (A_final(i, j) > 0.5f) ? 1.0f : 0.0f;
    } else {
        // multi-class → one-hot
        for (int j = 0; j < A_final.getColonnes(); ++j) {
            int max_idx = 0;
            float max_val = A_final(0, j);
            for (int i = 1; i < A_final.getLignes(); ++i) {
                if (A_final(i, j) > max_val) {
                    max_val = A_final(i, j);
                    max_idx = i;
                }
            }
            for (int i = 0; i < A_final.getLignes(); ++i) {
                A_final(i, j) = (i == max_idx) ? 1.0f : 0.0f;
            }
        }
    }

    return A_final;
}

// Fonction pour calculer la précision
float accuracy(const Matrice& Y_pred, const Matrice& Y_true) {
    if (Y_pred.getLignes() != Y_true.getLignes() || Y_pred.getColonnes() != Y_true.getColonnes()) {
        throw std::invalid_argument("Dimensions incompatibles pour le calcul de précision");
    }

    int correct = 0;
    int total = Y_pred.getLignes() * Y_pred.getColonnes();

    for (int i = 0; i < Y_pred.getLignes(); ++i) {
        for (int j = 0; j < Y_pred.getColonnes(); ++j) {
            if (std::abs(Y_pred(i, j) - Y_true(i, j)) < 0.5f) {
                correct++;
            }
        }
    }

    return static_cast<float>(correct) / total;
}