//
// Created by Trouba Maël on 19/09/2025.
//

#ifndef DL_NN_H
#define DL_NN_H

#include <vector>

// Déclaration de la classe Matrice
class Matrice {
private:
    std::vector<std::vector<float>> elements;
    int lignes, colonnes;

public:
    Matrice();
    Matrice(int l, int c);

    int getLignes() const;
    int getColonnes() const;

    float& operator()(int i, int j);
    float operator()(int i, int j) const;

    void afficher() const;

    Matrice operator+(const Matrice& m) const;
    Matrice operator-(const Matrice& m) const;
    Matrice operator-(float val) const;
    Matrice operator*(float val) const;

    friend Matrice operator-(float val, const Matrice& m);
    friend Matrice operator*(float val, const Matrice& m);

    Matrice dot(const Matrice& b) const;
    Matrice multiply_elementwise(const Matrice& m) const;
    Matrice T() const;
    Matrice sum_columns() const;
    Matrice sigmoid() const;
    Matrice sigmoid_prime() const;
    Matrice relu() const;
    Matrice relu_prime() const;
    Matrice softmax() const;
    float logLoss(const Matrice& y_true) const;
};

// Déclaration de la struct Layer
struct Layer {
    Matrice W;
    Matrice b;
    Matrice Z;
    Matrice A;

    Layer(int input_size, int output_size);
};

// Déclaration des fonctions
Matrice forward_pass(const Matrice& X, std::vector<Layer>& reseau);
void backprop(const Matrice& X, const Matrice& Y, float learning_rate, std::vector<Layer>& reseau);
Matrice predict(const Matrice& X, std::vector<Layer>& reseau);
float accuracy(const Matrice& Y_pred, const Matrice& Y_true);

#endif //DL_NN_H