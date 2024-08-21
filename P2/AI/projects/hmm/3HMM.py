import copy
import math


def create_matrix(string):
    list_str = string.split()
    list_val = [float(i) for i in list_str]
    rows = int(list_val[0])
    cols = int(list_val[1])
    list_val = list_val[2:]
    matrix = [list_val[i * cols:(i + 1) * cols] for i in range(rows)]
    return matrix


def create_sequence(string):
    list_str = string.split()
    list_val = [int(i) for i in list_str]
    return list_val[1:], list_val[0]


def multiply(A, B):
    if len(A[0]) != len(B):
        print(len(A[0]), len(B))
    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(B[0])):
            value = 0
            for k in range(len(B)):
                value += A[i][k] * B[k][j]
            row.append(value)
        result.append(row)
    return result


def transpose(A):
    transpose = [[A[j][i] for j in range(len(A[0]))] for i in range(len(A))]
    return transpose


def create_string(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    string = str(rows) + ' ' + str(cols)
    for i in range(rows):
        for j in range(cols):
            string += ' ' + str(round(matrix[i][j],6))
    return string


def alpha_pass_scale_up(A, B, pi, sequence, nb_em):
    c = [0 for i in range(nb_em)]
    alpha_list = [[0 for j in range(len(A))] for i in range(nb_em)]
    for j in range(len(A)):
        alpha_list[0][j] = pi[0][j] * B[j][int(sequence[0])]
    c[0] = 1/sum(alpha_list[0])

    for i in range(len(alpha_list[0])):
        alpha_list[0][i] = c[0] * alpha_list[0][i]

    for t in range(1, nb_em):
        for j in range(len(A)):
            for k in range(len(A)):
                alpha_list[t][j] += alpha_list[t - 1][k] * A[k][j] * B[j][int(sequence[t])]
        c[t] = 1/sum(alpha_list[t])
        for i in range(len(alpha_list[0])):
            alpha_list[t][i] = c[t] * alpha_list[t][i]
    return alpha_list, c


def beta_pass(A, B, sequence, nb_em, scale_factor):
    beta = [[1 for j in range(len(A))] for i in range(nb_em)]
    for t in range(nb_em - 2, -1, -1):
        for i in range(len(A)):
            beta[t][i] = 0
            for j in range(len(A[0])):
                beta[t][i] += beta[t + 1][j] * B[j][sequence[t + 1]] * A[i][j]
            beta[t][i] *= scale_factor[t]
    return beta





def baum_welch_update(A, B, current_state_dis, sequence, nb_em):
    ext_trans_matrix = [[0 for j in range(len(A[0]))] for i in range(len(A))]
    ext_em_matrix = [[0 for j in range(len(B[0]))] for i in range(len(B))]
    alpha, scale_factor = alpha_pass_scale_up(A, B, current_state_dis, sequence, nb_em)
    beta = beta_pass(A, B, sequence, nb_em, scale_factor)
    gama = []
    gama_sum = []
    for t in range(nb_em - 1):
        gama_t = []
        gama_sum_t = []
        for i in range(len(A)):
            gama_t_i = []
            gama_sum_t_i = 0
            for j in range(len(A[0])):

                gama_t_i += [alpha[t][i] * A[i][j] * B[j][sequence[t + 1]] * beta[t + 1][j]]
                gama_sum_t_i += alpha[t][i] * A[i][j] * B[j][sequence[t + 1]] * beta[t + 1][j]
            gama_t += [gama_t_i]
            gama_sum_t += [gama_sum_t_i]
        gama += [gama_t]
        gama_sum += [gama_sum_t]

    # Transposez la liste de listes pour regrouper les éléments à la même position
    transposed_list = zip(*gama_sum)

    # Calcul des sommes pour chaque position
    gama_sum_i = [sum(elements) for elements in transposed_list]
    for i in range(len(A)):
        for j in range(len(A[0])):
            sum_t = 0
            for t in range(nb_em - 1):
                sum_t += gama[t][i][j]
            ext_trans_matrix[i][j] = sum_t / gama_sum_i[i]
        for k in range(len(B[0])):
            for t in range(nb_em - 1):
                if sequence[t] == k:
                    ext_em_matrix[i][k] += gama_sum[t][i]
            ext_em_matrix[i][k] /= gama_sum_i[i]
    return ext_trans_matrix, ext_em_matrix, scale_factor, [gama_sum[0]]

def baum_welch(A, B, current_state_dis, sequence, nb_em):

    max_iter = 200
    iter = 1
    oldlogprob = float('-inf')
    ext_trans_matrix, ext_em_matrix, scale_factor, pi = baum_welch_update(A, B, current_state_dis, sequence, nb_em)
    logprob = 0
    for j in range(len(scale_factor)):
        logprob -= math.log(scale_factor[j])
    while (iter < max_iter) and abs((logprob - oldlogprob)/logprob) > 0.00001:
        iter += 1
        oldlogprob = logprob
        ext_trans_matrix, ext_em_matrix, scale_factor, pi = baum_welch_update(ext_trans_matrix, ext_em_matrix, pi, sequence, nb_em)
        logprob = 0
        for j in range(len(scale_factor)):
            logprob -= math.log(scale_factor[j])
    print(iter)
    return ext_trans_matrix, ext_em_matrix


def main():
    A = input()
    A = create_matrix(A)

    B = input()
    B = create_matrix(B)

    current_state_dis = input()
    current_state_dis = create_matrix(current_state_dis)

    sequence = input()
    sequence, nb_em = create_sequence(sequence)

    ext_trans_matrix, ext_em_matrix = baum_welch(A, B, current_state_dis, sequence, nb_em)
    print(create_string(ext_trans_matrix))
    print(create_string(ext_em_matrix))


if __name__ == "__main__":
    main()
