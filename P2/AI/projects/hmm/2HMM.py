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
    string = ''
    for i in range(rows):
        for j in range(cols):
            string += ' ' + str(matrix[i][j])
    return string


def alpha_pass(A, B, current_state_dis, sequence):
    result = 0
    alpha = [[B[j][sequence[1]] * current_state_dis[0][j]] for j in range(len(current_state_dis[0]))]
    for t in range(2, sequence[0]+1):
        alpha = [[B[j][sequence[t]] * multiply(transpose(A), alpha)[j][0]] for j in range(len(current_state_dis[0]))]
    for i in range(len(alpha)):
        result += round(alpha[i][0], 6)
    return result


def find_state(A, B, current_state_dis, sequence, nb_em):
    delta_matrice = [[0 for j in range(len(A))] for i in range(nb_em)]
    delta_arg_matrice = [[0 for j in range(len(A))] for i in range(nb_em-1)]
    for i in range(len(A)):
        delta_matrice[0][i] = B[i][sequence[0]]*current_state_dis[0][i]

    for t in range(1, nb_em):
        for i in range(len(A)):
            delta_max = A[0][i] * delta_matrice[t - 1][0] * B[i][sequence[t]]
            arm_max = 0
            for j in range(1, len(A)):
                val = A[j][i]*delta_matrice[t-1][j]*B[i][sequence[t]]
                if val > delta_max:
                    delta_max = val
                    arm_max = j
            delta_matrice[t][i] = delta_max
            delta_arg_matrice[t-1][i] = arm_max


    max_proba_state = delta_matrice[nb_em-1][0]
    max_state = 0
    for i in range(1, len(A)):
        val = delta_matrice[nb_em-1][i]
        if val > max_proba_state:
            max_proba_state = val
            max_state = i
    most_likely_sequence = [max_state]
    for t in range(nb_em-2, -1, -1):
        most_likely_sequence = [delta_arg_matrice[t][most_likely_sequence[0]]] + most_likely_sequence
    return most_likely_sequence

def main():
    A = input()
    A = create_matrix(A)

    B = input()
    B = create_matrix(B)

    current_state_dis = input()
    current_state_dis = create_matrix(current_state_dis)

    sequence = input()
    sequence, nb_em = create_sequence(sequence)

    most_probable_sequence_state = find_state(A, B, current_state_dis,sequence, nb_em)
    print(create_string([most_probable_sequence_state]))


if __name__ == "__main__":
    main()
