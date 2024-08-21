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
    return list_val


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


def main():
    A = input()
    A = create_matrix(A)

    B = input()
    B = create_matrix(B)

    current_state_dis = input()
    current_state_dis = create_matrix(current_state_dis)

    sequence = input()
    sequence = create_sequence(sequence)

    proba = alpha_pass(A, B, current_state_dis, sequence)
    print(str(proba))


if __name__ == "__main__":
    main()
